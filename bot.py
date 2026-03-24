"""
AI Финансовый Агент v4.2 — исправленные модели и парсинг
"""
import os
import logging
import tempfile
import json
import re
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from groq import Groq
import gspread
from google.oauth2.service_account import Credentials

# ── ENV ─────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_CREDENTIALS")
CHAT_ID = os.getenv("CHAT_ID")

groq_client = Groq(api_key=GROQ_API_KEY)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── DATA CLASSES ─────────────────────────────────────────────────────────────
@dataclass
class Expense:
    amount: float
    currency: str = "UAH"
    category: str = "Другое"
    description: str = ""
    date: datetime = field(default_factory=datetime.now)
    raw_text: str = ""
    confidence: float = 0.0

# ── КОНСТАНТЫ ────────────────────────────────────────────────────────────────
CATEGORIES = ["Еда / продукты", "Транспорт", "Развлечения", "Здоровье / аптека", "Никотин", "Другое"]

CATEGORY_EMOJIS = {
    "Еда / продукты": "🍔", "Транспорт": "🚗", "Развлечения": "🎮",
    "Здоровье / аптека": "💊", "Никотин": "🚬", "Другое": "📦"
}

CURRENCY_SYMBOLS = {"UAH": "₴", "USD": "$", "EUR": "€"}
CURRENCY_NAMES = {
    "грн": "UAH", "гривен": "UAH", "гривна": "UAH", "гривны": "UAH", "₴": "UAH",
    "доллар": "USD", "доллара": "USD", "долларов": "USD", "бакс": "USD", "бакса": "USD", 
    "баксов": "USD", "$": "USD", "usd": "USD",
    "евро": "EUR", "€": "EUR", "eur": "EUR"
}

# Глобальное хранилище
user_contexts: Dict[int, dict] = {}

def get_user_context(chat_id: int):
    if chat_id not in user_contexts:
        user_contexts[chat_id] = {
            "last_expense": None,
            "pending_action": None,
            "history": []
        }
    return user_contexts[chat_id]

# ── AI PARSING ─────────────────────────────────────────────────────────────

# ИСПРАВЛЕНО: устаревшая модель заменена на актуальную
AI_MODEL = "llama-3.3-70b-versatile"  # или "llama-3.1-8b-instant" для скорости

SYSTEM_PROMPT = """Ты — финансовый ассистент. Извлеки ВСЕ траты из текста.

КРИТИЧЕСКИ ВАЖНО:
1. Каждая трата — отдельный объект в массиве items
2. Для КАЖДОЙ траты определи свою категорию и описание по контексту
3. Примеры:
   - "600 на снюс и 850 на топливо" → 
     [{"amount":600,"desc":"снюс","cat":"Никотин"}, {"amount":850,"desc":"топливо","cat":"Транспорт"}]
   - "кофе 150 и булочку 80" → 
     [{"amount":150,"desc":"кофе","cat":"Еда / продукты"}, {"amount":80,"desc":"булочка","cat":"Еда / продукты"}]

Категории: Еда / продукты, Транспорт, Развлечения, Здоровье / аптека, Никотин, Другое

Правила категорий:
- Никотин: снюс, сигареты, вейп, кальян, табак
- Транспорт: бензин, топливо, заправка, такси, метро, мойка машины, парковка
- Еда: кофе, ресторан, продукты, еда, булочка, пицца
- Развлечения: кино, игры, steam, подписка
- Здоровье: аптека, лекарства, врач

Верни строго JSON:
{
  "intent": "expense",
  "items": [
    {
      "amount": число,
      "currency": "UAH|USD|EUR", 
      "category": "категория",
      "description": "конкретное описание из текста",
      "date": "YYYY-MM-DD"
    }
  ],
  "confidence": 0.0-1.0
}"""

async def ai_parse(text: str, chat_id: int) -> Dict:
    """AI-парсинг с fallback"""
    ctx = get_user_context(chat_id)
    
    # Добавляем историю для контекста
    history = ""
    if ctx["history"]:
        recent = ctx["history"][-2:]
        history = "Последние сообщения:\n" + "\n".join([f"- {h['text'][:50]}" for h in recent])
    
    prompt = f"""{SYSTEM_PROMPT}

{history}

Текущее сообщение: "{text}"

Верни только JSON."""

    try:
        response = groq_client.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Сохраняем историю
        ctx["history"].append({
            "text": text,
            "parsed": result.get("items", []),
            "time": datetime.now().isoformat()
        })
        if len(ctx["history"]) > 5:
            ctx["history"] = ctx["history"][-5:]
        
        return result
        
    except Exception as e:
        logger.error(f"AI parse error: {e}")
        return smart_fallback_parse(text)

def smart_fallback_parse(text: str) -> Dict:
    """Умный fallback парсер — ищет пары сумма+описание"""
    text_lower = text.lower()
    items = []
    
    # Паттерн: сумма + валюта + предлог + описание
    # "600 гривен на снюсик", "850 на топливо", "150 за кофе"
    pattern = r'(\d+(?:[.,]\d+)?)\s*(?:грн|гривен|₴|долларов|баксов|\$|евро|€)?\s*(?:на|за|в|для|в\s+)?\s*([^,\.и\d][^,\.]*?)(?=\s*(?:и|,|\.|\d|$))'
    
    matches = list(re.finditer(pattern, text_lower))
    
    for match in matches:
        amount_str = match.group(1).replace(",", ".")
        amount = float(amount_str)
        desc_raw = match.group(2).strip()
        
        # Очищаем описание
        desc = re.sub(r'\s+(?:и|а|но|или)\s+.*$', '', desc_raw).strip()
        if len(desc) > 30:
            desc = desc[:30]
        
        # Определяем валюту
        currency = "UAH"
        if any(x in text_lower[max(0, match.start()-10):match.start()] for x in ["доллар", "бакс", "$", "usd"]):
            currency = "USD"
        elif any(x in text_lower[max(0, match.start()-10):match.start()] for x in ["евро", "€", "eur"]):
            currency = "EUR"
        
        # Определяем категорию по описанию
        category = "Другое"
        desc_lower = desc.lower()
        
        if any(w in desc_lower for w in ["снюс", "сигарет", "вейп", "кальян", "табак", "никотин"]):
            category = "Никотин"
        elif any(w in desc_lower for w in ["бензин", "топливо", "заправка", "такси", "метро", "мойка", "парковка", "транспорт", "машин"]):
            category = "Транспорт"
        elif any(w in desc_lower for w in ["кофе", "кафе", "ресторан", "еду", "пицца", "булочка", "продукты", "обед", "ужин", "завтрак", "гамбургер", "шаурма"]):
            category = "Еда / продукты"
        elif any(w in desc_lower for w in ["кино", "игра", "steam", "подписка", "бар", "клуб", "развлеч"]):
            category = "Развлечения"
        elif any(w in desc_lower for w in ["аптека", "лекарств", "врач", "больница", "анализ", "здоровье"]):
            category = "Здоровье / аптека"
        
        items.append({
            "amount": amount,
            "currency": currency,
            "category": category,
            "description": desc.capitalize(),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "confidence": 0.7
        })
    
    # Если не нашли паттерном, ищем просто числа
    if not items:
        numbers = re.findall(r'(\d+(?:[.,]\d+)?)\s*(?:грн|₴)?', text_lower)
        if numbers:
            # Одно число без контекста
            items.append({
                "amount": float(numbers[0].replace(",", ".")),
                "currency": "UAH",
                "category": "Другое",
                "description": "трата",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "confidence": 0.5
            })
    
    return {
        "intent": "expense",
        "items": items,
        "confidence": 0.7 if items else 0.3
    }

# ── DATE PARSING ─────────────────────────────────────────────────────────────

def parse_date(text: str) -> datetime:
    """Парсер дат"""
    text_lower = text.lower()
    now = datetime.now()
    
    if any(w in text_lower for w in ["сегодня", "сьогодні"]):
        return now
    if any(w in text_lower for w in ["вчера", "вчора"]):
        return now - timedelta(days=1)
    if "позавчера" in text_lower:
        return now - timedelta(days=2)
    
    # "N дней назад"
    match = re.search(r'(\d+)\s*дн[яеяй]\s*назад', text_lower)
    if match:
        return now - timedelta(days=int(match.group(1)))
    
    # Дни недели
    days = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
    for i, day in enumerate(days):
        if day in text_lower:
            diff = (now.weekday() - i) % 7
            if diff == 0:
                diff = 7
            return now - timedelta(days=diff)
    
    return now

# ── GOOGLE SHEETS ────────────────────────────────────────────────────────────

_gs_client = None
_spreadsheet = None

def get_gs_client():
    global _gs_client
    if _gs_client is None:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        if GOOGLE_CREDENTIALS:
            creds = Credentials.from_service_account_info(json.loads(GOOGLE_CREDENTIALS), scopes=scopes)
        else:
            creds = Credentials.from_service_account_file("credentials.json", scopes=scopes)
        _gs_client = gspread.authorize(creds)
    return _gs_client

def get_spreadsheet():
    global _spreadsheet
    if _spreadsheet is None:
        _spreadsheet = get_gs_client().open_by_key(GOOGLE_SHEET_ID)
    return _spreadsheet

def get_sheet(name="Expenses"):
    sp = get_spreadsheet()
    try:
        return sp.worksheet(name)
    except:
        return sp.add_worksheet(title=name, rows=1000, cols=10)

def init_db():
    sp = get_spreadsheet()
    sheets = ["Expenses", "Settings"]
    for s in sheets:
        try:
            sp.worksheet(s)
        except:
            ws = sp.add_worksheet(title=s, rows=1000, cols=10)
            if s == "Expenses":
                ws.append_row(["Дата", "Сумма", "Валюта", "Категория", "Описание", "Raw", "UserID", "Time"])

def save_expense(exp: Expense, user_id: int):
    ws = get_sheet("Expenses")
    ws.append_row([
        exp.date.strftime("%d.%m.%Y %H:%M"),
        exp.amount,
        exp.currency,
        exp.category,
        exp.description,
        exp.raw_text[:100],
        str(user_id),
        datetime.now().isoformat()
    ])

def get_expenses(user_id: int, days: int = 30) -> List[Dict]:
    ws = get_sheet("Expenses")
    try:
        records = ws.get_all_records()
    except:
        return []
    
    cutoff = datetime.now() - timedelta(days=days)
    result = []
    
    for r in records:
        if str(r.get("UserID")) != str(user_id):
            continue
        try:
            d = datetime.strptime(r["Дата"][:10], "%d.%m.%Y")
            if d >= cutoff:
                result.append(r)
        except:
            continue
    return result

def save_setting(user_id: int, key: str, value: str):
    ws = get_sheet("Settings")
    try:
        records = ws.get_all_records()
        for i, r in enumerate(records, start=2):
            if str(r.get("UserID")) == str(user_id) and r.get("Key") == key:
                ws.update_cell(i, 3, value)
                return
    except:
        pass
    ws.append_row([str(user_id), key, value])

def get_settings(user_id: int) -> Dict:
    try:
        ws = get_sheet("Settings")
        records = ws.get_all_records()
        return {r["Key"]: r["Value"] for r in records if str(r.get("UserID")) == str(user_id)}
    except:
        return {}

# ── PROCESSING ───────────────────────────────────────────────────────────────

async def process_expenses(update: Update, context: ContextTypes.DEFAULT_TYPE, items: List[Dict], raw_text: str):
    """Обработка списка трат"""
    chat_id = update.effective_chat.id
    
    if not items:
        await update.message.reply_text("🤔 Не нашёл сумму. Попробуй: «Снюс 800»")
        return
    
    lines = ["✅ *Записано!*\n"]
    total_uah = 0
    
    for item in items:
        # Парсим дату
        date_str = item.get("date", datetime.now().strftime("%Y-%m-%d"))
        try:
            exp_date = datetime.fromisoformat(date_str)
        except:
            exp_date = parse_date(raw_text)
        
        exp = Expense(
            amount=float(item.get("amount", 0)),
            currency=item.get("currency", "UAH"),
            category=item.get("category", "Другое"),
            description=item.get("description", "трата"),
            date=exp_date,
            raw_text=raw_text
        )
        
        save_expense(exp, chat_id)
        
        emoji = CATEGORY_EMOJIS.get(exp.category, "💰")
        symbol = CURRENCY_SYMBOLS.get(exp.currency, "₴")
        lines.append(f"{emoji} {exp.description}: *{exp.amount:,.0f} {symbol}* ({exp.category})")
        
        if exp.currency == "UAH":
            total_uah += exp.amount
    
    if len(items) > 1:
        lines.append(f"\n💰 *Итого: {total_uah:,.0f} ₴*")
    
    # Проверка бюджета
    settings = get_settings(chat_id)
    budget = float(settings.get("monthly_budget", 0))
    if budget > 0:
        month_exp = get_expenses(chat_id, 30)
        spent = sum(float(r["Сумма"]) for r in month_exp if r.get("Валюта") == "UAH")
        pct = min(int(spent / budget * 100), 100)
        if pct >= 90:
            lines.append(f"\n🔴 *Бюджет на {pct}%!*")
        elif pct >= 70:
            lines.append(f"\n🟡 Бюджет на {pct}%")
    
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def process_intent(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    """Главный обработчик"""
    chat_id = update.effective_chat.id
    ctx = get_user_context(chat_id)
    
    # Проверяем ожидание подтверждения категории
    if ctx["pending_action"] == "confirm_cat" and ctx["last_expense"]:
        cat = text.strip()
        if cat in CATEGORIES:
            ctx["last_expense"].category = cat
            save_expense(ctx["last_expense"], chat_id)
            emoji = CATEGORY_EMOJIS.get(cat, "💰")
            await update.message.reply_text(
                f"✅ {emoji} {cat}: *{ctx['last_expense'].amount:,.0f} ₴*",
                parse_mode="Markdown"
            )
            ctx["pending_action"] = None
            ctx["last_expense"] = None
            return
    
    # AI парсинг
    parsed = await ai_parse(text, chat_id)
    intent = parsed.get("intent", "unknown")
    items = parsed.get("items", [])
    
    if intent == "expense":
        # Если уверенность низкая и одна трата — спрашиваем категорию
        if parsed.get("confidence", 1) < 0.6 and len(items) == 1:
            exp = Expense(
                amount=items[0]["amount"],
                currency=items[0].get("currency", "UAH"),
                description=items[0].get("description", "трата")
            )
            ctx["last_expense"] = exp
            ctx["pending_action"] = "confirm_cat"
            
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton(cat, callback_data=f"cat_{cat}")] 
                for cat in CATEGORIES
            ])
            await update.message.reply_text(
                f"🤔 *{exp.amount:,.0f} ₴* — {exp.description}\nКакая категория?",
                parse_mode="Markdown",
                reply_markup=kb
            )
            return
        
        await process_expenses(update, context, items, text)
    
    elif intent == "budget_set":
        amt = items[0]["amount"] if items else 0
        save_setting(chat_id, "monthly_budget", str(amt))
        await update.message.reply_text(f"💰 Бюджет: *{amt:,.0f} ₴/мес*", parse_mode="Markdown")
    
    elif intent == "salary_set":
        day = 25  # default
        amt = None
        nums = re.findall(r'\d+', text)
        if nums:
            day = int(nums[0])
            if len(nums) > 1:
                amt = float(nums[1])
        save_setting(chat_id, "salary_day", str(day))
        if amt:
            save_setting(chat_id, "salary_amount", str(amt))
        msg = f"💵 Зарплата: *{day}-е число*"
        if amt:
            msg += f", {amt:,.0f} ₴"
        await update.message.reply_text(msg, parse_mode="Markdown")
    
    else:
        # Запрос статистики или непонятно
        if any(w in text.lower() for w in ["сколько", "статус", "отчёт", "анализ", "тратил"]):
            await show_stats(update, context, text)
        else:
            await update.message.reply_text(
                "🤔 Не понял. Примеры:\n"
                "• «Снюс 800, бензин 1200»\n"
                "• «Вчера кофе 150»\n"
                "• «Бюджет 20000»"
            )

async def show_stats(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    """Показ статистики"""
    chat_id = update.effective_chat.id
    
    # Определяем период
    days = 30
    if any(w in text.lower() for w in ["недел", "week", "7 дн"]):
        days = 7
    
    expenses = get_expenses(chat_id, days)
    if not expenses:
        await update.message.reply_text("📭 Нет данных.")
        return
    
    total = sum(float(r["Сумма"]) for r in expenses)
    by_cat = defaultdict(float)
    for r in expenses:
        by_cat[r["Категория"]] += float(r["Сумма"])
    
    period = "неделю" if days == 7 else "месяц"
    lines = [f"📊 *За {period}*\n", f"💰 *{total:,.0f} ₴* всего\n", "*По категориям:*"]
    
    for cat, amt in sorted(by_cat.items(), key=lambda x: -x[1]):
        pct = int(amt / total * 100) if total > 0 else 0
        emoji = CATEGORY_EMOJIS.get(cat, "•")
        lines.append(f"{emoji} {cat}: {amt:,.0f} ₴ ({pct}%)")
    
    settings = get_settings(chat_id)
    budget = float(settings.get("monthly_budget", 0))
    if budget > 0 and days == 30:
        left = budget - total
        pct = min(int(total / budget * 100), 100)
        lines.append(f"\n💳 Бюджет: {pct}%\n📉 Осталось: {left:,.0f} ₴")
    
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

# ── HANDLERS ─────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *AI Финансовый Агент v4.2*\n\n"
        "Просто напиши или скажи голосом:\n"
        "• «600 на снюс и 850 на топливо»\n"
        "• «Вчера кофе 150, сегодня такси 80»\n"
        "• «Сколько потратил на неделе?»\n\n"
        "Я пойму контекст и разделю на категории!",
        parse_mode="Markdown",
        reply_markup=ReplyKeyboardMarkup([
            [KeyboardButton("📊 Статус"), KeyboardButton("💰 Бюджет")],
            [KeyboardButton("📈 Анализ"), KeyboardButton("❓ Помощь")]
        ], resize_keyboard=True)
    )

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка голоса"""
    msg = await update.message.reply_text("🎙 Распознаю...")
    
    try:
        file = await context.bot.get_file(update.message.voice.file_id)
        
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            await file.download_to_drive(tmp.name)
            path = tmp.name
        
        with open(path, "rb") as f:
            transcript = groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                language="ru",
                response_format="text"
            )
        
        os.unlink(path)
        
        text = str(transcript).strip()
        logger.info(f"Voice: {text}")
        
        await msg.edit_text(f"📝 _{text}_", parse_mode="Markdown")
        
        # Обрабатываем распознанный текст
        await process_intent(update, context, text)
        
    except Exception as e:
        logger.error(f"Voice error: {e}")
        await msg.edit_text("❌ Ошибка распознавания. Попробуй текстом.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текста и кнопок"""
    text = update.message.text
    
    # Кнопки меню
    if text == "📊 Статус":
        await show_stats(update, context, "месяц")
        return
    if text == "📈 Анализ":
        await show_stats(update, context, "неделя")
        return
    if text == "💰 Бюджет":
        await update.message.reply_text("Напиши: «Бюджет 25000»")
        return
    if text == "❓ Помощь":
        await update.message.reply_text(
            "📝 *Примеры сообщений:*\n\n"
            "• «Снюс 800» — одна трата\n"
            "• «600 на снюс и 850 на бензин» — две траты\n"
            "• «Вчера кофе 150» — с датой\n"
            "• «Бюджет 20000» — установить бюджет\n"
            "• «Зарплата 25 числа» — день зарплаты\n"
            "• «Сколько потратил?» — статистика",
            parse_mode="Markdown"
        )
        return
    
    await process_intent(update, context, text)

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Inline кнопки"""
    query = update.callback_query
    await query.answer()
    data = query.data
    
    if data.startswith("cat_"):
        cat = data[4:]
        chat_id = query.message.chat_id
        ctx = get_user_context(chat_id)
        
        if ctx["pending_action"] == "confirm_cat" and ctx["last_expense"]:
            ctx["last_expense"].category = cat
            save_expense(ctx["last_expense"], chat_id)
            emoji = CATEGORY_EMOJIS.get(cat, "💰")
            
            await query.edit_message_text(
                f"✅ {emoji} {cat}: *{ctx['last_expense'].amount:,.0f} ₴* ({ctx['last_expense'].description})",
                parse_mode="Markdown"
            )
            ctx["pending_action"] = None
            ctx["last_expense"] = None

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    init_db()
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    logger.info(f"AI Агент v4.2 запущен! Модель: {AI_MODEL}")
    app.run_polling()

if __name__ == "__main__":
    main()
