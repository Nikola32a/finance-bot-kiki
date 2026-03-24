"""
AI Финансовый Агент v4.0 — контекстное понимание, память, проактивность
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
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler, ConversationHandler
from groq import Groq
import gspread
from google.oauth2.service_account import Credentials
import httpx

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
    confidence: float = 0.0  # уверенность AI в категории

@dataclass
class Debt:
    id: str
    name: str
    amounts: List[Dict[str, Any]]  # [{"amount": 500, "currency": "USD"}]
    date: datetime
    note: str = ""
    reminder_date: Optional[datetime] = None

@dataclass
class UserContext:
    last_expense: Optional[Expense] = None
    last_debt: Optional[Debt] = None
    conversation_history: List[Dict] = field(default_factory=list)
    preferences: Dict = field(default_factory=dict)
    pending_action: Optional[str] = None  # "awaiting_category", "awaiting_amount" и т.д.

# ── КОНСТАНТЫ ────────────────────────────────────────────────────────────────
CATEGORIES = ["Еда / продукты", "Транспорт", "Развлечения", "Здоровье / аптека", "Никотин", "Другое"]

CURRENCY_SYMBOLS = {"UAH": "₴", "USD": "$", "EUR": "€", "GBP": "£"}
CURRENCY_NAMES = {"грн": "UAH", "гривен": "UAH", "гривна": "UAH", "₴": "UAH",
                  "доллар": "USD", "доллара": "USD", "долларов": "USD", "баксов": "USD", "$": "USD",
                  "евро": "EUR", "€": "EUR"}

MONTH_NAMES = ["Январь","Февраль","Март","Апрель","Май","Июнь",
               "Июль","Август","Сентябрь","Октябрь","Ноябрь","Декабрь"]

# Глобальное хранилище контекста пользователей
user_contexts: Dict[int, UserContext] = {}

# ── AI CORE SYSTEM ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Ты — интеллектуальный финансовый ассистент. Твоя задача — понимать естественный язык пользователя и извлекать финансовые данные.

ПРАВИЛА ПОНИМАНИЯ:
1. Дата: "вчера", "позавчера", "3 дня назад", "15 марта", "в прошлый понедельник" → преобразуй в точную дату
2. Сумма: распознавай числа прописью ("тысяча двести", "полторы тысячи")
3. Валюта: определяй по контексту (грн, доллары, баксы, евро, ₴, $, €)
4. Категория: анализируй описание глубоко, не просто по ключевым словам:
   - "Поехал на море, бензин, дорога, еда в дороге" → раздели на Транспорт и Еда
   - "Купил снюс и кофе" → два разных товара, возможно разные категории
   - "Steam, игры, подписка" → Развлечения
   - "Врач, анализы, таблетки" → Здоровье

5. Контекст долгов:
   - "Дал в долг Саше 5000" → новый долг
   - "Саша вернул 3000" → частичное погашение
   - "Вернул долг Саше" → полное погашение (если сумма не указана — уточни)

6. Бюджет и зарплата:
   - "Бюджет на месяц 20000" → установка бюджета
   - "Зарплата 25 числа 35000" → установка зарплаты
   - "Сколько осталось до зарплаты?" → запрос статуса

7. Аналитика:
   - "Сколько потратил на этой неделе?" → отчёт за неделю
   - "Где больше всего денег уходит?" → анализ категорий
   - "Сравни с прошлым месяцем" → сравнительный анализ

ВСЕГДА отвечай в формате JSON с полями:
{
  "intent": "expense|debt_new|debt_payment|budget_set|salary_set|query_report|query_status|clarification_needed|other",
  "data": { ... },
  "confidence": 0.0-1.0,
  "clarification_question": "..." или null,
  "suggested_actions": ["..."]
}"""

def get_user_context(chat_id: int) -> UserContext:
    if chat_id not in user_contexts:
        user_contexts[chat_id] = UserContext()
    return user_contexts[chat_id]

async def ai_understand(text: str, chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> Dict:
    """Главная функция AI-понимания с контекстом"""
    user_ctx = get_user_context(chat_id)
    
    # Формируем контекст разговора
    history_text = ""
    if user_ctx.conversation_history:
        recent = user_ctx.conversation_history[-3:]
        history_text = "\n".join([f"User: {h['user']}\nAI: {h['ai']}" for h in recent])
    
    # Добавляем текущий контекст (что ждём от пользователя)
    pending_info = ""
    if user_ctx.pending_action:
        pending_info = f"\nОЖИДАЕТСЯ: {user_ctx.pending_action}"
    
    prompt = f"""{SYSTEM_PROMPT}

ИСТОРИЯ РАЗГОВОРА:
{history_text}
{pending_info}

ТЕКУЩЕЕ СООБЩЕНИЕ: "{text}"

Проанализируй и верни JSON. Если не уверен в категории или сумме — задай уточняющий вопрос."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",  # более умная модель для сложного понимания
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Сохраняем в историю
        user_ctx.conversation_history.append({
            "user": text,
            "ai": result.get("clarification_question") or result.get("intent"),
            "timestamp": datetime.now().isoformat()
        })
        
        # Ограничиваем историю
        if len(user_ctx.conversation_history) > 10:
            user_ctx.conversation_history = user_ctx.conversation_history[-10:]
        
        return result
        
    except Exception as e:
        logger.error(f"AI understand error: {e}")
        # Fallback на простое распознавание
        return fallback_parse(text)

def fallback_parse(text: str) -> Dict:
    """Резервный парсер если AI недоступен"""
    numbers = re.findall(r'\d+(?:[.,]\d+)?', text)
    amount = float(numbers[0].replace(",", ".")) if numbers else 0
    
    # Определяем категорию по простым правилам
    category = "Другое"
    text_lower = text.lower()
    
    category_keywords = {
        "Еда / продукты": ["еда", "продукты", "кафе", "ресторан", "пицца", "суши", "кофе", "атб", "сільпо"],
        "Транспорт": ["такси", "бензин", "метро", "автобус", "поезд", "uber", "bolt"],
        "Развлечения": ["кино", "игра", "steam", "netflix", "подписка", "бар"],
        "Здоровье / аптека": ["аптека", "врач", "лекарства", "анализы", "стоматолог"],
        "Никотин": ["снюс", "сигареты", "вейп", "кальян"]
    }
    
    for cat, keywords in category_keywords.items():
        if any(kw in text_lower for kw in keywords):
            category = cat
            break
    
    return {
        "intent": "expense",
        "data": {
            "amount": amount,
            "currency": "UAH",
            "category": category,
            "description": text[:50],
            "date": datetime.now().isoformat()
        },
        "confidence": 0.5,
        "clarification_question": None,
        "suggested_actions": []
    }

async def ai_converse(chat_id: int, message: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Свободный разговор с AI о финансах"""
    user_ctx = get_user_context(chat_id)
    
    # Получаем статистику для контекста
    stats = await get_user_stats(chat_id)
    
    prompt = f"""Ты — дружелюбный финансовый советник. Отвечай кратко, по-человечески, с эмодзи.

Контекст пользователя:
- Потрачено в этом месяце: {stats.get('month_total', 0)} ₴
- Топ категория: {stats.get('top_category', 'нет данных')}
- До зарплаты: {stats.get('days_to_salary', '?')} дней

История: {[h['user'] for h in user_ctx.conversation_history[-3:]]}

Вопрос: {message}

Дай полезный, конкретный ответ. Если спрашивает совета — дай 2-3 конкретных действия."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"AI converse error: {e}")
        return "Извини, не смог обработать запрос. Попробуй переформулировать."

# ── SMART PARSING ────────────────────────────────────────────────────────────

def parse_date(date_str: str) -> datetime:
    """Умный парсер дат на русском/украинском"""
    now = datetime.now()
    text = date_str.lower().strip()
    
    # Сегодня, вчера, позавчера
    if text in ["сегодня", "сьогодні"]:
        return now
    if text in ["вчера", "вчора"]:
        return now - timedelta(days=1)
    if text in ["позавчера", "позавчора"]:
        return now - timedelta(days=2)
    
    # "3 дня назад", "неделю назад"
    num_map = {"один": 1, "два": 2, "три": 3, "четыре": 4, "пять": 5, 
               "неделю": 7, "месяц": 30}
    match = re.search(r'(\w+)\s+(?:день|дня|дней|недел[юья]|месяц)\s+назад', text)
    if match:
        word = match.group(1)
        days = num_map.get(word, int(word) if word.isdigit() else 0)
        return now - timedelta(days=days)
    
    # "в понедельник", "во вторник"
    days = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
    for i, day in enumerate(days):
        if day in text:
            target_day = i
            current_day = now.weekday()
            diff = (current_day - target_day) % 7
            if diff == 0:
                diff = 7  # прошлая неделя
            return now - timedelta(days=diff)
    
    # Форматы дат
    formats = ["%d.%m.%Y", "%d.%m", "%d %B", "%d %b"]
    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt)
            if parsed.year == 1900:
                parsed = parsed.replace(year=now.year)
            return parsed
        except:
            continue
    
    return now

def parse_amount(text: str) -> tuple[float, str]:
    """Парсит сумму и валюту, включая числа прописью"""
    text_lower = text.lower()
    
    # Определяем валюту
    currency = "UAH"
    for word, code in CURRENCY_NAMES.items():
        if word in text_lower:
            currency = code
            break
    
    # Числа прописью
    num_words = {
        'сто': 100, 'двести': 200, 'триста': 300, 'четыреста': 400, 'пятьсот': 500,
        'тысяча': 1000, 'тысячи': 1000, 'тысяч': 1000, 'полтысячи': 500,
        'полторы тысячи': 1500, 'полторы': 1.5,
        'две тысячи': 2000, 'три тысячи': 3000, 'пять тысяч': 5000,
        'десять': 10, 'двадцать': 20, 'тридцать': 30, 'сорок': 40,
        'пятьдесят': 50, 'шестьдесят': 60, 'семьдесят': 70,
        'восемьдесят': 80, 'девяносто': 90
    }
    
    # Ищем числа прописью
    for word, value in sorted(num_words.items(), key=lambda x: -len(x[0])):
        if word in text_lower:
            # Проверяем множители ("тысяча", "тысячи")
            if 'тысяч' in word and value >= 1000:
                # Ищем множитель перед словом
                prev_text = text_lower[:text_lower.find(word)]
                mult_match = re.search(r'(\d+)\s*$', prev_text)
                if mult_match:
                    return float(mult_match.group(1)) * 1000, currency
            return float(value), currency
    
    # Обычные числа
    numbers = re.findall(r'\d+(?:[.,]\d+)?', text)
    if numbers:
        return float(numbers[0].replace(",", ".")), currency
    
    return 0, currency

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
        _gs_client = get_gs_client()
        _spreadsheet = _gs_client.open_by_key(GOOGLE_SHEET_ID)
    return _spreadsheet

def get_worksheet(name="Expenses"):
    sp = get_spreadsheet()
    try:
        return sp.worksheet(name)
    except:
        return sp.add_worksheet(title=name, rows=1000, cols=10)

def init_sheets():
    """Инициализация структуры таблиц"""
    sp = get_spreadsheet()
    
    # Основная таблица расходов
    try:
        ws = sp.worksheet("Expenses")
    except:
        ws = sp.add_worksheet(title="Expenses", rows=1000, cols=10)
        ws.append_row(["Дата", "Сумма", "Валюта", "Категория", "Описание", "Raw Text", "User ID", "Timestamp"])
    
    # Таблица долгов
    try:
        ws = sp.worksheet("Debts")
    except:
        ws = sp.add_worksheet(title="Debts", rows=500, cols=10)
        ws.append_row(["ID", "Кому", "Суммы (JSON)", "Дата", "Статус", "Примечание", "User ID"])
    
    # Настройки пользователей
    try:
        ws = sp.worksheet("Settings")
    except:
        ws = sp.add_worksheet(title="Settings", rows=500, cols=5)
        ws.append_row(["User ID", "Key", "Value", "Updated"])

# ── DATA OPERATIONS ───────────────────────────────────────────────────────────

def save_expense_ai(expense: Expense, user_id: int):
    """Сохранение с AI-контекстом"""
    ws = get_worksheet("Expenses")
    ws.append_row([
        expense.date.strftime("%d.%m.%Y %H:%M"),
        expense.amount,
        expense.currency,
        expense.category,
        expense.description,
        expense.raw_text,
        str(user_id),
        datetime.now().isoformat()
    ])

def get_expenses(user_id: int, days: int = 30) -> List[Dict]:
    """Получение расходов с фильтрацией"""
    ws = get_worksheet("Expenses")
    records = ws.get_all_records()
    
    cutoff = datetime.now() - timedelta(days=days)
    result = []
    
    for r in records:
        if str(r.get("User ID")) != str(user_id):
            continue
        try:
            date = datetime.strptime(r["Дата"][:10], "%d.%m.%Y")
            if date >= cutoff:
                result.append(r)
        except:
            continue
    
    return result

async def get_user_stats(chat_id: int) -> Dict:
    """Статистика для AI-контекста"""
    expenses = get_expenses(chat_id, 30)
    
    if not expenses:
        return {"month_total": 0, "top_category": "нет данных", "days_to_salary": None}
    
    total = sum(float(r["Сумма"]) for r in expenses)
    
    by_cat = defaultdict(float)
    for r in expenses:
        by_cat[r["Категория"]] += float(r["Сумма"])
    
    top_cat = max(by_cat.items(), key=lambda x: x[1])[0] if by_cat else "нет данных"
    
    # Дни до зарплаты
    settings = get_user_settings(chat_id)
    salary_day = settings.get("salary_day")
    days_to_salary = None
    if salary_day:
        now = datetime.now()
        if now.day < int(salary_day):
            days_to_salary = int(salary_day) - now.day
        else:
            next_month = (now.replace(day=1) + timedelta(days=32)).replace(day=1)
            next_salary = next_month.replace(day=min(int(salary_day), 28))
            days_to_salary = (next_salary - now).days
    
    return {
        "month_total": total,
        "top_category": top_cat,
        "days_to_salary": days_to_salary,
        "by_category": dict(by_cat)
    }

def get_user_settings(user_id: int) -> Dict:
    """Получение настроек пользователя"""
    try:
        ws = get_worksheet("Settings")
        records = ws.get_all_records()
        settings = {}
        for r in records:
            if str(r.get("User ID")) == str(user_id):
                settings[r["Key"]] = r["Value"]
        return settings
    except:
        return {}

def save_user_setting(user_id: int, key: str, value: str):
    """Сохранение настройки"""
    ws = get_worksheet("Settings")
    # Ищем существующую запись
    records = ws.get_all_records()
    for i, r in enumerate(records, start=2):
        if str(r.get("User ID")) == str(user_id) and r.get("Key") == key:
            ws.update_cell(i, 3, value)
            ws.update_cell(i, 4, datetime.now().isoformat())
            return
    
    # Новая запись
    ws.append_row([str(user_id), key, value, datetime.now().isoformat()])

# ── AI ANALYTICS ─────────────────────────────────────────────────────────────

async def generate_insight(chat_id: int) -> str:
    """Генерация умных инсайтов через AI"""
    stats = await get_user_stats(chat_id)
    expenses = get_expenses(chat_id, 30)
    
    if not expenses:
        return "📭 Пока недостаточно данных для анализа."
    
    # Формируем данные для анализа
    data_for_ai = {
        "total": stats["month_total"],
        "by_category": stats["by_category"],
        "recent_expenses": expenses[-10:],
        "days_to_salary": stats["days_to_salary"]
    }
    
    prompt = f"""Проанализируй траты пользователя и дай 1-2 конкретных, actionable совета.
Будь дружелюбным, используй эмодзи, конкретные цифры.

Данные: {json.dumps(data_for_ai, ensure_ascii=False)}

Примеры хороших ответов:
"🍔 На еду уходит 45% бюджета — это много! Попробуй готовить дома 2 раза в неделю, сэкономишь ~2000₴"
"🚬 Траты на никотин: 3500₴/мес = 42000₴/год. За эти деньги можно..."
"📉 До зарплаты 5 дней, осталось 15% бюджета — пора сократить траты"

Дай 1-2 совета максимум. Будь конкретным."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Insight generation error: {e}")
        return "💡 Продолжай записывать траты — скоро смогу дать персональные советы!"

# ── HANDLERS ───────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Старт с AI-приветствием"""
    chat_id = update.effective_chat.id
    
    welcome_text = """👋 Привет! Я твой AI-финансовый ассистент.

🧠 *Что я умею:*
• Понимаю естественный язык: "Вчера вечером потратил 1200 на бензин"
• Запоминаю контекст: "Ещё 300 за кофе" — пойму, что это продолжение
• Анализирую: "Где больше всего трачу?" — покажу инсайты
• Напоминаю: "Через 3 дня зарплата, осталось 10% бюджета"

💡 *Просто напиши:* "Снюс 800", "Дал в долг Саше 5000", или "Сколько потратил на этой неделе?"

Начни с любой траты!"""

    kb = ReplyKeyboardMarkup([
        [KeyboardButton("📊 Анализ"), KeyboardButton("💰 Статус")],
        [KeyboardButton("💡 Совет"), KeyboardButton("📈 Отчёт")]
    ], resize_keyboard=True)
    
    await update.message.reply_text(welcome_text, parse_mode="Markdown", reply_markup=kb)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Главный обработчик с AI-пониманием"""
    chat_id = update.effective_chat.id
    text = update.message.text
    user_ctx = get_user_context(chat_id)
    
    # Сбрасываем pending action если получили новое сообщение
    if user_ctx.pending_action and not text.startswith("/"):
        # Проверяем, не ответ ли это на предыдущий вопрос
        pass  # обработаем ниже
    
    # AI-анализ намерений
    understanding = await ai_understand(text, chat_id, context)
    
    intent = understanding.get("intent", "other")
    data = understanding.get("data", {})
    confidence = understanding.get("confidence", 0)
    clarification = understanding.get("clarification_question")
    
    # Если нужно уточнение
    if clarification:
        user_ctx.pending_action = f"awaiting_{intent}"
        await update.message.reply_text(f"🤔 {clarification}")
        return
    
    # Обработка намерений
    if intent == "expense":
        await process_expense_ai(update, context, data, user_ctx)
    elif intent == "debt_new":
        await process_debt_new(update, context, data)
    elif intent == "debt_payment":
        await process_debt_payment(update, context, data)
    elif intent == "budget_set":
        await process_budget_set(update, context, data)
    elif intent == "salary_set":
        await process_salary_set(update, context, data)
    elif intent == "query_report":
        await process_query_report(update, context, data)
    elif intent == "query_status":
        await process_query_status(update, context)
    else:
        # Свободный разговор
        response = await ai_converse(chat_id, text, context)
        await update.message.reply_text(response)

async def process_expense_ai(update: Update, context: ContextTypes.DEFAULT_TYPE, data: Dict, user_ctx: UserContext):
    """Обработка расхода с AI"""
    # Парсим дату если есть
    date_str = data.get("date", datetime.now().isoformat())
    if isinstance(date_str, str):
        expense_date = parse_date(date_str)
    else:
        expense_date = datetime.now()
    
    # Создаём объект расхода
    expense = Expense(
        amount=float(data.get("amount", 0)),
        currency=data.get("currency", "UAH"),
        category=data.get("category", "Другое"),
        description=data.get("description", ""),
        date=expense_date,
        raw_text=update.message.text,
        confidence=data.get("confidence", 0.8)
    )
    
    # Если уверенность низкая — уточняем категорию
    if expense.confidence < 0.7:
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton(f"{cat}", callback_data=f"cat_confirm_{cat}_{expense.amount}_{expense.currency}")]
            for cat in CATEGORIES
        ])
        await update.message.reply_text(
            f"🤔 *{expense.amount} {CURRENCY_SYMBOLS.get(expense.currency, '₴')}* — {expense.description}\n\n"
            f"Какая категория?",
            parse_mode="Markdown",
            reply_markup=kb
        )
        user_ctx.last_expense = expense
        user_ctx.pending_action = "awaiting_category"
        return
    
    # Сохраняем
    save_expense_ai(expense, update.effective_chat.id)
    user_ctx.last_expense = expense
    
    # Формируем ответ
    emoji_map = {
        "Еда / продукты": "🍔", "Транспорт": "🚗", "Развлечения": "🎮",
        "Здоровье / аптека": "💊", "Никотин": "🚬", "Другое": "📦"
    }
    
    response = (f"✅ *Записано!*\n\n"
                f"{emoji_map.get(expense.category, '💰')} {expense.category}\n"
                f"💵 {expense.amount:,.0f} {CURRENCY_SYMBOLS.get(expense.currency, '₴')}\n"
                f"📝 {expense.description}")
    
    # Добавляем контекст
    if expense.date.date() != datetime.now().date():
        response += f"\n📅 {expense.date.strftime('%d.%m.%Y')}"
    
    # Проверяем бюджет
    stats = await get_user_stats(update.effective_chat.id)
    settings = get_user_settings(update.effective_chat.id)
    budget = float(settings.get("monthly_budget", 0))
    
    if budget > 0:
        spent = stats["month_total"]
        pct = min(int(spent / budget * 100), 100)
        if pct >= 90:
            response += f"\n\n🔴 *Внимание!* Бюджет использован на {pct}%"
        elif pct >= 75:
            response += f"\n\n🟡 Бюджет использован на {pct}%"
    
    await update.message.reply_text(response, parse_mode="Markdown")

async def process_debt_new(update: Update, context: ContextTypes.DEFAULT_TYPE, data: Dict):
    """Новый долг"""
    # Реализация...
    await update.message.reply_text("💸 Долг записан! (реализация в процессе)")

async def process_debt_payment(update: Update, context: ContextTypes.DEFAULT_TYPE, data: Dict):
    """Погашение долга"""
    await update.message.reply_text("✅ Погашение записано! (реализация в процессе)")

async def process_budget_set(update: Update, context: ContextTypes.DEFAULT_TYPE, data: Dict):
    """Установка бюджета"""
    amount = data.get("amount", 0)
    save_user_setting(update.effective_chat.id, "monthly_budget", str(amount))
    await update.message.reply_text(f"💰 Бюджет установлен: *{amount:,.0f} ₴/мес*", parse_mode="Markdown")

async def process_salary_set(update: Update, context: ContextTypes.DEFAULT_TYPE, data: Dict):
    """Установка зарплаты"""
    day = data.get("day", 25)
    amount = data.get("amount")
    save_user_setting(update.effective_chat.id, "salary_day", str(day))
    if amount:
        save_user_setting(update.effective_chat.id, "salary_amount", str(amount))
    
    msg = f"💵 Зарплата: *{day}-е число*"
    if amount:
        msg += f", {amount:,.0f} ₴"
    await update.message.reply_text(msg, parse_mode="Markdown")

async def process_query_report(update: Update, context: ContextTypes.DEFAULT_TYPE, data: Dict):
    """Запрос отчёта"""
    period = data.get("period", "month")
    
    if period == "week":
        expenses = get_expenses(update.effective_chat.id, 7)
        title = "📅 *Отчёт за неделю*"
    else:
        expenses = get_expenses(update.effective_chat.id, 30)
        title = "📆 *Отчёт за месяц*"
    
    if not expenses:
        await update.message.reply_text("📭 Нет данных за этот период.")
        return
    
    total = sum(float(r["Сумма"]) for r in expenses)
    by_cat = defaultdict(float)
    for r in expenses:
        by_cat[r["Категория"]] += float(r["Сумма"])
    
    lines = [title, f"\n💰 *Всего: {total:,.0f} ₴*\n", "*По категориям:*"]
    for cat, amt in sorted(by_cat.items(), key=lambda x: -x[1]):
        pct = int(amt / total * 100)
        lines.append(f"• {cat}: {amt:,.0f} ₴ ({pct}%)")
    
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def process_query_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Статус финансов"""
    stats = await get_user_stats(update.effective_chat.id)
    settings = get_user_settings(update.effective_chat.id)
    
    lines = ["💳 *Финансовый статус*\n"]
    lines.append(f"📊 Потрачено: *{stats['month_total']:,.0f} ₴*")
    
    budget = float(settings.get("monthly_budget", 0))
    if budget > 0:
        left = budget - stats['month_total']
        pct = int(stats['month_total'] / budget * 100)
        lines.append(f"💰 Бюджет: {budget:,.0f} ₴ ({pct}%)")
        lines.append(f"📉 Осталось: {left:,.0f} ₴")
    
    if stats['days_to_salary']:
        lines.append(f"⏰ До зарплаты: {stats['days_to_salary']} дней")
    
    # Добавляем AI-инсайт
    insight = await generate_insight(update.effective_chat.id)
    lines.append(f"\n{insight}")
    
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка inline кнопок"""
    query = update.callback_query
    await query.answer()
    data = query.data
    
    if data.startswith("cat_confirm_"):
        # Подтверждение категории
        parts = data.split("_")
        cat = parts[2]
        amount = float(parts[3])
        currency = parts[4] if len(parts) > 4 else "UAH"
        
        chat_id = query.message.chat_id
        user_ctx = get_user_context(chat_id)
        
        if user_ctx.last_expense:
            user_ctx.last_expense.category = cat
            save_expense_ai(user_ctx.last_expense, chat_id)
            
            await query.edit_message_text(
                f"✅ *Записано!*\n\n📝 {user_ctx.last_expense.description}\n"
                f"💵 {amount:,.0f} {CURRENCY_SYMBOLS.get(currency, '₴')}\n"
                f"📁 {cat}",
                parse_mode="Markdown"
            )
            user_ctx.pending_action = None

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Голосовые сообщения с AI"""
    await update.message.reply_text("🎙 Распознаю голос...")
    
    try:
        file = await context.bot.get_file(update.message.voice.file_id)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            await file.download_to_drive(tmp.name)
            path = tmp.name
        
        # Распознаём через Groq Whisper
        with open(path, "rb") as f:
            transcript = groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                language="ru"
            )
        
        text = transcript.text
        os.unlink(path)
        
        await update.message.reply_text(f"📝 *Распознал:* _{text}_", parse_mode="Markdown")
        
        # Обрабатываем как текст
        update.message.text = text
        await handle_message(update, context)
        
    except Exception as e:
        logger.error(f"Voice error: {e}")
        await update.message.reply_text("❌ Не удалось распознать голосовое сообщение.")

# ── PROACTIVE FEATURES ─────────────────────────────────────────────────────────

async def scheduled_insight(context: ContextTypes.DEFAULT_TYPE):
    """Проактивные напоминания"""
    job = context.job
    chat_id = job.data.get("chat_id")
    
    stats = await get_user_stats(chat_id)
    settings = get_user_settings(chat_id)
    
    # Проверяем бюджет
    budget = float(settings.get("monthly_budget", 0))
    if budget > 0:
        pct = stats['month_total'] / budget
        if 0.75 <= pct < 0.8:
            await context.bot.send_message(
                chat_id=chat_id,
                text="🟡 *Внимание!* Вы использовали 75% бюджета месяца.\n\nПодумайте о сокращении необязательных трат на этой неделе.",
                parse_mode="Markdown"
            )

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    init_sheets()
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Команды
    app.add_handler(CommandHandler("start", cmd_start))
    
    # Callbacks
    app.add_handler(CallbackQueryHandler(handle_callback))
    
    # Голос
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    
    # Текст (главный обработчик)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Планировщик проактивных уведомлений
    if app.job_queue:
        # Ежедневная проверка в 19:00
        app.job_queue.run_daily(
            scheduled_insight,
            time=datetime.strptime("19:00", "%H:%M").time(),
            data={"chat_id": CHAT_ID} if CHAT_ID else None
        )
    
    logger.info("AI Финансовый Агент запущен! v4.0")
    app.run_polling()

if __name__ == "__main__":
    main()