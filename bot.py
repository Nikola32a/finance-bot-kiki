"""
AI Финансовый Агент v5.0 — полноценный агент с долгами, зарплатой, аналитикой, AI-советами
"""
import os
import logging
import tempfile
import json
import re
import asyncio
import random
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv

load_dotenv()

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler, ConversationHandler
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

# ── CONSTANTS ───────────────────────────────────────────────────────────────
AI_MODEL = "llama-3.3-70b-versatile"

CATEGORIES = ["Еда / продукты", "Транспорт", "Развлечения", "Здоровье / аптека", "Никотин", "Другое"]

CATEGORY_EMOJIS = {
    "Еда / продукты": "🍔", "Транспорт": "🚗", "Развлечения": "🎮",
    "Здоровье / аптека": "💊", "Никотин": "🚬", "Другое": "📦"
}

CURRENCY_SYMBOLS = {"UAH": "₴", "USD": "$", "EUR": "€", "GBP": "£"}

MONTH_NAMES = ["Январь","Февраль","Март","Апрель","Май","Июнь",
               "Июль","Август","Сентябрь","Октябрь","Ноябрь","Декабрь"]
MONTH_NAMES_GEN = ["января","февраля","марта","апреля","мая","июня",
                   "июля","августа","сентября","октября","ноября","декабря"]

# Эквиваленты для инсайтов
EQUIVALENTS = [
    (2000, "🍕 100 пицц"), (3000, "🎮 3 игры в Steam"), (5000, "✈️ билет в Европу"),
    (8000, "📱 бюджетный смартфон"), (15000, "💻 ноутбук"), (25000, "📱 iPhone"),
    (40000, "🏖 неделя на море"), (60000, "🚗 взнос на авто"), (100000, "🌍 отпуск мечты")
]

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

@dataclass
class Debt:
    id: str
    name: str
    amounts: List[Dict[str, Any]]  # [{"amount": 500, "currency": "USD"}]
    date: datetime
    note: str = ""
    status: str = "active"

# ── GLOBAL STATE ─────────────────────────────────────────────────────────────

user_contexts: Dict[int, Dict] = {}
debts: Dict[str, Debt] = {}
debt_counter = 0

def get_user_context(chat_id: int) -> Dict:
    if chat_id not in user_contexts:
        user_contexts[chat_id] = {
            "last_expense": None,
            "pending_action": None,
            "conversation_history": [],
            "preferences": {}
        }
    return user_contexts[chat_id]

# ── AI SYSTEM PROMPTS ─────────────────────────────────────────────────────────

EXPENSE_PROMPT = """Ты финансовый ассистент. Извлеки ВСЕ траты из текста.

ПРАВИЛА:
1. Каждая трата — отдельный объект в items
2. Определи категорию по смыслу:
   - Никотин: снюс, сигареты, вейп, табак, кальян
   - Транспорт: бензин, топливо, заправка, такси, метро, мойка, парковка
   - Еда: кофе, ресторан, продукты, еда, доставка
   - Развлечения: кино, игры, steam, подписка, бар
   - Здоровье: аптека, лекарства, врач, спортзал
   - Другое: одежда, техника, подарки

3. Даты: сегодня, вчера, 3 дня назад → YYYY-MM-DD
4. Валюты: грн/₴=UAH, доллары/баксы/$=USD, евро/€=EUR

Верни JSON:
{
  "intent": "expense",
  "items": [{"amount": число, "currency": "UAH", "category": "...", "description": "...", "date": "YYYY-MM-DD"}],
  "confidence": 0.0-1.0
}"""

DEBT_PROMPT = """Извлеки информацию о долге из текста.

Определи:
- intent: "debt_new" (дал в долг), "debt_payment" (вернули долг), или "unknown"
- name: имя человека
- amounts: массив [{"amount": число, "currency": "UAH|USD|EUR"}]
- note: за что (опционально)
- is_partial: true если частичное погашение

Примеры:
- "Дал в долг Саше 5000 грн на покушать" → debt_new, Саша, 5000 UAH, "на покушать"
- "Саша вернул 3000" → debt_payment, Саша, 3000 UAH
- "Вернул долг Саше полностью" → debt_payment, Саша, полное погашение

Верни JSON с полями: intent, name, amounts, note, is_partial"""

CONVERSATION_PROMPT = """Ты — дружелюбный финансовый советник. Отвечай кратко, тепло, с эмодзи.

Контекст пользователя:
{context}

Вопрос: {question}

Дай полезный, конкретный ответ. Если просит совет — предложи 2-3 конкретных действия.
Если вопрос не про финансы — вежливо направь к теме."""

# ── AI CORE FUNCTIONS ───────────────────────────────────────────────────────

async def ai_parse_expenses(text: str) -> Dict:
    """AI парсинг трат"""
    try:
        response = groq_client.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "user", "content": EXPENSE_PROMPT + f"\n\nТекст: \"{text}\""}],
            max_tokens=600,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"AI expense parse error: {e}")
        return smart_fallback_parse(text)

async def ai_parse_debt(text: str) -> Dict:
    """AI парсинг долгов"""
    try:
        response = groq_client.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "user", "content": DEBT_PROMPT + f"\n\nТекст: \"{text}\""}],
            max_tokens=400,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"AI debt parse error: {e}")
        return {"intent": "unknown"}

async def ai_conversation(text: str, context_data: Dict) -> str:
    """Свободный разговор с AI"""
    context_str = json.dumps(context_data, ensure_ascii=False, indent=2)
    prompt = CONVERSATION_PROMPT.format(context=context_str, question=text)
    
    try:
        response = groq_client.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"AI conversation error: {e}")
        return "🤔 Не совсем понял. Можешь переформулировать?"

def smart_fallback_parse(text: str) -> Dict:
    """Умный fallback парсер"""
    text_lower = text.lower()
    items = []
    
    # Паттерн: сумма + (валюта) + предлог + описание
    pattern = r'(\d+(?:[.,]\d+)?)\s*(?:грн|гривен|₴|долларов|баксов|\$|евро|€)?\s*(?:на|за|в|для)?\s*([^,\.и\d][^,\.]*?)(?=\s*(?:и|,|\.|\d|$))'
    
    for match in re.finditer(pattern, text_lower):
        amount = float(match.group(1).replace(",", "."))
        desc = match.group(2).strip()
        if len(desc) > 30:
            desc = desc[:30]
        
        # Определяем валюту
        currency = "UAH"
        pre_text = text_lower[max(0, match.start()-15):match.start()]
        if any(x in pre_text for x in ["доллар", "бакс", "$"]):
            currency = "USD"
        elif any(x in pre_text for x in ["евро", "€"]):
            currency = "EUR"
        
        # Категория
        category = "Другое"
        desc_lower = desc.lower()
        if any(w in desc_lower for w in ["снюс", "сигарет", "вейп", "табак"]):
            category = "Никотин"
        elif any(w in desc_lower for w in ["бензин", "топливо", "заправка", "такси", "метро", "мойка"]):
            category = "Транспорт"
        elif any(w in desc_lower for w in ["кофе", "кафе", "ресторан", "еду", "пицца", "продукты"]):
            category = "Еда / продукты"
        elif any(w in desc_lower for w in ["кино", "игра", "steam", "подписка"]):
            category = "Развлечения"
        elif any(w in desc_lower for w in ["аптека", "лекарств", "врач"]):
            category = "Здоровье / аптека"
        
        items.append({
            "amount": amount,
            "currency": currency,
            "category": category,
            "description": desc.capitalize(),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "confidence": 0.6
        })
    
    if not items:
        # Ищем просто числа
        nums = re.findall(r'(\d+(?:[.,]\d+)?)\s*(?:грн|₴)?', text_lower)
        if nums:
            items.append({
                "amount": float(nums[0].replace(",", ".")),
                "currency": "UAH",
                "category": "Другое",
                "description": "трата",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "confidence": 0.4
            })
    
    return {
        "intent": "expense" if items else "unknown",
        "items": items,
        "confidence": 0.6 if items else 0.2
    }

# ── DATE PARSING ─────────────────────────────────────────────────────────────

def parse_smart_date(text: str) -> datetime:
    """Умный парсер дат"""
    text_lower = text.lower()
    now = datetime.now()
    
    if any(w in text_lower for w in ["сегодня", "сьогодні"]):
        return now
    if any(w in text_lower for w in ["вчера", "вчора"]):
        return now - timedelta(days=1)
    if "позавчера" in text_lower:
        return now - timedelta(days=2)
    
    # N дней назад
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
    
    # ДД.ММ.ГГГГ
    match = re.search(r'(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?', text_lower)
    if match:
        d, m, y = match.groups()
        year = int(y) + 2000 if y and len(y) == 2 else (int(y) if y else now.year)
        try:
            return datetime(year, int(m), int(d))
        except:
            pass
    
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
    """Получить или создать лист"""
    sp = get_spreadsheet()
    try:
        return sp.worksheet(name)
    except gspread.WorksheetNotFound:
        # Создаём новый лист
        ws = sp.add_worksheet(title=name, rows=1000, cols=10)
        
        # Инициализируем заголовки в зависимости от типа
        if name == "Expenses":
            ws.append_row(["Дата", "Сумма", "Валюта", "Категория", "Описание", "Raw Text", "UserID", "Timestamp"])
        elif name == "Debts":
            ws.append_row(["ID", "Кому", "Суммы JSON", "Дата", "Статус", "Примечание", "UserID", "Created"])
        elif name == "Settings":
            ws.append_row(["UserID", "Key", "Value", "Updated"])
        elif name == "Memory":
            ws.append_row(["UserID", "Keyword", "Category", "Count", "LastUsed"])
        
        return ws

def init_db():
    """Инициализация всех таблиц при старте"""
    sp = get_spreadsheet()
    
    required_sheets = ["Expenses", "Debts", "Settings", "Memory"]
    
    for sheet_name in required_sheets:
        try:
            sp.worksheet(sheet_name)
            logger.info(f"Sheet '{sheet_name}' found")
        except gspread.WorksheetNotFound:
            logger.info(f"Creating sheet '{sheet_name}'...")
            ws = sp.add_worksheet(title=sheet_name, rows=1000, cols=10)
            
            if sheet_name == "Expenses":
                ws.append_row(["Дата", "Сумма", "Валюта", "Категория", "Описание", "Raw Text", "UserID", "Timestamp"])
            elif sheet_name == "Debts":
                ws.append_row(["ID", "Кому", "Суммы JSON", "Дата", "Статус", "Примечание", "UserID", "Created"])
            elif sheet_name == "Settings":
                ws.append_row(["UserID", "Key", "Value", "Updated"])
            elif sheet_name == "Memory":
                ws.append_row(["UserID", "Keyword", "Category", "Count", "LastUsed"])
            
            logger.info(f"Sheet '{sheet_name}' created")

# ── DATA OPERATIONS ───────────────────────────────────────────────────────────

def save_expense(exp: 'Expense', user_id: int):
    """Сохранить трату"""
    try:
        ws = get_sheet("Expenses")
        ws.append_row([
            exp.date.strftime("%d.%m.%Y %H:%M"),
            float(exp.amount),
            exp.currency,
            exp.category,
            exp.description,
            exp.raw_text[:100] if exp.raw_text else "",
            str(user_id),
            datetime.now().isoformat()
        ])
        logger.info(f"Expense saved: {exp.amount} {exp.currency} - {exp.category}")
    except Exception as e:
        logger.error(f"Save expense error: {e}")
        raise

def get_expenses(user_id: int, days: int = 30) -> List[Dict]:
    """Получить траты за период"""
    try:
        ws = get_sheet("Expenses")
        records = ws.get_all_records()
    except Exception as e:
        logger.error(f"Get expenses error: {e}")
        return []
    
    cutoff = datetime.now() - timedelta(days=days)
    result = []
    
    for r in records:
        if str(r.get("UserID")) != str(user_id):
            continue
        
        try:
            date_str = r.get("Дата", "")
            if not date_str:
                continue
            
            # Парсим дату (формат: DD.MM.YYYY HH:MM)
            d = datetime.strptime(date_str[:10], "%d.%m.%Y")
            if d >= cutoff:
                result.append(r)
        except Exception as e:
            continue
    
    return result

def save_debt(debt: 'Debt', user_id: int):
    """Сохранить долг"""
    try:
        ws = get_sheet("Debts")
        amounts_json = json.dumps(debt.amounts, ensure_ascii=False)
        ws.append_row([
            debt.id,
            debt.name,
            amounts_json,
            debt.date.strftime("%d.%m.%Y"),
            debt.status,
            debt.note,
            str(user_id),
            datetime.now().isoformat()
        ])
        logger.info(f"Debt saved: {debt.id} - {debt.name}")
    except Exception as e:
        logger.error(f"Save debt error: {e}")

def load_debts(user_id: int):
    """Загрузить активные долги пользователя"""
    global debts, debt_counter
    
    try:
        ws = get_sheet("Debts")
        records = ws.get_all_records()
        
        # Очищаем старые долги этого пользователя
        debts.clear()
        
        for r in records:
            if str(r.get("UserID")) != str(user_id):
                continue
            if r.get("Статус") != "active":
                continue
            
            did = str(r["ID"])
            try:
                amounts = json.loads(r["Суммы JSON"])
            except:
                # Fallback для старых записей
                try:
                    amt = float(r["Суммы JSON"])
                    amounts = [{"amount": amt, "currency": "UAH"}]
                except:
                    amounts = []
            
            debts[did] = Debt(
                id=did,
                name=r["Кому"],
                amounts=amounts,
                date=datetime.strptime(r["Дата"], "%d.%m.%Y"),
                note=r.get("Примечание", ""),
                status=r.get("Статус", "active")
            )
            
            # Обновляем глобальный счётчик
            try:
                num = int(did)
                if num > debt_counter:
                    debt_counter = num
            except:
                pass
        
        logger.info(f"Loaded {len(debts)} debts for user {user_id}")
        
    except Exception as e:
        logger.error(f"Load debts error: {e}")

def update_debt_status(debt_id: str, status: str):
    """Обновить статус долга в таблице"""
    try:
        ws = get_sheet("Debts")
        records = ws.get_all_records()
        
        for i, r in enumerate(records, start=2):  # start=2 because row 1 is header
            if str(r.get("ID")) == debt_id:
                ws.update_cell(i, 5, status)  # Column 5 = Статус
                logger.info(f"Debt {debt_id} status updated to {status}")
                
                # Обновляем в памяти
                if debt_id in debts:
                    debts[debt_id].status = status
                return True
        
        return False
    except Exception as e:
        logger.error(f"Update debt status error: {e}")
        return False

def update_debt_amounts(debt_id: str, new_amounts: List[Dict]):
    """Обновить суммы долга"""
    try:
        ws = get_sheet("Debts")
        records = ws.get_all_records()
        
        for i, r in enumerate(records, start=2):
            if str(r.get("ID")) == debt_id:
                ws.update_cell(i, 3, json.dumps(new_amounts, ensure_ascii=False))
                logger.info(f"Debt {debt_id} amounts updated")
                
                if debt_id in debts:
                    debts[debt_id].amounts = new_amounts
                return True
        
        return False
    except Exception as e:
        logger.error(f"Update debt amounts error: {e}")
        return False

def save_setting(user_id: int, key: str, value: str):
    """Сохранить настройку"""
    try:
        ws = get_sheet("Settings")
        
        # Ищем существующую запись
        records = ws.get_all_records()
        for i, r in enumerate(records, start=2):
            if str(r.get("UserID")) == str(user_id) and r.get("Key") == key:
                # Обновляем
                ws.update_cell(i, 3, value)
                ws.update_cell(i, 4, datetime.now().isoformat())
                logger.info(f"Setting updated: {key} = {value}")
                return
        
        # Новая запись
        ws.append_row([str(user_id), key, value, datetime.now().isoformat()])
        logger.info(f"Setting created: {key} = {value}")
        
    except Exception as e:
        logger.error(f"Save setting error: {e}")

def get_settings(user_id: int) -> Dict[str, str]:
    """Получить все настройки пользователя"""
    try:
        ws = get_sheet("Settings")
        records = ws.get_all_records()
        
        settings = {}
        for r in records:
            if str(r.get("UserID")) == str(user_id):
                settings[r.get("Key", "")] = r.get("Value", "")
        
        return settings
    except Exception as e:
        logger.error(f"Get settings error: {e}")
        return {}

def get_setting(user_id: int, key: str, default=None) -> Optional[str]:
    """Получить одну настройку"""
    settings = get_settings(user_id)
    return settings.get(key, default)

def save_memory(user_id: int, keyword: str, category: str):
    """Сохранить предпочтение категории для слова"""
    if not keyword or len(keyword) < 2:
        return
    
    try:
        ws = get_sheet("Memory")
        records = ws.get_all_records()
        
        # Ищем существующую запись
        for i, r in enumerate(records, start=2):
            if str(r.get("UserID")) == str(user_id) and r.get("Keyword") == keyword:
                # Увеличиваем счётчик использования
                count = int(r.get("Count", 1)) + 1
                ws.update_cell(i, 4, count)
                ws.update_cell(i, 5, datetime.now().isoformat())
                logger.info(f"Memory updated: {keyword} -> {category} (x{count})")
                return
        
        # Новая запись
        ws.append_row([str(user_id), keyword.lower(), category, 1, datetime.now().isoformat()])
        logger.info(f"Memory created: {keyword} -> {category}")
        
    except Exception as e:
        logger.error(f"Save memory error: {e}")

def get_memory(user_id: int, keyword: str) -> Optional[str]:
    """Получить запомненную категорию для слова"""
    if not keyword:
        return None
    
    try:
        ws = get_sheet("Memory")
        records = ws.get_all_records()
        
        keyword_lower = keyword.lower()
        
        # Ищем точное совпадение или содержание
        for r in records:
            if str(r.get("UserID")) == str(user_id):
                mem_keyword = r.get("Keyword", "").lower()
                if mem_keyword == keyword_lower or keyword_lower in mem_keyword:
                    return r.get("Category")
        
        return None
    except Exception as e:
        logger.error(f"Get memory error: {e}")
        return None

def get_all_memory(user_id: int) -> Dict[str, str]:
    """Получить всю память пользователя"""
    try:
        ws = get_sheet("Memory")
        records = ws.get_all_records()
        
        memory = {}
        for r in records:
            if str(r.get("UserID")) == str(user_id):
                keyword = r.get("Keyword", "")
                if keyword:
                    memory[keyword] = r.get("Category", "Другое")
        
        return memory
    except Exception as e:
        logger.error(f"Get all memory error: {e}")
        return {}

# ── ANALYTICS FUNCTIONS ─────────────────────────────────────────────────────

def analyze_expenses(records: List[Dict]) -> Dict:
    """Анализ трат"""
    if not records:
        return None
    
    total = sum(float(r["Сумма"]) for r in records if r.get("Сумма"))
    by_category = defaultdict(float)
    by_day = defaultdict(float)
    by_description = defaultdict(lambda: {"count": 0, "total": 0.0})
    
    for r in records:
        amt = float(r["Сумма"]) if r.get("Сумма") else 0
        cat = r.get("Категория", "Другое")
        desc = r.get("Описание", "").lower()
        
        by_category[cat] += amt
        
        # По дням недели
        try:
            d = datetime.strptime(r["Дата"][:10], "%d.%m.%Y")
            day_name = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"][d.weekday()]
            by_day[day_name] += amt
        except:
            pass
        
        # Частые траты
        if desc:
            by_description[desc]["count"] += 1
            by_description[desc]["total"] += amt
    
    # "Утечки" — частые мелкие траты
    leaks = {k: v for k, v in by_description.items() if v["count"] >= 3 and v["total"] > 500}
    
    return {
        "total": total,
        "count": len(records),
        "by_category": dict(by_category),
        "by_day": dict(by_day),
        "leaks": leaks,
        "avg_per_day": total / max(len(set(r["Дата"][:10] for r in records)), 1)
    }

def get_month_name(n: int, genitive: bool = False) -> str:
    return (MONTH_NAMES_GEN if genitive else MONTH_NAMES)[n-1]

def format_amount(amount: float, currency: str = "UAH") -> str:
    return f"{amount:,.0f} {CURRENCY_SYMBOLS.get(currency, '₴')}"

# ── BUDGET & SALARY ─────────────────────────────────────────────────────────

def get_budget_status(user_id: int) -> Optional[Dict]:
    settings = get_settings(user_id)
    budget = settings.get("monthly_budget")
    if not budget:
        return None
    
    try:
        budget = float(budget)
    except:
        return None
    
    expenses = get_expenses(user_id, 30)
    spent = sum(float(r["Сумма"]) for r in expenses if r.get("Валюта") == "UAH")
    left = budget - spent
    percent = min(int(spent / budget * 100), 100) if budget > 0 else 0
    
    return {
        "budget": budget,
        "spent": spent,
        "left": left,
        "percent": percent,
        "status": "critical" if percent >= 90 else "warning" if percent >= 75 else "ok"
    }

def get_salary_status(user_id: int) -> Optional[Dict]:
    settings = get_settings(user_id)
    day = settings.get("salary_day")
    if not day:
        return None
    
    try:
        day = int(day)
    except:
        return None
    
    amount = settings.get("salary_amount")
    if amount:
        try:
            amount = float(amount)
        except:
            amount = None
    
    now = datetime.now()
    if now.day < day:
        next_salary = now.replace(day=day)
        days_left = day - now.day
    else:
        next_month = (now.replace(day=1) + timedelta(days=32)).replace(day=1)
        next_salary = next_month.replace(day=min(day, 28))
        days_left = (next_salary - now).days
    
    expenses = get_expenses(user_id, 30)
    spent = sum(float(r["Сумма"]) for r in expenses if r.get("Валюта") == "UAH")
    
    result = {
        "day": day,
        "next_date": next_salary,
        "days_left": days_left,
        "spent": spent
    }
    
    if amount:
        result["amount"] = amount
        result["left"] = amount - spent
        result["daily_budget"] = (amount - spent) / max(days_left, 1) if days_left > 0 else 0
    
    return result

# ── AI INSIGHTS ─────────────────────────────────────────────────────────────

async def generate_ai_insight(user_id: int) -> str:
    """Генерация AI-инсайта на основе данных"""
    expenses = get_expenses(user_id, 30)
    if len(expenses) < 5:
        return "📊 Пока мало данных. Записывай больше трат — через неделю я дам персональные советы!"
    
    analysis = analyze_expenses(expenses)
    budget = get_budget_status(user_id)
    
    # Формируем контекст для AI
    context = {
        "month_total": analysis["total"],
        "top_category": max(analysis["by_category"].items(), key=lambda x: x[1])[0] if analysis["by_category"] else "нет",
        "categories": analysis["by_category"],
        "budget_percent": budget["percent"] if budget else None,
        "days_to_salary": get_salary_status(user_id)["days_left"] if get_salary_status(user_id) else None
    }
    
    prompt = f"""Проанализируй финансы пользователя и дай 1-2 конкретных, полезных совета.
Будь дружелюбным, используй эмодзи, конкретные цифры.

Данные: {json.dumps(context, ensure_ascii=False)}

Примеры хороших советов:
"🍔 На еду уходит 45% — это выше нормы. Попробуй готовить дома 2 раза в неделю, сэкономишь ~2000₴"
"📉 До зарплаты 5 дней, осталось 15% бюджета. Сегодня лучше без крупных трат!"
"☕ Кофе на вынос: 150₴ × 20 дней = 3000₴/мес. Термос с собой = 1500₴ экономии"

Дай 1-2 коротких совета:"""

    try:
        response = groq_client.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"AI insight error: {e}")
        # Fallback инсайт
        return generate_fallback_insight(analysis, budget)

def generate_fallback_insight(analysis: Dict, budget: Optional[Dict]) -> str:
    """Резервный инсайт если AI недоступен"""
    tips = []
    
    # Анализ категорий
    total = analysis["total"]
    by_cat = analysis["by_category"]
    
    if "Еда / продукты" in by_cat:
        food_pct = by_cat["Еда / продукты"] / total * 100
        if food_pct > 40:
            tips.append(f"🍔 На еду {food_pct:.0f}% бюджета. Попробуй планировать меню — сэкономишь ~{by_cat['Еда / продукты'] * 0.2:,.0f} ₴")
    
    if "Никотин" in by_cat:
        nic = by_cat["Никотин"]
        tips.append(f"🚬 Никотин: {nic:,.0f} ₴/мес = {nic*12:,.0f} ₴/год. Подумай о снижении 😉")
    
    if "Развлечения" in by_cat:
        ent = by_cat["Развлечения"]
        if ent > total * 0.25:
            tips.append(f"🎮 Развлечения: {ent/total*100:.0f}%. Можно сократить на 20% = +{ent*0.2:,.0f} ₴")
    
    # Утечки
    if analysis["leaks"]:
        top_leak = max(analysis["leaks"].items(), key=lambda x: x[1]["total"])
        tips.append(f"💸 «{top_leak[0]}» — {top_leak[1]['count']} раз = {top_leak[1]['total']:,.0f} ₴")
    
    # Бюджет
    if budget:
        if budget["status"] == "critical":
            tips.append(f"🔴 Бюджет на {budget['percent']}%! Осталось {budget['left']:,.0f} ₴ — только необходимое.")
        elif budget["status"] == "warning":
            tips.append(f"🟡 Бюджет на {budget['percent']}%. До зарплаты осторожнее с тратами.")
    
    if not tips:
        return "💡 Продолжай записывать траты. Через неделю я смогу найти паттерны и дать советы!"
    
    return "💡 *Советы:*\n" + "\n".join(f"• {t}" for t in tips[:2])

# ── DEBT FUNCTIONS ─────────────────────────────────────────────────────────────

def format_amounts(amounts: List[Dict]) -> str:
    """Форматирование сумм долга"""
    parts = []
    for a in amounts:
        sym = CURRENCY_SYMBOLS.get(a.get("currency", "UAH"), "₴")
        parts.append(f"{a['amount']:,.0f}{sym}")
    return " + ".join(parts) if parts else "0₴"

def find_debt_by_name(name: str) -> Optional[Tuple[str, Debt]]:
    """Поиск долга по имени"""
    name_lower = name.lower()
    for did, debt in debts.items():
        if name_lower in debt.name.lower() or debt.name.lower() in name_lower:
            return did, debt
    return None

async def process_debt_new(update: Update, context: ContextTypes.DEFAULT_TYPE, parsed: Dict):
    """Создание нового долга"""
    global debt_counter
    
    name = parsed.get("name")
    amounts = parsed.get("amounts", [])
    note = parsed.get("note", "")
    
    if not name or not amounts:
        await update.message.reply_text("🤔 Не понял кто и сколько. Попробуй: «Дал в долг Саше 5000»")
        return
    
    debt_counter += 1
    did = str(debt_counter)
    
    debt = Debt(
        id=did,
        name=name,
        amounts=amounts,
        date=datetime.now(),
        note=note,
        status="active"
    )
    
    debts[did] = debt
    save_debt(debt, update.effective_chat.id)
    
    # Устанавливаем напоминание через 2 недели по умолчанию
    if context.job_queue:
        context.job_queue.run_once(
            send_debt_reminder,
            when=timedelta(weeks=2),
            data={"debt_id": did, "chat_id": update.effective_chat.id},
            name=f"debt_{did}"
        )
    
    amt_str = format_amounts(amounts)
    note_str = f"\n📝 {note}" if note else ""
    
    await update.message.reply_text(
        f"💸 *Долг записан!*\n\n"
        f"👤 *{name}*\n"
        f"💰 {amt_str}{note_str}\n"
        f"📅 {datetime.now().strftime('%d.%m.%Y')}\n\n"
        f"⏰ Напомню через 2 недели",
        parse_mode="Markdown"
    )

async def process_debt_payment(update: Update, context: ContextTypes.DEFAULT_TYPE, parsed: Dict):
    """Обработка возврата долга"""
    name = parsed.get("name")
    amounts = parsed.get("amounts", [])
    is_partial = parsed.get("is_partial", True)
    
    if not name:
        await update.message.reply_text("🤔 Кто вернул деньги?")
        return
    
    found = find_debt_by_name(name)
    if not found:
        await update.message.reply_text(f"🤔 Не нашёл долга для *{name}*. Проверь имя или создай новый.", parse_mode="Markdown")
        return
    
    did, debt = found
    
    if not is_partial and not amounts:
        # Полное погашение без суммы
        debt.status = "paid"
        update_debt_status(did, "paid")
        debts.pop(did, None)
        
        await update.message.reply_text(
            f"✅ *{debt.name}* вернул долг полностью!\n"
            f"💰 {format_amounts(debt.amounts)}\n\n"
            f"🎉 Долг закрыт!",
            parse_mode="Markdown"
        )
        return
    
    # Частичное погашение
    lines = [f"💰 *{debt.name}* вернул:\n"]
    remaining = []
    
    for paid_amt in amounts:
        cur = paid_amt.get("currency", "UAH")
        paid = float(paid_amt["amount"])
        
        # Ищем соответствующую валюту в долге
        found_cur = False
        for i, debt_amt in enumerate(debt.amounts):
            if debt_amt.get("currency", "UAH") == cur:
                found_cur = True
                old = float(debt_amt["amount"])
                new = old - paid
                
                if new <= 0:
                    lines.append(f"✅ {CURRENCY_SYMBOLS.get(cur, '₴')}: закрыто ({paid:,.0f})")
                else:
                    debt_amt["amount"] = new
                    remaining.append(debt_amt)
                    lines.append(f"💸 {CURRENCY_SYMBOLS.get(cur, '₴')}: {paid:,.0f} → остаток *{new:,.0f}*")
                break
        
        if not found_cur:
            # Новая валюта? Странно, но обработаем
            lines.append(f"⚠️ {CURRENCY_SYMBOLS.get(cur, '₴')}: {paid:,.0f} (не найдено в долге)")
    
    # Обновляем или закрываем долг
    if remaining:
        debt.amounts = remaining
        update_debt_amounts(did, remaining)
        lines.append(f"\n📊 Остаток: {format_amounts(remaining)}")
        
        # Кнопки действий
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Закрыть долг", callback_data=f"debt_close_{did}")],
            [InlineKeyboardButton("⏰ Напомнить позже", callback_data=f"debt_remind_{did}")]
        ])
        
        await update.message.reply_text(
            "\n".join(lines),
            parse_mode="Markdown",
            reply_markup=kb
        )
    else:
        # Полностью погашено
        debt.status = "paid"
        update_debt_status(did, "paid")
        debts.pop(did, None)
        lines.append(f"\n🎉 *Долг полностью закрыт!*")
        
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def show_debts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показать все долги"""
    user_id = update.effective_chat.id
    load_debts(user_id)  # Перезагружаем актуальные данные
    
    active_debts = {k: v for k, v in debts.items() if v.status == "active"}
    
    if not active_debts:
        await update.message.reply_text("✅ *Активных долгов нет!*", parse_mode="Markdown")
        return
    
    lines = ["💸 *Активные долги:*\n"]
    totals = defaultdict(float)
    
    for did, debt in active_debts.items():
        days_ago = (datetime.now() - debt.date).days
        note = f" — _{debt.note}_" if debt.note else ""
        
        lines.append(f"👤 *{debt.name}* — {format_amounts(debt.amounts)}{note}")
        lines.append(f"   📅 {debt.date.strftime('%d.%m.%Y')} ({days_ago} дн. назад)")
        
        for a in debt.amounts:
            totals[a.get("currency", "UAH")] += float(a["amount"])
        
        lines.append("")
    
    # Итоги по валютам
    if totals:
        lines.append("*Итого:*")
        for cur in ["UAH", "USD", "EUR"]:
            if cur in totals:
                lines.append(f"💰 {format_amounts([{'amount': totals[cur], 'currency': cur}])}")
    
    # Кнопки управления
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Обновить", callback_data="debts_refresh")],
        [InlineKeyboardButton("➕ Добавить долг", callback_data="debt_add")]
    ])
    
    await update.message.reply_text(
        "\n".join(lines),
        parse_mode="Markdown",
        reply_markup=kb
    )

async def send_debt_reminder(context: ContextTypes.DEFAULT_TYPE):
    """Отправка напоминания о долге"""
    job = context.job
    data = job.data
    did = data.get("debt_id")
    chat_id = data.get("chat_id")
    
    if did not in debts:
        return
    
    debt = debts[did]
    
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Вернули", callback_data=f"debt_paid_{did}")],
        [InlineKeyboardButton("⏰ Напомнить ещё", callback_data=f"debt_remind_{did}")]
    ])
    
    note = f"\n📝 {debt.note}" if debt.note else ""
    
    await context.bot.send_message(
        chat_id=chat_id,
        text=f"💸 *Напоминание о долге*\n\n"
             f"👤 *{debt.name}* должен {format_amounts(debt.amounts)}{note}\n"
             f"📅 С {debt.date.strftime('%d.%m.%Y')}\n\n"
             f"Долг вернули?",
        parse_mode="Markdown",
        reply_markup=kb
    )

# ── REPORT FUNCTIONS ─────────────────────────────────────────────────────────-

async def show_stats(update: Update, context: ContextTypes.DEFAULT_TYPE, period: str = "month"):
    """Показать статистику"""
    user_id = update.effective_chat.id
    days = 7 if period == "week" else 30
    
    expenses = get_expenses(user_id, days)
    if not expenses:
        await update.message.reply_text("📭 Нет данных за этот период.")
        return
    
    analysis = analyze_expenses(expenses)
    
    period_name = "неделю" if days == 7 else "месяц"
    now = datetime.now()
    
    lines = [f"📊 *Статистика за {period_name}*\n"]
    lines.append(f"💰 *Всего: {analysis['total']:,.0f} ₴* ({analysis['count']} трат)")
    lines.append(f"📈 В среднем: {analysis['avg_per_day']:,.0f} ₴/день\n")
    
    # По категориям
    lines.append("*По категориям:*")
    for cat, amt in sorted(analysis["by_category"].items(), key=lambda x: -x[1]):
        pct = amt / analysis["total"] * 100
        emoji = CATEGORY_EMOJIS.get(cat, "•")
        lines.append(f"{emoji} {cat}: {amt:,.0f} ₴ ({pct:.0f}%)")
    
    # По дням недели
    if analysis["by_day"]:
        lines.append(f"\n*По дням:*")
        for day in ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]:
            if day in analysis["by_day"]:
                lines.append(f"• {day}: {analysis['by_day'][day]:,.0f} ₴")
    
    # Бюджет
    budget = get_budget_status(user_id)
    if budget:
        bar = "█" * (budget["percent"] // 10) + "░" * (10 - budget["percent"] // 10)
        status_emoji = "🔴" if budget["status"] == "critical" else "🟡" if budget["status"] == "warning" else "🟢"
        lines.append(f"\n{status_emoji} *Бюджет:* [{bar}] {budget['percent']}%")
        lines.append(f"💳 Осталось: {budget['left']:,.0f} ₴")
    
    # Зарплата
    salary = get_salary_status(user_id)
    if salary:
        lines.append(f"\n💵 *Зарплата:* {salary['days_left']} дней")
        if salary.get("daily_budget"):
            lines.append(f"📊 Можно тратить: ~{salary['daily_budget']:,.0f} ₴/день")
    
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def show_comparison(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Сравнение с прошлым месяцем"""
    user_id = update.effective_chat.id
    
    # Текущий месяц
    now = datetime.now()
    current_month = now.month
    current_year = now.year
    
    # Получаем все расходы
    all_expenses = get_expenses(user_id, 90)  # 3 месяца
    
    # Группируем по месяцам
    months_data = defaultdict(list)
    for r in all_expenses:
        try:
            d = datetime.strptime(r["Дата"][:10], "%d.%m.%Y")
            months_data[(d.year, d.month)].append(r)
        except:
            continue
    
    if len(months_data) < 2:
        await update.message.reply_text("📭 Нужно минимум 2 месяца данных для сравнения.")
        return
    
    lines = ["🪞 *Сравнение месяцев*\n"]
    
    # Сортируем месяцы
    sorted_months = sorted(months_data.items(), reverse=True)[:3]
    
    prev_total = None
    for (year, month), records in sorted_months:
        analysis = analyze_expenses(records)
        total = analysis["total"]
        name = f"{get_month_name(month)} {year}"
        
        if prev_total:
            diff_pct = (total - prev_total) / prev_total * 100
            arrow = "📈" if diff_pct > 0 else "📉"
            sign = "+" if diff_pct > 0 else ""
            lines.append(f"*{name}:* {total:,.0f} ₴ {arrow} {sign}{diff_pct:.0f}%")
        else:
            lines.append(f"*{name}:* {total:,.0f} ₴ (текущий)")
        
        # Топ категория
        if analysis["by_category"]:
            top_cat = max(analysis["by_category"].items(), key=lambda x: x[1])
            emoji = CATEGORY_EMOJIS.get(top_cat[0], "•")
            lines.append(f"  └ {emoji} {top_cat[0]}: {top_cat[1]:,.0f} ₴")
        
        prev_total = total
        lines.append("")
    
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def show_habits(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Анализ привычек (частые траты)"""
    user_id = update.effective_chat.id
    expenses = get_expenses(user_id, 90)
    
    if len(expenses) < 10:
        await update.message.reply_text("📭 Недостаточно данных. Нужно минимум 10 трат за 3 месяца.")
        return
    
    # Анализируем повторяющиеся траты
    by_desc = defaultdict(lambda: {"count": 0, "total": 0.0, "avg": 0})
    for r in expenses:
        desc = r.get("Описание", "").lower().strip()
        if len(desc) < 3:
            continue
        amt = float(r["Сумма"]) if r.get("Сумма") else 0
        by_desc[desc]["count"] += 1
        by_desc[desc]["total"] += amt
    
    # Фильтруем привычки (минимум 3 раза, минимум 500 всего)
    habits = {k: v for k, v in by_desc.items() if v["count"] >= 3 and v["total"] >= 500}
    
    if not habits:
        await update.message.reply_text("📊 Пока не выявлено явных привычек. Продолжай записывать!")
        return
    
    lines = ["💸 *Стоимость привычек*\n"]
    
    for desc, data in sorted(habits.items(), key=lambda x: -x[1]["total"])[:5]:
        monthly = data["total"] / 3  # ~в месяц
        yearly = monthly * 12
        
        # Ищем эквивалент
        equiv = None
        for threshold, label in EQUIVALENTS:
            if yearly >= threshold * 0.8:
                equiv = label
                break
        
        lines.append(f"*{desc.capitalize()}*")
        lines.append(f"  📅 {data['count']} раз = {data['total']:,.0f} ₴")
        lines.append(f"  📆 ~{monthly:,.0f} ₴/мес = {yearly:,.0f} ₴/год")
        if equiv:
            lines.append(f"  💡 Это = {equiv}")
        lines.append("")
    
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

# ── MAIN MESSAGE PROCESSOR ───────────────────────────────────────────────────

async def process_message(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    """Главный процессор сообщений"""
    chat_id = update.effective_chat.id
    user_ctx = get_user_context(chat_id)
    
    # Проверяем ожидающие действия
    if user_ctx.get("pending_action"):
        action = user_ctx["pending_action"]
        
        if action == "confirm_category" and user_ctx.get("last_expense"):
            cat = text.strip()
            if cat in CATEGORIES:
                exp = user_ctx["last_expense"]
                exp.category = cat
                save_expense(exp, chat_id)
                save_memory(chat_id, exp.description.lower(), cat)
                
                emoji = CATEGORY_EMOJIS.get(cat, "💰")
                await update.message.reply_text(
                    f"✅ {emoji} {cat}: *{exp.amount:,.0f} ₴* ({exp.description})",
                    parse_mode="Markdown"
                )
                user_ctx["pending_action"] = None
                user_ctx["last_expense"] = None
                return
            else:
                await update.message.reply_text(f"Выбери категорию из списка: {', '.join(CATEGORIES)}")
                return
        
        elif action == "confirm_debt_name" and user_ctx.get("pending_debt"):
            # Уточнение имени для долга
            pass  # Реализовать при необходимости
    
    # Определяем тип сообщения
    text_lower = text.lower()
    
    # 1. Долги
    if any(kw in text_lower for kw in ["долг", "одолжил", "дал в долг", "вернул", "отдал", "забрал"]):
        parsed = await ai_parse_debt(text)
        intent = parsed.get("intent")
        
        if intent == "debt_new":
            await process_debt_new(update, context, parsed)
            return
        elif intent == "debt_payment":
            await process_debt_payment(update, context, parsed)
            return
    
    # 2. Бюджет
    if "бюджет" in text_lower:
        nums = re.findall(r'\d+', text)
        if nums:
            amount = float(nums[0])
            save_setting(chat_id, "monthly_budget", str(amount))
            await update.message.reply_text(f"💰 Бюджет установлен: *{amount:,.0f} ₴/мес*", parse_mode="Markdown")
            return
    
    # 3. Зарплата
    if any(kw in text_lower for kw in ["зарплата", "зп", "аванс", "получаю"]):
        nums = re.findall(r'\d+', text)
        if nums:
            day = int(nums[0])
            amount = float(nums[1]) if len(nums) > 1 else None
            save_setting(chat_id, "salary_day", str(day))
            if amount:
                save_setting(chat_id, "salary_amount", str(amount))
            
            msg = f"💵 Зарплата: *{day}-е число*"
            if amount:
                msg += f", {amount:,.0f} ₴"
            await update.message.reply_text(msg, parse_mode="Markdown")
            return
    
    # 4. Запросы статистики (ключевые слова)
    if any(kw in text_lower for kw in ["сколько", "статус", "отчёт", "тратил", "потратил", "баланс"]):
        if any(kw in text_lower for kw in ["недел", "week"]):
            await show_stats(update, context, "week")
        else:
            await show_stats(update, context, "month")
        return
    
    # 5. Сравнение
    if any(kw in text_lower for kw in ["сравни", "прошлый", "раньше", "динамика"]):
        await show_comparison(update, context)
        return
    
    # 6. Привычки
    if any(kw in text_lower for kw in ["привычк", "часто", "регулярно", "паттерн"]):
        await show_habits(update, context)
        return
    
    # 7. Совет
    if any(kw in text_lower for kw in ["совет", "рекомендация", "что делать", "как сэкономить", "помоги"]):
        insight = await generate_ai_insight(chat_id)
        await update.message.reply_text(insight, parse_mode="Markdown")
        return
    
    # 8. Траты (по умолчанию)
    parsed = await ai_parse_expenses(text)
    
    if parsed.get("intent") == "expense" and parsed.get("items"):
        items = parsed["items"]
        
        # Проверяем уверенность
        if parsed.get("confidence", 1) < 0.6 and len(items) == 1:
            # Низкая уверенность — спрашиваем категорию
            exp = Expense(
                amount=items[0]["amount"],
                currency=items[0].get("currency", "UAH"),
                description=items[0].get("description", "трата")
            )
            user_ctx["last_expense"] = exp
            user_ctx["pending_action"] = "confirm_category"
            
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
        
        # Сохраняем траты
        await process_expenses(update, context, items, text)
        
        # Сохраняем в память для обучения
        for item in items:
            if item.get("description"):
                save_memory(chat_id, item["description"].lower(), item.get("category", "Другое"))
    else:
        # Не распознали — свободный разговор с AI
        # Собираем контекст
        expenses = get_expenses(chat_id, 7)
        context_data = {
            "recent_total": sum(float(r["Сумма"]) for r in expenses) if expenses else 0,
            "has_budget": get_budget_status(chat_id) is not None,
            "has_salary": get_salary_status(chat_id) is not None
        }
        
        response = await ai_conversation(text, context_data)
        await update.message.reply_text(response)

async def process_expenses(update: Update, context: ContextTypes.DEFAULT_TYPE, items: List[Dict], raw_text: str):
    """Обработка списка трат"""
    chat_id = update.effective_chat.id
    lines = ["✅ *Записано!*\n"]
    total_uah = 0
    
    for item in items:
        # Парсим дату
        date_str = item.get("date")
        if date_str:
            try:
                exp_date = datetime.fromisoformat(date_str)
            except:
                exp_date = parse_smart_date(raw_text)
        else:
            exp_date = parse_smart_date(raw_text)
        
        # Проверяем память пользователя
        desc = item.get("description", "трата")
        mem_cat = get_memory(chat_id, desc.lower())
        category = mem_cat if mem_cat else item.get("category", "Другое")
        
        exp = Expense(
            amount=float(item.get("amount", 0)),
            currency=item.get("currency", "UAH"),
            category=category,
            description=desc,
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
    budget = get_budget_status(chat_id)
    if budget:
        if budget["status"] == "critical":
            lines.append(f"\n🔴 *Бюджет на {budget['percent']}%!* Осталось {budget['left']:,.0f} ₴")
        elif budget["status"] == "warning":
            lines.append(f"\n🟡 Бюджет использован на {budget['percent']}%")
    
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

# ── HANDLERS ─────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Старт"""
    chat_id = update.effective_chat.id
    load_debts(chat_id)  # Загружаем долги при старте
    
    await update.message.reply_text(
        "👋 *AI Финансовый Агент v5.0*\n\n"
        "🎙 *Голосом или текстом* — я всё пойму:\n"
        "• «600 на снюс и 850 на топливо» — несколько трат\n"
        "• «Дал в долг Саше 5000» — учёт долгов\n"
        "• «Сколько потратил на неделе?» — аналитика\n"
        "• «Бюджет 20000, зарплата 25 числа» — планирование\n\n"
        "💡 *AI-функции:*\n"
        "• Умное распознавание категорий\n"
        "• Напоминания о долгах\n"
        "• Анализ привычек\n"
        "• Персональные советы\n\n"
        "Просто напиши или скажи!",
        parse_mode="Markdown",
        reply_markup=ReplyKeyboardMarkup([
            [KeyboardButton("📊 Статус"), KeyboardButton("💸 Долги")],
            [KeyboardButton("📈 Анализ"), KeyboardButton("💡 Совет")],
            [KeyboardButton("🪞 Сравнение"), KeyboardButton("💸 Привычки")]
        ], resize_keyboard=True)
    )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текста"""
    text = update.message.text
    
    # Обработка кнопок меню
    if text == "📊 Статус":
        await show_stats(update, context, "month")
        return
    if text == "📈 Анализ":
        await show_stats(update, context, "week")
        return
    if text == "💸 Долги":
        await show_debts(update, context)
        return
    if text == "💡 Совет":
        insight = await generate_ai_insight(update.effective_chat.id)
        await update.message.reply_text(insight, parse_mode="Markdown")
        return
    if text == "🪞 Сравнение":
        await show_comparison(update, context)
        return
    if text == "💸 Привычки":
        await show_habits(update, context)
        return
    
    # Основная обработка
    await process_message(update, context, text)

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
        
        # Обрабатываем как текст
        await process_message(update, context, text)
        
    except Exception as e:
        logger.error(f"Voice error: {e}")
        await msg.edit_text("❌ Ошибка распознавания. Попробуй текстом.")

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка inline кнопок"""
    query = update.callback_query
    await query.answer()
    data = query.data
    
    chat_id = query.message.chat_id
    
    if data.startswith("cat_"):
        # Подтверждение категории
        cat = data[4:]
        user_ctx = get_user_context(chat_id)
        
        if user_ctx.get("pending_action") == "confirm_category" and user_ctx.get("last_expense"):
            exp = user_ctx["last_expense"]
            exp.category = cat
            save_expense(exp, chat_id)
            save_memory(chat_id, exp.description.lower(), cat)
            
            emoji = CATEGORY_EMOJIS.get(cat, "💰")
            await query.edit_message_text(
                f"✅ {emoji} {cat}: *{exp.amount:,.0f} ₴* ({exp.description})",
                parse_mode="Markdown"
            )
            user_ctx["pending_action"] = None
            user_ctx["last_expense"] = None
    
    elif data == "debts_refresh":
        await show_debts(update, context)
    
    elif data == "debt_add":
        await query.edit_message_text(
            "💸 *Новый долг*\n\nНапиши: «Дал в долг [имя] [сумма] [валюта]»\n"
            "Пример: «Дал в долг Саше 5000 грн на покушать»",
            parse_mode="Markdown"
        )
    
    elif data.startswith("debt_close_"):
        did = data[11:]
        if did in debts:
            debt = debts[did]
            update_debt_status(did, "paid")
            debts.pop(did, None)
            await query.edit_message_text(
                f"✅ Долг *{debt.name}* закрыт!\n🎉 {format_amounts(debt.amounts)}",
                parse_mode="Markdown"
            )
    
    elif data.startswith("debt_remind_"):
        did = data[12:]
        if did in debts and context.job_queue:
            # Откладываем напоминание на 3 дня
            context.job_queue.run_once(
                send_debt_reminder,
                when=timedelta(days=3),
                data={"debt_id": did, "chat_id": chat_id},
                name=f"debt_{did}_remind"
            )
            await query.edit_message_text("⏰ Напомню через 3 дня.")
    
    elif data.startswith("debt_paid_"):
        did = data[10:]
        if did in debts:
            debt = debts[did]
            # Помечаем как погашенный
            update_debt_status(did, "paid")
            debts.pop(did, None)
            await query.edit_message_text(
                f"🎉 Отлично! Долг *{debt.name}* погашен.\n💰 {format_amounts(debt.amounts)}",
                parse_mode="Markdown"
            )

# ── SCHEDULED TASKS ─────────────────────────────────────────────────────────-

async def scheduled_budget_check(context: ContextTypes.DEFAULT_TYPE):
    """Ежедневная проверка бюджета"""
    # Реализовать при необходимости
    pass

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    init_db()
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    # Scheduled jobs
    if app.job_queue and CHAT_ID:
        # Ежедневный инсайт в 19:00
        app.job_queue.run_daily(
            lambda ctx: ctx.bot.send_message(
                chat_id=CHAT_ID, 
                text=asyncio.run(generate_ai_insight(int(CHAT_ID)))
            ),
            time=datetime.strptime("19:00", "%H:%M").time()
        )
    
    logger.info(f"AI Финансовый Агент v5.0 запущен! Модель: {AI_MODEL}")
    app.run_polling()

if __name__ == "__main__":
    main()
