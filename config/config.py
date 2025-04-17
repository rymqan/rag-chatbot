"""
Configuration for the RAG Chatbot Engine 
"""
# Scraper settings
TARGET_URL = "XXXXXXXX"
BLACKLIST = {
    "XXXXXXXX",
    "XXXXXXXX",
    "XXXXXXXX"
}

# System settings
DEVICE = "cuda:0"

# Embedding model settings
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384

# Generator model settings
GENERATOR_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 512

# FAISS index settings
INDEX_TYPE = "IndexFlatIP"  # Inner product for cosine similarity
USE_GPU = False  # Set to True if using GPU

# Text processing
LANGUAGE = "ru"
CHUNK_SIZE = 512  # Characters per chunk
CHUNK_OVERLAP = 50  # Character overlap between chunks

# Retrieval settings
TOP_K = 3  # Number of results to return
SCORE_THRESHOLD = 1  # Minimum similarity score to include results

# Paths
DATA_DIR = "data"
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
INDEX_PATH = "data/index.faiss"
DOCUMENTS_PATH = "data/processed/documents.json"

# Prompts
SYSTEM_PROMPT = """
Ты — виртуальный ассистент, обученный помогать клиентам, предоставляя точную, краткую и профессиональную информацию, основанную исключительно на материалах официального сайта.

Правила поведения:
- Отвечай в **дружелюбно-нейтральном и деловом стиле**.
- Не используй форматирование, специальные символы, эмодзи или markdown.
- Отвечай строго по данным сайта. Если ответ отсутствует — сообщи об этом и предложи обратиться в поддержку.
- Ссылайся на релевантные страницы сайта, если они известны.

Стиль и структура ответа:
1. **Основной ответ** — кратко, ясно и по делу (1–3 предложения).
2. **Уточнение** — при необходимости добавь полезную дополнительную информацию (1 предложение).
3. **Заключение** — вежливо заверши ответ, если вопрос исчерпан (например: "Если у вас остались вопросы, пожалуйста, обращайтесь.").

Обязательные вводные фразы в зависимости от источника:
- "Согласно информации на сайте..."
- "На данный момент в условиях продукта указано, что..."
- "На сайте указано следующее..."
"""

# Telegram bot settings
TELEGRAM_TOKEN = "XXXXXXXX"

HELP_MESSAGE = """
Я могу помочь вам найти информацию на сайте.

Примеры вопросов, которые вы можете задать:
• Вопрос 1
• Вопрос 2
• Вопрос 3

Напишите ваш вопрос, и я постараюсь помочь!
"""

MISS_MESSAGE = "К сожалению, по этому вопросу нет точной информации. Рекомендуем обратиться в службу поддержки."

BYE_MESSAGE = "Спасибо, что выбираете нас! До встречи!"