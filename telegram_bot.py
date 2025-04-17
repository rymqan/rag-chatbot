from telegram import Update, ForceReply
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

from retriever.retriever import Retriever
from generator.generator import generate_answer
from config.config import (
    INDEX_PATH, DOCUMENTS_PATH, 
    HELP_MESSAGE, MISS_MESSAGE,
    TELEGRAM_TOKEN
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_html(
        rf"Здравствуйте, {user.mention_html()}! Я — виртуальный ассистент. Задайте интересующий вопрос. Для подробностей напишите /help",
        reply_markup=ForceReply(selective=True),
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = (
        HELP_MESSAGE
    )
    await update.message.reply_text(message)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.message.text.strip()
    retriever = context.application.bot_data["retriever"]
    results = retriever.retrieve(query)

    if not results:
        await update.message.reply_text(MISS_MESSAGE)
        return

    answer = generate_answer(query, results)
    await update.message.reply_text(answer)

start_handler = CommandHandler("start", start)
help_handler = CommandHandler("help", help_command)
message_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)

def main():
    retriever = Retriever(index_path=INDEX_PATH, documents_path=DOCUMENTS_PATH)

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.bot_data["retriever"] = retriever  # Pass retriever to handlers

    application.add_handler(start_handler)
    application.add_handler(help_handler)
    application.add_handler(message_handler)

    application.run_polling()

if __name__ == "__main__":
    main()
