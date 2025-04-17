# RAG Chatbot Engine

A retrieval-based chatbot engine for website that can answer questions about products and services using natural language processing (NLP) techniques.

## Overview

This system uses a modern retrieval-based approach to:

1. Scrapes the content of the website
2. Process and index the scraped data
3. Understand user questions in natural language
4. Retrieve relevant information using dense vector similarity
5. Return answers with links to their original sources

## Features

- Text preprocessing and chunking for optimal retrieval
- Multilingual support with focus on Russian language
- Dense vector embeddings using Sentence Transformers
- Fast similarity search with FAISS
- Retrieval with source attribution
- Interactive query interface for testing

## Project Structure

```
rag-chatbot/
├── config/              # Configuration components
│   └── config.py/       # Retrieval and generation parameters
├── data/                # Data storage
│   ├── raw/             # Scraped data
│   ├── processed/       # Processed chunks
│   └── index.faiss      # FAISS index file
├── generator/           # Text generator components
│   └── generator.py     # Transformer-based generation
├── retriever/           # Core retrieval components
│   ├── preprocessor.py  # Text processing
│   ├── embedder.py      # Vector embeddings
│   ├── index.py         # FAISS indexing
│   └── retriever.py     # Main retrieval class
├── main.py              # Main application entry with CLI
├── README.md            # Project readme document
├── requirements.txt     # Dependencies
├── scraper.py           # Website scraper logic
└── telegram_bot.py      # Telegram bot logic
```

## Installation

1. Clone the repository
   ```
   git clone https://github.com/rymqan/rag-chatbot.git
   cd rag-chatbot
   ```
2. Create a virtual environment:
   ```
   python -m venv rag_chatbot_env
   source rag_chatbot_env/bin/activate  # On Windows: rag_chatbot_env\Scripts\activate.bat
   ```
3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Download the spaCy Russian language model:
   ```
   python -m spacy download ru_core_news_md
   ```

## Usage

### Data Preparation

Modify the website link to be parsed in the config file:

```python
# config/config.py
EMBEDDING_MODEL = "https://your-preferred-website"
```

Scrape the website data with:

```
python scraper.py
```

It will be saved in the `data/raw/` directory as JSON files with the following structure:

```json
[
  {
    "title": "Product/Service Title",
    "url": "Website link",
    "text": "Full text content",
    "metadata": {
      "additional_field1": "value1",
      "additional_field2": "value2"
    }
  }
]
```

Then process the data with:

```
python main.py --prepare
```

This will:
1. Process and chunk the text
2. Generate embeddings
3. Build a FAISS index for similarity search

### Interactive Query Mode

To start the interactive query interface:

```
python main.py --query
```

Type your questions about parsed website products and services to see the system in action.

### Telegram Bot Integration

1. Set the Telegram Token as an environment variable `TELEGRAM_TOKEN`
2. Run the bot that utilizes the retriever:

```
python telegram_bot.py
```

## Customization

### Changing the Embedding Model

To use a different sentence transformer model, modify `EMBEDDING_MODEL` in the config file:

```python
# config/config.py
EMBEDDING_MODEL = "your-preferred-model"
```

Tested embedding models for Russian language:
- paraphrase-multilingual-MiniLM-L12-v2
- all-MiniLM-L6-v2

### Adjusting Retrieval Parameters

You can tune retrieval parameters in the config file:

```python
# config/config.py
TOP_K = 3  # Number of results to return
SCORE_THRESHOLD = 1.0  # Minimum similarity score
```

### Changing the Generator Model

To use a different transformer model, modify `GENERATOR_MODEL` in the config file:

```python
# config/config.py
GENERATOR_MODEL = "your-preferred-model"
```

Tested text generation models for Russian language:
- Qwen/Qwen2.5-0.5B-Instruct
- Vikhrmodels/QVikhr-2.5-1.5B-Instruct-r

## Performance Considerations

- For large datasets (>100K documents), consider using:
  - Quantized FAISS indices (e.g., IndexIVFPQ)
  - Batch processing for embeddings
  - GPU acceleration if available

## Troubleshooting

Common issues:

- **No results returned**: Check SCORE_THRESHOLD value, might be too high
- **Irrelevant results**: Try a different embedding model or improve chunking
- **Slow performance**: Consider optimizing FAISS index type or using GPU
