"""
Main entry point for RAG Chatbot Engine
"""
import os
import argparse

from retriever.preprocessor import process_documents
from retriever.embedder import embed_documents
from retriever.index import create_and_save_index
from retriever.retriever import Retriever
from generator.generator import generate_answer
from config.config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, 
    INDEX_PATH, DOCUMENTS_PATH,
    MISS_MESSAGE, BYE_MESSAGE
)

def prepare_data():
    """Process raw data into chunks and generate embeddings and index"""
    # Process documents
    _ = process_documents(RAW_DATA_PATH, DOCUMENTS_PATH)
    
    # Generate embeddings
    _, embeddings = embed_documents(DOCUMENTS_PATH)
    
    # Create index
    create_and_save_index(embeddings, INDEX_PATH)
    

def query_interactive(retriever: Retriever):
    """Interactive query mode for testing the retriever"""
    while True:
        query = input("Вопрос: ")
        
        if query.lower() in ['quit', 'exit', 'q']:
            print(BYE_MESSAGE)
            break
        
        if not query.strip():
            continue
        
        # Retrieve results
        results = retriever.retrieve(query)
        
        if not results:
            print(MISS_MESSAGE)
            continue

        answer = generate_answer(query, results)
        print(answer)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="RAG Chatbot Engine")
    parser.add_argument("--prepare", action="store_true", help="Prepare data (process, embed, index)")
    parser.add_argument("--query", action="store_true", help="Enter interactive query mode")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    
    # Handle commands
    if args.prepare:
        prepare_data()
    
    # Only initialize retriever if needed for query mode
    if args.query:
        # Initialize retriever
        retriever = Retriever(
            index_path=INDEX_PATH if os.path.exists(INDEX_PATH) else None,
            documents_path=DOCUMENTS_PATH if os.path.exists(DOCUMENTS_PATH) else None
        )
        query_interactive(retriever)
    
    # If no arguments provided, show help
    if not (args.prepare or args.query or args.test):
        parser.print_help()

if __name__ == "__main__":
    main()
