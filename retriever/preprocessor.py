"""
Text preprocessing utilities for the RAG Chatbot
"""
import re
import spacy
from typing import List, Dict, Any
import json
import os
from tqdm import tqdm

from config.config import LANGUAGE, CHUNK_SIZE, CHUNK_OVERLAP

class TextPreprocessor:
    """Text preprocessing class for cleaning and tokenizing text"""
    
    def __init__(self, lang: str = LANGUAGE):
        """Initialize the preprocessor with language model"""
        self.lang = lang
        self.nlp = spacy.load(f"{lang}_core_news_md")
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and normalizing"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize punctuation
        text = re.sub(r'[«»„""]', '"', text)
        
        # Trim whitespace
        text = text.strip()
        
        return text
    
    def process(self, text: str) -> str:
        """Process text through the full preprocessing pipeline"""
        # Clean the text first
        clean_text = self.clean_text(text)
        
        # Process with spaCy
        doc = self.nlp(clean_text)
        
        # Additional processing can be added here
        
        return clean_text
    
    def chunk_text(self, text: str, metadata: Dict[str, Any], 
                  chunk_size: int = CHUNK_SIZE, 
                  chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks while preserving paragraph structure.
        Returns a list of dictionaries with text and metadata.
        """
        # Clean the text first
        text = self.clean_text(text)
        
        # If text is smaller than chunk size, return as is
        if len(text) <= chunk_size:
            return [{
                "text": text,
                "metadata": metadata
            }]
        
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph exceeds chunk size and we already have content
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    "text": current_chunk,
                    "metadata": metadata
                })
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + "\n\n" + para
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "metadata": metadata
            })
        
        return chunks


def process_documents(input_path: str, output_path: str) -> List[Dict[str, Any]]:
    """
    Process all documents from the input directory and save chunks to output file.
    Expected input format: JSON files with documents containing text and metadata.
    """
    preprocessor = TextPreprocessor()
    all_chunks = []
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Process all files in the input directory
    for filename in tqdm(os.listdir(input_path)):
        if filename.endswith('.json'):
            file_path = os.path.join(input_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            # Handle both single document and list of documents
            if not isinstance(documents, list):
                documents = [documents]
            
            for doc in documents:
                if 'text' not in doc:
                    continue
                
                # Create metadata with source URL
                metadata = {
                    "source_url": doc.get('url', 'unknown'),
                    "source_file": filename
                }
                
                # Add any additional metadata from the document
                if 'metadata' in doc and isinstance(doc['metadata'], dict):
                    metadata.update(doc['metadata'])
                
                # Chunk the document
                chunks = preprocessor.chunk_text(doc['text'], metadata)
                
                # Add document ID to each chunk
                for i, chunk in enumerate(chunks):
                    chunk["id"] = f"{os.path.splitext(filename)[0]}_{i}"
                
                all_chunks.extend(chunks)
    
    # Save all chunks to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    return all_chunks
