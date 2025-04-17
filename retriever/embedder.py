"""
Embedding generation for RAG Chatbot
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Union
import os
import json
import torch

from config.config import EMBEDDING_MODEL

class Embedder:
    """Generate embeddings for text using sentence-transformers"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Initialize the embedder with the specified model"""
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def embed_text(self, text: Union[str, List[str]], 
                  batch_size: int = 32, 
                  show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for text or list of texts
        Returns numpy array of embeddings
        """
        # Handle single text
        if isinstance(text, str):
            text = [text]
        
        # Check available device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        embeddings = self.model.encode(
            text, 
            batch_size=batch_size,
            show_progress_bar=show_progress,
            device=device,
            convert_to_numpy=True
        )
        
        return embeddings

def embed_documents(documents_path: str, embeddings_output_path: Optional[str] = None) -> tuple:
    """
    Generate embeddings for all documents and return texts and embeddings.
    Also saves embeddings to file if output path is provided.
    """
    # Load documents
    with open(documents_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Extract texts
    texts = [doc['text'] for doc in documents]
    
    # Create embedder and generate embeddings
    embedder = Embedder()
    embeddings = embedder.embed_text(texts, show_progress=True)
    
    # Save embeddings if output path is provided
    if embeddings_output_path:
        os.makedirs(os.path.dirname(embeddings_output_path), exist_ok=True)
        np.save(embeddings_output_path, embeddings)
    
    return texts, embeddings
