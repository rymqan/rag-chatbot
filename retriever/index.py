"""
FAISS indexing functionality for RAG Chatbot
"""
import numpy as np
import faiss
import os
from typing import Tuple
from pathlib import Path

from config.config import EMBEDDING_DIMENSION, INDEX_TYPE

class FAISSIndex:
    """Wrapper for FAISS index used for vector similarity search"""
    
    def __init__(self, dimension: int = EMBEDDING_DIMENSION, index_type: str = INDEX_TYPE):
        """Initialize FAISS index with specified parameters"""
        self.dimension = dimension
        self.index_type = index_type
        
        # Create appropriate index based on type
        if index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        elif index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(dimension)  # L2 distance
    
    def add_embeddings(self, embeddings: np.ndarray) -> None:
        """Add embeddings to the index"""
        if len(embeddings.shape) == 1:
            # Single embedding, reshape to 2D
            embeddings = embeddings.reshape(1, -1)
        
        # Ensure correct type
        embeddings = embeddings.astype(np.float32)
        
        # Add to index
        self.index.add(embeddings)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors in the index
        Returns distances and indices
        """
        if len(query_embedding.shape) == 1:
            # Single embedding, reshape to 2D
            query_embedding = query_embedding.reshape(1, -1)
        
        # Ensure correct type
        query_embedding = query_embedding.astype(np.float32)
        
        # Search in index
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices

    def save(self, path: str) -> None:
        """Save index to file"""
        path = Path(path)
        
        # Ensure directory exists
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise
        index_to_save = self.index
        
        # Save index
        try:
            faiss.write_index(index_to_save, str(path))
        except Exception as e:
            raise
    
    def load(self, path: str) -> None:
        """Load index from file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file not found: {path}")
        
        # Load index
        self.index = faiss.read_index(path)

def create_and_save_index(embeddings: np.ndarray, index_path: str) -> FAISSIndex:
    """Create FAISS index from embeddings and save to file"""
    # Initialize index
    index = FAISSIndex()
    
    # Add embeddings
    index.add_embeddings(embeddings)
    
    # Save index
    index.save(index_path)
    
    return index
