"""
Main retrieval class for RAG Chatbot
"""
import os
import json
from typing import List, Dict, Any, Optional

from config.config import (
    EMBEDDING_MODEL, TOP_K, SCORE_THRESHOLD
)
from retriever.preprocessor import TextPreprocessor
from retriever.embedder import Embedder
from retriever.index import FAISSIndex

class Retriever:
    """Main retrieval class that combines preprocessing, embedding, and indexing"""
    
    def __init__(self, 
                model_name: str = EMBEDDING_MODEL,
                index_path: Optional[str] = None,
                documents_path: Optional[str] = None):
        """Initialize the retriever with model, index, and documents"""
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor()
        
        # Initialize embedder
        self.embedder = Embedder(model_name)
        
        # Initialize index
        self.index = FAISSIndex()
        
        # Load index if path is provided
        if index_path and os.path.exists(index_path):
            self.index.load(index_path)
        
        # Load documents
        self.documents = []
        if documents_path and os.path.exists(documents_path):
            with open(documents_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
    
    def index_documents(self, documents: List[Dict[str, Any]], index_path: Optional[str] = None) -> None:
        """Process and index the provided documents"""
        # Save documents for later retrieval
        self.documents = documents
        
        # Extract texts from documents
        texts = [doc['text'] for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedder.embed_text(texts, show_progress=True)
        
        # Create and add to index
        self.index.add_embeddings(embeddings)
        
        # Save index if path is provided
        if index_path:
            self.index.save(index_path)
    
    def save_index_and_documents(self, index_path: str, documents_path: str) -> None:
        """Save index and documents to disk"""
        # Save index
        self.index.save(index_path)
        
        # Save documents
        os.makedirs(os.path.dirname(documents_path), exist_ok=True)
        with open(documents_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
    
    def search(self, query: str, top_k: int = TOP_K, 
              threshold: float = SCORE_THRESHOLD) -> List[Dict[str, Any]]:
        """
        Search for documents relevant to the query
        Returns list of results with text, metadata, and relevance score
        """
        # Preprocess query
        processed_query = self.preprocessor.process(query)
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(processed_query)
        
        # Search in index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Normalize scores (FAISS returns IP similarity, higher is better)
        # Convert to 0-1 range for easier interpretation
        scores = distances[0]  # Get first row of distances (for single query)
        
        # Format results
        results = []
        for i, doc_idx in enumerate(indices[0]):  # Get first row of indices
            if doc_idx == -1 or doc_idx >= len(self.documents):
                continue  # Invalid index
            
            score = float(scores[i])
            
            # Skip results below threshold
            if score < threshold:
                continue
            
            document = self.documents[doc_idx]
            results.append({
                "text": document["text"],
                "metadata": document.get("metadata", {}),
                "score": score,
                "id": document.get("id", f"doc_{doc_idx}")
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results
    
    def rerank_results(self, query: str, results: List[Dict[str, Any]], 
                      num_results: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Optional second-stage reranking for better precision
        Uses more expensive cross-encoder scoring
        """
        # TODO: Implement cross-encoder reranking if needed
        return results[:num_results]
    
    def retrieve(self, query: str, top_k: int = TOP_K, 
                threshold: float = SCORE_THRESHOLD,
                use_reranking: bool = False) -> List[Dict[str, Any]]:
        """
        Main retrieval method that handles the full pipeline:
        - Query processing
        - Retrieval
        - Optional reranking
        - Formatting results with source links
        """
        # Get raw search results
        results = self.search(query, top_k=top_k*2 if use_reranking else top_k, threshold=threshold)
        
        # Apply reranking if specified
        if use_reranking and len(results) > 0:
            results = self.rerank_results(query, results, num_results=top_k)
        
        # Format results with source links
        formatted_results = []
        for result in results:
            # Extract source URL from metadata
            source_url = result.get("metadata", {}).get("source_url", "")
            
            formatted_results.append({
                "text": result["text"],
                "source_url": source_url,
                "title": result.get("metadata", {}).get("title", "info"),
                "score": result["score"]
            })
        
        return formatted_results
    