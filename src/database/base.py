from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

class VectorDatabase(ABC):
    """Abstract base class for vector databases"""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the database with configuration"""
        pass
    
    @abstractmethod
    def create_database(self, database_name: str, texts: List[str], embeddings: List[List[float]], 
                       metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        """Create a new vector database"""
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 5, score_threshold: float = 0.8, 
              filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search the database for similar vectors"""
        pass
    
    @abstractmethod
    def delete_database(self, database_name: str) -> None:
        """Delete a vector database"""
        pass
    
    @abstractmethod
    def list_databases(self) -> List[str]:
        """List all available databases"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources"""
        pass 