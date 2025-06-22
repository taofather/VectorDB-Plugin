import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml

from langchain_community.vectorstores import TileDB
from ..base import VectorDatabase

class TileDBVectorDB(VectorDatabase):
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.config = None
        self.db = None
        self.embeddings = None
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize TileDB with configuration"""
        self.config = config
        self.db = None
        self.embeddings = None
    
    def create_database(self, database_name: str, texts: List[str], embeddings: List[List[float]], 
                       metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        """Create a new TileDB vector database"""
        persist_directory = self.root_dir / "Vector_DB" / database_name
        
        # Create TileDB vector store
        TileDB.from_embeddings(
            text_embeddings=[(txt, emb) for txt, emb in zip(texts, embeddings)],
            embedding=self.embeddings,
            metadatas=metadatas,
            ids=ids,
            metric="euclidean",
            index_uri=str(persist_directory),
            index_type="FLAT",
            allow_dangerous_deserialization=True,
        )
        
        # Update config
        self._update_config_with_database(database_name)
    
    def search(self, query: str, k: int = 5, score_threshold: float = 0.8, 
              filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search the TileDB database"""
        if not self.db:
            raise ValueError("Database not initialized")
        
        results = self.db.similarity_search_with_score(
            query,
            k=k,
            filter=filter_dict,
            score_threshold=score_threshold
        )
        
        return [
            {
                'text': doc.page_content,
                'metadata': doc.metadata,
                'score': score
            }
            for doc, score in results
        ]
    
    def delete_database(self, database_name: str) -> None:
        """Delete a TileDB database"""
        db_path = self.root_dir / "Vector_DB" / database_name
        if db_path.exists():
            shutil.rmtree(db_path)
        
        # Remove from config
        self._remove_database_from_config(database_name)
    
    def list_databases(self) -> List[str]:
        """List all available TileDB databases"""
        config_path = self.root_dir / "config.yaml"
        if not config_path.exists():
            return []
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            return list(config.get('created_databases', {}).keys())
    
    def cleanup(self) -> None:
        """Clean up TileDB resources"""
        self.db = None
        self.embeddings = None
    
    def _update_config_with_database(self, database_name: str) -> None:
        """Update config.yaml with new database information"""
        config_path = self.root_dir / "config.yaml"
        if not config_path.exists():
            return
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file) or {}
        
        if 'created_databases' not in config:
            config['created_databases'] = {}
        
        config['created_databases'][database_name] = {
            'model': self.config.get('EMBEDDING_MODEL_NAME'),
            'chunk_size': self.config.get('database', {}).get('chunk_size'),
            'chunk_overlap': self.config.get('database', {}).get('chunk_overlap')
        }
        
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.safe_dump(config, file, allow_unicode=True)
    
    def _remove_database_from_config(self, database_name: str) -> None:
        """Remove database from config.yaml"""
        config_path = self.root_dir / "config.yaml"
        if not config_path.exists():
            return
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        if 'created_databases' in config and database_name in config['created_databases']:
            del config['created_databases'][database_name]
        
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.safe_dump(config, file, allow_unicode=True) 