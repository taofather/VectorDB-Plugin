from typing import Dict, Any
import yaml
from pathlib import Path

from .base import VectorDatabase
from .tiledb.implementation import TileDBVectorDB
from .pgvector.implementation import PGVectorDB

class VectorDBFactory:
    @staticmethod
    def create_database(config: Dict[str, Any]) -> VectorDatabase:
        """Create a vector database instance based on configuration"""
        db_type = config.get('database', {}).get('type', 'tiledb')
        
        if db_type == 'pgvector':
            return PGVectorDB()
        else:
            return TileDBVectorDB()
    
    @staticmethod
    def get_database_type() -> str:
        """Get the current database type from config"""
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        if not config_path.exists():
            return 'tiledb'  # default
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            return config.get('database', {}).get('type', 'tiledb') 