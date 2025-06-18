from typing import Dict, Any
from config_manager import ConfigManager

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
        config = ConfigManager().get_config()
        return config.get('database', {}).get('type', 'tiledb') 