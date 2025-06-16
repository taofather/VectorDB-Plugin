import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
import logging
from config_manager import ConfigManager

from ..base import VectorDatabase

class PGVectorDB(VectorDatabase):
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.config = None
        self.conn = None
        self.embeddings = None
        self.config_manager = ConfigManager()
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize PostgreSQL connection"""
        self.config = config
        pg_config = config.get('postgresql', {})
        
        try:
            self.conn = psycopg2.connect(
                host=pg_config.get('host', 'localhost'),
                port=pg_config.get('port', 5432),
                user=pg_config.get('user', 'postgres'),
                password=pg_config.get('password', ''),
                database=pg_config.get('database', 'vectordb')
            )
            
            # Enable pgvector extension
            with self.conn.cursor() as cur:
                cur.execute('CREATE EXTENSION IF NOT EXISTS vector;')
                self.conn.commit()
                
        except Exception as e:
            raise ValueError(f"Failed to connect to PostgreSQL: {e}")
    
    def create_database(self, database_name: str, texts: List[str], embeddings: List[List[float]], 
                       metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        """Create a new pgvector database"""
        if not self.conn:
            raise ValueError("Database not initialized")
        
        # Create table for this database
        table_name = f"vectors_{database_name}"
        with self.conn.cursor() as cur:
            # Create table with vector column
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id TEXT PRIMARY KEY,
                    text TEXT,
                    embedding vector,
                    metadata JSONB
                );
            """)
            
            # Create index for vector similarity search
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx 
                ON {table_name} 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            # Insert data
            data = [(id_, text, embedding, metadata) 
                   for id_, text, embedding, metadata in zip(ids, texts, embeddings, metadatas)]
            
            execute_values(cur, f"""
                INSERT INTO {table_name} (id, text, embedding, metadata)
                VALUES %s
            """, data)
            
            self.conn.commit()
        
        # Update config
        self._update_config_with_database(database_name)
    
    def search(self, query: str, k: int = 5, score_threshold: float = 0.8, 
              filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search the pgvector database"""
        if not self.conn:
            raise ValueError("Database not initialized")
        
        if not self.embeddings:
            raise ValueError("No embeddings available for search")
        
        query_embedding = self.embeddings.encode([query])[0]
        
        # Get the current database name from config
        config_data = self.config_manager.get_config()
        database_name = config_data.get('database', {}).get('database_to_search', '')
        if not database_name:
            raise ValueError("No database selected for search")
        
        table_name = f"vectors_{database_name}"
        
        with self.conn.cursor() as cur:
            # Search using cosine similarity
            cur.execute(f"""
                SELECT id, text, metadata, 1 - (embedding <=> %s::vector) as similarity
                FROM {table_name}
                WHERE 1 - (embedding <=> %s::vector) > %s
                ORDER BY similarity DESC
                LIMIT %s;
            """, (query_embedding.tolist(), query_embedding.tolist(), score_threshold, k))
            
            results = []
            for row in cur.fetchall():
                result = {
                    'id': row[0],
                    'text': row[1],
                    'metadata': row[2],
                    'similarity': row[3]
                }
                results.append(result)
            
            return results
    
    def delete_database(self, database_name: str) -> None:
        """Delete a pgvector database"""
        if not self.conn:
            raise ValueError("Database not initialized")
        
        table_name = f"vectors_{database_name}"
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table_name};")
            self.conn.commit()
        
        # Remove from config
        self._remove_database_from_config(database_name)
    
    def list_databases(self) -> List[str]:
        """List all available pgvector databases"""
        if not self.conn:
            raise ValueError("Database not initialized")
        
        tables = self._get_vector_tables()
        return [table.replace('vectors_', '') for table in tables]
    
    def cleanup(self) -> None:
        """Clean up PostgreSQL connection"""
        if self.conn:
            self.conn.close()
        self.conn = None
        self.embeddings = None
    
    def _get_vector_tables(self) -> List[str]:
        """Get list of vector tables in the database"""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE 'vectors_%';
            """)
            return [row[0] for row in cur.fetchall()]
    
    def _update_config_with_database(self, database_name: str) -> None:
        """Update config.yaml with new database information"""
        config_data = self.config_manager.get_config()
        
        if 'created_databases' not in config_data:
            config_data['created_databases'] = {}
        
        config_data['created_databases'][database_name] = {
            'model': self.config.get('EMBEDDING_MODEL_NAME'),
            'chunk_size': self.config.get('database', {}).get('chunk_size'),
            'chunk_overlap': self.config.get('database', {}).get('chunk_overlap')
        }
        
        self.config_manager.save_config(config_data)
    
    def _remove_database_from_config(self, database_name: str) -> None:
        """Remove database from config.yaml"""
        config_data = self.config_manager.get_config()
        
        if 'created_databases' in config_data and database_name in config_data['created_databases']:
            del config_data['created_databases'][database_name]
        
        self.config_manager.save_config(config_data) 