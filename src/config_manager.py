from pathlib import Path
import yaml
import logging
from typing import Any, Dict, Optional

class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the configuration manager"""
        self.root_dir = Path(__file__).resolve().parent.parent
        self.config_path = self.root_dir / "config.yaml"
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file, create if missing (always at project root)"""
        try:
            if not self.config_path.exists():
                self._create_default_config()
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            raise
    
    def _create_default_config(self) -> None:
        """Create default configuration file at project root (../config.yaml)"""
        default_config = {
            'database': {
                'type': 'tiledb',
                'chunk_size': 1000,
                'chunk_overlap': 200
            },
            'EMBEDDING_MODEL_NAME': 'BAAI/bge-small-en-v1.5',
            'created_databases': {},
            'theme': 'dark',
            'postgresql': {
                'host': 'localhost',
                'port': 5432,
                'user': 'postgres',
                'password': '',
                'database': 'vectordb'
            }
        }
        self.save_config(default_config)
    
    def save_config(self, config: Dict[str, Any] = None) -> None:
        """Save configuration to file"""
        if config is not None:
            self.config = config
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(self.config, f, default_flow_style=False)
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get the entire configuration"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        self.config.update(updates)
        self.save_config()
    
    def get_path(self, *parts: str) -> Path:
        """Get a path relative to the root directory"""
        return self.root_dir.joinpath(*parts)
    
    # Convenience methods for commonly used paths
    @property
    def docs_dir(self) -> Path:
        """Get the documents directory path"""
        return self.get_path('Docs_for_DB')
    
    @property
    def vector_db_dir(self) -> Path:
        """Get the vector database directory path"""
        return self.get_path('Vector_DB')
    
    @property
    def models_dir(self) -> Path:
        """Get the models directory path"""
        return self.get_path('Models')
    
    @property
    def themes_dir(self) -> Path:
        """Get the themes directory path"""
        return self.get_path('themes')
    
    @property
    def tokenizer_dir(self) -> Path:
        """Get the tokenizer directory path"""
        return self.get_path('Tokenizer')
    
    def get_database_type(self) -> str:
        """Get the current database type"""
        return self.config.get('database', {}).get('type', 'tiledb')
    
    def get_theme(self) -> str:
        """Get the current theme"""
        return self.config.get('theme', 'dark')
    
    def set_theme(self, theme: str) -> None:
        """Set the current theme"""
        self.config['theme'] = theme
        self.save_config()
    
    def get_postgresql_config(self) -> Dict[str, Any]:
        """Get PostgreSQL configuration"""
        return self.config.get('postgresql', {})
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist"""
        directories = [
            self.docs_dir,
            self.vector_db_dir,
            self.models_dir / 'vector',
            self.models_dir / 'tts',
            self.themes_dir,
            self.tokenizer_dir
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True) 