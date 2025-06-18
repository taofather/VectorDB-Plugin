import sys
import subprocess
import os
from pathlib import Path
import platform
import yaml
import psycopg2
from config_manager import ConfigManager

def install_requirements():
    """Install required packages from requirements.txt"""
    print("\nInstalling required packages...")
    
    # Install priority packages first
    priority_packages = [
        "torch",
        "torchvision",
        "torchaudio",
        "psycopg2-binary",
        "tiledb",
        "langchain-community",
        "numpy",
        "PySide6",
        "PyYAML"
    ]
    
    print("\nInstalling priority packages...")
    for package in priority_packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {str(e)}")
            if package == "flash-attn":
                print("Skipping flash-attn as it requires CUDA...")
                continue
            return False
    
    # Try to install flash-attn only if CUDA is available
    if platform.system() != 'Darwin':  # Not macOS
        print("\nInstalling flash-attn...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "flash-attn"])
        except subprocess.CalledProcessError as e:
            print(f"Error installing flash-attn: {str(e)}")
            print("Skipping flash-attn as it requires CUDA...")
    
    return True

def create_directory_structure():
    """Create necessary directories for both TileDB and pgvector"""
    config = ConfigManager()
    
    directories = [
        config.vector_db_dir / "tiledb",
        config.vector_db_dir / "pgvector",
        config.models_dir / "vector",
        config.models_dir / "tts",
        config.docs_dir,
        config.themes_dir
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def initialize_postgresql():
    """Initialize PostgreSQL database and pgvector extension"""
    print("\nInitializing PostgreSQL database...")
    
    # Load config from root directory
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}
        pg_config = config_data.get('postgresql', {})
    else:
        print("⚠ config.yaml not found in root directory")
        return False
    
    # Default values if not configured
    host = pg_config.get('host', 'localhost')
    port = pg_config.get('port', 5433)
    user = pg_config.get('user', 'postgres')
    password = pg_config.get('password', 'postgres')
    database = pg_config.get('database', 'vectordb')
    
    try:
        # Try to connect to PostgreSQL
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        
        # Enable pgvector extension
        with conn.cursor() as cur:
            cur.execute('CREATE EXTENSION IF NOT EXISTS vector;')
            conn.commit()
            print("✓ PostgreSQL connection successful")
            print("✓ pgvector extension enabled")
        
        conn.close()
        return True
        
    except psycopg2.Error as e:
        print(f"⚠ PostgreSQL connection failed: {e}")
        print("Make sure PostgreSQL is running with the correct configuration:")
        print(f"  Host: {host}, Port: {port}, User: {user}, Database: {database}")
        print("You can start PostgreSQL using: docker-compose up -d")
        return False

def main():
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    # Create directory structure first
    print("\nCreating directory structure...")
    create_directory_structure()
    
    # Install requirements
    if not install_requirements():
        print("\nSetup failed. Please check the errors above.")
        sys.exit(1)
    
    # Initialize PostgreSQL
    if not initialize_postgresql():
        print("\nPostgreSQL initialization failed, but setup can continue.")
        print("Make sure to start PostgreSQL before creating databases.")
    
    print("\nSetup completed successfully!")

if __name__ == "__main__":
    main() 