import sys
import subprocess
import os
from pathlib import Path
import platform

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
    # Get the root directory (one level up from src)
    root_dir = Path(__file__).resolve().parent.parent
    
    directories = [
        "Vector_DB/tiledb",
        "Vector_DB/pgvector",
        "Models/vector",
        "Models/tts",
        "Docs_for_DB",
        "themes"
    ]
    
    for directory in directories:
        (root_dir / directory).mkdir(parents=True, exist_ok=True)

def main():
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    # Ask for database type
    print("\nSelect database type:")
    print("1. TileDB (default)")
    print("2. PostgreSQL with pgvector")
    
    choice = input("Enter your choice (1/2): ").strip()
    
    # Create directory structure first
    print("\nCreating directory structure...")
    create_directory_structure()
    
    # Install requirements
    if not install_requirements():
        print("\nSetup failed. Please check the errors above.")
        sys.exit(1)
    
    print("\nSetup completed successfully!")

if __name__ == "__main__":
    main() 