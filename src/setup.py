import subprocess
import sys
import os
import platform
from constants import libs, priority_libs

def install_requirements():
    print("Installing required packages...")
    
    # Check if running on Mac M2
    is_mac_m2 = platform.system() == 'Darwin' and platform.machine() == 'arm64'
    
    # Install priority libraries first
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if python_version in priority_libs:
        print("\nInstalling priority packages...")
        for package in priority_libs[python_version]["GPU"]:
            print(f"Installing {package}...")
            try:
                if is_mac_m2 and package in ["torch", "torchvision", "torchaudio"]:
                    # Install PyTorch with Metal support for Mac M2
                    subprocess.run([sys.executable, "-m", "pip", "install", "--pre", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/nightly/cpu"], check=True)
                else:
                    subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error installing {package}: {e}")
                return False
    
    # Install common libraries
    print("\nInstalling common packages...")
    for package in libs:
        print(f"Installing {package}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
            return False
    
    return True

if __name__ == "__main__":
    # Check Python version
    if sys.version_info.major != 3 or sys.version_info.minor not in [11, 12]:
        print("Error: This program requires Python 3.11 or 3.12")
        sys.exit(1)
    
    # Install requirements
    if install_requirements():
        print("\nSetup completed successfully!")
        print("You can now run the program with: python gui.py")
    else:
        print("\nSetup failed. Please check the errors above.")
        sys.exit(1) 