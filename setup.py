"""
Setup script for Human Recognition and Activity Understanding System.
"""

import subprocess
import sys
from pathlib import Path


def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    print("=" * 60)
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("\n✓ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error installing packages: {e}")
        print("\nTry installing manually:")
        print("  pip install -r requirements.txt")
        return False
    
    return True


def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    print("=" * 60)
    
    directories = [
        "known_faces",
        "test_videos",
        "output",
        "output/screenshots"
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    return True


def create_sample_faces_readme():
    """Create README in known_faces directory."""
    readme_path = Path("known_faces/README.txt")
    
    content = """Face Recognition Database
========================

Add photos of people you want to recognize in this directory.

Structure:
----------
known_faces/
├── person1/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── person2/
│   ├── photo1.jpg
│   └── photo2.jpg
└── person3/
    └── photo1.jpg

Instructions:
-------------
1. Create a new folder for each person (use their name)
2. Add 2-5 clear photos of their face
3. Supported formats: JPG, PNG, BMP
4. Photos should be well-lit and show the face clearly
5. The system will automatically load faces on startup

Example:
--------
To add "John Doe":
1. Create folder: known_faces/John_Doe/
2. Add photos: john1.jpg, john2.jpg, john3.jpg
3. Restart the application

The system will create a cache file (face_encodings.pkl) for faster loading.
If you add new faces, delete this cache file to rebuild the database.
"""
    
    with open(readme_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Created: {readme_path}")


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    print("=" * 60)
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required!")
        return False
    
    print("✓ Python version is compatible")
    return True


def main():
    """Main setup function."""
    print("\n" + "=" * 60)
    print("Human Recognition System - Setup")
    print("=" * 60 + "\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    print()
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    print()
    
    # Create sample README
    create_sample_faces_readme()
    
    print()
    
    # Install requirements
    response = input("\nInstall required packages? (y/n): ")
    if response.lower() == 'y':
        if not install_requirements():
            sys.exit(1)
    else:
        print("\nSkipping package installation.")
        print("Remember to install manually: pip install -r requirements.txt")
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Add face photos to known_faces/ directory")
    print("2. Run the system: python src/main.py")
    print("3. See README.md for more information")
    print()


if __name__ == "__main__":
    main()
