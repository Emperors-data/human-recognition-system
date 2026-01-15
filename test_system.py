"""
Quick test script to verify all modules are working.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    print("=" * 60)
    
    try:
        print("  - Importing detection module...", end=" ")
        from detection import PersonDetector, DetectionConfig
        print("✓")
        
        print("  - Importing face recognition module...", end=" ")
        from face_recognition import FaceRecognizer, FaceDatabase
        print("✓")
        
        print("  - Importing pose module...", end=" ")
        from pose import PoseEstimator, PoseAnalyzer
        print("✓")
        
        print("  - Importing activity module...", end=" ")
        from activity import ActivityClassifier, ActivityRules
        print("✓")
        
        print("  - Importing tracking module...", end=" ")
        from tracking import PersonTracker
        print("✓")
        
        print("  - Importing utils module...", end=" ")
        from utils import Visualizer, ConfigManager
        print("✓")
        
        print("\n✓ All modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    print("=" * 60)
    
    try:
        from utils import ConfigManager
        
        config = ConfigManager("config.yaml")
        
        print(f"  - Detection model: {config.get('detection.model')}")
        print(f"  - Face tolerance: {config.get('face_recognition.tolerance')}")
        print(f"  - Pose complexity: {config.get('pose.model_complexity')}")
        print(f"  - Activity method: {config.get('activity.method')}")
        
        print("\n✓ Configuration loaded successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Configuration error: {e}")
        return False


def test_face_database():
    """Test face database."""
    print("\nTesting face database...")
    print("=" * 60)
    
    try:
        from face_recognition import FaceDatabase
        
        db = FaceDatabase("known_faces", "face_encodings.pkl")
        info = db.get_database_info()
        
        print(f"  - Total encodings: {info['total_encodings']}")
        print(f"  - Unique persons: {info['unique_persons']}")
        print(f"  - Known names: {info['known_names']}")
        
        if info['total_encodings'] == 0:
            print("\n⚠ No faces in database. Add faces to known_faces/ directory.")
        else:
            print("\n✓ Face database loaded successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Face database error: {e}")
        return False


def test_dependencies():
    """Test that all required dependencies are installed."""
    print("\nTesting dependencies...")
    print("=" * 60)
    
    dependencies = [
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("yaml", "pyyaml"),
        ("torch", "torch"),
        ("ultralytics", "ultralytics"),
        ("face_recognition", "face-recognition"),
        ("mediapipe", "mediapipe"),
        ("deep_sort_realtime", "deep-sort-realtime")
    ]
    
    all_installed = True
    
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - NOT INSTALLED")
            all_installed = False
    
    if all_installed:
        print("\n✓ All dependencies installed!")
    else:
        print("\n✗ Some dependencies missing. Run: pip install -r requirements.txt")
    
    return all_installed


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Human Recognition System - Quick Test")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test dependencies
    results.append(("Dependencies", test_dependencies()))
    
    print()
    
    # Test imports
    results.append(("Module Imports", test_imports()))
    
    print()
    
    # Test configuration
    results.append(("Configuration", test_config()))
    
    print()
    
    # Test face database
    results.append(("Face Database", test_face_database()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! System is ready to use.")
        print("\nRun the system with: python src/main.py")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Check config.yaml exists")
        print("  - Add faces to known_faces/ directory")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
