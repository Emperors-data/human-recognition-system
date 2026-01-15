"""
Face database management - Simplified for DeepFace.
"""

import os
from typing import Dict, List
from pathlib import Path


class FaceDatabase:
    """
    Manages face database for DeepFace recognition.
    Note: DeepFace doesn't require pre-computed encodings like face_recognition.
    """
    
    def __init__(self, database_path: str, cache_file: str = None):
        """
        Initialize face database.
        
        Args:
            database_path: Path to directory containing known faces
            cache_file: Not used with DeepFace (kept for compatibility)
        """
        self.database_path = Path(database_path)
        self.known_face_paths = {}  # name -> list of image paths
        
        # Load database
        self._load_database()
    
    def _load_database(self):
        """Load face image paths from database directory."""
        if not self.database_path.exists():
            print(f"Creating database directory: {self.database_path}")
            self.database_path.mkdir(parents=True, exist_ok=True)
            print("No faces in database. Add images to known_faces/ directory.")
            return
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        # Iterate through person directories
        for person_dir in self.database_path.iterdir():
            if not person_dir.is_dir():
                continue
            
            person_name = person_dir.name
            self.known_face_paths[person_name] = []
            
            # Load all images for this person
            for image_path in person_dir.iterdir():
                if image_path.suffix.lower() not in image_extensions:
                    continue
                
                self.known_face_paths[person_name].append(str(image_path))
            
            if self.known_face_paths[person_name]:
                print(f"Loaded {len(self.known_face_paths[person_name])} images for {person_name}")
        
        print(f"Total faces loaded: {sum(len(paths) for paths in self.known_face_paths.values())}")
    
    def get_known_names(self) -> List[str]:
        """Get list of unique known names."""
        return list(self.known_face_paths.keys())
    
    def get_database_info(self) -> Dict:
        """Get information about the database."""
        return {
            'total_encodings': sum(len(paths) for paths in self.known_face_paths.values()),
            'unique_persons': len(self.known_face_paths),
            'known_names': self.get_known_names()
        }
    
    def rebuild_database(self):
        """Rebuild the database from scratch."""
        self.known_face_paths = {}
        self._load_database()
