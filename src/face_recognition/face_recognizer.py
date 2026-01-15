"""
Fast face recognition using OpenCV's LBPH Face Recognizer.
Much faster than DeepFace for real-time applications.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import pickle


class FaceRecognizer:
    """
    Real-time face recognition using OpenCV's LBPH algorithm.
    """
    
    def __init__(self, config, database_path: str = "known_faces"):
        """
        Initialize face recognizer.
        
        Args:
            config: Configuration dictionary
            database_path: Path to known faces directory
        """
        self.config = config
        self.database_path = Path(database_path)
        self.model_path = Path("face_recognizer_model.yml")
        
        # Initialize face detector (Haar Cascade - fast)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize LBPH face recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8
        )
        
        # Load or train model
        self.label_to_name = {}
        self.name_to_label = {}
        
        if self.model_path.exists():
            print(f"Loading trained face recognizer from {self.model_path}")
            self._load_model()
        else:
            print(f"Training face recognizer on {database_path}")
            self._train_model()
        
        print(f"Face recognizer initialized with OpenCV LBPH")
        print(f"Known persons: {list(self.label_to_name.values())}")
    
    def _train_model(self):
        """Train the face recognizer on known faces."""
        if not self.database_path.exists():
            print(f"Creating database directory: {self.database_path}")
            self.database_path.mkdir(parents=True, exist_ok=True)
            print("No faces to train. Add images to known_faces/ directory.")
            return
        
        faces = []
        labels = []
        current_label = 0
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        # Load all known faces
        for person_dir in self.database_path.iterdir():
            if not person_dir.is_dir():
                continue
            
            person_name = person_dir.name
            self.name_to_label[person_name] = current_label
            self.label_to_name[current_label] = person_name
            
            print(f"Training on {person_name}...")
            
            for image_path in person_dir.iterdir():
                if image_path.suffix.lower() not in image_extensions:
                    continue
                
                # Load image
                img = cv2.imread(str(image_path))
                if img is None:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                detected_faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                # Use the largest face
                if len(detected_faces) > 0:
                    # Sort by area and take largest
                    detected_faces = sorted(detected_faces, key=lambda x: x[2]*x[3], reverse=True)
                    x, y, w, h = detected_faces[0]
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Resize to standard size
                    face_roi = cv2.resize(face_roi, (200, 200))
                    
                    faces.append(face_roi)
                    labels.append(current_label)
            
            current_label += 1
        
        if len(faces) == 0:
            print("No faces found for training!")
            return
        
        # Train the recognizer
        print(f"Training on {len(faces)} face images...")
        self.recognizer.train(faces, np.array(labels))
        
        # Save the model
        self.recognizer.save(str(self.model_path))
        
        # Save label mappings
        with open(self.model_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump({
                'label_to_name': self.label_to_name,
                'name_to_label': self.name_to_label
            }, f)
        
        print(f"✓ Model trained and saved to {self.model_path}")
        print(f"✓ Trained on {len(self.label_to_name)} persons")
    
    def _load_model(self):
        """Load pre-trained model."""
        self.recognizer.read(str(self.model_path))
        
        # Load label mappings
        with open(self.model_path.with_suffix('.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.label_to_name = data['label_to_name']
            self.name_to_label = data['name_to_label']
        
        print(f"✓ Loaded model for {len(self.label_to_name)} persons")
    
    def recognize_person(self, frame: np.ndarray, 
                        person_bbox: Tuple[int, int, int, int]) -> Tuple[str, float]:
        """
        Recognize a person within a bounding box.
        
        Args:
            frame: Input image (BGR)
            person_bbox: Person bounding box (x1, y1, x2, y2)
        
        Returns:
            Tuple of (name, confidence)
        """
        if len(self.label_to_name) == 0:
            return ("Unknown", 0.0)
        
        x1, y1, x2, y2 = person_bbox
        h, w = frame.shape[:2]
        
        # Ensure bbox is within frame
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Extract person region
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            return ("Unknown", 0.0)
        
        gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in person region
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return ("Unknown", 0.0)
        
        # Use the largest face
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        fx, fy, fw, fh = faces[0]
        face_roi = gray[fy:fy+fh, fx:fx+fw]
        
        # Resize to standard size
        face_roi = cv2.resize(face_roi, (200, 200))
        
        # Recognize
        label, confidence = self.recognizer.predict(face_roi)
        
        # Convert confidence to similarity (lower is better for LBPH)
        # Typical range: 0-100 (lower = more confident)
        # Convert to 0-1 scale (higher = more confident)
        similarity = max(0, 1.0 - (confidence / 100.0))
        
        # Threshold for recognition
        threshold = self.config.get('confidence_threshold', 0.5)
        
        if similarity >= threshold:
            name = self.label_to_name.get(label, "Unknown")
            return (name, similarity)
        else:
            return ("Unknown", similarity)
    
    def retrain(self):
        """Retrain the model on current database."""
        print("Retraining face recognizer...")
        self._train_model()
    
    def get_database_info(self):
        """Get information about the face database."""
        return {
            'total_persons': len(self.label_to_name),
            'known_names': list(self.label_to_name.values()),
            'model_trained': self.model_path.exists()
        }


def test_face_recognition():
    """Test face recognition with webcam."""
    config = {'confidence_threshold': 0.5}
    
    recognizer = FaceRecognizer(config, "known_faces")
    
    # Print database info
    info = recognizer.get_database_info()
    print(f"Database info: {info}")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Use whole frame as bbox for testing
        h, w = frame.shape[:2]
        name, conf = recognizer.recognize_person(frame, (0, 0, w, h))
        
        # Display result
        text = f"{name} ({conf:.2f})"
        cv2.putText(frame, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Face Recognition Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_face_recognition()
