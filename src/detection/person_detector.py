"""
YOLOv8-Pose based person detection and pose estimation.
Combines detection and pose in a single model for better performance.
"""

import cv2
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO


class PersonDetector:
    """
    Person detection using YOLOv8-Pose model.
    Detects persons AND their keypoints in one pass.
    """
    
    def __init__(self, config):
        """
        Initialize YOLOv8-Pose detector.
        
        Args:
            config: Detection configuration
        """
        self.config = config
        self.model_path = config.model.replace('yolov8', 'yolov8') + '-pose.pt' if 'pose' not in config.model else config.model
        self.confidence_threshold = config.confidence_threshold
        self.device = config.device
        
        # Load YOLOv8-Pose model
        print(f"Loading {self.model_path} model...")
        try:
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def detect_with_pose(self, frame: np.ndarray) -> Tuple[List[Tuple], List[np.ndarray]]:
        """
        Detect persons and their poses in one pass.
        
        Args:
            frame: Input image (BGR format)
        
        Returns:
            Tuple of (detections, keypoints_list)
            - detections: List of (x1, y1, x2, y2, confidence)
            - keypoints_list: List of keypoint arrays (17, 3) for each detection
        """
        if self.model is None:
            return [], []
        
        try:
            # Run inference
            results = self.model(frame, verbose=False, device=self.device)
            
            detections = []
            keypoints_list = []
            
            for result in results:
                # Get boxes
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    # Get keypoints if available
                    has_keypoints = result.keypoints is not None
                    
                    for i, box in enumerate(boxes):
                        conf = float(box.conf[0])
                        
                        if conf >= self.confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            detections.append((int(x1), int(y1), int(x2), int(y2), conf))
                            
                            # Get corresponding keypoints
                            if has_keypoints:
                                try:
                                    # Get keypoints for this detection
                                    kpts_data = result.keypoints.data[i].cpu().numpy()  # Shape: (17, 3) or (17, 2)
                                    
                                    # Ensure we have (17, 3) format [x, y, conf]
                                    if kpts_data.shape[1] == 2:
                                        # Add confidence column (all 1.0)
                                        kpts_with_conf = np.concatenate([
                                            kpts_data,
                                            np.ones((17, 1))
                                        ], axis=1)
                                    else:
                                        kpts_with_conf = kpts_data
                                    
                                    keypoints_list.append(kpts_with_conf)
                                except Exception as e:
                                    print(f"Warning: Could not extract keypoints for detection {i}: {e}")
                                    keypoints_list.append(np.zeros((17, 3)))
                            else:
                                # No keypoints available
                                keypoints_list.append(np.zeros((17, 3)))
            
            print(f"Detected {len(detections)} persons")  # Debug output
            return detections, keypoints_list
            
        except Exception as e:
            print(f"Error during detection: {e}")
            return [], []
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect persons only (for compatibility).
        
        Args:
            frame: Input image (BGR format)
        
        Returns:
            List of (x1, y1, x2, y2, confidence) tuples
        """
        detections, _ = self.detect_with_pose(frame)
        return detections


# Keep DetectionConfig for compatibility
class DetectionConfig:
    """Configuration for person detection."""
    
    def __init__(self, config_dict: dict = None):
        """Initialize detection configuration."""
        if config_dict is None:
            config_dict = {}
        
        self.model = config_dict.get('model', 'yolov8n-pose')
        self.confidence_threshold = config_dict.get('confidence_threshold', 0.5)
        self.device = config_dict.get('device', 'cpu')
        self.person_class_id = 0  # COCO person class


def test_detection():
    """Test person detection with webcam."""
    config = DetectionConfig({
        'model': 'yolov8n-pose',
        'confidence_threshold': 0.5,
        'device': 'cpu'
    })
    
    detector = PersonDetector(config)
    
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect with pose
        detections, keypoints_list = detector.detect_with_pose(frame)
        
        # Draw results
        for (x1, y1, x2, y2, conf), keypoints in zip(detections, keypoints_list):
            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw keypoints
            for kpt in keypoints:
                x, y, c = kpt
                if c > 0.5:  # Only draw confident keypoints
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
        
        cv2.imshow('YOLOv8-Pose Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_detection()
