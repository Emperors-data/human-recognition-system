"""
Pose estimation using YOLOv8-Pose keypoints.
Works with 17 COCO keypoints from YOLO model.
"""

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple


# YOLO-Pose keypoint indices (COCO format - 17 keypoints)
KEYPOINT_NAMES = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle'     # 16
]

# Skeleton connections for visualization
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),  # nose to eyes
    (1, 3), (2, 4),  # eyes to ears
    (0, 5), (0, 6),  # nose to shoulders
    (5, 6),          # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10), # right arm
    (5, 11), (6, 12),# shoulders to hips
    (11, 12),        # hips
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16)  # right leg
]


class PoseEstimator:
    """
    Pose estimation using YOLOv8-Pose keypoints.
    """
    
    def __init__(self, config):
        """
        Initialize pose estimator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.min_keypoint_confidence = config.get('min_detection_confidence', 0.5)
        
        print("Pose estimator initialized (YOLOv8-Pose mode)")
    
    def create_pose_from_keypoints(self, keypoints: np.ndarray, 
                                   bbox: Tuple[int, int, int, int]) -> Optional[Dict]:
        """
        Create pose data structure from YOLO keypoints.
        
        Args:
            keypoints: Array of shape (17, 3) with [x, y, confidence]
            bbox: Person bounding box
        
        Returns:
            Dictionary with pose data
        """
        if keypoints is None or len(keypoints) == 0:
            return None
        
        # Convert to landmark format
        landmarks = []
        for i, (x, y, conf) in enumerate(keypoints):
            landmarks.append({
                'x': float(x),
                'y': float(y),
                'z': 0.0,  # YOLO doesn't provide z
                'visibility': float(conf),
                'name': KEYPOINT_NAMES[i]
            })
        
        return {
            'landmarks': landmarks,
            'bbox': bbox,
            'world_landmarks': None,
            'keypoint_count': len(keypoints)
        }
    
    def estimate_pose(self, frame: np.ndarray, 
                     person_bbox: Optional[Tuple[int, int, int, int]] = None,
                     keypoints: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        Estimate pose from YOLO keypoints.
        
        Args:
            frame: Input image (not used, kept for compatibility)
            person_bbox: Person bounding box
            keypoints: YOLO keypoints array (17, 3)
        
        Returns:
            Dictionary containing pose landmarks
        """
        if keypoints is None:
            return None
        
        return self.create_pose_from_keypoints(keypoints, person_bbox)
    
    def draw_pose(self, frame: np.ndarray, pose_data: Dict, 
                  color: Tuple[int, int, int] = (0, 255, 0),
                  thickness: int = 2) -> np.ndarray:
        """
        Draw pose skeleton on frame.
        
        Args:
            frame: Input image
            pose_data: Pose data from estimate_pose()
            color: Color for skeleton (BGR)
            thickness: Line thickness
        
        Returns:
            Annotated frame
        """
        if pose_data is None or 'landmarks' not in pose_data:
            return frame
        
        landmarks = pose_data['landmarks']
        
        # Draw skeleton connections
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_kpt = landmarks[start_idx]
                end_kpt = landmarks[end_idx]
                
                # Only draw if both keypoints are confident
                if start_kpt['visibility'] > self.min_keypoint_confidence and \
                   end_kpt['visibility'] > self.min_keypoint_confidence:
                    start_point = (int(start_kpt['x']), int(start_kpt['y']))
                    end_point = (int(end_kpt['x']), int(end_kpt['y']))
                    
                    cv2.line(frame, start_point, end_point, color, thickness)
        
        # Draw keypoints
        for landmark in landmarks:
            if landmark['visibility'] > self.min_keypoint_confidence:
                center = (int(landmark['x']), int(landmark['y']))
                cv2.circle(frame, center, 4, color, -1)
                cv2.circle(frame, center, 5, (255, 255, 255), 1)
        
        return frame
    
    def get_landmark_names(self) -> List[str]:
        """Get list of landmark names."""
        return KEYPOINT_NAMES


def test_pose_estimation():
    """Test pose estimation."""
    config = {
        'min_detection_confidence': 0.5
    }
    
    estimator = PoseEstimator(config)
    
    # Create dummy keypoints for testing
    keypoints = np.random.rand(17, 3)
    keypoints[:, 0] *= 640  # x coordinates
    keypoints[:, 1] *= 480  # y coordinates
    keypoints[:, 2] = 0.9   # confidence
    
    bbox = (100, 100, 300, 400)
    
    # Create pose
    pose_data = estimator.create_pose_from_keypoints(keypoints, bbox)
    
    print(f"Created pose with {len(pose_data['landmarks'])} landmarks")
    print(f"Landmark names: {estimator.get_landmark_names()}")
    
    # Test visualization
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = estimator.draw_pose(frame, pose_data)
    
    cv2.imshow('Pose Test', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_pose_estimation()
