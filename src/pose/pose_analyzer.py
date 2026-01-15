"""
Pose analysis for extracting features from YOLOv8-Pose keypoints.
"""

import numpy as np
from typing import Dict, Optional, Tuple


class PoseAnalyzer:
    """
    Analyzes pose keypoints to extract features for activity classification.
    Works with 17 COCO keypoints from YOLOv8-Pose.
    """
    
    def __init__(self):
        """Initialize pose analyzer."""
        # Keypoint indices (COCO format)
        self.NOSE = 0
        self.LEFT_EYE = 1
        self.RIGHT_EYE = 2
        self.LEFT_EAR = 3
        self.RIGHT_EAR = 4
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_ELBOW = 7
        self.RIGHT_ELBOW = 8
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12
        self.LEFT_KNEE = 13
        self.RIGHT_KNEE = 14
        self.LEFT_ANKLE = 15
        self.RIGHT_ANKLE = 16
    
    def analyze_pose(self, pose_data: Optional[Dict]) -> Dict:
        """
        Analyze pose and extract features.
        
        Args:
            pose_data: Pose data from PoseEstimator
        
        Returns:
            Dictionary of pose features
        """
        if pose_data is None or 'landmarks' not in pose_data:
            return {'valid': False}
        
        landmarks = pose_data['landmarks']
        
        # Extract features
        features = {
            'valid': True,
            'orientation': self._get_body_orientation(landmarks),
            'joint_angles': self._calculate_joint_angles(landmarks),
            'hand_near_face': self._check_hand_near_face(landmarks),
            'body_center': self._get_body_center(landmarks),
            'keypoint_confidence': self._get_average_confidence(landmarks)
        }
        
        return features
    
    def _get_body_center(self, landmarks: list) -> Optional[Tuple[float, float]]:
        """Calculate body center from hip keypoints."""
        left_hip = landmarks[self.LEFT_HIP]
        right_hip = landmarks[self.RIGHT_HIP]
        
        if left_hip['visibility'] > 0.5 and right_hip['visibility'] > 0.5:
            center_x = (left_hip['x'] + right_hip['x']) / 2
            center_y = (left_hip['y'] + right_hip['y']) / 2
            return (center_x, center_y)
        
        return None
    
    def _get_average_confidence(self, landmarks: list) -> float:
        """Get average confidence of all keypoints."""
        confidences = [lm['visibility'] for lm in landmarks]
        return np.mean(confidences) if confidences else 0.0
    
    def _get_body_orientation(self, landmarks: list) -> str:
        """
        Determine body orientation (standing, sitting, lying).
        
        Uses hip and shoulder positions relative to ankles.
        """
        # Get key points
        left_shoulder = landmarks[self.LEFT_SHOULDER]
        right_shoulder = landmarks[self.RIGHT_SHOULDER]
        left_hip = landmarks[self.LEFT_HIP]
        right_hip = landmarks[self.RIGHT_HIP]
        left_knee = landmarks[self.LEFT_KNEE]
        right_knee = landmarks[self.RIGHT_KNEE]
        
        # Check if we have enough confident keypoints
        min_conf = 0.5
        if not (left_hip['visibility'] > min_conf and right_hip['visibility'] > min_conf):
            return 'unknown'
        
        # Calculate hip-knee angle (vertical alignment)
        hip_y = (left_hip['y'] + right_hip['y']) / 2
        knee_y = (left_knee['y'] + right_knee['y']) / 2 if \
                 (left_knee['visibility'] > min_conf and right_knee['visibility'] > min_conf) else hip_y + 100
        
        # Calculate shoulder-hip vertical distance
        shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2 if \
                     (left_shoulder['visibility'] > min_conf and right_shoulder['visibility'] > min_conf) else hip_y - 100
        
        torso_height = hip_y - shoulder_y
        leg_height = knee_y - hip_y
        
        # Determine orientation
        if torso_height < 50:  # Very small torso = lying down
            return 'lying'
        elif leg_height < torso_height * 0.5:  # Legs bent significantly = sitting
            return 'sitting'
        else:  # Legs extended = standing
            return 'standing'
    
    def _calculate_joint_angles(self, landmarks: list) -> Dict[str, float]:
        """Calculate joint angles."""
        angles = {}
        
        # Left elbow angle
        angles['left_elbow'] = self._calculate_angle(
            landmarks[self.LEFT_SHOULDER],
            landmarks[self.LEFT_ELBOW],
            landmarks[self.LEFT_WRIST]
        )
        
        # Right elbow angle
        angles['right_elbow'] = self._calculate_angle(
            landmarks[self.RIGHT_SHOULDER],
            landmarks[self.RIGHT_ELBOW],
            landmarks[self.RIGHT_WRIST]
        )
        
        # Left knee angle
        angles['left_knee'] = self._calculate_angle(
            landmarks[self.LEFT_HIP],
            landmarks[self.LEFT_KNEE],
            landmarks[self.LEFT_ANKLE]
        )
        
        # Right knee angle
        angles['right_knee'] = self._calculate_angle(
            landmarks[self.RIGHT_HIP],
            landmarks[self.RIGHT_KNEE],
            landmarks[self.RIGHT_ANKLE]
        )
        
        return angles
    
    def _calculate_angle(self, point1: Dict, point2: Dict, point3: Dict) -> float:
        """
        Calculate angle between three points.
        
        Args:
            point1, point2, point3: Keypoint dictionaries
        
        Returns:
            Angle in degrees (0-180)
        """
        # Check confidence
        if point1['visibility'] < 0.5 or point2['visibility'] < 0.5 or point3['visibility'] < 0.5:
            return 180.0  # Default to straight
        
        # Create vectors
        v1 = np.array([point1['x'] - point2['x'], point1['y'] - point2['y']])
        v2 = np.array([point3['x'] - point2['x'], point3['y'] - point2['y']])
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _check_hand_near_face(self, landmarks: list) -> Tuple[bool, bool]:
        """
        Check if hands are near face.
        
        Returns:
            Tuple of (left_hand_near_face, right_hand_near_face)
        """
        nose = landmarks[self.NOSE]
        left_wrist = landmarks[self.LEFT_WRIST]
        right_wrist = landmarks[self.RIGHT_WRIST]
        
        # Distance threshold (in pixels)
        threshold = 100
        
        left_near = False
        right_near = False
        
        if nose['visibility'] > 0.5 and left_wrist['visibility'] > 0.5:
            dist = np.sqrt((nose['x'] - left_wrist['x'])**2 + (nose['y'] - left_wrist['y'])**2)
            left_near = dist < threshold
        
        if nose['visibility'] > 0.5 and right_wrist['visibility'] > 0.5:
            dist = np.sqrt((nose['x'] - right_wrist['x'])**2 + (nose['y'] - right_wrist['y'])**2)
            right_near = dist < threshold
        
        return (left_near, right_near)


def test_pose_analyzer():
    """Test pose analyzer."""
    analyzer = PoseAnalyzer()
    
    # Create dummy pose data
    landmarks = []
    for i in range(17):
        landmarks.append({
            'x': 100 + i * 10,
            'y': 200 + i * 5,
            'z': 0,
            'visibility': 0.9
        })
    
    pose_data = {'landmarks': landmarks}
    
    # Analyze
    features = analyzer.analyze_pose(pose_data)
    
    print("Pose Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_pose_analyzer()
