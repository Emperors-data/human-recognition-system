"""
Rule-based activity classification logic.
"""

import numpy as np
from typing import Dict, List, Optional


class ActivityRules:
    """
    Rule-based activity classification using pose features.
    """
    
    def __init__(self):
        """Initialize activity rules."""
        self.activity_names = [
            "standing",
            "sitting",
            "walking",
            "sleeping",
            "using_phone",
            "writing",
            "talking",
            "unknown"
        ]
    
    def classify_activity(self, pose_features: Dict, 
                         movement_history: Optional[List[tuple]] = None) -> tuple:
        """
        Classify activity based on pose features and movement history.
        
        Args:
            pose_features: Dictionary of pose features from PoseAnalyzer
            movement_history: List of (x, y) positions over recent frames
        
        Returns:
            Tuple of (activity_name, confidence)
        """
        if not pose_features.get('valid', False):
            # Default to sitting for stationary people when pose is unavailable
            return ('sitting', 0.7)
        
        orientation = pose_features.get('orientation', 'unknown')
        joint_angles = pose_features.get('joint_angles', {})
        hand_near_face = pose_features.get('hand_near_face', (False, False))
        
        # Calculate movement if history is available
        movement_speed = 0.0
        if movement_history and len(movement_history) > 1:
            movement_speed = self._calculate_movement_speed(movement_history)
        
        # Activity detection rules - prioritize based on movement and orientation
        activities_scores = {}
        
        # 1. Sleeping detection (lying down)
        if orientation == 'lying':
            activities_scores['sleeping'] = 0.9
        
        # 2. Walking detection (requires significant movement)
        # Only classify as walking if there's substantial movement
        elif movement_speed > 15.0:  # Increased threshold - must be actually moving
            activities_scores['walking'] = 0.8
        
        # 3. Sitting detection (default for low/no movement)
        elif orientation == 'sitting' or movement_speed < 5.0:
            # If minimal movement, likely sitting
            # Check if writing (hand near desk level, sitting)
            if self._is_writing_posture(joint_angles, hand_near_face):
                activities_scores['writing'] = 0.8
            # Check if using phone (hand near face, sitting)
            elif any(hand_near_face):
                activities_scores['using_phone'] = 0.85
            else:
                activities_scores['sitting'] = 0.9  # High confidence for sitting when not moving
        
        # 4. Standing activities (moderate movement)
        elif orientation == 'standing':
            # Check for phone usage (hand near face, standing)
            if any(hand_near_face):
                activities_scores['using_phone'] = 0.75
            # Check for talking (hands gesturing)
            elif self._is_talking_posture(joint_angles, hand_near_face):
                activities_scores['talking'] = 0.7
            else:
                activities_scores['standing'] = 0.8
        
        # Return activity with highest confidence
        if activities_scores:
            best_activity = max(activities_scores.items(), key=lambda x: x[1])
            return best_activity
        else:
            return ('unknown', 0.0)
    
    def _calculate_movement_speed(self, movement_history: List[tuple]) -> float:
        """
        Calculate average movement speed from position history.
        
        Args:
            movement_history: List of (x, y) positions
        
        Returns:
            Average movement speed
        """
        if len(movement_history) < 2:
            return 0.0
        
        distances = []
        for i in range(1, len(movement_history)):
            prev_x, prev_y = movement_history[i-1]
            curr_x, curr_y = movement_history[i]
            distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _is_writing_posture(self, joint_angles: Dict, 
                           hand_near_face: tuple) -> bool:
        """
        Detect writing posture.
        
        Writing characteristics:
        - Sitting position
        - One arm bent (elbow angle ~90 degrees)
        - Hand not near face
        """
        if not joint_angles:
            return False
        
        # Check elbow angles (writing typically has one elbow bent ~90 degrees)
        left_elbow = joint_angles.get('left_elbow', 180)
        right_elbow = joint_angles.get('right_elbow', 180)
        
        # Writing posture: elbow bent between 60-120 degrees
        left_writing = 60 < left_elbow < 120
        right_writing = 60 < right_elbow < 120
        
        # Hand should not be near face
        hands_away = not any(hand_near_face)
        
        return (left_writing or right_writing) and hands_away
    
    def _is_talking_posture(self, joint_angles: Dict, 
                           hand_near_face: tuple) -> bool:
        """
        Detect talking posture.
        
        Talking characteristics:
        - Standing position
        - Hands may be gesturing (varied arm positions)
        - Possible hand movements near face
        """
        if not joint_angles:
            return False
        
        # Check for varied arm positions (gesturing)
        left_elbow = joint_angles.get('left_elbow', 180)
        right_elbow = joint_angles.get('right_elbow', 180)
        
        # Gesturing: arms in varied positions
        left_gesturing = 30 < left_elbow < 150
        right_gesturing = 30 < right_elbow < 150
        
        return left_gesturing or right_gesturing
    
    def get_activity_color(self, activity: str) -> tuple:
        """
        Get color for activity visualization.
        
        Args:
            activity: Activity name
        
        Returns:
            BGR color tuple
        """
        color_map = {
            'standing': (0, 255, 0),      # Green
            'sitting': (255, 200, 0),     # Cyan
            'walking': (0, 165, 255),     # Orange
            'sleeping': (128, 0, 128),    # Purple
            'using_phone': (0, 255, 255), # Yellow
            'writing': (255, 0, 255),     # Magenta
            'talking': (255, 128, 0),     # Light Blue
            'unknown': (128, 128, 128)    # Gray
        }
        return color_map.get(activity, (128, 128, 128))
