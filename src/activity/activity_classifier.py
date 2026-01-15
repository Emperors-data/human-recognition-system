"""
Activity classification with temporal smoothing.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple
from .activity_rules import ActivityRules


class ActivityClassifier:
    """
    Classifies human activities with temporal smoothing.
    """
    
    def __init__(self, config):
        """
        Initialize activity classifier.
        
        Args:
            config: Configuration dictionary with activity settings
        """
        self.config = config
        self.method = config.get('method', 'rule_based')
        self.smoothing_window = config.get('smoothing_window', 5)
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        
        # Initialize rule-based classifier
        self.rules = ActivityRules()
        
        # Store activity history for each person (keyed by person ID)
        self.activity_history = {}
        self.movement_history = {}
        
        print(f"Activity classifier initialized (method: {self.method})")
    
    def classify(self, person_id: int, pose_features: Dict, 
                body_center: Optional[Tuple[float, float]] = None) -> Tuple[str, float]:
        """
        Classify activity for a person.
        
        Args:
            person_id: Unique person identifier
            pose_features: Pose features from PoseAnalyzer
            body_center: Optional (x, y) body center position
        
        Returns:
            Tuple of (activity_name, confidence)
        """
        # Initialize history for new person
        if person_id not in self.activity_history:
            self.activity_history[person_id] = deque(maxlen=self.smoothing_window)
            self.movement_history[person_id] = deque(maxlen=self.smoothing_window)
        
        # Update movement history
        if body_center is not None:
            self.movement_history[person_id].append(body_center)
        
        # Get movement history for this person
        movement = list(self.movement_history[person_id]) if person_id in self.movement_history else None
        
        # Classify activity
        if self.method == 'rule_based':
            activity, confidence = self.rules.classify_activity(pose_features, movement)
        else:
            # Placeholder for ML-based classification
            activity, confidence = ('unknown', 0.0)
        
        # Add to history
        self.activity_history[person_id].append((activity, confidence))
        
        # Apply temporal smoothing
        smoothed_activity, smoothed_confidence = self._smooth_activity(person_id)
        
        return smoothed_activity, smoothed_confidence
    
    def _smooth_activity(self, person_id: int) -> Tuple[str, float]:
        """
        Apply temporal smoothing to activity predictions.
        
        Args:
            person_id: Person identifier
        
        Returns:
            Tuple of (smoothed_activity, smoothed_confidence)
        """
        if person_id not in self.activity_history or len(self.activity_history[person_id]) == 0:
            return ('unknown', 0.0)
        
        history = list(self.activity_history[person_id])
        
        # Count activity occurrences
        activity_counts = {}
        total_confidence = {}
        
        for activity, confidence in history:
            if activity not in activity_counts:
                activity_counts[activity] = 0
                total_confidence[activity] = 0.0
            activity_counts[activity] += 1
            total_confidence[activity] += confidence
        
        # Find most common activity
        most_common_activity = max(activity_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate average confidence for most common activity
        avg_confidence = total_confidence[most_common_activity] / activity_counts[most_common_activity]
        
        return most_common_activity, avg_confidence
    
    def get_activity_color(self, activity: str) -> Tuple[int, int, int]:
        """
        Get color for activity visualization.
        
        Args:
            activity: Activity name
        
        Returns:
            BGR color tuple
        """
        return self.rules.get_activity_color(activity)
    
    def reset_person(self, person_id: int):
        """
        Reset history for a person.
        
        Args:
            person_id: Person identifier
        """
        if person_id in self.activity_history:
            del self.activity_history[person_id]
        if person_id in self.movement_history:
            del self.movement_history[person_id]
    
    def cleanup_old_persons(self, active_person_ids: List[int]):
        """
        Remove history for persons no longer being tracked.
        
        Args:
            active_person_ids: List of currently active person IDs
        """
        # Remove history for persons not in active list
        inactive_ids = set(self.activity_history.keys()) - set(active_person_ids)
        for person_id in inactive_ids:
            self.reset_person(person_id)
