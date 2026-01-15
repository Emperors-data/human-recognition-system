"""
Person tracking using Deep SORT.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from deep_sort_realtime.deepsort_tracker import DeepSort


class PersonTracker:
    """
    Multi-person tracking with state management.
    """
    
    def __init__(self, config):
        """
        Initialize person tracker.
        
        Args:
            config: Configuration dictionary with tracking settings
        """
        self.config = config
        
        # Initialize Deep SORT tracker
        self.tracker = DeepSort(
            max_age=config.get('max_age', 30),
            n_init=config.get('min_hits', 3),
            max_iou_distance=config.get('max_iou_distance', 0.7),
            max_cosine_distance=config.get('max_cosine_distance', 0.3),
            nn_budget=config.get('nn_budget', 100),
            embedder="mobilenet",  # Feature extractor for appearance
            embedder_gpu=True
        )
        
        # Store person states
        self.person_states = {}  # track_id -> state dict
        
        print("Person tracker initialized")
    
    def update(self, detections: List[Tuple[int, int, int, int, float]], 
              frame: np.ndarray) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (x1, y1, x2, y2, confidence) tuples
            frame: Current frame for appearance features
        
        Returns:
            List of tracked persons with IDs and bounding boxes
        """
        if len(detections) == 0:
            # Update tracker with empty detections
            tracks = self.tracker.update_tracks([], frame=frame)
            return self._process_tracks(tracks)
        
        # Convert detections to Deep SORT format: ([left, top, width, height], confidence, class)
        deep_sort_detections = []
        for (x1, y1, x2, y2, conf) in detections:
            left = x1
            top = y1
            width = x2 - x1
            height = y2 - y1
            deep_sort_detections.append(([left, top, width, height], conf, 'person'))
        
        # Update tracker
        tracks = self.tracker.update_tracks(deep_sort_detections, frame=frame)
        
        return self._process_tracks(tracks)
    
    def _process_tracks(self, tracks) -> List[Dict]:
        """
        Process tracks and maintain person states.
        
        Args:
            tracks: List of Track objects from Deep SORT
        
        Returns:
            List of person dictionaries
        """
        persons = []
        active_track_ids = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            active_track_ids.append(track_id)
            
            # Get bounding box
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            
            # Initialize state for new track
            if track_id not in self.person_states:
                self.person_states[track_id] = {
                    'track_id': track_id,
                    'name': 'Unknown',
                    'face_confidence': 0.0,
                    'activity': 'unknown',
                    'activity_confidence': 0.0,
                    'pose_data': None,
                    'first_seen_frame': 0
                }
            
            # Create person dict
            person = {
                'track_id': track_id,
                'bbox': (x1, y1, x2, y2),
                'name': self.person_states[track_id]['name'],
                'face_confidence': self.person_states[track_id]['face_confidence'],
                'activity': self.person_states[track_id]['activity'],
                'activity_confidence': self.person_states[track_id]['activity_confidence'],
                'pose_data': self.person_states[track_id]['pose_data']
            }
            
            persons.append(person)
        
        # Clean up inactive tracks
        self._cleanup_inactive_tracks(active_track_ids)
        
        return persons
    
    def update_person_name(self, track_id: int, name: str, confidence: float):
        """
        Update person's recognized name.
        
        Args:
            track_id: Track ID
            name: Recognized name
            confidence: Recognition confidence
        """
        if track_id in self.person_states:
            # Only update if new confidence is higher or name was unknown
            if (confidence > self.person_states[track_id]['face_confidence'] or 
                self.person_states[track_id]['name'] == 'Unknown'):
                self.person_states[track_id]['name'] = name
                self.person_states[track_id]['face_confidence'] = confidence
    
    def update_person_activity(self, track_id: int, activity: str, confidence: float):
        """
        Update person's activity.
        
        Args:
            track_id: Track ID
            activity: Activity name
            confidence: Activity confidence
        """
        if track_id in self.person_states:
            self.person_states[track_id]['activity'] = activity
            self.person_states[track_id]['activity_confidence'] = confidence
    
    def update_person_pose(self, track_id: int, pose_data: Dict):
        """
        Update person's pose data.
        
        Args:
            track_id: Track ID
            pose_data: Pose data from PoseEstimator
        """
        if track_id in self.person_states:
            self.person_states[track_id]['pose_data'] = pose_data
    
    def _cleanup_inactive_tracks(self, active_track_ids: List[int]):
        """
        Remove states for inactive tracks.
        
        Args:
            active_track_ids: List of currently active track IDs
        """
        inactive_ids = set(self.person_states.keys()) - set(active_track_ids)
        for track_id in inactive_ids:
            del self.person_states[track_id]
    
    def get_person_state(self, track_id: int) -> Optional[Dict]:
        """
        Get state for a specific person.
        
        Args:
            track_id: Track ID
        
        Returns:
            Person state dictionary or None
        """
        return self.person_states.get(track_id)
    
    def get_all_states(self) -> Dict[int, Dict]:
        """
        Get all person states.
        
        Returns:
            Dictionary of track_id -> state
        """
        return self.person_states.copy()
    
    def reset(self):
        """Reset tracker and clear all states."""
        self.tracker = DeepSort(
            max_age=self.config.get('max_age', 30),
            n_init=self.config.get('min_hits', 3),
            max_iou_distance=self.config.get('max_iou_distance', 0.7),
            max_cosine_distance=self.config.get('max_cosine_distance', 0.3),
            nn_budget=self.config.get('nn_budget', 100),
            embedder="mobilenet",
            embedder_gpu=True
        )
        self.person_states = {}
