"""
Visualization utilities for drawing annotations on frames.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import time


class Visualizer:
    """
    Handles all visualization and drawing operations.
    """
    
    def __init__(self, config):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration dictionary with visualization settings
        """
        self.config = config
        self.show_bbox = config.get('show_bbox', True)
        self.show_skeleton = config.get('show_skeleton', True)
        self.show_activity = config.get('show_activity', True)
        self.show_face_name = config.get('show_face_name', True)
        self.show_fps = config.get('show_fps', True)
        self.show_person_count = config.get('show_person_count', True)
        
        self.bbox_thickness = config.get('bbox_thickness', 2)
        self.skeleton_thickness = config.get('skeleton_thickness', 2)
        self.font_scale = config.get('font_scale', 0.6)
        self.font_thickness = config.get('font_thickness', 2)
        
        # FPS calculation
        self.fps_history = []
        self.last_time = time.time()
    
    def draw_person(self, frame: np.ndarray, person: Dict, 
                   pose_estimator=None, activity_classifier=None) -> np.ndarray:
        """
        Draw all annotations for a person.
        
        Args:
            frame: Input frame
            person: Person dictionary with tracking info
            pose_estimator: Optional PoseEstimator for drawing skeleton
            activity_classifier: Optional ActivityClassifier for activity colors
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        track_id = person['track_id']
        bbox = person['bbox']
        name = person.get('name', 'Unknown')
        activity = person.get('activity', 'unknown')
        activity_confidence = person.get('activity_confidence', 0.0)
        pose_data = person.get('pose_data')
        
        # Get activity color
        if activity_classifier:
            color = activity_classifier.get_activity_color(activity)
        else:
            color = (0, 255, 0)
        
        # Draw bounding box
        if self.show_bbox:
            annotated = self._draw_bbox(annotated, bbox, color, track_id)
        
        # Draw skeleton
        if self.show_skeleton and pose_data is not None and pose_estimator:
            annotated = pose_estimator.draw_pose(annotated, pose_data, color, self.skeleton_thickness)
        
        # Draw name and activity labels
        if self.show_face_name or self.show_activity:
            annotated = self._draw_labels(annotated, bbox, name, activity, 
                                         activity_confidence, color, track_id)
        
        return annotated
    
    def _draw_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                  color: Tuple[int, int, int], track_id: int) -> np.ndarray:
        """Draw bounding box with track ID."""
        x1, y1, x2, y2 = bbox
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.bbox_thickness)
        
        # Draw track ID
        id_label = f"ID: {track_id}"
        cv2.putText(frame, id_label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, self.font_thickness)
        
        return frame
    
    def _draw_labels(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                    name: str, activity: str, activity_confidence: float,
                    color: Tuple[int, int, int], track_id: int) -> np.ndarray:
        """Draw name and activity labels above the bounding box."""
        x1, y1, x2, y2 = bbox
        
        labels = []
        
        # Build label text - always show identifier and activity
        person_label = name if name != 'Unknown' else f"Person {track_id}"
        activity_label = activity.capitalize() if activity != 'unknown' else 'Detecting...'
        
        # Combine into single line
        label_text = f"{person_label} | {activity_label}"
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness
        )
        
        # Calculate position above bbox
        label_y = y1 - 40  # Position above the ID label
        label_x = x1
        
        # Ensure label doesn't go off screen
        if label_y - text_height - 10 < 0:
            label_y = y2 + text_height + 20  # Put below if not enough space above
        
        # Draw semi-transparent background for better readability
        padding = 5
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (label_x - padding, label_y - text_height - padding),
            (label_x + text_width + padding, label_y + baseline + padding),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw the text
        cv2.putText(
            frame, label_text, (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), self.font_thickness
        )
        
        return frame
    
    def draw_hud(self, frame: np.ndarray, person_count: int, fps: float = None) -> np.ndarray:
        """
        Draw heads-up display with system information.
        
        Args:
            frame: Input frame
            person_count: Number of persons detected
            fps: Optional FPS value
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Calculate FPS if not provided
        if fps is None:
            fps = self._calculate_fps()
        
        # Draw semi-transparent background for HUD
        hud_height = 80
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (300, hud_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
        
        y_offset = 25
        
        # Draw FPS
        if self.show_fps:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(annotated, fps_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        # Draw person count
        if self.show_person_count:
            count_text = f"Persons: {person_count}"
            cv2.putText(annotated, count_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated
    
    def _calculate_fps(self) -> float:
        """Calculate FPS based on frame times."""
        current_time = time.time()
        time_diff = current_time - self.last_time
        self.last_time = current_time
        
        if time_diff > 0:
            fps = 1.0 / time_diff
            self.fps_history.append(fps)
            
            # Keep only last 30 frames for smoothing
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            
            return np.mean(self.fps_history)
        
        return 0.0
    
    def draw_all_persons(self, frame: np.ndarray, persons: List[Dict],
                        pose_estimator=None, activity_classifier=None) -> np.ndarray:
        """
        Draw annotations for all persons.
        
        Args:
            frame: Input frame
            persons: List of person dictionaries
            pose_estimator: Optional PoseEstimator
            activity_classifier: Optional ActivityClassifier
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for person in persons:
            annotated = self.draw_person(annotated, person, pose_estimator, activity_classifier)
        
        # Draw HUD
        annotated = self.draw_hud(annotated, len(persons))
        
        return annotated
    
    def save_screenshot(self, frame: np.ndarray, path: str):
        """
        Save screenshot to file.
        
        Args:
            frame: Frame to save
            path: Output path
        """
        try:
            cv2.imwrite(path, frame)
            print(f"Screenshot saved: {path}")
        except Exception as e:
            print(f"Error saving screenshot: {e}")
    
    def create_video_writer(self, output_path: str, frame_width: int, 
                           frame_height: int, fps: float = 30.0):
        """
        Create video writer for saving output.
        
        Args:
            output_path: Output video path
            frame_width: Frame width
            frame_height: Frame height
            fps: Frames per second
        
        Returns:
            VideoWriter object
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        return writer
