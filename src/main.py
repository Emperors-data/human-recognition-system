"""
Main integration pipeline for human recognition and activity understanding system.
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from detection import PersonDetector, DetectionConfig
from face_recognition import FaceRecognizer
from pose import PoseEstimator, PoseAnalyzer
from activity import ActivityClassifier
from tracking import PersonTracker
from utils import Visualizer, ConfigManager


class HumanRecognitionSystem:
    """
    Integrated system for human recognition and activity understanding.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the complete system.
        
        Args:
            config_path: Path to configuration file
        """
        print("=" * 60)
        print("Human Recognition and Activity Understanding System")
        print("=" * 60)
        
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        
        # Initialize modules
        print("\nInitializing modules...")
        
        # Detection
        detection_config = DetectionConfig(self.config_manager.get_section('detection'))
        self.detector = PersonDetector(detection_config)
        
        # Face recognition
        face_config = self.config_manager.get_section('face_recognition')
        self.face_recognizer = FaceRecognizer(
            face_config, 
            face_config.get('face_database_path', 'known_faces')
        )
        
        # Pose estimation
        pose_config = self.config_manager.get_section('pose')
        self.pose_estimator = PoseEstimator(pose_config)
        self.pose_analyzer = PoseAnalyzer()
        
        # Activity classification
        activity_config = self.config_manager.get_section('activity')
        self.activity_classifier = ActivityClassifier(activity_config)
        
        # Tracking
        tracking_config = self.config_manager.get_section('tracking')
        self.tracker = PersonTracker(tracking_config)
        
        # Visualization
        viz_config = self.config_manager.get_section('visualization')
        self.visualizer = Visualizer(viz_config)
        
        print("\n✓ All modules initialized successfully!")
        print("=" * 60)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the entire pipeline.
        
        Args:
            frame: Input frame (BGR)
        
        Returns:
            Annotated frame
        """
        # 1. Detect persons AND get poses (YOLOv8-Pose does both)
        detections, keypoints_list = self.detector.detect_with_pose(frame)
        
        # 2. Update tracker
        persons = self.tracker.update(detections, frame)
        
        # 3. Process each person
        for i, person in enumerate(persons):
            track_id = person['track_id']
            bbox = person['bbox']
            
            # Get corresponding keypoints (match by index)
            person_keypoints = keypoints_list[i] if i < len(keypoints_list) else None
            
            # Face recognition (every frame for real-time)
            try:
                name, face_conf = self.face_recognizer.recognize_person(frame, bbox)
                if name != "Unknown":
                    print(f"✓ Recognized: {name} ({face_conf:.2f})")
                self.tracker.update_person_name(track_id, name, face_conf)
                person['name'] = name
                person['face_confidence'] = face_conf
            except Exception as e:
                print(f"Face recognition error: {e}")
                person['name'] = "Unknown"
                person['face_confidence'] = 0.0
            
            # Pose estimation (use YOLO keypoints)
            pose_data = self.pose_estimator.estimate_pose(frame, bbox, person_keypoints)
            self.tracker.update_person_pose(track_id, pose_data)
            person['pose_data'] = pose_data
            
            # Pose analysis
            pose_features = self.pose_analyzer.analyze_pose(pose_data)
            
            # Activity classification
            body_center = pose_features.get('body_center')
            activity, activity_conf = self.activity_classifier.classify(
                track_id, pose_features, body_center
            )
            self.tracker.update_person_activity(track_id, activity, activity_conf)
            person['activity'] = activity
            person['activity_confidence'] = activity_conf
        
        # 4. Visualize results
        annotated_frame = self.visualizer.draw_all_persons(
            frame, persons, self.pose_estimator, self.activity_classifier
        )
        
        return annotated_frame
    
    def run_webcam(self, camera_index: int = 0):
        """
        Run system on webcam feed.
        
        Args:
            camera_index: Camera device index
        """
        print(f"\nStarting webcam (index: {camera_index})...")
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save screenshot")
        print("  - Press 'p' to pause/resume")
        print("  - Press 'r' to reset tracker")
        print()
        
        # Use CAP_DSHOW backend for Windows (more reliable)
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        # Set camera properties
        input_config = self.config_manager.get_section('input')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_config.get('frame_width', 1280))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_config.get('frame_height', 720))
        
        paused = False
        frame_count = 0
        screenshot_count = 0
        
        output_config = self.config_manager.get_section('output')
        window_name = output_config.get('window_name', 'Human Recognition System')
        
        # Video writer (optional)
        video_writer = None
        if output_config.get('save_video', False):
            output_path = output_config.get('output_path', 'output/result.mp4')
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fps = input_config.get('fps', 30)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_writer = self.visualizer.create_video_writer(
                output_path, frame_width, frame_height, fps
            )
            print(f"Recording to: {output_path}")
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Failed to read frame")
                        break
                    
                    # Process frame
                    annotated_frame = self.process_frame(frame)
                    
                    # Save to video if enabled
                    if video_writer is not None:
                        video_writer.write(annotated_frame)
                    
                    frame_count += 1
                else:
                    # Use last frame when paused
                    pass
                
                # Display
                if output_config.get('display_window', True):
                    cv2.imshow(window_name, annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_path = Path(output_config.get('screenshot_path', 'output/screenshots'))
                    screenshot_path.mkdir(parents=True, exist_ok=True)
                    filename = screenshot_path / f"screenshot_{screenshot_count:04d}.jpg"
                    self.visualizer.save_screenshot(annotated_frame, str(filename))
                    screenshot_count += 1
                elif key == ord('p'):
                    # Pause/resume
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    print(f"\n{status}")
                elif key == ord('r'):
                    # Reset tracker
                    self.tracker.reset()
                    print("\nTracker reset")
        
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()
            print(f"\nProcessed {frame_count} frames")
    
    def run_video(self, video_path: str, output_path: str = None):
        """
        Run system on video file.
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save output video
        """
        print(f"\nProcessing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS, {total_frames} frames")
        
        # Create video writer if output path specified
        video_writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            video_writer = self.visualizer.create_video_writer(
                output_path, frame_width, frame_height, fps
            )
            print(f"Output will be saved to: {output_path}")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated_frame = self.process_frame(frame)
                
                # Save to output video
                if video_writer is not None:
                    video_writer.write(annotated_frame)
                
                # Display
                cv2.imshow('Processing Video', annotated_frame)
                
                frame_count += 1
                
                # Print progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    elapsed = time.time() - start_time
                    fps_actual = frame_count / elapsed if elapsed > 0 else 0
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - {fps_actual:.1f} FPS")
                
                # Allow quitting
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcessing interrupted")
                    break
        
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()
            
            elapsed = time.time() - start_time
            print(f"\nProcessed {frame_count} frames in {elapsed:.2f}s ({frame_count/elapsed:.2f} FPS)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Human Recognition and Activity Understanding System'
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--camera', type=int, default=None,
                       help='Camera index for webcam mode')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to input video file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output video file')
    
    args = parser.parse_args()
    
    # Initialize system
    system = HumanRecognitionSystem(args.config)
    
    # Run appropriate mode
    if args.video:
        # Video file mode
        output_path = args.output or 'output/result.mp4'
        system.run_video(args.video, output_path)
    else:
        # Webcam mode (default)
        camera_index = args.camera if args.camera is not None else 0
        system.run_webcam(camera_index)


if __name__ == "__main__":
    main()
