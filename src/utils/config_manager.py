"""
Configuration management utilities.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """
    Manages system configuration from YAML file.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            print(f"Config file not found: {self.config_path}")
            print("Using default configuration")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Configuration loaded from: {self.config_path}")
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default configuration")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'detection': {
                'model': 'yolov8n',
                'confidence_threshold': 0.5,
                'nms_iou_threshold': 0.45,
                'device': 'cpu'
            },
            'face_recognition': {
                'tolerance': 0.6,
                'model': 'hog',
                'num_jitters': 1,
                'face_database_path': 'known_faces',
                'encodings_cache': 'face_encodings.pkl'
            },
            'pose': {
                'model_complexity': 1,
                'min_detection_confidence': 0.5,
                'min_tracking_confidence': 0.5,
                'enable_segmentation': False,
                'smooth_landmarks': True
            },
            'activity': {
                'method': 'rule_based',
                'smoothing_window': 5,
                'confidence_threshold': 0.6
            },
            'tracking': {
                'max_age': 30,
                'min_hits': 3,
                'iou_threshold': 0.3,
                'max_iou_distance': 0.7,
                'max_cosine_distance': 0.3,
                'nn_budget': 100
            },
            'visualization': {
                'show_bbox': True,
                'show_skeleton': True,
                'show_activity': True,
                'show_face_name': True,
                'show_fps': True,
                'show_person_count': True,
                'bbox_thickness': 2,
                'skeleton_thickness': 2,
                'font_scale': 0.6,
                'font_thickness': 2
            },
            'input': {
                'camera_index': 0,
                'video_path': None,
                'frame_width': 1280,
                'frame_height': 720,
                'fps': 30
            },
            'output': {
                'save_video': False,
                'output_path': 'output/result.mp4',
                'save_screenshots': True,
                'screenshot_path': 'output/screenshots',
                'display_window': True,
                'window_name': 'Human Recognition System'
            },
            'performance': {
                'skip_frames': 0,
                'resize_factor': 1.0,
                'max_persons': 10
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'detection.model')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., 'detection', 'pose')
        
        Returns:
            Section dictionary
        """
        return self.config.get(section, {})
    
    def update(self, key: str, value: Any):
        """
        Update configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: New value
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: str = None):
        """
        Save configuration to file.
        
        Args:
            path: Optional path to save to (defaults to original config_path)
        """
        save_path = Path(path) if path else self.config_path
        
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            print(f"Configuration saved to: {save_path}")
        except Exception as e:
            print(f"Error saving config: {e}")
