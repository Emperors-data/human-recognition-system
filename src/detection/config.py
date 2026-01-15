"""
Detection configuration settings.
"""

class DetectionConfig:
    """Configuration for person detection."""
    
    def __init__(self, config_dict=None):
        """
        Initialize detection configuration.
        
        Args:
            config_dict: Dictionary containing configuration parameters
        """
        if config_dict is None:
            config_dict = {}
        
        self.model = config_dict.get('model', 'yolov5s')
        self.confidence_threshold = config_dict.get('confidence_threshold', 0.5)
        self.nms_iou_threshold = config_dict.get('nms_iou_threshold', 0.45)
        self.device = config_dict.get('device', 'cuda')
        
        # Person class ID in COCO dataset
        self.person_class_id = 0
    
    def __repr__(self):
        return (f"DetectionConfig(model={self.model}, "
                f"confidence={self.confidence_threshold}, "
                f"device={self.device})")
