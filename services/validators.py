"""Validators for temporal consistency and landmark visibility."""


class TemporalValidator:
    """
    Validates that a position is stable across multiple consecutive frames.
    
    Prevents flickering between states by requiring consistency over time.
    """
    
    def __init__(self, required_frames=3):
        """
        Initialize temporal validator.
        
        Args:
            required_frames: Number of consecutive frames required for stability
        """
        self.required_frames = required_frames
        self.frame_buffer = []
    
    def add_observation(self, position, confidence):
        """
        Add a frame observation to the buffer.
        
        Args:
            position: Current detected position
            confidence: Confidence level (0-1)
        """
        self.frame_buffer.append({"position": position, "confidence": confidence})
        if len(self.frame_buffer) > self.required_frames * 2:
            self.frame_buffer.pop(0)
    
    def is_stable_position(self, target_position):
        """
        Check if a position has been stable for required frames.
        
        Args:
            target_position: Position to check stability for
        
        Returns:
            tuple: (is_stable, average_confidence)
        """
        if len(self.frame_buffer) < self.required_frames:
            return False, 0.0
        
        recent = self.frame_buffer[-self.required_frames :]
        matches = sum(1 for obs in recent if obs["position"] == target_position)
        
        if matches >= self.required_frames:
            avg_conf = sum(obs["confidence"] for obs in recent) / len(recent)
            return True, avg_conf
        
        return False, 0.0
