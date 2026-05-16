"""Configuration and data models for exercise detection."""


class ExerciseConfig:
    """Configuration for exercise detection."""
    
    def __init__(
        self,
        name,
        required_landmarks,
        visibility_threshold=0.5,
        stability_frames=20,
        temporal_validation_frames=3,
        use_hysteresis=True,
        ready_position="DOWN",
        start_movement_position="DOWN",
        completion_position="UP",
    ):
        """
        Initialize exercise configuration.
        
        Args:
            name: Exercise name (e.g., "Bicep Curl")
            required_landmarks: List of landmark indices needed for this exercise
            visibility_threshold: Confidence threshold for landmarks (0-1)
            stability_frames: Frames needed to stabilize state
            temporal_validation_frames: Frames for temporal validation
            use_hysteresis: Whether to use hysteresis in detection
            ready_position: Initial position for exercise
            start_movement_position: Position where movement begins
            completion_position: Position where rep is counted
        """
        self.name = name
        self.required_landmarks = required_landmarks
        self.visibility_threshold = visibility_threshold
        self.stability_frames = stability_frames
        self.temporal_validation_frames = temporal_validation_frames
        self.use_hysteresis = use_hysteresis
        self.ready_position = ready_position
        self.start_movement_position = start_movement_position
        self.completion_position = completion_position
