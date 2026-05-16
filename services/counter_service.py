"""Exercise counting service - core inference logic for rep counting."""

from .validators import TemporalValidator


class RobustExerciseCounter:
    """
    Core exercise counter using state machine for robust rep counting.
    
    This is the key class that can be called from an API endpoint.
    It handles the state transitions and rep counting logic.
    """
    
    def __init__(self, detector, config):
        """
        Initialize exercise counter.
        
        Args:
            detector: ExerciseDetector instance (Bicep, Pushup, JumpJack)
            config: ExerciseConfig instance
        """
        self.detector = detector
        self.config = config
        self.state = "NOT_READY"
        self.counter = 0
        self.stable_frame_count = 0
        self.last_position = None
        self.temporal_validator = TemporalValidator(
            required_frames=config.temporal_validation_frames
        )
        self.debug_info_ext = {}
    
    def update(self, landmarks, landmarks_visible):
        """
        Update counter with new frame landmarks.
        
        This is the main method to call for each frame.
        
        Args:
            landmarks: MediaPipe landmarks from pose detector
            landmarks_visible: bool, whether landmarks are valid/visible
        
        Returns:
            dict: {
                'state': str,  # Current state (NOT_READY, READY, DOWN, UP)
                'counter': int,  # Current rep count
                'should_increment': bool,  # Whether rep was just completed
                'debug_info': dict  # Debug information
            }
        """
        should_increment = False
        debug_info = {"state": self.state, "position": None, "confidence": 0.0}
        
        # If landmarks not visible, go back to NOT_READY state
        if not landmarks_visible:
            self.state = "NOT_READY"
            self.stable_frame_count = 0
            return {
                "state": self.state,
                "counter": self.counter,
                "should_increment": should_increment,
                "debug_info": debug_info,
            }
        
        # Get movement detection from detector
        movement = self.detector.detect_movement(landmarks)
        position = movement["position"]
        confidence = movement["confidence"]
        
        # Add observation to temporal validator for smoothing
        self.temporal_validator.add_observation(position, confidence)
        
        debug_info["position"] = position
        debug_info["confidence"] = confidence
        debug_info["debug_info"] = movement["debug_info"]
        
        # ============================================================
        # STATE MACHINE FOR REP COUNTING
        # ============================================================
        # Typical sequence: NOT_READY -> READY -> DOWN -> UP -> DOWN -> UP
        #
        # NOT_READY: Waiting for user to get into starting position
        # READY: User is in starting position, can begin counting
        # DOWN: User has moved into starting of rep (first position)
        # UP: User has completed rep (second position) -> INCREMENT
        # Then cycle: UP -> DOWN -> UP
        # ============================================================
        
        if self.state == "NOT_READY":
            # Wait for stable ready position
            if position == self.config.ready_position:
                self.stable_frame_count += 1
            else:
                self.stable_frame_count = 0
            
            self.debug_info_ext["stable_count"] = self.stable_frame_count
            self.debug_info_ext["ready_pos"] = self.config.ready_position
            
            if self.stable_frame_count >= self.config.stability_frames:
                self.state = "READY"
                self.stable_frame_count = 0
        
        elif self.state == "READY":
            # Wait for movement to start
            if position == self.config.start_movement_position:
                self.state = "DOWN"
        
        elif self.state == "DOWN":
            # Wait for completion of rep
            is_stable, _ = self.temporal_validator.is_stable_position(
                self.config.completion_position
            )
            if is_stable:
                self.state = "UP"
                self.counter += 1
                should_increment = True
        
        elif self.state == "UP":
            # Wait for next rep to start
            is_stable, _ = self.temporal_validator.is_stable_position(
                self.config.start_movement_position
            )
            if is_stable:
                self.state = "DOWN"
        
        self.last_position = position
        debug_info["state"] = self.state
        
        return {
            "state": self.state,
            "counter": self.counter,
            "should_increment": should_increment,
            "debug_info": debug_info,
        }
