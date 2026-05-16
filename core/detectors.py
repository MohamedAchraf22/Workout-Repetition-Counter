"""Exercise detection logic using MediaPipe pose landmarks."""

import numpy as np
import mediapipe as mp

from .geometry import calculate_angle
from .models import ExerciseConfig


class ExerciseDetector:
    """Base class for exercise detection."""
    
    def __init__(self, config):
        """
        Initialize detector with exercise config.
        
        Args:
            config: ExerciseConfig instance
        """
        self.config = config
    
    def get_required_landmarks(self):
        """Return list of required landmark indices."""
        return self.config.required_landmarks
    
    def validate_landmarks(self, landmarks):
        """
        Validate that all required landmarks are visible and confident.
        
        Args:
            landmarks: MediaPipe landmarks list
        
        Returns:
            tuple: (all_valid, visibility_scores)
        """
        visibility_scores = []
        for idx in self.config.required_landmarks:
            vis = landmarks[idx].visibility
            visibility_scores.append(vis)
        
        all_valid = all(v >= self.config.visibility_threshold for v in visibility_scores)
        return all_valid, visibility_scores
    
    def detect_movement(self, landmarks):
        """
        Detect movement from landmarks.
        
        Must be implemented by subclasses.
        
        Args:
            landmarks: MediaPipe landmarks list
        
        Returns:
            dict: {
                'position': str,  # Current position state
                'confidence': float,  # Confidence 0-1
                'debug_info': dict  # Debug information
            }
        """
        raise NotImplementedError()


class BicepCurlDetector(ExerciseDetector):
    """Detector for bicep curl exercise."""
    
    def detect_movement(self, landmarks):
        """
        Detect bicep curl movement.
        
        Uses angle between shoulder-elbow-wrist to classify UP or DOWN position.
        """
        mp_pose = mp.solutions.pose
        SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW.value
        WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value
        
        shoulder = [landmarks[SHOULDER].x, landmarks[SHOULDER].y]
        elbow = [landmarks[ELBOW].x, landmarks[ELBOW].y]
        wrist = [landmarks[WRIST].x, landmarks[WRIST].y]
        
        angle = calculate_angle(shoulder, elbow, wrist)
        
        if angle > 120:
            position = "DOWN"
            confidence = 0.9
        elif angle < 70:
            position = "UP"
            confidence = 0.9
        else:
            position = "BETWEEN"
            confidence = 0.5
        
        return {
            "position": position,
            "confidence": confidence,
            "debug_info": {"angle": angle},
        }


class PushupDetector(ExerciseDetector):
    """Detector for push-up exercise (left arm)."""
    
    def detect_movement(self, landmarks):
        """
        Detect push-up movement (left arm).
        
        Uses angle between shoulder-elbow-wrist to classify UP or DOWN position.
        """
        mp_pose = mp.solutions.pose
        SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW.value
        WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value
        
        shoulder = [landmarks[SHOULDER].x, landmarks[SHOULDER].y]
        elbow = [landmarks[ELBOW].x, landmarks[ELBOW].y]
        wrist = [landmarks[WRIST].x, landmarks[WRIST].y]
        
        angle = calculate_angle(shoulder, elbow, wrist)
        
        if angle < 90:
            position = "DOWN"
            confidence = 0.95
        elif angle > 140:
            position = "UP"
            confidence = 0.95
        else:
            position = "TRANSITION"
            confidence = 0.3
        
        return {
            "position": position,
            "confidence": confidence,
            "debug_info": {"angle": angle},
        }


class RightPushupDetector(ExerciseDetector):
    """Detector for push-up exercise (right arm)."""
    
    def detect_movement(self, landmarks):
        """
        Detect push-up movement (right arm).
        
        Uses angle between shoulder-elbow-wrist to classify UP or DOWN position.
        """
        mp_pose = mp.solutions.pose
        SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW.value
        WRIST = mp_pose.PoseLandmark.RIGHT_WRIST.value
        
        shoulder = [landmarks[SHOULDER].x, landmarks[SHOULDER].y]
        elbow = [landmarks[ELBOW].x, landmarks[ELBOW].y]
        wrist = [landmarks[WRIST].x, landmarks[WRIST].y]
        
        angle = calculate_angle(shoulder, elbow, wrist)
        
        if angle < 90:
            position = "DOWN"
            confidence = 0.95
        elif angle > 140:
            position = "UP"
            confidence = 0.95
        else:
            position = "TRANSITION"
            confidence = 0.3
        
        return {
            "position": position,
            "confidence": confidence,
            "debug_info": {"angle": angle},
        }


class JumpJackDetector(ExerciseDetector):
    """Detector for jump jack exercise with coordinated arm, leg, and jump detection."""
    
    def __init__(
        self,
        config,
        arm_open_threshold=130,
        arm_closed_threshold=80,
        leg_open_threshold=100,
        leg_closed_threshold=50,
        jump_threshold=0.15,
        jump_memory_duration=8,
        required_rising_frames=2,
        min_jump_displacement=0.015,
        velocity_threshold=0.003,
    ):
        """
        Initialize Jump Jack detector with thresholds.
        
        Args:
            config: ExerciseConfig instance
            arm_open_threshold: Angle threshold for arms open (degrees)
            arm_closed_threshold: Angle threshold for arms closed (degrees)
            leg_open_threshold: Angle threshold for legs open (degrees)
            leg_closed_threshold: Angle threshold for legs closed (degrees)
            jump_threshold: Threshold for jump detection
            jump_memory_duration: Frames to remember jump for posture opening delay
            required_rising_frames: Frames needed to confirm rising motion
            min_jump_displacement: Minimum vertical displacement for jump
            velocity_threshold: Velocity threshold for upward motion
        """
        super().__init__(config)
        self.arm_open_threshold = arm_open_threshold
        self.arm_closed_threshold = arm_closed_threshold
        self.leg_open_threshold = leg_open_threshold
        self.leg_closed_threshold = leg_closed_threshold
        self.jump_threshold = jump_threshold
        self.jump_memory_duration = jump_memory_duration
        self.min_jump_displacement = min_jump_displacement
        self.velocity_threshold = velocity_threshold
        
        # Jump detection state
        self.baseline_hip_y = None
        
        # ============================================================
        # Jump Memory System:
        # Why it's needed: Timing mismatch between jump motion and posture opening
        #
        # Problem: In real jump jacks, the sequence is:
        #   Frame 1: Person jumps (legs still closed)
        #   Frame 2-4: Person is in air, arms/legs start opening
        #   Frame 5-6: Person lands, arms/legs now fully open
        #
        # If we require jump AND open_posture in same frame, most reps fail!
        # Solution: Remember that a jump happened for several frames, allowing
        # posture opening to occur a few frames AFTER the jump is detected.
        #
        # This prevents:
        # - Jumps without posture change not being counted
        # - Posture changes without jumping being counted
        # - Mismatch between body motion timing and pose timing
        # ============================================================
        self.jump_memory_frames = 0
        
        self.rising_frames = 0
        self.required_rising_frames = required_rising_frames
        self.previous_body_center_y = None
        self.previous_left_ankle_y = None
        self.previous_right_ankle_y = None
        self.body_center_velocity_buffer = []
        self.ankle_motion_buffer = []
    
    def _detect_jump(self, landmarks):
        """
        Detect vertical jump motion from body center and ankle positions.
        
        Returns:
            dict: Jump detection info
        """
        mp_pose = mp.solutions.pose
        LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
        RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
        LEFT_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE.value
        RIGHT_ANKLE = mp_pose.PoseLandmark.RIGHT_ANKLE.value
        
        left_shoulder = [landmarks[LEFT_SHOULDER].x, landmarks[LEFT_SHOULDER].y]
        right_shoulder = [landmarks[RIGHT_SHOULDER].x, landmarks[RIGHT_SHOULDER].y]
        left_hip = [landmarks[LEFT_HIP].x, landmarks[LEFT_HIP].y]
        right_hip = [landmarks[RIGHT_HIP].x, landmarks[RIGHT_HIP].y]
        left_ankle = [landmarks[LEFT_ANKLE].x, landmarks[LEFT_ANKLE].y]
        right_ankle = [landmarks[RIGHT_ANKLE].x, landmarks[RIGHT_ANKLE].y]
        
        body_center_y = (
            left_hip[1] + right_hip[1] + left_shoulder[1] + right_shoulder[1]
        ) / 4.0
        left_ankle_y = left_ankle[1]
        right_ankle_y = right_ankle[1]
        
        if self.previous_body_center_y is None:
            self.previous_body_center_y = body_center_y
            self.previous_left_ankle_y = left_ankle_y
            self.previous_right_ankle_y = right_ankle_y
        
        if self.previous_left_ankle_y is None:
            self.previous_left_ankle_y = left_ankle_y
        if self.previous_right_ankle_y is None:
            self.previous_right_ankle_y = right_ankle_y
        
        body_center_velocity = self.previous_body_center_y - body_center_y
        left_ankle_velocity = self.previous_left_ankle_y - left_ankle_y
        right_ankle_velocity = self.previous_right_ankle_y - right_ankle_y
        
        displacement = abs(body_center_velocity)
        
        body_center_rising = body_center_velocity > self.velocity_threshold
        left_ankle_rising = left_ankle_velocity > self.velocity_threshold
        right_ankle_rising = right_ankle_velocity > self.velocity_threshold
        both_ankles_rising = left_ankle_rising and right_ankle_rising
        
        displacement_sufficient = displacement > self.min_jump_displacement
        
        if len(self.body_center_velocity_buffer) > 5:
            self.body_center_velocity_buffer.pop(0)
        self.body_center_velocity_buffer.append(body_center_velocity)
        
        if len(self.ankle_motion_buffer) > 5:
            self.ankle_motion_buffer.pop(0)
        self.ankle_motion_buffer.append(both_ankles_rising)
        
        body_center_consistent_rise = (
            len(self.body_center_velocity_buffer) > 0
            and all(
                v > self.velocity_threshold
                for v in self.body_center_velocity_buffer[-2:]
            )
            if len(self.body_center_velocity_buffer) >= 2
            else False
        )
        ankles_consistent_motion = (
            len(self.ankle_motion_buffer) > 0
            and sum(self.ankle_motion_buffer[-2:]) >= 1
            if len(self.ankle_motion_buffer) >= 2
            else False
        )
        
        jump_detected = (
            body_center_rising
            and both_ankles_rising
            and body_center_consistent_rise
            and displacement_sufficient
        )
        
        return {
            "jump_detected": jump_detected,
            "body_center_y": body_center_y,
            "body_center_velocity": body_center_velocity,
            "displacement": displacement,
            "displacement_sufficient": displacement_sufficient,
            "left_ankle_y": left_ankle_y,
            "right_ankle_y": right_ankle_y,
            "left_ankle_velocity": left_ankle_velocity,
            "right_ankle_velocity": right_ankle_velocity,
            "both_ankles_rising": both_ankles_rising,
            "body_center_rising": body_center_rising,
        }
    
    def detect_movement(self, landmarks):
        """
        Detect jump jack movement (arms, legs, and jump coordination).
        
        Returns position as TOGETHER, APART, or TRANSITION based on:
        - Arm angle (open/closed)
        - Leg angle (open/closed)
        - Vertical jump detection (recent jump memory)
        """
        mp_pose = mp.solutions.pose
        LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        LEFT_ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW.value
        RIGHT_ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW.value
        LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
        RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
        LEFT_KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value
        RIGHT_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE.value
        
        # Extract landmark positions
        left_shoulder = [landmarks[LEFT_SHOULDER].x, landmarks[LEFT_SHOULDER].y]
        right_shoulder = [landmarks[RIGHT_SHOULDER].x, landmarks[RIGHT_SHOULDER].y]
        left_elbow = [landmarks[LEFT_ELBOW].x, landmarks[LEFT_ELBOW].y]
        right_elbow = [landmarks[RIGHT_ELBOW].x, landmarks[RIGHT_ELBOW].y]
        left_hip = [landmarks[LEFT_HIP].x, landmarks[LEFT_HIP].y]
        right_hip = [landmarks[RIGHT_HIP].x, landmarks[RIGHT_HIP].y]
        left_knee = [landmarks[LEFT_KNEE].x, landmarks[LEFT_KNEE].y]
        right_knee = [landmarks[RIGHT_KNEE].x, landmarks[RIGHT_KNEE].y]
        
        # ============================================================
        # ARM AND LEG ANGLE DETECTION
        # ============================================================
        left_arm_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
        right_arm_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
        average_arm_angle = (left_arm_angle + right_arm_angle) / 2.0
        
        left_leg_angle = calculate_angle(left_knee, left_hip, right_hip)
        right_leg_angle = calculate_angle(right_knee, right_hip, left_hip)
        average_leg_angle = (left_leg_angle + right_leg_angle) / 2.0
        
        arms_open = average_arm_angle > self.arm_open_threshold
        legs_open = average_leg_angle > self.leg_open_threshold
        arms_closed = average_arm_angle < self.arm_closed_threshold
        legs_closed = average_leg_angle < self.leg_closed_threshold
        
        # ============================================================
        # VERTICAL JUMP DETECTION
        # ============================================================
        jump_info = self._detect_jump(landmarks)
        
        body_center_velocity = jump_info["body_center_velocity"]
        is_rising = body_center_velocity > 0
        
        if is_rising:
            self.rising_frames += 1
        else:
            self.rising_frames = 0
        
        current_jump_detected = (
            jump_info["jump_detected"] and self.rising_frames >= self.required_rising_frames
        )
        
        # ============================================================
        # JUMP MEMORY STATE MANAGEMENT
        # ============================================================
        # Set memory when upward jump is detected
        if current_jump_detected:
            self.jump_memory_frames = self.jump_memory_duration
        
        # Decrement memory every frame (automatic expiration)
        if self.jump_memory_frames > 0:
            self.jump_memory_frames -= 1
        
        # Check if jump is still "recent" (within memory window)
        recent_jump_detected = self.jump_memory_frames > 0
        
        # ============================================================
        # POSITION CLASSIFICATION
        # ============================================================
        # APART position requires ALL THREE conditions:
        #   1. arms_open    - Arms raised/spread overhead
        #   2. legs_open    - Legs spread apart
        #   3. recent_jump_detected - Recent upward motion (from memory)
        #
        # Why require all three?
        # - Prevents counting static poses without jumping
        # - Handles timing mismatch: jump happens first, posture opens 1-3 frames later
        # - Memory duration (8 frames @ 30fps) = ~267ms window for posture to open
        #
        # TOGETHER position requires:
        #   1. arms_closed - Arms at sides
        #   2. legs_closed - Legs together
        #
        # TRANSITION is any other combination (flickering prevention)
        
        if arms_open and legs_open and recent_jump_detected:
            position = "APART"
            confidence = 0.95
        elif arms_closed and legs_closed:
            position = "TOGETHER"
            confidence = 0.95
        else:
            position = "TRANSITION"
            confidence = 0.3
        
        # Update previous positions for next frame's velocity calculations
        # Must happen every frame for accurate motion tracking
        self.previous_body_center_y = jump_info["body_center_y"]
        self.previous_left_ankle_y = jump_info["left_ankle_y"]
        self.previous_right_ankle_y = jump_info["right_ankle_y"]
        
        left_ankle_velocity = jump_info["left_ankle_velocity"]
        right_ankle_velocity = jump_info["right_ankle_velocity"]
        both_ankles_rising = jump_info["both_ankles_rising"]
        
        return {
            "position": position,
            "confidence": confidence,
            "debug_info": {
                # Arm angles
                "left_arm_angle": round(left_arm_angle, 2),
                "right_arm_angle": round(right_arm_angle, 2),
                "average_arm_angle": round(average_arm_angle, 2),
                # Leg angles
                "left_leg_angle": round(left_leg_angle, 2),
                "right_leg_angle": round(right_leg_angle, 2),
                "average_leg_angle": round(average_leg_angle, 2),
                # State indicators
                "arms_open": arms_open,
                "legs_open": legs_open,
                "arms_closed": arms_closed,
                "legs_closed": legs_closed,
                "body_center_y": round(jump_info["body_center_y"], 4),
                "body_center_velocity": round(body_center_velocity, 4),
                "displacement": round(jump_info["displacement"], 4),
                "displacement_sufficient": jump_info["displacement_sufficient"],
                "left_ankle_velocity": round(left_ankle_velocity, 4),
                "right_ankle_velocity": round(right_ankle_velocity, 4),
                "both_ankles_rising": both_ankles_rising,
                "body_center_rising": jump_info["body_center_rising"],
                "is_rising": is_rising,
                "rising_frames": self.rising_frames,
                "current_jump_detected": current_jump_detected,
                "recent_jump_detected": recent_jump_detected,
                "jump_memory_frames": self.jump_memory_frames,
            },
        }
