"""Core ML logic for pose detection and exercise detection."""

from .geometry import calculate_angle
from .models import ExerciseConfig
from .detectors import (
    ExerciseDetector,
    BicepCurlDetector,
    PushupDetector,
    RightPushupDetector,
    JumpJackDetector,
)

__all__ = [
    "calculate_angle",
    "ExerciseConfig",
    "ExerciseDetector",
    "BicepCurlDetector",
    "PushupDetector",
    "RightPushupDetector",
    "JumpJackDetector",
]
