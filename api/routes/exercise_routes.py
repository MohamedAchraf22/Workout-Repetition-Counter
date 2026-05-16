"""
FastAPI routes for exercise counting.

This is the API layer that orchestrates requests/responses.
Business logic is kept in the services layer (counter_service.py).

ARCHITECTURE:
  Client uploads video
    ↓
  FastAPI endpoint receives file
    ↓
  Temporary file saved if needed
    ↓
  Call existing services.RobustExerciseCounter
    ↓
  Return JSON response

Key principle: Thin routes, logic stays in services.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional
import cv2
import tempfile
import os
import mediapipe as mp

from core import ExerciseConfig, BicepCurlDetector, JumpJackDetector, PushupDetector
from services import RobustExerciseCounter
from api.schemas.responses import ExerciseCountResponse, ErrorResponse


# Create router for exercise endpoints
router = APIRouter(prefix="/api", tags=["exercises"])


def process_video_for_counting(video_path: str, exercise_type: str) -> dict:
    """
    Internal helper: Process video using existing counter service.
    
    This function demonstrates the key principle:
    API routes should be THIN - actual logic lives in services.
    
    Args:
        video_path: Path to video file
        exercise_type: Type of exercise ("biceps", "pushup", "jumpjack")
    
    Returns:
        dict with keys: success, count, message
    
    Raises:
        ValueError: If exercise type unknown or video cannot be opened
    """
    
    # ================================================================
    # SETUP: Create exercise-specific configuration and detector
    # ================================================================
    mp_pose = mp.solutions.pose
    
    if exercise_type.lower() == "biceps":
        # Bicep curl uses arm landmarks
        required_landmarks = [
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_ELBOW.value,
            mp_pose.PoseLandmark.LEFT_WRIST.value,
        ]
        config = ExerciseConfig(
            name="Bicep Curl",
            required_landmarks=required_landmarks,
            visibility_threshold=0.5,
            stability_frames=30,
            temporal_validation_frames=2,
            ready_position="DOWN",
            start_movement_position="DOWN",
            completion_position="UP",
        )
        detector = BicepCurlDetector(config)
    
    elif exercise_type.lower() == "jumpjack":
        # Jump jacks use full body landmarks
        required_landmarks = [
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_ELBOW.value,
            mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            mp_pose.PoseLandmark.LEFT_HIP.value,
            mp_pose.PoseLandmark.RIGHT_HIP.value,
            mp_pose.PoseLandmark.LEFT_KNEE.value,
            mp_pose.PoseLandmark.RIGHT_KNEE.value,
        ]
        config = ExerciseConfig(
            name="Jump Jacks",
            required_landmarks=required_landmarks,
            visibility_threshold=0.5,
            stability_frames=1,
            temporal_validation_frames=2,
            ready_position="TOGETHER",
            start_movement_position="APART",
            completion_position="TOGETHER",
        )
        detector = JumpJackDetector(config)
    
    elif exercise_type.lower() == "pushup":
        # Pushups use arm and torso landmarks
        required_landmarks = [
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_ELBOW.value,
            mp_pose.PoseLandmark.LEFT_WRIST.value,
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        ]
        config = ExerciseConfig(
            name="Push-up",
            required_landmarks=required_landmarks,
            visibility_threshold=0.5,
            stability_frames=20,
            temporal_validation_frames=2,
            ready_position="UP",
            start_movement_position="UP",
            completion_position="DOWN",
        )
        detector = PushupDetector(config)
    
    else:
        raise ValueError(f"Unknown exercise type: {exercise_type}")
    
    # ================================================================
    # CORE LOGIC: Call existing service to process video
    # ================================================================
    counter = RobustExerciseCounter(detector, config)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    try:
        # Process each frame using MediaPipe pose detection
        with mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as pose:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect pose landmarks
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                
                # Check if landmarks are visible/valid
                landmarks_visible = False
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    landmarks_visible, _ = detector.validate_landmarks(landmarks)
                
                # Update counter (key service call)
                counter.update(landmarks, landmarks_visible)
        
        return {
            "success": True,
            "count": counter.counter,
            "message": f"Successfully processed {frame_count} frames",
        }
    
    finally:
        cap.release()


@router.post("/count", response_model=ExerciseCountResponse)
async def count_exercise_reps(
    file: UploadFile = File(...),
    exercise_type: str = "biceps"
) -> ExerciseCountResponse:
    """
    Count exercise repetitions from an uploaded video.
    
    ENDPOINT: POST /api/count
    
    PARAMETERS:
        file: Video file (MP4, AVI, MOV, etc.)
        exercise_type: Type of exercise - "biceps", "jumpjack", or "pushup"
    
    RESPONSE:
        {
            "success": true,
            "exercise": "biceps",
            "count": 12,
            "message": "Successfully processed X frames"
        }
    
    FLOW:
        1. Validate uploaded file
        2. Save to temporary location
        3. Call services.RobustExerciseCounter via process_video_for_counting()
        4. Return JSON result
        5. Clean up temporary file
    
    ERROR HANDLING:
        - 400: Invalid/missing file
        - 400: Invalid exercise type
        - 422: Processing failure
    
    NOTE: This is a THIN route that only orchestrates.
    All heavy lifting happens in the services layer.
    """
    
    # Validate file is provided
    if not file:
        raise HTTPException(
            status_code=400,
            detail="No file provided"
        )
    
    # Validate exercise type
    valid_exercises = ["biceps", "jumpjack", "pushup"]
    if exercise_type.lower() not in valid_exercises:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid exercise type. Must be one of: {', '.join(valid_exercises)}"
        )
    
    # Validate file extension
    valid_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv"]
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Supported: {', '.join(valid_extensions)}"
        )
    
    # Create temporary file for processing
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext,
            dir=tempfile.gettempdir()
        ) as temp:
            temp_file = temp.name
            # Write uploaded file to temp location
            contents = await file.read()
            temp.write(contents)
        
        # Process video using existing service
        result = process_video_for_counting(temp_file, exercise_type)
        
        return ExerciseCountResponse(
            success=result["success"],
            exercise=exercise_type.lower(),
            count=result["count"],
            message=result["message"]
        )
    
    except ValueError as e:
        # Validation errors (unknown exercise, can't open video)
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )
    
    except Exception as e:
        # Unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )
    
    finally:
        # Always clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except:
                pass  # Ignore cleanup errors


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        {"status": "ok", "service": "Exercise Counter API"}
    """
    return {
        "status": "ok",
        "service": "Exercise Counter API"
    }
