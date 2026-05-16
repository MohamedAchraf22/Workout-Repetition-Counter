"""PySimpleGUI user interface for exercise counter application."""

import os
from pathlib import Path

import PySimpleGUI as sg
import cv2
import mediapipe as mp
from datetime import datetime

from core import ExerciseConfig, BicepCurlDetector, PushupDetector, RightPushupDetector, JumpJackDetector
from services import RobustExerciseCounter
from utils import NormalizationStatsCollector


sg.theme("LightBlue2")
sg.set_options(font=("Helvetica", 10))


class ExerciseCounterUI:
    """Main UI for exercise counter application."""
    
    def __init__(self):
        """Initialize UI."""
        self.window = None
        self.selected_exercise = None
        self.selected_side = None
        self.config = {}
        self.video_source = 0  # Default to camera (0)
        mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing = mp_drawing
        self.mp_pose = mp.solutions.pose
    
    def create_main_menu(self):
        """Create main menu window."""
        layout = [
            [
                sg.Text(
                    "🏋️  EXERCISE COUNTER 🏋️",
                    font=("Helvetica", 18, "bold"),
                    justification="center",
                )
            ],
            [sg.Text("_" * 50, font=("Helvetica", 1))],
            [sg.Text("Select an exercise to begin:", font=("Helvetica", 12))],
            [sg.Text("")],
            [sg.Button("💪 Biceps", size=(20, 3), font=("Helvetica", 12, "bold"))],
            [sg.Text("")],
            [sg.Button("🤸 Push-ups", size=(20, 3), font=("Helvetica", 12, "bold"))],
            [sg.Text("")],
            [sg.Button("🤾 Jump Jacks", size=(20, 3), font=("Helvetica", 12, "bold"))],
            [sg.Text("")],
            [sg.Button("❌ Exit", size=(20, 2), font=("Helvetica", 11))],
        ]
        
        window = sg.Window(
            "Exercise Counter - Main Menu",
            layout,
            finalize=True,
            size=(400, 450),
        )
        return window
    
    def create_pushup_side_menu(self):
        """Create push-up side selection menu."""
        layout = [
            [
                sg.Text(
                    "🤸 PUSH-UPS - Select Side",
                    font=("Helvetica", 16, "bold"),
                    justification="center",
                )
            ],
            [sg.Text("_" * 50, font=("Helvetica", 1))],
            [
                sg.Text(
                    "Which side do you want to perform push-ups on?",
                    font=("Helvetica", 11),
                )
            ],
            [sg.Text("")],
            [sg.Button("👈 Left Arm", size=(20, 3), font=("Helvetica", 12, "bold"))],
            [sg.Text("")],
            [sg.Button("👉 Right Arm", size=(20, 3), font=("Helvetica", 12, "bold"))],
            [sg.Text("")],
            [sg.Button("⬅️  Back", size=(20, 2), font=("Helvetica", 11))],
        ]
        
        window = sg.Window(
            "Exercise Counter - Push-up Side",
            layout,
            finalize=True,
            size=(400, 350),
        )
        return window
    
    def create_config_screen(self, exercise_name, side=None):
        """Create exercise configuration screen."""
        exercise_params = {
            "Biceps": {
                "title": "💪 Biceps Configuration",
                "description": "Bicep Curl Exercise Settings",
                "params": [
                    {"name": "Stability Frames", "key": "stability_frames", "default": 30},
                    {
                        "name": "Temporal Validation Frames",
                        "key": "temporal_frames",
                        "default": 2,
                    },
                    {"name": "Visibility Threshold", "key": "visibility", "default": 0.5},
                ],
            },
            "Push-ups": {
                "title": "🤸 Push-ups Configuration",
                "description": f"Push-up Exercise Settings ({side} Arm)",
                "params": [
                    {"name": "Stability Frames", "key": "stability_frames", "default": 40},
                    {
                        "name": "Temporal Validation Frames",
                        "key": "temporal_frames",
                        "default": 4,
                    },
                    {"name": "Visibility Threshold", "key": "visibility", "default": 0.6},
                ],
            },
            "Jump Jacks": {
                "title": "🤾 Jump Jacks Configuration",
                "description": "Jump Jacks Exercise Settings (Full-Body Coordinated Motion Detection)",
                "params": [
                    {
                        "name": "Arm OPEN Threshold (°)",
                        "key": "arm_open_threshold",
                        "default": 110,
                        "type": "input",
                    },
                    {
                        "name": "Arm CLOSED Threshold (°)",
                        "key": "arm_closed_threshold",
                        "default": 60,
                        "type": "input",
                    },
                    {
                        "name": "Leg OPEN Threshold (°)",
                        "key": "leg_open_threshold",
                        "default": 100,
                        "type": "input",
                    },
                    {
                        "name": "Leg CLOSED Threshold (°)",
                        "key": "leg_closed_threshold",
                        "default": 95,
                        "type": "input",
                    },
                    {
                        "name": "Min Jump Displacement (0.0-1.0)",
                        "key": "min_jump_displacement",
                        "default": 0.015,
                        "type": "input",
                    },
                    {
                        "name": "Velocity Threshold (0.0-1.0)",
                        "key": "velocity_threshold",
                        "default": 0.003,
                        "type": "input",
                    },
                    {
                        "name": "Jump Memory Duration (frames)",
                        "key": "jump_memory_duration",
                        "default": 8,
                        "type": "input",
                    },
                    {
                        "name": "Required Rising Frames",
                        "key": "required_rising_frames",
                        "default": 2,
                        "type": "input",
                    },
                    {
                        "name": "Stability Frames",
                        "key": "stability_frames",
                        "default": 1,
                        "type": "input",
                    },
                    {
                        "name": "Temporal Validation Frames",
                        "key": "temporal_frames",
                        "default": 2,
                        "type": "input",
                    },
                    {
                        "name": "Visibility Threshold",
                        "key": "visibility",
                        "default": 0.5,
                        "type": "input",
                    },
                    {
                        "name": "Ready Position",
                        "key": "ready_position",
                        "default": "TOGETHER",
                        "type": "dropdown",
                        "options": ["TOGETHER", "APART", "TRANSITION"],
                    },
                    {
                        "name": "Start Movement Position",
                        "key": "start_movement_position",
                        "default": "APART",
                        "type": "dropdown",
                        "options": ["TOGETHER", "APART", "TRANSITION"],
                    },
                    {
                        "name": "Completion Position",
                        "key": "completion_position",
                        "default": "TOGETHER",
                        "type": "dropdown",
                        "options": ["TOGETHER", "APART", "TRANSITION"],
                    },
                ],
            },
        }
        
        config = exercise_params.get(exercise_name, exercise_params["Biceps"])
        
        layout = [
            [
                sg.Text(
                    config["title"],
                    font=("Helvetica", 16, "bold"),
                    justification="center",
                )
            ],
            [sg.Text("_" * 70, font=("Helvetica", 1))],
            [
                sg.Text(
                    config["description"],
                    font=("Helvetica", 11),
                    text_color="#555555",
                )
            ],
            [sg.Text("")],
        ]
        
        for param in config["params"]:
            param_type = param.get("type", "input")
            if param_type == "dropdown":
                layout.append(
                    [
                        sg.Text(
                            param["name"],
                            size=(25, 1),
                            font=("Helvetica", 10),
                        ),
                        sg.Combo(
                            param["options"],
                            default_value=param["default"],
                            key=param["key"],
                            size=(13, 1),
                            font=("Helvetica", 10),
                            readonly=True,
                        ),
                    ]
                )
            else:
                layout.append(
                    [
                        sg.Text(
                            param["name"],
                            size=(25, 1),
                            font=("Helvetica", 10),
                        ),
                        sg.InputText(
                            str(param["default"]),
                            key=param["key"],
                            size=(15, 1),
                            font=("Helvetica", 10),
                        ),
                    ]
                )
        
        layout.extend(
            [
                [sg.Text("")],
                [sg.Text("VIDEO SOURCE", font=("Helvetica", 11, "bold"))],
                [
                    sg.Radio(
                        "📷 Use Webcam",
                        group_id="source",
                        key="use_webcam",
                        default=True,
                        font=("Helvetica", 10),
                    )
                ],
                [
                    sg.Radio(
                        "🎥 Use Video File",
                        group_id="source",
                        key="use_video",
                        font=("Helvetica", 10),
                    )
                ],
                [
                    sg.InputText(
                        key="video_path",
                        size=(35, 1),
                        font=("Helvetica", 9),
                        disabled=True,
                    ),
                    sg.Button("Browse", key="browse_video", font=("Helvetica", 9)),
                ],
                [sg.Text("")],
                [sg.Text("_" * 70, font=("Helvetica", 1))],
                [
                    sg.Button(
                        "✅ START",
                        size=(15, 2),
                        font=("Helvetica", 12, "bold"),
                        button_color=("#FFFFFF", "#28a745"),
                    ),
                    sg.Button("⬅️  Back", size=(15, 2), font=("Helvetica", 12)),
                ],
            ]
        )
        
        window = sg.Window(
            f"Exercise Counter - {exercise_name} Configuration",
            layout,
            finalize=True,
            size=(650, 950),
        )
        
        return window
    
    def validate_config(self, values):
        """Validate user-entered configuration values."""
        errors = []
        
        try:
            stability = int(values["stability_frames"])
            if stability < 1:
                errors.append("Stability Frames must be >= 1")
        except ValueError:
            errors.append("Stability Frames must be a number")
        
        try:
            temporal = int(values["temporal_frames"])
            if temporal < 1:
                errors.append("Temporal Frames must be >= 1")
        except ValueError:
            errors.append("Temporal Frames must be a number")
        
        try:
            visibility = float(values["visibility"])
            if not (0 <= visibility <= 1):
                errors.append("Visibility must be 0-1")
        except ValueError:
            errors.append("Visibility must be a number")
        
        # Validate angle-based thresholds for Jump Jacks
        if "arm_open_threshold" in values:
            try:
                angle = float(values["arm_open_threshold"])
                if not (0 <= angle <= 180):
                    errors.append("Arm OPEN Threshold must be 0-180 degrees")
                else:
                    self.config["arm_open_threshold"] = angle
            except ValueError:
                errors.append("Arm OPEN Threshold must be a number")
        
        if "arm_closed_threshold" in values:
            try:
                angle = float(values["arm_closed_threshold"])
                if not (0 <= angle <= 180):
                    errors.append("Arm CLOSED Threshold must be 0-180 degrees")
                else:
                    self.config["arm_closed_threshold"] = angle
            except ValueError:
                errors.append("Arm CLOSED Threshold must be a number")
        
        if "leg_open_threshold" in values:
            try:
                angle = float(values["leg_open_threshold"])
                if not (0 <= angle <= 180):
                    errors.append("Leg OPEN Threshold must be 0-180 degrees")
                else:
                    self.config["leg_open_threshold"] = angle
            except ValueError:
                errors.append("Leg OPEN Threshold must be a number")
        
        if "leg_closed_threshold" in values:
            try:
                angle = float(values["leg_closed_threshold"])
                if not (0 <= angle <= 180):
                    errors.append("Leg CLOSED Threshold must be 0-180 degrees")
                else:
                    self.config["leg_closed_threshold"] = angle
            except ValueError:
                errors.append("Leg CLOSED Threshold must be a number")
        
        # Validate jump displacement threshold for Jump Jacks
        if "min_jump_displacement" in values:
            try:
                displacement = float(values["min_jump_displacement"])
                if not (0 <= displacement <= 1.0):
                    errors.append("Min Jump Displacement must be 0.0-1.0 (recommended: 0.01-0.03)")
                else:
                    self.config["min_jump_displacement"] = displacement
            except ValueError:
                errors.append("Min Jump Displacement must be a number")
        
        # Validate velocity threshold for Jump Jacks
        if "velocity_threshold" in values:
            try:
                velocity = float(values["velocity_threshold"])
                if not (0 <= velocity <= 1.0):
                    errors.append("Velocity Threshold must be 0.0-1.0 (recommended: 0.001-0.01)")
                else:
                    self.config["velocity_threshold"] = velocity
            except ValueError:
                errors.append("Velocity Threshold must be a number")
        
        # Validate jump threshold for Jump Jacks
        if "jump_threshold" in values:
            try:
                jump_thresh = float(values["jump_threshold"])
                if not (0 <= jump_thresh <= 1.0):
                    errors.append("Jump Threshold must be 0.0-1.0 (recommended: 0.1-0.3)")
                else:
                    self.config["jump_threshold"] = jump_thresh
            except ValueError:
                errors.append("Jump Threshold must be a number")
        
        # Validate jump memory duration for Jump Jacks
        if "jump_memory_duration" in values:
            try:
                jump_mem = int(values["jump_memory_duration"])
                if jump_mem < 1:
                    errors.append("Jump Memory Duration must be >= 1 frame (recommended: 4-12)")
                else:
                    self.config["jump_memory_duration"] = jump_mem
            except ValueError:
                errors.append("Jump Memory Duration must be an integer")
        
        # Validate required rising frames for Jump Jacks
        if "required_rising_frames" in values:
            try:
                rising_frames = int(values["required_rising_frames"])
                if rising_frames < 1:
                    errors.append("Required Rising Frames must be >= 1 (recommended: 1-3)")
                else:
                    self.config["required_rising_frames"] = rising_frames
            except ValueError:
                errors.append("Required Rising Frames must be an integer")
        
        # Validate counting cycle positions for Jump Jacks
        valid_positions = ["TOGETHER", "APART", "TRANSITION"]
        if "ready_position" in values:
            if values["ready_position"] not in valid_positions:
                errors.append("Ready Position must be TOGETHER, APART, or TRANSITION")
            else:
                self.config["ready_position"] = values["ready_position"]
        
        if "start_movement_position" in values:
            if values["start_movement_position"] not in valid_positions:
                errors.append("Start Movement Position must be TOGETHER, APART, or TRANSITION")
            else:
                self.config["start_movement_position"] = values["start_movement_position"]
        
        if "completion_position" in values:
            if values["completion_position"] not in valid_positions:
                errors.append("Completion Position must be TOGETHER, APART, or TRANSITION")
            else:
                self.config["completion_position"] = values["completion_position"]
        
        return errors
    
    def run_counter(self):
        """Run the main exercise counter loop with video/camera input."""
        exercise = self.selected_exercise
        
        # Prepare config
        mp_pose = self.mp_pose
        mp_drawing = self.mp_drawing
        
        # Create configuration object
        if exercise == "Biceps":
            mp_pose = self.mp_pose
            required_landmarks = [
                mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                mp_pose.PoseLandmark.LEFT_ELBOW.value,
                mp_pose.PoseLandmark.LEFT_WRIST.value,
            ]
            config = ExerciseConfig(
                name="Bicep Curl",
                required_landmarks=required_landmarks,
                visibility_threshold=float(self.config["visibility"]),
                stability_frames=int(self.config["stability_frames"]),
                temporal_validation_frames=int(self.config["temporal_frames"]),
                ready_position="DOWN",
                start_movement_position="DOWN",
                completion_position="UP",
            )
            detector = BicepCurlDetector(config)
        
        elif exercise == "Push-ups":
            if self.selected_side == "LEFT":
                required_landmarks = [
                    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                    mp_pose.PoseLandmark.LEFT_ELBOW.value,
                    mp_pose.PoseLandmark.LEFT_WRIST.value,
                ]
                detector_class = PushupDetector
            else:
                required_landmarks = [
                    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                    mp_pose.PoseLandmark.RIGHT_WRIST.value,
                ]
                detector_class = RightPushupDetector
            
            config = ExerciseConfig(
                name="Push-up",
                required_landmarks=required_landmarks,
                visibility_threshold=float(self.config["visibility"]),
                stability_frames=int(self.config["stability_frames"]),
                temporal_validation_frames=int(self.config["temporal_frames"]),
                ready_position="UP",
                start_movement_position="DOWN",
                completion_position="UP",
            )
            detector = detector_class(config)
        
        elif exercise == "Jump Jacks":
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
                visibility_threshold=float(self.config["visibility"]),
                stability_frames=int(self.config["stability_frames"]),
                temporal_validation_frames=int(self.config["temporal_frames"]),
                ready_position=self.config.get("ready_position", "TOGETHER"),
                start_movement_position=self.config.get("start_movement_position", "APART"),
                completion_position=self.config.get("completion_position", "TOGETHER"),
            )
            detector = JumpJackDetector(
                config,
                arm_open_threshold=float(self.config["arm_open_threshold"]),
                arm_closed_threshold=float(self.config["arm_closed_threshold"]),
                leg_open_threshold=float(self.config["leg_open_threshold"]),
                leg_closed_threshold=float(self.config["leg_closed_threshold"]),
                jump_threshold=float(self.config.get("jump_threshold", 0.15)),
                jump_memory_duration=int(self.config.get("jump_memory_duration", 8)),
                required_rising_frames=int(self.config.get("required_rising_frames", 2)),
                min_jump_displacement=float(self.config.get("min_jump_displacement", 0.015)),
                velocity_threshold=float(self.config.get("velocity_threshold", 0.003)),
            )
        
        counter = RobustExerciseCounter(detector, config)
        
        # Initialize stats collector for Jump Jacks debugging
        stats_collector = None
        if exercise == "Jump Jacks":
            stats_collector = NormalizationStatsCollector()
        
        # Ask if user wants to save the video
        save_video = sg.popup_yes_no(
            "Do you want to save the processed video?",
            title="Save Video",
            button_color=("white", "steelblue"),
        )
        
        save_folder = None
        output_video_path = None
        
        if save_video == "Yes":
            save_folder = sg.popup_get_folder(
                "Select folder to save video:",
                default_path=str(Path.home() / "Videos"),
            )
            if not save_folder:
                sg.popup_warning("Video will not be saved.", title="Save Cancelled")
                save_video = "No"
        
        # Run counter loop
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            if self.video_source == 0:
                error_msg = "❌ Cannot open camera!"
            else:
                error_msg = f"❌ Cannot open video file: {self.video_source}"
            sg.popup_error(error_msg, title="Video Source Error")
            return
        
        # Get frame delay for video files
        if self.video_source == 0:
            # Camera: use 10ms delay
            frame_delay = 10
        else:
            # Video file: play at natural speed, skip frames if needed
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                frame_delay = max(1, int(1000 / fps))
            else:
                frame_delay = 33  # Default to ~30 FPS
        
        window_name = f"{exercise} Counter - Press Q or ESC to quit, R to reset"
        if self.video_source != 0:
            source_info = f" [Video: {os.path.basename(self.video_source)}]"
            window_name = f"{exercise} Counter{source_info} - Press Q or ESC to quit, R to reset"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1536, 864)
        
        import time
        
        frame_times = []
        target_fps = cap.get(cv2.CAP_PROP_FPS) if self.video_source != 0 else 30
        
        video_writer = None
        frame_count = 0
        
        with mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as pose:
            while cap.isOpened():
                loop_start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Initialize VideoWriter on first frame
                if video_writer is None and save_video == "Yes":
                    frame_height, frame_width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_video_path = os.path.join(
                        save_folder, f"{exercise}_{timestamp}.mp4"
                    )
                    video_writer = cv2.VideoWriter(
                        output_video_path, fourcc, target_fps, (frame_width, frame_height)
                    )
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                landmarks_visible = False
                
                try:
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        landmarks_visible, _ = detector.validate_landmarks(landmarks)
                except:
                    pass
                
                counter_info = counter.update(landmarks, landmarks_visible)
                state = counter_info["state"]
                rep_count = counter_info["counter"]
                debug = counter_info["debug_info"]
                
                # Record angle statistics for Jump Jacks
                if (
                    stats_collector is not None
                    and "debug_info" in debug
                    and debug["debug_info"]
                ):
                    debug_data = debug["debug_info"]
                    arm_angle = debug_data.get("average_arm_angle", 0)
                    leg_angle = debug_data.get("average_leg_angle", 0)
                    position = debug["position"]
                    # Only record valid positions (not TRANSITION)
                    if (
                        position in ["TOGETHER", "APART"]
                        and arm_angle > 0
                        and leg_angle > 0
                    ):
                        stats_collector.record_frame(arm_angle, leg_angle, position)
                
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(245, 117, 66), thickness=4, circle_radius=2
                    ),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=4),
                )
                
                # Draw UI
                cv2.rectangle(image, (0, 0), (350, 100), (0, 100, 200), -1)
                cv2.putText(
                    image,
                    exercise.upper(),
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    image,
                    str(rep_count),
                    (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.2,
                    (0, 255, 0),
                    3,
                )
                
                state_color = {
                    "NOT_READY": (0, 0, 255),
                    "READY": (0, 255, 0),
                    "DOWN": (0, 165, 255),
                    "UP": (255, 0, 0),
                    "APART": (0, 165, 255),
                    "TOGETHER": (0, 255, 0),
                }.get(state, (255, 255, 255))
                
                cv2.putText(
                    image,
                    f"State: {state}",
                    (200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    state_color,
                    2,
                )
                cv2.putText(
                    image,
                    f'Pos: {debug["position"]}',
                    (200, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )
                
                # Additional debug info for Jump Jacks
                if (
                    exercise == "Jump Jacks"
                    and "debug_info" in debug
                    and debug["debug_info"]
                ):
                    debug_data = debug["debug_info"]
                    avg_arm_angle = debug_data.get("average_arm_angle", 0)
                    avg_leg_angle = debug_data.get("average_leg_angle", 0)
                    left_arm = debug_data.get("left_arm_angle", 0)
                    right_arm = debug_data.get("right_arm_angle", 0)
                    left_leg = debug_data.get("left_leg_angle", 0)
                    right_leg = debug_data.get("right_leg_angle", 0)
                    arms_open = debug_data.get("arms_open", False)
                    legs_open = debug_data.get("legs_open", False)
                    
                    # Jump detection metrics
                    body_center_y = debug_data.get("body_center_y", 0)
                    body_center_velocity = debug_data.get("body_center_velocity", 0)
                    left_ankle_velocity = debug_data.get("left_ankle_velocity", 0)
                    right_ankle_velocity = debug_data.get("right_ankle_velocity", 0)
                    both_ankles_rising = debug_data.get("both_ankles_rising", False)
                    body_center_rising = debug_data.get("body_center_rising", False)
                    current_jump_detected = debug_data.get("current_jump_detected", False)
                    is_rising = debug_data.get("is_rising", False)
                    rising_frames = debug_data.get("rising_frames", 0)
                    
                    recent_jump_detected = debug_data.get("recent_jump_detected", False)
                    jump_memory_frames = debug_data.get("jump_memory_frames", 0)
                    
                    displacement = debug_data.get("displacement", 0)
                    displacement_sufficient = debug_data.get("displacement_sufficient", False)
                    
                    # Display on LEFT side with clear colors (green=true, red=false)
                    y_pos = 120
                    line_height = 18
                    
                    # ARM ANGLES
                    arm_color = (0, 255, 0) if arms_open else (0, 0, 255)
                    cv2.putText(
                        image,
                        f"ARM: {avg_arm_angle:.0f}° (L:{left_arm:.0f}° R:{right_arm:.0f}°)",
                        (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        arm_color,
                        2,
                    )
                    y_pos += line_height
                    cv2.putText(
                        image,
                        f'  OPEN:{float(self.config.get("arm_open_threshold", 130)):.0f}° CLOSE:{float(self.config.get("arm_closed_threshold", 80)):.0f}°',
                        (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (150, 150, 255),
                        1,
                    )
                    y_pos += line_height + 5
                    
                    # LEG ANGLES
                    leg_color = (0, 255, 0) if legs_open else (0, 0, 255)
                    cv2.putText(
                        image,
                        f"LEG: {avg_leg_angle:.0f}° (L:{left_leg:.0f}° R:{right_leg:.0f}°)",
                        (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        leg_color,
                        2,
                    )
                    y_pos += line_height
                    cv2.putText(
                        image,
                        f'  OPEN:{float(self.config.get("leg_open_threshold", 100)):.0f}° CLOSE:{float(self.config.get("leg_closed_threshold", 50)):.0f}°',
                        (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (150, 150, 255),
                        1,
                    )
                    y_pos += line_height + 5
                    
                    # JUMP DETECTION
                    jump_color = (0, 255, 0) if current_jump_detected else (0, 0, 255)
                    cv2.putText(
                        image,
                        f"JUMP DETECTED: {current_jump_detected}",
                        (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        jump_color,
                        2,
                    )
                    y_pos += line_height
                    
                    # Body Center Velocity
                    rising_color = (0, 255, 0) if body_center_rising else (0, 0, 255)
                    cv2.putText(
                        image,
                        f"Body Vel: {body_center_velocity:.4f}",
                        (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        rising_color,
                        1,
                    )
                    y_pos += line_height
                    
                    # Ankle Motion
                    ankle_color = (0, 255, 0) if both_ankles_rising else (0, 0, 255)
                    cv2.putText(
                        image,
                        f"Both Ankles Rising: {both_ankles_rising}",
                        (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        ankle_color,
                        1,
                    )
                    y_pos += line_height
                    cv2.putText(
                        image,
                        f"  L-Ankle: {left_ankle_velocity:.4f} | R-Ankle: {right_ankle_velocity:.4f}",
                        (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (100, 150, 255),
                        1,
                    )
                    y_pos += line_height + 5
                    
                    # Displacement
                    disp_color = (0, 255, 0) if displacement_sufficient else (0, 0, 255)
                    cv2.putText(
                        image,
                        f"Displacement: {displacement:.4f}",
                        (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        disp_color,
                        1,
                    )
                    y_pos += line_height
                    cv2.putText(
                        image,
                        f'  Threshold: {float(self.config.get("min_jump_displacement", 0.015)):.4f}',
                        (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (100, 150, 255),
                        1,
                    )
                    y_pos += line_height + 5
                    
                    # Rising Frames
                    rising_frame_color = (0, 255, 0) if is_rising else (0, 0, 255)
                    cv2.putText(
                        image,
                        f"Rising: {is_rising} | Frames: {rising_frames}",
                        (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        rising_frame_color,
                        1,
                    )
                    y_pos += line_height + 5
                    
                    # Jump Memory
                    memory_color = (0, 255, 0) if recent_jump_detected else (0, 0, 255)
                    cv2.putText(
                        image,
                        f"Jump Memory: {jump_memory_frames}",
                        (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        memory_color,
                        1,
                    )
                    y_pos += line_height
                    
                    # Confidence
                    cv2.putText(
                        image,
                        f'Confidence: {debug.get("confidence", 0):.2f}',
                        (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (100, 255, 100),
                        1,
                    )
                    y_pos += line_height + 5
                    
                    # State info for NOT_READY
                    if state == "NOT_READY":
                        stable_count = counter.debug_info_ext.get("stable_count", 0)
                        ready_pos = counter.debug_info_ext.get("ready_pos", "")
                        cv2.putText(
                            image,
                            f'Ready: {stable_count}/{int(self.config["stability_frames"])} ({ready_pos})',
                            (15, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (100, 150, 255),
                            2,
                        )
                        y_pos += line_height
                        cv2.putText(
                            image,
                            f'NEED: Arm<{float(self.config.get("arm_closed_threshold", 80)):.0f}° AND Leg<{float(self.config.get("leg_closed_threshold", 50)):.0f}°',
                            (15, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 100, 100),
                            1,
                        )
                
                # Show controls
                cv2.putText(
                    image,
                    "Q/ESC = Quit | R = Reset",
                    (15, image.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )
                
                cv2.imshow(window_name, image)
                
                # Write frame to video file
                if video_writer is not None:
                    video_writer.write(image)
                    frame_count += 1
                
                # Maintain video playback speed
                frame_times.append(time.time())
                if len(frame_times) > 30:
                    frame_times.pop(0)
                
                elapsed = (time.time() - loop_start_time) * 1000
                target_frame_time = 1000.0 / target_fps
                wait_time = max(1, int(target_frame_time - elapsed))
                
                key = cv2.waitKey(wait_time) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key == ord("r"):
                    counter.counter = 0
                    print(f"Counter reset! Rep count: 0")
        
        cap.release()
        
        # Release video writer
        if video_writer is not None:
            video_writer.release()
        
        cv2.destroyAllWindows()
        
        # Analyze and print normalization statistics for Jump Jacks
        if stats_collector is not None:
            stats_collector.analyze_and_print()
        
        # Show final result
        if video_writer is not None:
            message = f"✅ Workout Complete!\n\nTotal Reps: {counter.counter}\n\n📁 Video saved to:\n{output_video_path}"
        else:
            message = f"✅ Workout Complete!\n\nTotal Reps: {counter.counter}"
        
        sg.popup(message, title="Session Finished", font=("Courier", 12))
    
    def run(self):
        """Main application loop."""
        while True:
            self.window = self.create_main_menu()
            event, values = self.window.read()
            self.window.close()
            
            if event == sg.WINDOW_CLOSED or event == "❌ Exit":
                break
            
            if event == "💪 Biceps":
                self.selected_exercise = "Biceps"
                self.show_config_screen()
            
            elif event == "🤸 Push-ups":
                self.window = self.create_pushup_side_menu()
                side_event, _ = self.window.read()
                self.window.close()
                
                if side_event == "👈 Left Arm":
                    self.selected_exercise = "Push-ups"
                    self.selected_side = "LEFT"
                    self.show_config_screen("Push-ups", "LEFT")
                
                elif side_event == "👉 Right Arm":
                    self.selected_exercise = "Push-ups"
                    self.selected_side = "RIGHT"
                    self.show_config_screen("Push-ups", "RIGHT")
            
            elif event == "🤾 Jump Jacks":
                self.selected_exercise = "Jump Jacks"
                self.show_config_screen()
    
    def show_config_screen(self, exercise_name=None, side=None):
        """Show configuration screen and handle user input."""
        if exercise_name is None:
            exercise_name = self.selected_exercise
        
        self.window = self.create_config_screen(exercise_name, side)
        
        while True:
            event, values = self.window.read(timeout=100)
            
            # Handle browse button click
            if event == "browse_video":
                if values["use_video"]:
                    file_path = sg.popup_get_file(
                        "Select a video file:",
                        file_types=(
                            ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
                            ("All Files", "*"),
                        ),
                    )
                    if file_path:
                        self.window["video_path"].update(file_path)
                else:
                    sg.popup_warning(
                        'Please select "Use Video File" first', title="Info"
                    )
                continue
            
            # Handle radio button changes
            if event == "use_webcam":
                self.window["video_path"].update(disabled=True)
                continue
            
            if event == "use_video":
                self.window["video_path"].update(disabled=False)
                continue
            
            if event == sg.WINDOW_CLOSED or event == "⬅️  Back":
                self.window.close()
                return
            
            if event == "✅ START":
                errors = self.validate_config(values)
                
                # Check if video file is selected when using video
                if values["use_video"] and not values["video_path"]:
                    errors.append("Please select a video file or use webcam")
                elif values["use_video"] and not os.path.exists(values["video_path"]):
                    errors.append("Video file does not exist")
                
                if errors:
                    sg.popup_error(
                        "\n".join(errors), title="Configuration Error"
                    )
                    continue
                
                self.config = values
                
                # Set video source
                if values["use_webcam"]:
                    self.video_source = 0
                else:
                    self.video_source = values["video_path"]
                
                self.window.close()
                
                # Run the counter
                self.run_counter()
                return


def main():
    """Application entry point."""
    try:
        ui = ExerciseCounterUI()
        ui.run()
    except Exception as e:
        sg.popup_error(f"Error: {str(e)}", title="Application Error")
    finally:
        pass


if __name__ == "__main__":
    main()
