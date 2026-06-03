# 🏋️ Exercise Counter - AI-Powered Repetition Counter

> A real-time exercise repetition counter powered by AI-based pose estimation and computer vision. Detect and count exercises like bicep curls, push-ups, and jump jacks with high accuracy.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

---

## 📋 Table of Contents
- [Features](#features)
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Docker Deployment](#docker-deployment)
- [Project Structure](#project-structure)
- [Supported Exercises](#supported-exercises)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)

---

## ✨ Features

- **Real-time Exercise Detection**: Uses MediaPipe pose estimation for accurate body landmark detection
- **Multiple Exercise Support**: Bicep curls, push-ups, and jump jacks (extensible for more exercises)
- **Dual Interface**: GUI for desktop use and REST API for programmatic access
- **Video File Processing**: Upload and process video files to get repetition counts
- **Database Integration**: Track workout history with PostgreSQL
- **Docker Ready**: Easy deployment with Docker and docker-compose
- **Robust Detection**: Validates pose visibility, applies temporal smoothing for accurate counting
- **CORS Enabled**: Full cross-origin support for web frontend integration

---

## 📝 Project Overview

**Exercise Counter** is an intelligent fitness application that leverages computer vision and machine learning to automatically count exercise repetitions from video footage. The system combines:

- **MediaPipe Pose Detection**: Detects 33 body landmarks in real-time
- **Custom Exercise Detectors**: Specialized logic for each exercise type
- **Robust Counting Algorithm**: Validates movements with temporal smoothing
- **REST API**: FastAPI backend for remote video processing
- **Modular Architecture**: Separation of concerns across UI, API, and services layers

### Use Cases
- Gym equipment integration
- Fitness app backend
- Home workout tracking
- Physical therapy monitoring
- Fitness class automation

---

## 🏗️ Architecture

The project follows a clean, modular architecture:

```
┌─────────────────────────────────────────────────┐
│           User Interfaces                       │
│  ┌──────────────────┐  ┌──────────────────┐   │
│  │   GUI (Tkinter)  │  │  REST API (Fast) │   │
│  └────────┬─────────┘  └────────┬─────────┘   │
└───────────┼──────────────────────┼─────────────┘
            │                      │
┌───────────┼──────────────────────┼─────────────┐
│           │   Services Layer     │             │
│       ┌───▼──────────────────────▼──┐          │
│       │  RobustExerciseCounter      │          │
│       └───┬───────────────────────┬─┘          │
└───────────┼───────────────────────┼────────────┘
            │                       │
┌───────────┼───────────────────────┼────────────┐
│           │   Core ML Logic       │            │
│    ┌──────▼─────┐  ┌─────────────▼──┐         │
│    │  Detectors │  │   Geometry &   │         │
│    │(Exercise)  │  │   Validators   │         │
│    └─────┬──────┘  └────────┬───────┘         │
└──────────┼───────────────────┼─────────────────┘
           │                   │
┌──────────┼───────────────────┼─────────────────┐
│          │  MediaPipe Pose   │                 │
│          │  Estimation       │                 │
└──────────┼───────────────────┼─────────────────┘
           │
        Video Input
```

### Layer Description

| Layer | Purpose | Components |
|-------|---------|------------|
| **UI Layer** | User interaction | GUI (Tkinter), API (FastAPI) |
| **Services** | Business logic | `RobustExerciseCounter`, validators |
| **Core ML** | Detection logic | Exercise detectors, geometry calculations |
| **Database** | Data persistence | SQLAlchemy ORM, PostgreSQL |

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- pip or conda
- (Optional) Docker & Docker Compose
- Webcam or video files for processing

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd pose_detection
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Database Setup (Optional, for API mode)
```bash
# Install PostgreSQL or use Docker
docker run --name workout_db -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres:15
```

---

## ⚡ Quick Start

### Mode 1: GUI Application
Launch the desktop application with real-time exercise counting:

```bash
python main.py
```

Features:
- Real-time webcam feed
- Live repetition counter
- Visual feedback with skeleton overlay
- Save results to database

### Mode 2: REST API Server
Start the FastAPI server for programmatic access:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Access the API:
- **API Docs**: http://localhost:8000/docs (interactive Swagger UI)
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/health

### Mode 3: Python Script
Process videos programmatically:

```python
from example_api_usage import count_reps_from_video

result = count_reps_from_video(
    video_path="path/to/video.mp4",
    exercise_type="biceps"
)

print(f"Total reps: {result['total_reps']}")
print(f"Status: {result['message']}")
```

---

## 📡 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Count Exercise Repetitions
**Endpoint:** `POST /api/count`

**Description:** Upload a video file and get the count of exercise repetitions.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | ✓ | Video file (MP4, AVI, MOV, MKV, FLV) |
| `username` | String | ✓ | Username for tracking |
| `exercise_type` | String | ✗ | Exercise type: `biceps`, `pushup`, or `jumpjack` (default: `biceps`) |

**Request Example:**
```bash
curl -X POST "http://localhost:8000/api/count" \
  -F "file=@workout.mp4" \
  -F "username=john_doe" \
  -F "exercise_type=biceps"
```

**Response (Success):**
```json
{
  "success": true,
  "exercise": "biceps",
  "count": 15,
  "message": "Successfully processed 240 frames",
  "debug_info": null
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "ValueError",
  "message": "Invalid exercise type. Must be one of: biceps, jumpjack, pushup"
}
```

**Status Codes:**
| Code | Meaning |
|------|---------|
| 200 | Success - repetitions counted |
| 400 | Bad request - invalid parameters or file |
| 422 | Unprocessable entity - video processing failed |
| 500 | Server error |

**Supported Formats:**
- `.mp4` - MPEG-4 video
- `.avi` - Audio Video Interleave
- `.mov` - QuickTime movie
- `.mkv` - Matroska video
- `.flv` - Flash video

---

#### 2. Health Check
**Endpoint:** `GET /api/health`

**Description:** Check if the API service is running.

**Request Example:**
```bash
curl "http://localhost:8000/api/health"
```

**Response:**
```json
{
  "status": "ok",
  "service": "Exercise Counter API"
}
```

---

### Request Examples

#### Python (Requests Library)
```python
import requests

url = "http://localhost:8000/api/count"
with open("video.mp4", "rb") as f:
    files = {"file": f}
    data = {"username": "john_doe", "exercise_type": "biceps"}
    response = requests.post(url, files=files, data=data)

print(response.json())
# Output:
# {
#   "success": true,
#   "exercise": "biceps",
#   "count": 12,
#   "message": "Successfully processed 240 frames"
# }
```

#### JavaScript (Fetch API)
```javascript
const formData = new FormData();
formData.append("file", videoFile);
formData.append("username", "john_doe");
formData.append("exercise_type", "biceps");

const response = await fetch("http://localhost:8000/api/count", {
  method: "POST",
  body: formData,
  headers: {
    "Accept": "application/json"
  }
});

const result = await response.json();
console.log(`Reps: ${result.count}`);
```

#### cURL
```bash
# Basic request
curl -X POST "http://localhost:8000/api/count" \
  -F "file=@myworkout.mp4" \
  -F "username=john_doe" \
  -F "exercise_type=biceps"

# Save response to file
curl -X POST "http://localhost:8000/api/count" \
  -F "file=@myworkout.mp4" \
  -F "username=john_doe" \
  -F "exercise_type=pushup" \
  -o response.json
```



---

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

Run the entire stack (API + PostgreSQL database) with one command:

```bash
docker-compose up -d
```

This will start:
- **API Service**: Running on `http://localhost:8000`
- **PostgreSQL Database**: Running on `localhost:5432`

### Environment Variables

Configure via `docker-compose.yml`:
```yaml
environment:
  DATABASE_URL: postgresql://postgres:postgres@db:5432/workoutdb
  PYTHONUNBUFFERED: 1
```

### Build and Run Separately

```bash
# Build the image
docker build -t exercise-counter:latest .

# Run the container
docker run -p 8000:8000 exercise-counter:latest

# With database
docker run \
  -e DATABASE_URL=postgresql://user:pass@db:5432/workoutdb \
  -p 8000:8000 \
  exercise-counter:latest
```

### Accessing Services

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Database**: localhost:5432

### Stop Services

```bash
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

---

## 📂 Project Structure

```
pose_detection/
├── api/                          # REST API layer
│   ├── routes/
│   │   └── exercise_routes.py   # FastAPI endpoints
│   └── schemas/
│       └── responses.py          # Pydantic response models
│
├── core/                         # Core ML logic
│   ├── detectors.py             # Exercise detector implementations
│   ├── geometry.py              # Angle/distance calculations
│   └── models.py                # ExerciseConfig class
│
├── services/                     # Business logic
│   ├── counter_service.py       # RobustExerciseCounter
│   └── validators.py            # Input validation logic
│
├── db/                           # Database layer
│   ├── database.py              # SQLAlchemy setup
│   ├── models.py                # ORM models (WorkoutResult)
│   └── session.py               # Database session management
│
├── ui/                           # GUI components
│   └── gui.py                   # Tkinter interface
│
├── utils/                        # Utility functions
│   └── stats.py                 # Statistics calculations
│
├── main.py                       # Entry point (GUI or API)
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container image
├── docker-compose.yml            # Multi-container orchestration
└── README.md                     # This file
```

---

## 🏃 Supported Exercises

### 1. Bicep Curl
- **Muscles**: Biceps, forearms
- **Detection**: Arm angle (shoulder-elbow-wrist)
- **Position**: Down (rest) → Up (peak contraction)
- **Key Points**: Uses left arm, requires 30 frame stability

### 2. Push-Up
- **Muscles**: Chest, shoulders, triceps
- **Detection**: Body angle, elbow angle
- **Position**: Up (starting) → Down (chest to floor)
- **Key Points**: Full body coordination, moderate validation

### 3. Jump Jack
- **Muscles**: Full body cardio
- **Detection**: Leg spread position
- **Position**: Together (rest) → Apart (jump)
- **Key Points**: Uses all landmarks, minimal stability requirement

### Adding Custom Exercises

1. **Create a new detector class** in `core/detectors.py`:
```python
class SquatDetector(ExerciseDetector):
    def detect_movement(self, landmarks):
        # Calculate knee angle
        # Return movement state
        pass
```

2. **Define detection logic** using geometry utilities from `core/geometry.py`

3. **Update `exercise_routes.py`** to add the new exercise type

---

## 🛠️ Technologies Used

### Core Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| **MediaPipe** | 0.10.20 | Pose landmark detection |
| **OpenCV** | 4.8.0.74 | Video processing |
| **NumPy** | ≥1.26.0 | Numerical computations |

### Backend
| Library | Version | Purpose |
|---------|---------|---------|
| **FastAPI** | 0.104.1 | REST API framework |
| **Uvicorn** | 0.24.0 | ASGI server |
| **Pydantic** | (FastAPI dep) | Data validation |
| **SQLAlchemy** | 2.0.49 | ORM |

### Database
| Tool | Version | Purpose |
|------|---------|---------|
| **PostgreSQL** | 15 | Production database |
| **psycopg2** | 2.9.12 | PostgreSQL adapter |

### UI
| Library | Purpose |
|---------|---------|
| **PySimpleGUI** | Desktop GUI (built-in with Python) |

### Development
| Tool | Purpose |
|------|---------|
| **Jupyter** | Interactive notebooks |
| **Docker** | Containerization |

---




### 📦 Dependencies
pytest==7.4.0  
requests==2.31.0  

---

## ⚙️ Installation Guide

### 💻 Command
pip install -r requirements.txt  

---

