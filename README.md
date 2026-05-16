📌 Exercise Counter - Project Dependencies

This project uses Computer Vision and Web APIs to detect and count exercises in real time.

🧠 Core Libraries (Computer Vision & ML)
OpenCV → video processing and camera input
MediaPipe → pose estimation and landmark detection
NumPy → numerical computations
opencv-python==4.8.0.74
mediapipe==0.10.20
numpy>=1.26.0
🌐 Backend (FastAPI Integration)

Used for building REST APIs and real-time communication.

fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
🖥️ GUI Support (Optional)

Used for image rendering and UI enhancements.

Pillow==10.0.0
🧪 Development & Testing (Optional)

Used during development and API testing.

pytest==7.4.0
requests==2.31.0
⚙️ Installation

To install all dependencies:

pip install -r requirements.txt
🚀 Notes
Make sure you're using Python 3.9+
It's recommended to use a virtual environment (venv)
MediaPipe may require compatible Python versions
