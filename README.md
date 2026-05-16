# 📌 Exercise Counter - Project Dependencies

## 📝 Overview
This project uses Computer Vision and Web APIs to detect and count exercises in real time using AI-based pose estimation and a backend API system.

---

## 🧠 Core Libraries (Computer Vision & Machine Learning)

### 🔍 Description
These libraries handle video processing, pose detection, and numerical computations.

- OpenCV → video processing and camera input  
- MediaPipe → pose estimation and landmark detection  
- NumPy → numerical computations  

### 📦 Dependencies
opencv-python==4.8.0.74  
mediapipe==0.10.20  
numpy>=1.26.0  

---

## 🌐 Backend (FastAPI Integration)

### 🔍 Description
Used to build REST APIs and handle real-time communication between components.

### 📦 Dependencies
fastapi==0.104.1  
uvicorn[standard]==0.24.0  
python-multipart==0.0.6  

---

## 🖥️ GUI Support (Optional)

### 🔍 Description
Used for image rendering and user interface enhancements.

### 📦 Dependencies
Pillow==10.0.0  

---

## 🧪 Development & Testing

### 🔍 Description
Tools used for testing and development workflow.

### 📦 Dependencies
pytest==7.4.0  
requests==2.31.0  

---

## ⚙️ Installation Guide

### 💻 Command
pip install -r requirements.txt  

---

## 🚀 Important Notes

- Python 3.9+ is recommended  
- Always use a virtual environment (venv)  
- MediaPipe may require specific Python versions for compatibility  
