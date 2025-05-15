# Face Recognition System using Python and OpenCV

### 👩‍💻 Developed by: Ishika Sheth & Priya Shukla

---

## 📌 Project Overview

This project is a real-time **Face Recognition System** that uses classical machine learning techniques and computer vision to detect and recognize human faces from a webcam. It is designed to be lightweight, fast, and functional even on basic hardware — making it ideal for applications like **attendance systems**, **access control**, and **personal security**.

---

## 🎯 Features

- Real-time face detection and recognition using webcam
- Detects multiple faces from group images
- Face registration through image capture
- Model training using LBPH (Local Binary Patterns Histogram)
- Simple and user-friendly Flask web interface
- Works offline, requires no internet connection

---

## 🧠 Technologies Used

- **Python 3.x**
- **OpenCV** – for image processing and face detection
- **LBPH** – for face recognition model
- **Haar Cascades** – for face detection
- **Flask** – for web interface
- **Pickle** – for saving trained models
- **NumPy**

---

## 🏗️ Project Structure

Face Recognition/
├── app.py # Flask application
├── train.py # Model training script
├── dataset.py # Script to capture images
├── recognizer.yml # Trained LBPH model
├── labels.pickle # Label encoding for names
├── static/ # CSS and assets
├── templates/ # HTML templates
└── dataset/ # Collected face images


---

## 🛠️ Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/your-username/face-recognition-project.git
cd face-recognition-project

Install dependencies

Make sure Python is installed. Then install required libraries:

pip install opencv-python
pip install opencv-contrib-python
pip install numpy
pip install flask

Capture images for training

python dataset.py
Follow the instructions to capture 20+ images for each person using the webcam.

Train the model

python train.py

Run the web application
python app.py

How It Works
Capture Face Images: Use webcam to collect labeled face data

Train Model: LBPH algorithm is used to learn facial patterns

Detect and Recognize Faces: Real-time video feed is scanned for faces, matched against trained data

Display Results: Name and confidence score shown on screen

📸 Demo
Check out our project demo on YouTube:
🔗https://www.youtube.com/watch?v=-ZMc8d8-4ts
