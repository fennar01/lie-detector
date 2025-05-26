import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

st.set_page_config(page_title="Lie Detector", layout="centered")
st.title("Lie Detector - Real-Time Facial Video Analysis")

st.write("""
This app analyzes facial video data and displays a live probability that the subject is lying.

**Current Feature:**
- Real-time facial detection and tracking

**Features coming soon:**
- Sentiment analysis
- Blood perfusion analysis
- Microexpression detection
- Nervousness analysis

---
""")

st.header("Live Video Input: Face Detection Demo")

# Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Streamlit video capture
run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to access webcam.")
                break
            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # To improve performance
            image.flags.writeable = False
            results = face_detection.process(image)
            image.flags.writeable = True
            # Draw face detections
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
            # Display
            FRAME_WINDOW.image(image)
            # Streamlit workaround for breaking loop
            if not st.session_state.get('run', True):
                break
    cap.release()
else:
    st.info("Click 'Start Webcam' to begin face detection.")

st.header("Lie Probability")
st.metric(label="Probability of Lying", value="-- %") 