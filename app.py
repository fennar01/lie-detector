import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from collections import namedtuple

# Placeholder for a simple emotion classifier (to be replaced with a real model)
class DummyEmotionModel:
    def __init__(self):
        self.labels = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted']
    def predict(self, face_img):
        # Randomly pick an emotion for demonstration
        idx = np.random.randint(0, len(self.labels))
        return self.labels[idx], np.random.uniform(0.5, 1.0)

# Initialize the dummy model (replace with a real model for production)
emotion_model = DummyEmotionModel()

st.set_page_config(page_title="Lie Detector", layout="centered")
st.title("Lie Detector - Real-Time Facial Video Analysis")

st.write("""
This app analyzes facial video data and displays a live probability that the subject is lying.

**Current Features:**
- Real-time facial detection and tracking
- Facial expression sentiment analysis (demo)
- Blood perfusion analysis (demo)

**Features coming soon:**
- Microexpression detection
- Nervousness analysis

---
""")

st.header("Live Video Input: Face Detection, Sentiment & Blood Perfusion Analysis Demo")

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
            annotated_image = image.copy()
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    x1 = int(bboxC.xmin * iw)
                    y1 = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)
                    x2, y2 = x1 + w, y1 + h
                    # Crop face for emotion and perfusion analysis
                    face_img = image[max(0, y1):min(ih, y2), max(0, x1):min(iw, x2)]
                    label = ""
                    if face_img.size > 0:
                        # Sentiment analysis
                        face_pil = Image.fromarray(face_img).resize((48, 48)).convert('L')
                        face_np = np.array(face_pil)
                        emotion, confidence = emotion_model.predict(face_np)
                        label += f"{emotion} ({confidence*100:.0f}%) "
                        # Blood perfusion analysis (mean hue value)
                        face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                        face_hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
                        mean_hue = np.mean(face_hsv[:,:,0])
                        label += f"Perfusion: {mean_hue:.1f}"
                        # Draw bounding box and label
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(annotated_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    mp_drawing.draw_detection(annotated_image, detection)
            FRAME_WINDOW.image(annotated_image)
            # Streamlit workaround for breaking loop
            if not st.session_state.get('run', True):
                break
    cap.release()
else:
    st.info("Click 'Start Webcam' to begin face detection, sentiment, and blood perfusion analysis.")

st.header("Lie Probability")
st.metric(label="Probability of Lying", value="-- %") 