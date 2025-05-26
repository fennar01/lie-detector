import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from collections import namedtuple
import random

# Placeholder for a simple emotion classifier (to be replaced with a real model)
class DummyEmotionModel:
    def __init__(self):
        self.labels = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted']
    def predict(self, face_img):
        # Randomly pick an emotion for demonstration
        idx = np.random.randint(0, len(self.labels))
        return self.labels[idx], np.random.uniform(0.5, 1.0)

# Demo microexpression detector (randomized for now)
def demo_microexpression(face_img):
    # In a real implementation, use facial landmarks and temporal analysis
    microexp = random.choice(['none', 'brow raise', 'lip twitch', 'eye squint'])
    intensity = np.random.uniform(0, 1)
    return microexp, intensity

# Demo nervousness/behavioral cue analysis (randomized for now)
def demo_nervousness(face_img):
    # In a real implementation, use blink rate, gaze aversion, fidgeting, etc.
    blink_rate = np.random.uniform(0, 1)  # 0=low, 1=high
    gaze_aversion = random.choice([True, False])
    return blink_rate, gaze_aversion

# Demo probability scoring system
def compute_lie_probability(emotion_conf, perfusion, microexp_intensity, blink_rate, gaze_aversion):
    # Weighted sum (demo logic)
    prob = 0.2 * (1 - emotion_conf)  # less confidence = more likely lying
    prob += 0.2 * (abs(perfusion - 90) / 90)  # deviation from mean hue
    prob += 0.2 * microexp_intensity
    prob += 0.2 * blink_rate
    prob += 0.2 * (1 if gaze_aversion else 0)
    return min(max(prob, 0), 1)

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
- Microexpression detection (demo)
- Nervousness/behavioral cue analysis (demo)
- Live probability scoring & alert system (demo)

---
""")

st.header("Live Video Input: All Features Demo")

# Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Streamlit video capture
run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])
alert_placeholder = st.empty()
prob_placeholder = st.empty()

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
            lie_prob = 0
            alert = None
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
                        label += f"Perfusion: {mean_hue:.1f} "
                        # Microexpression detection (demo)
                        microexp, microexp_intensity = demo_microexpression(face_img)
                        label += f"Microexp: {microexp} ({microexp_intensity:.2f}) "
                        # Nervousness/behavioral cue analysis (demo)
                        blink_rate, gaze_aversion = demo_nervousness(face_img)
                        label += f"Blink: {blink_rate:.2f} GazeAv: {gaze_aversion} "
                        # Probability scoring
                        prob = compute_lie_probability(confidence, mean_hue, microexp_intensity, blink_rate, gaze_aversion)
                        lie_prob = max(lie_prob, prob)  # show max if multiple faces
                        # Draw bounding box and label
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(annotated_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    mp_drawing.draw_detection(annotated_image, detection)
            FRAME_WINDOW.image(annotated_image)
            prob_placeholder.metric(label="Probability of Lying", value=f"{lie_prob*100:.1f} %")
            if lie_prob > 0.7:
                alert_placeholder.error(f"ALERT: High probability of lying detected! ({lie_prob*100:.1f}%)")
            else:
                alert_placeholder.empty()
            # Streamlit workaround for breaking loop
            if not st.session_state.get('run', True):
                break
    cap.release()
else:
    st.info("Click 'Start Webcam' to begin full demo.")
    prob_placeholder.metric(label="Probability of Lying", value="-- %")
    alert_placeholder.empty() 