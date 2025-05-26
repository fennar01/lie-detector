import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from fer import FER
import random

# Initialize the FER emotion detector
fer_detector = FER(mtcnn=True)

# Demo microexpression detector (randomized for now)
def demo_microexpression(face_img):
    microexp = random.choice(['none', 'brow raise', 'lip twitch', 'eye squint'])
    intensity = np.random.uniform(0, 1)
    return microexp, intensity

# Demo nervousness/behavioral cue analysis (randomized for now)
def demo_nervousness(face_img):
    blink_rate = np.random.uniform(0, 1)
    gaze_aversion = random.choice([True, False])
    return blink_rate, gaze_aversion

# Demo probability scoring system
def compute_lie_probability(emotion_conf, perfusion, microexp_intensity, blink_rate, gaze_aversion):
    prob = 0.2 * (1 - emotion_conf)
    prob += 0.2 * (abs(perfusion - 90) / 90)
    prob += 0.2 * microexp_intensity
    prob += 0.2 * blink_rate
    prob += 0.2 * (1 if gaze_aversion else 0)
    return min(max(prob, 0), 1)

st.set_page_config(page_title="Lie Detector", layout="centered")
st.title("Lie Detector - Real-Time Facial Video Analysis")

st.write("""
This app analyzes facial video data and displays a live probability that the subject is lying.

**Current Features:**
- Real-time facial detection and tracking
- Facial expression sentiment analysis (real model)
- Blood perfusion analysis (demo)
- Microexpression detection (demo)
- Nervousness/behavioral cue analysis (demo)
- Live probability scoring & alert system (demo)

---
""")

st.header("Live Video Input: All Features Demo")

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

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
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                    face_img = image[max(0, y1):min(ih, y2), max(0, x1):min(iw, x2)]
                    label = ""
                    if face_img.size > 0:
                        # Real sentiment analysis using FER
                        try:
                            face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                            emotions = fer_detector.detect_emotions(face_bgr)
                            if emotions and len(emotions) > 0:
                                top_emotion, emotion_conf = max(emotions[0]["emotions"].items(), key=lambda x: x[1])
                                label += f"{top_emotion} ({emotion_conf*100:.0f}%) "
                            else:
                                top_emotion, emotion_conf = 'neutral', 0.5
                                label += f"neutral (50%) "
                        except Exception as e:
                            top_emotion, emotion_conf = 'neutral', 0.5
                            label += f"neutral (50%) "
                        # Blood perfusion analysis (mean hue value)
                        face_hsv = cv2.cvtColor(face_img, cv2.COLOR_RGB2HSV)
                        mean_hue = np.mean(face_hsv[:,:,0])
                        label += f"Perfusion: {mean_hue:.1f} "
                        # Microexpression detection (demo)
                        microexp, microexp_intensity = demo_microexpression(face_img)
                        label += f"Microexp: {microexp} ({microexp_intensity:.2f}) "
                        # Nervousness/behavioral cue analysis (demo)
                        blink_rate, gaze_aversion = demo_nervousness(face_img)
                        label += f"Blink: {blink_rate:.2f} GazeAv: {gaze_aversion} "
                        # Probability scoring
                        prob = compute_lie_probability(emotion_conf, mean_hue, microexp_intensity, blink_rate, gaze_aversion)
                        lie_prob = max(lie_prob, prob)
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
            if not st.session_state.get('run', True):
                break
    cap.release()
else:
    st.info("Click 'Start Webcam' to begin full demo.")
    prob_placeholder.metric(label="Probability of Lying", value="-- %")
    alert_placeholder.empty() 