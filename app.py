import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from fer import FER
import random
from collections import deque

# Initialize the FER emotion detector
fer_detector = FER(mtcnn=True)

# Initialize Mediapipe FaceMesh for landmarks
mp_face_mesh = mp.solutions.face_mesh

# Eye aspect ratio for blink detection
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_idx):
    # Compute EAR for one eye
    p = np.array([landmarks[i] for i in eye_idx])
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    return (A + B) / (2.0 * C)

# Microexpression: track landmark movement over a short window
def microexpression_intensity(landmark_history):
    if len(landmark_history) < 2:
        return 0.0
    diffs = [np.linalg.norm(np.array(landmark_history[i]) - np.array(landmark_history[i-1])) for i in range(1, len(landmark_history))]
    return float(np.mean(diffs))

# Gaze aversion: compare nose and eye positions
def gaze_aversion(landmarks):
    # Simple: if nose tip x is far from midpoint of eyes x
    left_eye = np.mean([landmarks[i] for i in LEFT_EYE_IDX], axis=0)
    right_eye = np.mean([landmarks[i] for i in RIGHT_EYE_IDX], axis=0)
    eyes_mid_x = (left_eye[0] + right_eye[0]) / 2
    nose_tip_x = landmarks[1][0]  # landmark 1 is nose tip
    return abs(nose_tip_x - eyes_mid_x) > 0.04  # threshold is empirical

# Probability scoring system
def compute_lie_probability(emotion_conf, perfusion, microexp_intensity, blink_rate, gaze_aversion):
    prob = 0.2 * (1 - emotion_conf)
    prob += 0.2 * (abs(perfusion - 90) / 90)
    prob += 0.2 * min(microexp_intensity * 10, 1)  # scale for demo
    prob += 0.2 * min(blink_rate / 0.3, 1)  # scale for demo
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
- Microexpression detection (landmark-based)
- Nervousness/behavioral cue analysis (blink/gaze)
- Live probability scoring & alert system

---
""")

st.header("Live Video Input: All Features Demo")

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])
alert_placeholder = st.empty()
prob_placeholder = st.empty()

# Buffers for blink and microexpression analysis
landmark_histories = {}
blink_queues = {}
BLINK_WINDOW = 30  # frames
MICROEXP_WINDOW = 10  # frames

if run:
    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection, \
         mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        frame_count = 0
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to access webcam.")
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = face_detection.process(image)
            mesh_results = face_mesh.process(image)
            image.flags.writeable = True
            annotated_image = image.copy()
            lie_prob = 0
            alert = None
            ih, iw, _ = image.shape
            if results.detections and mesh_results.multi_face_landmarks:
                for detection, face_landmarks in zip(results.detections, mesh_results.multi_face_landmarks):
                    bboxC = detection.location_data.relative_bounding_box
                    x1 = int(bboxC.xmin * iw)
                    y1 = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)
                    x2, y2 = x1 + w, y1 + h
                    face_img = image[max(0, y1):min(ih, y2), max(0, x1):min(iw, x2)]
                    label = ""
                    # Get landmarks as np array (normalized)
                    landmarks = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])
                    # Track landmark history for microexpression
                    face_id = 0  # single face for now
                    if face_id not in landmark_histories:
                        landmark_histories[face_id] = deque(maxlen=MICROEXP_WINDOW)
                        blink_queues[face_id] = deque(maxlen=BLINK_WINDOW)
                    landmark_histories[face_id].append(landmarks)
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
                    # Microexpression detection (landmark movement)
                    microexp_intensity = microexpression_intensity([h[1][1] for h in enumerate(landmark_histories[face_id])])
                    label += f"Microexp: {microexp_intensity:.2f} "
                    # Blink detection (eye aspect ratio)
                    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_IDX)
                    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX)
                    ear = (left_ear + right_ear) / 2
                    blink = ear < 0.21  # threshold for blink
                    blink_queues[face_id].append(blink)
                    blink_rate = np.mean(blink_queues[face_id])
                    label += f"Blink: {blink_rate:.2f} "
                    # Gaze aversion
                    gaze_away = gaze_aversion(landmarks)
                    label += f"GazeAv: {gaze_away} "
                    # Probability scoring
                    prob = compute_lie_probability(emotion_conf, mean_hue, microexp_intensity, blink_rate, gaze_away)
                    lie_prob = max(lie_prob, prob)
                    # Draw bounding box and label
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(annotated_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    mp_drawing.draw_detection(annotated_image, detection)
                    # Draw face mesh
                    for lm in landmarks:
                        cx, cy = int(lm[0] * iw), int(lm[1] * ih)
                        cv2.circle(annotated_image, (cx, cy), 1, (255,0,0), -1)
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