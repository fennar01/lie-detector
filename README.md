# Lie Detector

A real-time lie detector that analyzes facial video data to estimate the probability that a subject is lying. The system leverages advanced techniques including facial recognition, sentiment analysis, blood perfusion analysis, nervousness detection, and microexpression analysis.

## Features
- **Live Video Analysis**: Processes facial video input in real time.
- **Real-Time Facial Detection and Tracking**: Detects and tracks faces in the webcam feed using Mediapipe.
- **Sentiment Analysis (In Progress)**: Analyzes emotional state from facial expressions using deep learning models and, optionally, from speech using NLP models.
- **Blood Perfusion Analysis (In Progress)**: Estimates blood flow changes in the face (e.g., blushing, pallor) using color analysis and subtle skin tone variations.
- **Microexpression Detection (In Progress)**: Detects rapid, involuntary facial movements that may indicate concealed emotions using deep learning and facial landmark tracking.
- **Nervousness & Behavioral Cue Analysis (In Progress)**: Detects subtle nervous tics, fidgeting, gaze aversion, and other behavioral cues that may indicate stress or deception.
- **Lie Probability Score**: Continuously updates the probability that the subject is lying. *(Coming soon)*
- **User Alerts**: Notifies the user when the probability of lying exceeds a threshold. *(Coming soon)*

## Sentiment Analysis (In Progress)
The app analyzes the subject's emotional state in real time using facial expression recognition. This leverages pre-trained deep learning models (e.g., from the `transformers` library or custom CNNs) to classify emotions such as happiness, sadness, anger, surprise, and fear. Optionally, audio sentiment analysis may be added in the future.

## Blood Perfusion Analysis (In Progress)
The app estimates blood perfusion (blood flow) in the subject's face by analyzing subtle changes in skin color and tone over time. This can be achieved using color space transformations (e.g., RGB to HSV), region-of-interest tracking, and signal processing to detect physiological changes such as blushing or pallor, which may correlate with stress or deception.

## Microexpression Detection (In Progress)
The app detects microexpressions—brief, involuntary facial movements that can reveal concealed emotions. This uses facial landmark tracking (e.g., with Mediapipe or dlib) and deep learning models trained to recognize rapid changes in facial muscle movement. Microexpression detection is a key component for advanced lie detection and emotional analysis.

## Nervousness & Behavioral Cue Analysis (In Progress)
The next feature will analyze nervousness and behavioral cues such as gaze aversion, fidgeting, blinking rate, and other subtle movements. This will use facial landmark tracking, pose estimation, and temporal analysis to identify patterns associated with stress or deception.

## Tech Stack
- Python (OpenCV, dlib, scikit-learn, deep learning frameworks)
- Web frontend (Streamlit or Flask for demo UI)
- Optional: TensorFlow/PyTorch for microexpression models

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/fennar01/lie-detector.git
   cd lie-detector
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```

## Usage
- Launch the app and provide access to your webcam.
- The system will display a live video feed with real-time face detection, sentiment analysis, blood perfusion analysis, and (soon) microexpression and behavioral cue analysis.
- Additional features (lie probability, alerts, etc.) will be added soon.

## Contributing
Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Roadmap
- [x] Project scaffolding and documentation
- [x] Real-time facial detection and tracking
- [~] Sentiment analysis integration (in progress)
- [~] Blood perfusion analysis module (in progress)
- [~] Microexpression detection (in progress)
- [~] Nervousness and behavioral cue analysis (in progress)
- [ ] Live probability scoring and alert system
- [ ] Web-based user interface
- [ ] Model training and dataset curation
- [ ] Extensive testing and validation
- [ ] Deployment and cloud support

## Disclaimer
This project is for research and educational purposes only. Lie detection is a complex and nuanced field; results should not be considered definitive or used for critical decisions. 