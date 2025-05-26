# Lie Detector

A real-time lie detector that analyzes facial video data to estimate the probability that a subject is lying. The system leverages advanced techniques including facial recognition, sentiment analysis, blood perfusion analysis, nervousness detection, and microexpression analysis.

## Features
- **Live Video Analysis**: Processes facial video input in real time.
- **Lie Probability Score**: Continuously updates the probability that the subject is lying.
- **Sentiment Analysis**: Analyzes emotional state from facial expressions and speech (if available).
- **Facial Recognition**: Identifies and tracks the subject's face.
- **Blood Perfusion Detection**: Estimates blood flow changes in the face (e.g., blushing, pallor) using color analysis.
- **Nervousness & Microexpressions**: Detects subtle facial movements and nervous tics.
- **User Alerts**: Notifies the user when the probability of lying exceeds a threshold.

## Tech Stack
- Python (OpenCV, dlib, scikit-learn, deep learning frameworks)
- Web frontend (Streamlit or Flask for demo UI)
- Optional: TensorFlow/PyTorch for microexpression models

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/lie-detector.git
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
- The system will display a live video feed with a real-time probability score.
- Alerts will be shown if the subject is likely lying.

## Contributing
Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Roadmap
- [x] Project scaffolding and documentation
- [ ] Real-time facial detection and tracking
- [ ] Sentiment analysis integration
- [ ] Blood perfusion analysis module
- [ ] Microexpression detection (deep learning)
- [ ] Nervousness and behavioral cue analysis
- [ ] Live probability scoring and alert system
- [ ] Web-based user interface
- [ ] Model training and dataset curation
- [ ] Extensive testing and validation
- [ ] Deployment and cloud support

## Disclaimer
This project is for research and educational purposes only. Lie detection is a complex and nuanced field; results should not be considered definitive or used for critical decisions. 