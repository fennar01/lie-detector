import streamlit as st

st.set_page_config(page_title="Lie Detector", layout="centered")
st.title("Lie Detector - Real-Time Facial Video Analysis")

st.write("""
This app will analyze facial video data and display a live probability that the subject is lying.

**Features coming soon:**
- Real-time facial detection
- Sentiment analysis
- Blood perfusion analysis
- Microexpression detection
- Nervousness analysis

---
""")

st.header("Live Video Input (Coming Soon)")
st.info("Video input and analysis will be available in future versions.")

st.header("Lie Probability")
st.metric(label="Probability of Lying", value="-- %") 