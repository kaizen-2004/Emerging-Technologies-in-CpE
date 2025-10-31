import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import subprocess
import pickle
from insightface.app import FaceAnalysis

# -----------------------------
# Configuration
# -----------------------------
MODEL_SVM = "models/svm_model.pkl"
LABEL_ENCODER = "models/label_encoder.pkl"

# Initialize session state
for key, default in {
    'detection_running': False,
    'log_messages': [],
    'camera_available': False,
    'detection_mode': "image_upload"
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -----------------------------
# Helper functions
# -----------------------------
def load_face_model():
    """Load SVM + LabelEncoder + InsightFace"""
    st.session_state.clf = pickle.load(open(MODEL_SVM, "rb"))
    st.session_state.le = pickle.load(open(LABEL_ENCODER, "rb"))
    st.session_state.fa = FaceAnalysis(providers=['CPUExecutionProvider'])
    st.session_state.fa.prepare(ctx_id=0, det_size=(640,640))

def log_status(status):
    timestamp = time.strftime('%H:%M:%S')
    st.session_state.log_messages.append(f"[{timestamp}] {status}")
    if len(st.session_state.log_messages) > 50:
        st.session_state.log_messages.pop(0)

def process_frame(frame):
    """Run facial recognition on a frame"""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = st.session_state.fa.get(img_rgb)
    names = []

    if not faces:
        return frame, names

    for face in faces:
        emb = face.embedding / np.linalg.norm(face.embedding)
        pred_idx = st.session_state.clf.predict([emb])[0]
        probs = st.session_state.clf.predict_proba([emb])[0]
        name = st.session_state.le.inverse_transform([pred_idx])[0]
        names.append(name)
        prob = probs[pred_idx]

        # Draw bbox
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{name} ({prob:.2f})", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Draw landmarks
        if face.landmark is not None:
            for (x, y) in face.landmark.astype(int):
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    return frame, names

def capture_screen():
    """Wayland-compatible screen capture using grim"""
    proc = subprocess.run(["grim", "-t", "png", "-"], capture_output=True)
    img_arr = np.frombuffer(proc.stdout, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return img

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Facial Recognition App", layout="wide")
st.title("📷 Facial Recognition App")
st.markdown("---")

# Detection mode selection
mode = st.radio("Detection Mode:", ["Image Upload", "Camera", "Screen Capture"])
st.session_state.detection_mode = mode.lower().replace(" ", "_")

# Load models once
if 'fa' not in st.session_state:
    with st.spinner("Loading models..."):
        load_face_model()

# Layout: main content + log
col1, col2 = st.columns([2,1])

with col1:
    # -----------------------------
    # Image Upload Mode
    # -----------------------------
    if st.session_state.detection_mode == "image_upload":
        uploaded_file = st.file_uploader("Upload Image", type=['png','jpg','jpeg'])
        if uploaded_file:
            image = np.array(Image.open(uploaded_file))
            frame, names = process_frame(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            if names:
                log_status(f"Detected: {', '.join(names)}")
            else:
                log_status("No faces detected")

    # -----------------------------
    # Camera Mode
    # -----------------------------
    elif st.session_state.detection_mode == "camera":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open camera")
        else:
            video_placeholder = st.empty()
            start = st.button("Start Detection")
            stop = st.button("Stop Detection")

            if start:
                st.session_state.detection_running = True
            if stop:
                st.session_state.detection_running = False

            while st.session_state.detection_running:
                ret, frame = cap.read()
                if not ret:
                    break
                frame, names = process_frame(frame)
                video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                if names:
                    log_status(f"Detected: {', '.join(names)}")
                time.sleep(0.05)
            cap.release()

    # -----------------------------
    # Screen Capture Mode
    # -----------------------------
    elif st.session_state.detection_mode == "screen_capture":
        video_placeholder = st.empty()
        start = st.button("Start Screen Capture")
        stop = st.button("Stop Screen Capture")
        if start:
            st.session_state.detection_running = True
        if stop:
            st.session_state.detection_running = False

        while st.session_state.detection_running:
            frame = capture_screen()
            frame, names = process_frame(frame)
            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            if names:
                log_status(f"Detected: {', '.join(names)}")
            time.sleep(0.05)

with col2:
    st.subheader("📋 Detection Log")
    log_container = st.empty()
    with log_container:
        for msg in reversed(st.session_state.log_messages[-20:]):
            st.text(msg)
