import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

# -----------------------------
# Configuration
# -----------------------------
FACE_DATABASE_FILE = "face_database.pkl"
DET_SIZE = (320, 320)
THRESHOLD = 0.4  # Cosine distance threshold

# -----------------------------
# Load face database
# -----------------------------
with open(FACE_DATABASE_FILE, "rb") as f:
    face_database = pickle.load(f)

print(f"✅ Loaded database: {len(face_database)} persons")

# -----------------------------
# Initialize InsightFace
# -----------------------------
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=DET_SIZE)

# -----------------------------
# Real-time recognition
# -----------------------------
cap = cv2.VideoCapture(0)
print("🎥 Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    
    for face in faces:
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        embedding = face.embedding

        # Match against database
        name = "Unknown"
        min_dist = THRESHOLD
        for person, embeddings in face_database.items():
            for db_emb in embeddings:
                dist = cosine(embedding, db_emb)
                if dist < min_dist:
                    min_dist = dist
                    name = person

        # Draw bounding box and label
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Draw facial landmarks (2D)

        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            for (lx, ly) in face.landmark_2d_106:
                # Larger circle, thicker for visibility
                cv2.circle(frame, (int(lx), int(ly)), 1, (0, 255, 0), -1)


    cv2.imshow("Face Recognition with Landmarks", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
