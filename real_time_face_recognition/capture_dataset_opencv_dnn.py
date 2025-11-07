import cv2
import os
import numpy as np

# ================= CONFIGURATION =================
person_name = input("Enter person name: ")
save_dir = os.path.join("dataset-raw", person_name)
os.makedirs(save_dir, exist_ok=True)

max_images = 50       # Number of images to capture
frame_interval = 10    # Capture every N frames
conf_threshold = 0.5  # Minimum confidence for face detection
# ================================================

# Load OpenCV DNN face detector
model_file = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
config_file = "deploy.prototxt"
if not os.path.exists(model_file) or not os.path.exists(config_file):
    raise FileNotFoundError("Make sure deploy.prototxt and caffemodel are in the working directory.")

net = cv2.dnn.readNetFromCaffe(config_file, model_file)

def detect_faces(frame, conf_threshold=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            faces.append((x1, y1, x2-x1, y2-y1))
    return faces

# Open webcam
cap = cv2.VideoCapture(0)
count = 0
frame_count = 0

print("Press 'q' to quit anytime.")

while count < max_images:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    faces = detect_faces(frame, conf_threshold)
    for (x, y, w, h) in faces:
        if frame_count % frame_interval == 0:
            face_crop = frame[y:y+h, x:x+w]
            filename = os.path.join(save_dir, f"{count+1}.jpg")
            cv2.imwrite(filename, face_crop)
            count += 1
            print(f"Saved: {filename}")
        # Draw rectangle for visualization
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Capture Dataset", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Capture complete! {count} images saved for {person_name}.")
