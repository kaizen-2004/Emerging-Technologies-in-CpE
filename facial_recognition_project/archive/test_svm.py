#!/usr/bin/env python3
"""
test_svm.py
Run inference on a single image with visualization.
Usage:
    python test_svm.py --image path/to/image.jpg
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import argparse
import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis

MODEL_SVM = "models/svm_model.pkl"
LABEL_ENCODER = "models/label_encoder.pkl"

def load_model_and_faceanalyzer():
    print("Loading SVM + label encoder...")
    clf = pickle.load(open(MODEL_SVM, "rb"))
    le = pickle.load(open(LABEL_ENCODER, "rb"))

    print("Loading InsightFace (CPU/GPU)...")
    fa = FaceAnalysis(providers=['CPUExecutionProvider'])
    fa.prepare(ctx_id=0, det_size=(640,640))
    return clf, le, fa

def run_inference(img_path, clf, le, fa):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("Cannot read:", img_path)
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = fa.get(img_rgb)

    if not faces:
        print("No face detected.")
        return

    face = max(faces, key=lambda x: x.bbox[2]*x.bbox[3])
    emb = face.embedding
    emb = emb / np.linalg.norm(emb)

    pred_idx = clf.predict([emb])[0]
    probs = clf.predict_proba([emb])[0]
    pred_name = le.inverse_transform([pred_idx])[0]
    pred_prob = probs[pred_idx]

    # Draw bounding box
    x1, y1, x2, y2 = map(int, face.bbox)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw label
    label = f"{pred_name}: {pred_prob:.2f}"
    cv2.putText(img_bgr, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Draw facial landmarks
    if hasattr(face, "landmark") and face.landmark is not None:
        for (x, y) in face.landmark.astype(int):
            cv2.circle(img_bgr, (x, y), 2, (0, 0, 255), -1)

    # Show image in resizable window
    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Recognition", 800, 600)
    cv2.imshow("Face Recognition", img_bgr)
    print(f"Predicted: {pred_name} (confidence {pred_prob:.3f})")

    # Top-3 predictions
    topk = np.argsort(probs)[::-1][:3]
    print("Top-3 predictions:")
    for i in topk:
        print(f"  {le.inverse_transform([i])[0]}: {probs[i]:.3f}")

    # Use this to allow normal window close instead of waiting for key
    while cv2.getWindowProperty("Face Recognition", cv2.WND_PROP_VISIBLE) >= 1:
        if cv2.waitKey(100) & 0xFF == 27:  # optional escape key
            break

    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    args = parser.parse_args()

    clf, le, fa = load_model_and_faceanalyzer()
    run_inference(args.image, clf, le, fa)

if __name__ == "__main__":
    main()
