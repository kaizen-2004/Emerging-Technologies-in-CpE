"""
extract_embeddings.py
Loads processed aligned faces (112x112), uses InsightFace FaceAnalysis to extract embeddings,
saves embeddings + label encoder into embeddings/faces_embeddings.npz and embeddings/labels.npy
"""
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

PROC_DIR = "dataset_processed"
OUT_DIR = "embeddings"
EMB_FILE = os.path.join(OUT_DIR, "faces_embeddings.npz")
LABELS_FILE = os.path.join(OUT_DIR, "labels.npy")

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def run():
    print("Loading InsightFace embedding model (GPU)...")
    fa = FaceAnalysis(
    name="buffalo_l",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
    # Name left blank to use default model pack; prepare loads models for embedding too
    fa.prepare(ctx_id=0, det_size=(640, 640))

    ensure_dir(OUT_DIR)

    X, y = [], []
    persons = [p for p in os.listdir(PROC_DIR) if os.path.isdir(os.path.join(PROC_DIR, p))]
    for person in persons:
        person_dir = os.path.join(PROC_DIR, person)
        imgs = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for im in tqdm(imgs, desc=f"Embedding {person}", unit="img"):
            path = os.path.join(person_dir, im)
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # fa.get will detect faces; since images are already aligned, we expect one face
            faces = fa.get(img_rgb)
            if not faces:
                print("No face found during embedding:", path)
                continue

            face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
            emb = face.embedding  # 512-d float array
            if emb is None:
                print("No embedding:", path)
                continue

            # L2-normalize embedding (recommended)
            emb = emb / np.linalg.norm(emb)

            X.append(emb.astype(np.float32))
            y.append(person)

    X = np.asarray(X)
    y = np.asarray(y)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    np.savez(EMB_FILE, X=X, y=y_enc)
    np.save(LABELS_FILE, le.classes_)
    print(f"Saved embeddings -> {EMB_FILE} and labels -> {LABELS_FILE}")
    print("Finished.")

if __name__ == "__main__":
    run()
