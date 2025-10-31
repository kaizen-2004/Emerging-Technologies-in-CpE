"""
preprocess.py
Detects faces using InsightFace FaceAnalysis, aligns & crops them,
saves 112x112 uint8 images to dataset_processed/<person>/
"""

import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm

RAW_DIR = "dataset_raw"
OUT_DIR = "dataset_processed"
IMG_SIZE = (112, 112)

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def align_crop(face_kps, img, output_size=IMG_SIZE):
    """
    Align & crop using 5 facial landmarks (left_eye, right_eye, nose, left_mouth, right_mouth).
    We'll perform similarity transform to canonical landmarks for 112x112.
    """
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]
    ], dtype=np.float32)

    if output_size[1] == 112:
        src[:, 0] = src[:, 0] + 8.0  # as in common ArcFace preprocessing

    dst = np.array(face_kps, dtype=np.float32)  # 5x2

    # compute similarity transform
    tform = cv2.estimateAffinePartial2D(dst, src, method=cv2.LMEDS)[0]
    if tform is None:
        return None
    warped = cv2.warpAffine(img, tform, output_size, borderValue=0.0)
    return warped

def run():
    print("Loading InsightFace detector (GPU)...")
    fa = FaceAnalysis(providers=['CPUExecutionProvider'])  # GPU on Colab
    fa.prepare(ctx_id=0, det_size=(640, 640))

    ensure_dir(OUT_DIR)

    persons = [p for p in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, p))]
    for person in persons:
        in_person = os.path.join(RAW_DIR, person)
        out_person = os.path.join(OUT_DIR, person)
        ensure_dir(out_person)

        images = [f for f in os.listdir(in_person) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for img_name in tqdm(images, desc=f"Processing {person}", unit="img"):
            path = os.path.join(in_person, img_name)
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                print("Cannot read:", path)
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            faces = fa.get(img_rgb)
            if not faces:
                print("No face detected:", path)
                continue

            # choose largest face (if multiple)
            face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])

            # get 5 landmarks as list of tuples [(x,y), ...]
            kps = face.kps  # shape (5,2)
            aligned = align_crop(kps, img_rgb)
            if aligned is None:
                print("Alignment failed:", path)
                continue

            # save as uint8 BGR (OpenCV)
            aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
            out_path = os.path.join(out_person, img_name)
            cv2.imwrite(out_path, aligned_bgr)
    print("Preprocessing complete.")

if __name__ == "__main__":
    run()
