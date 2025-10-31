"""
train_svm.py
Load embeddings, train a linear SVM, save model and label encoder
"""
import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

EMB_FILE = "embeddings/faces_embeddings.npz"
LABELS_FILE = "embeddings/labels.npy"
OUT_MODEL = "models/svm.pkl"
OUT_LABELS = "models/label_encoder.pkl"

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def run():
    if not os.path.exists(EMB_FILE):
        raise FileNotFoundError(f"{EMB_FILE} not found. Run extract_embeddings.py first.")

    ensure_dir("models")

    data = np.load(EMB_FILE)
    X, y = data["X"], data["y"]
    classes = np.load(LABELS_FILE, allow_pickle=True)

    # small dataset? use stratify to keep class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = SVC(kernel="linear", probability=True)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", acc)
    print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=classes))

    with open(OUT_MODEL, "wb") as f:
        pickle.dump(clf, f)
    with open(OUT_LABELS, "wb") as f:
        pickle.dump(classes, f)

    print("Saved SVM model and label encoder to models/")

if __name__ == "__main__":
    run()
