# Face Recognition Project

A comprehensive face recognition system built with Python, InsightFace, and OpenCV that demonstrates end-to-end facial recognition capabilities with an interactive interface.

## 🚀 Features

- **Face Detection & Recognition**: Uses InsightFace for state-of-the-art face detection and recognition
- **Interactive Display**: Real-time recognition with mouse hover functionality
- **Data Augmentation**: Automatic dataset expansion with bounding box preservation
- **Bounding Boxes & Landmarks**: Visualizes face detection with colored landmarks
- **Adjustable Threshold**: Fine-tune recognition sensitivity
- **Resizable Windows**: Flexible interface that adapts to user preferences

## 📁 Project Structure

```
facial_recognition_project/
├── dataset_raw/                          # Original images (organized by person)
├── dataset_processed/
│   ├── original_annotated/              # Standardized images for annotation
│   ├── augmented/                       # Automatically generated variations
│   └── annotations.json                 # Combined annotations
├── annotations/                         # LabelImg XML files
├── models/                              # Trained InsightFace model
└── detection_examples/                  # Sample detection outputs
```

## 🛠️ Installation & Setup

1. **Install dependencies:**
```bash
pip install opencv-python insightface albumentations numpy scikit-learn pillow
```

2. **Prepare dataset:**
   - Organize images in `dataset_raw/` with subfolders for each person
   - Run standardization and augmentation pipeline
   - Annotate faces using LabelImg

3. **Train the model:**
```bash
python train_model.py
```

4. **Launch interactive display:**
```bash
python interactive.py
```

## 🎮 Usage

### Interactive Display Controls:
- **Mouse Hover**: See real-time recognition on thumbnails
- **+/- Keys**: Adjust recognition threshold
- **'o' Key**: Find optimal threshold automatically
- **'r' Key**: Generate accuracy report
- **'s' Key**: Save detection examples
- **'f' Key**: Toggle fullscreen
- **'q' Key**: Quit application


## 📊 Performance

The system achieves:
- **90%+ accuracy** with proper training data
- **Real-time recognition** on CPU
- **10+ persons** with 30+ embeddings each
- **Automatic data augmentation** (10x dataset expansion)
