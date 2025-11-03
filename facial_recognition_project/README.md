# Face Recognition Project in Emerging Technologies in CpE

## **Steve Villa and Lovely Nacional**
## **BSCpE 4A**

## **Submitted to: Prof. Ador Utulo**


A comprehensive face recognition system built with Python, InsightFace, and OpenCV that demonstrates end-to-end facial recognition capabilities with an interactive interface.

##  Features

- **Face Detection & Recognition**: Uses InsightFace for state-of-the-art face detection and recognition
- **Interactive Display**: Real-time recognition with mouse hover functionality
- **Data Augmentation**: Automatic dataset expansion with bounding box preservation
- **Bounding Boxes & Landmarks**: Visualizes face detection with colored landmarks
- **Adjustable Threshold**: Fine-tune recognition sensitivity
- **Resizable Windows**: Flexible interface that adapts to user preferences

##  Project Structure

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

##  Installation & Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Running the program:**
```text
Open the main.ipynb and follow the instructions inside
```


##  Usage

### Interactive Display Controls:
- **Mouse Hover**: See real-time recognition on thumbnails
- **+/- Keys**: Adjust recognition threshold
- **'o' Key**: Find optimal threshold automatically
- **'r' Key**: Generate accuracy report
- **'s' Key**: Save detection examples
- **'f' Key**: Toggle fullscreen
- **'q' Key**: Quit application


##  Performance

The system achieves:
- **90%+ accuracy** with proper training data
- **Real-time recognition** on CPU
- **10 persons** with 5 embeddings each
- **Automatic data augmentation** (10x dataset expansion

## Interface

<img width="1360" height="758" alt="image" src="https://github.com/user-attachments/assets/671d8217-ab75-4bef-b284-c74cf144a1f4" />

