import cv2
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

class EnhancedInteractiveFaceDisplay:
    def __init__(self, face_recognition_model):
        self.face_model = face_recognition_model
        self.collage = None
        self.image_positions = {}
        
    def create_collage(self, images_per_row=5):
        """Create collage from ORIGINAL_ANNOTATED images (not augmented)"""
        print("=== Creating Enhanced Interactive Collage from ORIGINAL Images ===")
        
        base_dir = Path("/home/steve/Python/Emerging-Technologies-in-CpE/facial_recognition_project")
        original_path = base_dir / "dataset_processed" / "augmented"
        
        all_images = []
        image_info = []
        
        for person in sorted(original_path.iterdir()):
            if person.is_dir():
                image_files = list(person.glob('*.jpg'))
                
                for image_file in image_files:
                    image = cv2.imread(str(image_file))
                    if image is not None:
                        image = cv2.resize(image, (180, 180))
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        all_images.append(image_rgb)
                        image_info.append({
                            'image': image,
                            'person': person.name,
                            'path': image_file
                        })
        
        total_images = len(all_images)
        rows = (total_images + images_per_row - 1) // images_per_row
        collage_height = rows * 180
        collage_width = images_per_row * 180
        
        self.collage = np.zeros((collage_height, collage_width, 3), dtype=np.uint8)
        self.image_positions = {}
        
        for idx, (image, info) in enumerate(zip(all_images, image_info)):
            row = idx // images_per_row
            col = idx % images_per_row
            
            y_start = row * 180
            y_end = y_start + 180
            x_start = col * 180
            x_end = x_start + 180
            
            self.collage[y_start:y_end, x_start:x_end] = image
            self.image_positions[(x_start, y_start, x_end, y_end)] = info
        
        print(f"Created collage with {total_images} ORIGINAL images ({rows} rows)")
        return self.collage

    def recognize_face_with_details(self, image_path):
        """Recognize faces with bounding boxes and landmarks"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return []
            
            faces = self.face_model.model.get(image)
            results = []
            
            for face in faces:
                if hasattr(face, 'embedding') and hasattr(face, 'bbox'):
                    query_embedding = face.embedding.reshape(1, -1)
                    best_match = "Unknown"
                    best_similarity = 0
                    
                    for person_id, embeddings in self.face_model.face_database.items():
                        for db_embedding in embeddings:
                            db_embedding = db_embedding.reshape(1, -1)
                            similarity = cosine_similarity(query_embedding, db_embedding)[0][0]
                            
                            if similarity > best_similarity and similarity > self.face_model.threshold:
                                best_similarity = similarity
                                best_match = person_id
                    
                    landmarks = face.kps if hasattr(face, 'kps') else None
                    
                    results.append({
                        'identity': best_match,
                        'confidence': best_similarity,
                        'bbox': face.bbox.astype(int),
                        'landmarks': landmarks.astype(int) if landmarks is not None else None
                    })
            
            return results
        except Exception as e:
            print(f"Recognition error for {image_path}: {e}")
            return []

    def draw_face_details(self, image, results):
        """Draw bounding boxes and facial landmarks on image - NO FIXED SIZE"""
        display_image = image.copy()
        
        for result in results:
            bbox = result['bbox']
            landmarks = result['landmarks']
            
            # Draw bounding box
            cv2.rectangle(display_image, 
                         (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                         (0, 255, 0), 2)
            
            # Draw facial landmarks if available
            if landmarks is not None:
                for i, landmark in enumerate(landmarks):
                    if i == 0:  # Right eye
                        color = (255, 0, 0)
                    elif i == 1:  # Left eye
                        color = (255, 255, 0)
                    elif i == 2:  # Nose
                        color = (0, 255, 255)
                    else:  # Mouth corners
                        color = (0, 0, 255)
                    
                    cv2.circle(display_image, tuple(landmark), 3, color, -1)
            
            # Draw identity and confidence
            identity = result['identity']
            confidence = result['confidence']
            
            if confidence > 0.8:
                text_color = (0, 255, 0)
            elif confidence > 0.6:
                text_color = (0, 255, 255)
            else:
                text_color = (0, 0, 255)
            
            text = f"{identity} ({confidence:.2f})"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_image,
                         (bbox[0], bbox[1] - text_size[1] - 10),
                         (bbox[0] + text_size[0], bbox[1]),
                         (0, 0, 0), -1)
            
            cv2.putText(display_image, text,
                       (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        return display_image  # Return original size, let window handle resizing

    def mouse_callback(self, event, x, y, flags, param):
        """Enhanced mouse callback with detailed face visualization"""
        if event == cv2.EVENT_MOUSEMOVE:
            display_image = self.collage.copy()
            detailed_view = None
            
            for bbox, info in self.image_positions.items():
                x1, y1, x2, y2 = bbox
                if x1 <= x <= x2 and y1 <= y <= y2:
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    recognition_results = self.recognize_face_with_details(info['path'])
                    
                    if recognition_results:
                        original_image = cv2.imread(str(info['path']))
                        detailed_view = self.draw_face_details(original_image, recognition_results)
                        
                        result = recognition_results[0]
                        hover_info = f"{result['identity']} ({result['confidence']:.2f})"
                        
                        if result['confidence'] > 0.8:
                            color = (0, 255, 0)
                        elif result['confidence'] > 0.6:
                            color = (0, 255, 255)
                        else:
                            color = (0, 0, 255)
                            
                        cv2.putText(display_image, hover_info, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Update displays
                    self.update_resizable_display(display_image, detailed_view, info)
                    return
            
            # Reset display
            cv2.putText(display_image, "Hover over ORIGINAL images to see face recognition", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            self.update_resizable_display(display_image, None, None)

    def update_resizable_display(self, collage_image, detailed_image, image_info):
        """Update both windows with resizable capability"""
        # Update collage window (left)
        cv2.imshow('Face Recognition Collage - ORIGINAL Images', collage_image)
        
        # Update detailed view window (right)
        if detailed_image is not None and image_info is not None:
            # Get current window size to maintain user's preferred size
            try:
                # Try to get current window size, fallback to reasonable default
                window_size = cv2.getWindowImageRect('Face Details - Bounding Box & Landmarks')
                if window_size[2] > 0 and window_size[3] > 0:  # Valid size
                    display_size = (window_size[2], window_size[3])
                else:
                    display_size = (600, 600)  # Fallback size
            except:
                display_size = (600, 600)  # Fallback size
            
            # Resize detailed image to fit current window size while maintaining aspect ratio
            h, w = detailed_image.shape[:2]
            window_w, window_h = display_size
            
            # Calculate scaling factor to fit within window while maintaining aspect ratio
            scale = min(window_w / w, window_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized_detailed = cv2.resize(detailed_image, (new_w, new_h))
            
            # Create canvas with window size
            canvas = np.zeros((window_h, window_w, 3), dtype=np.uint8)
            
            # Center the resized image on canvas
            y_offset = (window_h - new_h) // 2
            x_offset = (window_w - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_detailed
            
            # Add info text
            info_text = f"{image_info['person']}"
            cv2.putText(canvas, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Face Details - Bounding Box & Landmarks', canvas)
        else:
            # Empty state with resizable placeholder
            try:
                window_size = cv2.getWindowImageRect('Face Details - Bounding Box & Landmarks')
                if window_size[2] > 0 and window_size[3] > 0:
                    canvas_size = (window_size[2], window_size[3])
                else:
                    canvas_size = (600, 600)
            except:
                canvas_size = (600, 600)
            
            empty_view = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
            cv2.putText(empty_view, "Hover over images to see", 
                       (canvas_size[0]//4, canvas_size[1]//2 - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            
            cv2.imshow('Face Details - Bounding Box & Landmarks', empty_view)

    def display_interactive(self):
        """Start the enhanced interactive display with RESIZABLE windows"""
        self.create_collage()
        
        # Create RESIZABLE windows
        cv2.namedWindow('Face Recognition Collage - ORIGINAL Images', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Face Details - Bounding Box & Landmarks', cv2.WINDOW_NORMAL)
        
        # Set initial window sizes
        cv2.resizeWindow('Face Recognition Collage - ORIGINAL Images', self.collage.shape[1], self.collage.shape[0])
        cv2.resizeWindow('Face Details - Bounding Box & Landmarks', 800, 600)  # Wider initial size
        
        cv2.setMouseCallback('Face Recognition Collage - ORIGINAL Images', self.mouse_callback)
        
        print("\n🎯 ENHANCED INTERACTIVE DISPLAY READY!")
        print("   - Hover over ORIGINAL images to see recognition results")
        print("   - Right window shows bounding boxes and facial landmarks")
        print("   - ✅ WINDOWS ARE NOW RESIZABLE - drag corners to adjust size!")
        print("   - Landmark colors: Blue=Right Eye, Cyan=Left Eye, Yellow=Nose, Red=Mouth")
        print("   - Press 'q' to quit, 'r' for report, 's' to save examples")
        print("   - Press 'f' to toggle fullscreen on detailed view")
        
        # Initial display
        display_image = self.collage.copy()
        cv2.putText(display_image, "Hover over ORIGINAL images to see face recognition", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Initial empty detailed view
        empty_view = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(empty_view, "Hover over ORIGINAL images to see", (150, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(empty_view, "face details with landmarks", (170, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(empty_view, "Drag window corners to resize", (180, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('Face Recognition Collage - ORIGINAL Images', display_image)
        cv2.imshow('Face Details - Bounding Box & Landmarks', empty_view)
        
        fullscreen = False
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.generate_detailed_report()
            elif key == ord('s'):
                self.save_detection_examples()
            elif key == ord('f'):
                # Toggle fullscreen on detailed view
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty('Face Details - Bounding Box & Landmarks', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    print("🖥️  Detailed view: FULLSCREEN")
                else:
                    cv2.setWindowProperty('Face Details - Bounding Box & Landmarks', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    print("🖥️  Detailed view: WINDOWED")
        
        cv2.destroyAllWindows()

    def generate_detailed_report(self):
        """Generate detailed report with landmark information"""
        print("\n" + "="*60)
        print("DETAILED FACE DETECTION REPORT - ORIGINAL IMAGES")
        print("="*60)
        
        total_faces = 0
        correct_predictions = 0
        
        for bbox, info in self.image_positions.items():
            results = self.recognize_face_with_details(info['path'])
            actual_person = info['person']
            
            if results:
                for result in results:
                    total_faces += 1
                    
                    predicted_person = result['identity']
                    confidence = result['confidence']
                    
                    is_correct = (predicted_person == actual_person)
                    if is_correct:
                        correct_predictions += 1
                    
                    status = "✅" if is_correct else "❌"
                    
                    print(f"{status} {actual_person:12} -> {predicted_person:12} (conf: {confidence:.3f})")
        
        if total_faces > 0:
            accuracy = (correct_predictions / total_faces) * 100
            print(f"\n📊 Summary:")
            print(f"   Total faces detected: {total_faces}")
            print(f"   Correct predictions: {correct_predictions}")
            print(f"   Accuracy: {accuracy:.1f}%")

    def save_detection_examples(self):
        """Save examples of face detection with landmarks from ORIGINAL images"""
        print("\n=== Saving Detection Examples from ORIGINAL Images ===")
        
        base_dir = Path("/home/steve/Python/Emerging-Technologies-in-CpE/facial_recognition_project")
        examples_dir = base_dir / "detection_examples_original"
        examples_dir.mkdir(exist_ok=True)
        
        saved_count = 0
        
        for i, (bbox, info) in enumerate(list(self.image_positions.items())[:5]):
            results = self.recognize_face_with_details(info['path'])
            
            if results:
                original_image = cv2.imread(str(info['path']))
                detailed_image = self.draw_face_details(original_image, results)
                
                output_path = examples_dir / f"example_{i+1:02d}_{info['person']}.jpg"
                cv2.imwrite(str(output_path), detailed_image)
                saved_count += 1
                print(f"✅ Saved: {output_path.name}")
        
        print(f"Saved {saved_count} detection examples to {examples_dir}")



class InsightFaceTrainer:
    def __init__(self, model_name='buffalo_l'):
        import insightface
        from insightface.app import FaceAnalysis
        self.model = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
        self.model.prepare(ctx_id=0, det_size=(640, 640))
        self.face_database = {}
        self.threshold = 0.4
    
    def load_database(self, input_path):
        """Load face database from file"""
        with open(input_path, 'rb') as f:
            self.face_database = pickle.load(f)
        print(f"✅ Face database loaded from {input_path}")

def main_after_training():
    """Complete workflow after successful training - USING ORIGINAL IMAGES"""
    base_dir = Path("/home/steve/Python/Emerging-Technologies-in-CpE/facial_recognition_project")
    
    print("🚀 LAUNCHING FACE RECOGNITION SYSTEM - RESIZABLE WINDOWS")
    print("=" * 60)
    
    # 1. Load the trained model
    print("📦 Step 1: Loading trained model...")
    trained_model = InsightFaceTrainer()
    model_path = base_dir / "models" / "face_database.pkl"
    
    if not model_path.exists():
        print("❌ Model file not found! Please train first.")
        return
    
    trained_model.load_database(model_path)
    print(f"   ✅ Loaded: {len(trained_model.face_database)} persons, {sum(len(emb) for emb in trained_model.face_database.values())} embeddings")
    
    # 2. Launch interactive display with RESIZABLE windows
    
    display = EnhancedInteractiveFaceDisplay(trained_model)
    display.display_interactive()
    
    print("\n🎉 Face recognition system completed!")

# RUN THE COMPLETE WORKFLOW
main_after_training()