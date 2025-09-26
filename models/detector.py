import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import cv2
import numpy as np 
import time
from pathlib import Path

class YoloDetector:
    def __init__(self, conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        
        # Try to load best.pt, fallback to yolov8n.pt
        model_path = Path(__file__).parent / "best.pt"
        if model_path.exists():
            self.model = YOLO(str(model_path))
            print("✅ Using custom model: best.pt")
        else:
            self.model = YOLO("yolov8n.pt")
            print("✅ Using YOLOv8n model")
            
        self.class_names = ['person', 'car', 'truck', 'bus', 'traffic light', 'traffic sign']

    def detect(self, image):
        start_time = time.time()

        # Ensure image is in correct format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB for YOLO
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        results = self.model(image_rgb, conf=self.conf_threshold, verbose=False)

        # Extract the results
        boxes = []
        scores = []
        class_names = []

        if len(results) > 0 and results[0].boxes is not None: 
            for box in results[0].boxes: 
                # Get coordinates in correct format
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())

                if class_id < len(self.model.names):
                    class_name = self.model.names[class_id]
                    
                    # Ensure coordinates are valid
                    if x2 > x1 and y2 > y1:
                        boxes.append([float(x1), float(y1), float(x2), float(y2)])
                        scores.append(float(confidence))
                        class_names.append(class_name)

        # Draw boxes on original image (BGR format)
        result_image = self._draw_boxes(image.copy(), boxes, scores, class_names)

        return {
            'boxes': boxes,
            'scores': scores,
            'class_names': class_names,
            'image': result_image,
            'inference_time': time.time() - start_time
        }
    
    def _draw_boxes(self, image, boxes, scores, class_names):
        """Draw bounding boxes on image."""
        colors = {
            'person': (0, 255, 0),      # Green
            'bicycle': (255, 255, 0),   # Yellow  
            'car': (255, 0, 0),         # Red
            'motorcycle': (255, 0, 255), # Magenta
            'airplane': (0, 255, 255),  # Cyan
            'bus': (255, 165, 0),       # Orange
            'train': (128, 0, 128),     # Purple
            'truck': (255, 20, 147),    # Deep Pink
            'boat': (0, 191, 255),      # Deep Sky Blue
            'traffic light': (255, 255, 255),  # White
            'fire hydrant': (139, 69, 19),     # Saddle Brown
            'stop sign': (255, 69, 0),         # Red Orange
            'parking meter': (128, 128, 128),   # Gray
        }
        
        for box, score, class_name in zip(boxes, scores, class_names):
            x1, y1, x2, y2 = map(int, box)
            color = colors.get(class_name, (0, 0, 255))  # Default red
            
            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            
            # Draw box with thickness based on confidence
            thickness = max(1, int(2 * score))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label background
            label = f"{class_name}: {score:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Ensure label fits within image
            label_y = max(text_height + 5, y1)
            cv2.rectangle(
                image, 
                (x1, label_y - text_height - 5), 
                (x1 + text_width, label_y), 
                color, -1
            )
            
            # Draw label text
            cv2.putText(
                image, label, (x1, label_y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
        
        return image 
    
    def detect_video(self, video_path, output_path=None, show_progress=True):
        """Simple video detection"""
        print(f"🎥 Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video info: {width}x{height}, {fps}FPS, {total_frames} frames")
        
        # Setup video writer if output path given
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Will save to: {output_path}")

        # Process video
        frame_count = 0
        total_detections = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            result = self.detect(frame)
            
            # Add info to frame
            detections = len(result['boxes'])
            total_detections += detections
            current_fps = 1.0 / result["inference_time"] if result["inference_time"] > 0 else 0
            
            # Add text overlay
            info_frame = result["image"].copy()
            cv2.putText(info_frame, f"Frame: {frame_count}/{total_frames}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(info_frame, f"FPS: {current_fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(info_frame, f"Objects: {detections}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save frame if output specified
            if writer:
                writer.write(info_frame)
            
            # Show progress
            if show_progress and frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"📈 Progress: {progress:.1f}% | Objects found: {total_detections}")
            
            # Show video (optional - comment out for faster processing)
            cv2.imshow("Video Detection", info_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⏹️ Stopped by user")
                break
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Final stats
        print(f"✅ Processing complete!")
        print(f"   Frames processed: {frame_count}")
        print(f"   Total objects detected: {total_detections}")
        print(f"   Average objects per frame: {total_detections/frame_count:.1f}")
        
        return True

    def detect_webcam(self, save_output=False):
        """Simple webcam detection"""
        print("📹 Starting webcam detection...")
        print("Press 'q' to quit, 's' to save current frame")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return False
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Cannot read from webcam")
                break
            
            # Run detection
            result = self.detect(frame)
            
            # Add info
            detections = len(result['boxes'])
            current_fps = 1.0 / result["inference_time"] if result["inference_time"] > 0 else 0
            
            # Add overlay
            info_frame = result["image"].copy()
            height = info_frame.shape[0]
            cv2.putText(info_frame, f"Live Detection - FPS: {current_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(info_frame, f"Objects: {detections}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(info_frame, "Press 'q' to quit, 's' to save", (10, height-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow("Webcam Detection", info_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_name = f"webcam_detection_{frame_count}.jpg"
                cv2.imwrite(save_name, info_frame)
                print(f"💾 Saved: {save_name}")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"📊 Processed {frame_count} webcam frames")
        return True

