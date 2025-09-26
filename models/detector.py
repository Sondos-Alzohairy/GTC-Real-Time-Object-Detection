import torch
import torch.nn as nn
import torch.nn.functional as f
from ultralytics import YOLO
import cv2
import numpy as np 
from typing import List,  Tuple, Dict
import time
import os 


class YoloDetector:

    def __init__(self, conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.model = YOLO("new_model.pt")
        self.class_names =  ['person', 'car', 'truck', 'bus', 'traffic light', 'traffic sign']

    def detect(self, image):
        start_time = time.time()

        results = self.model(image, conf=self.conf_threshold, verbose=True)

        # Extract the result
        boxes = []
        scores = []
        class_names = []

        if len(results) > 0 and results[0].boxes is not None : 
            for box in results[0].boxes: 
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id =  int(box.cls[0].cpu().numpy())

                if class_id < len(self.model.names):
                    class_name = self.model.names[class_id]
                    boxes.append([x1, y1, x2, y2])
                    scores.append(confidence)
                    class_names.append(class_name)

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
            'car': (255, 0, 0),         # Red
            'truck': (255, 0, 255),     # Magenta
            'bus': (255, 255, 0),       # Yellow
            'traffic light': (255, 255, 255),  # White
            'traffic sign': (128, 128, 128)    # Gray
        }
        
        for box, score, class_name in zip(boxes, scores, class_names):
            x1, y1, x2, y2 = map(int, box)
            color = colors.get(class_name, (0, 0, 255))  # Default red
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {score:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image 
    
    def detect_video(self, video_path, output_path = None):
        """detect a videos version"""
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))


        frame_count = 0
        while True : 
            ret, frame = cap.read()

            if not ret : 
                break
            result = self.detect(frame)
            fps_current =  1.0 / result["inference_time"]
            cv2.putText(result["image"], f"FPS: {fps_current :.1f}", (10, 30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1 ,(0,255,0),2)

            if output_path : 
                writer.write(result["image"])

            cv2.imshow("Detection", result["image"])

            if cv2.waitKey(1) & 0xFF  == ord("q"):
                break
            frame_count += 1
            if frame_count % 30 == 0 :
                print(f"Processed {frame_count} frames")
            
            cap.release()
            if output_path : 
                writer.release()
            cv2.destroyAllWindows()
            print(f"Process complete")

            