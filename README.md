# Real-Time Object Detection for Autonomous Vehicles

## Project Overview
Autonomous vehicles need to detect and classify objects in real time to ensure safety and make accurate driving decisions. This project implements a real-time object detection system capable of identifying pedestrians, vehicles, traffic signs, and road obstacles under diverse environmental conditions.

The project uses deep learning models (e.g., YOLO, Faster R-CNN, SSD) to process video streams and provide real-time detection with bounding boxes.

## Features
- Detects key objects on the road: pedestrians, vehicles, traffic signs, obstacles.
- Real-time processing with FPS measurement.
- Evaluation metrics include mean Average Precision (mAP) and accuracy.
- Interactive web-based demo for video input and live detection visualization.

## Project Workflow
1. **Data Preparation**
   - Use publicly available datasets such as KITTI, BDD100K, or COCO.
   - Preprocess images and annotations into a unified format.

2. **Exploratory Data Analysis & Augmentation**
   - Analyze class distributions (pedestrians, vehicles, signs).
   - Apply augmentations like brightness adjustment, rotation, and cropping.

3. **Model Training & Validation**
   - Train deep learning object detection models.
   - Evaluate model performance using mAP, accuracy, and FPS.

4. **Deployment**
   - Deploy a web app that accepts video streams.
   - Show real-time detection with bounding boxes and labels.
