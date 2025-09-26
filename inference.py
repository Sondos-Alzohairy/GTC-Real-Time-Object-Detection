from ultralytics import YOLO
import sys
import cv2
import numpy as np
from pathlib import Path
import time

# Add current directory to path
sys.path.append(".")
from models.detector import YoloDetector    

def run_image_inference():
    """Run inference on a single image"""
    print("🔍 Running image inference...")
    
    # Initialize detector
    model = YoloDetector()
    
    # Test image path
    test_image_path = "testimg.jpg"
    
    if not Path(test_image_path).exists():
        print(f"❌ Image not found: {test_image_path}")
        print("💡 Please add a test image named 'testimg.jpg' to the project folder")
        return
    
    # Load and process image
    print(f"📸 Loading image: {test_image_path}")
    image = cv2.imread(test_image_path)
    
    if image is None:
        print(f"❌ Could not load image: {test_image_path}")
        return
    
    print(f"📐 Image shape: {image.shape}")
    
    # Run detection
    print("🎯 Running detection...")
    start_time = time.time()
    result = model.detect(image)
    end_time = time.time()
    
    # Print results
    print(f"⏱️ Inference time: {result['inference_time']:.3f}s")
    print(f"🎯 Objects detected: {len(result['boxes'])}")
    
    for i, (box, score, class_name) in enumerate(zip(result['boxes'], result['scores'], result['class_names'])):
        x1, y1, x2, y2 = box
        print(f"   {i+1}. {class_name}: {score:.3f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    
    # Save result
    output_path = "inference_result.jpg"
    cv2.imwrite(output_path, result["image"])
    print(f"💾 Result saved: {output_path}")

def run_video_inference():
    """Run inference on a video file"""
    print("\n🎥 Running video inference...")
    
    # Initialize detector
    model = YoloDetector()
    
    # Test video path
    video_path = "traffic.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        print("💡 Please add a test video named 'traffic.mp4' to the project folder")
        return
    
    # Run video detection
    print(f"🎬 Processing video: {video_path}")
    output_path = "inference_video_result2.mp4"
    
    start_time = time.time()
    result = model.detect_video(video_path, output_path)
    end_time = time.time()
    
    print(f"⏱️ Total processing time: {end_time - start_time:.2f}s")
    print(f"💾 Video result saved: {output_path}")

def run_webcam_inference():
    """Run real-time inference on webcam"""
    print("\n📹 Running webcam inference...")
    print("Press 'q' to quit")
    
    # Initialize detector
    model = YoloDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    frame_count = 0
    total_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        result = model.detect(frame)
        
        # Calculate FPS
        frame_count += 1
        total_time += result['inference_time']
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        # Add FPS text
        cv2.putText(result["image"], f"FPS: {avg_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add detection count
        cv2.putText(result["image"], f"Objects: {len(result['boxes'])}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show result
        cv2.imshow("Real-time Detection", result["image"])
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"📊 Average FPS: {avg_fps:.1f}")

def run_batch_inference():
    """Run inference on multiple images in a folder"""
    print("\n📁 Running batch inference...")
    
    # Initialize detector
    model = YoloDetector()
    
    # Look for images in test folder
    test_folder = Path("test_images")
    if not test_folder.exists():
        print(f"❌ Test folder not found: {test_folder}")
        print("💡 Create a 'test_images' folder and add images to it")
        return
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(test_folder.glob(f"*{ext}"))
        image_paths.extend(test_folder.glob(f"*{ext.upper()}"))
    
    if not image_paths:
        print(f"❌ No images found in {test_folder}")
        return
    
    print(f"📸 Found {len(image_paths)} images")
    
    # Create output folder
    output_folder = Path("batch_results")
    output_folder.mkdir(exist_ok=True)
    
    # Process each image
    total_detections = 0
    total_time = 0
    
    for i, img_path in enumerate(image_paths, 1):
        print(f"🔍 Processing ({i}/{len(image_paths)}): {img_path.name}")
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"   ❌ Could not load {img_path.name}")
            continue
        
        # Run detection
        result = model.detect(image)
        
        # Update stats
        detections = len(result['boxes'])
        total_detections += detections
        total_time += result['inference_time']
        
        print(f"   🎯 Found {detections} objects in {result['inference_time']:.3f}s")
        
        # Save result
        output_path = output_folder / f"result_{img_path.name}"
        cv2.imwrite(str(output_path), result["image"])
    
    # Print summary
    avg_time = total_time / len(image_paths)
    avg_detections = total_detections / len(image_paths)
    
    print(f"\n📊 Batch Processing Summary:")
    print(f"   Images processed: {len(image_paths)}")
    print(f"   Total detections: {total_detections}")
    print(f"   Average detections per image: {avg_detections:.1f}")
    print(f"   Average processing time: {avg_time:.3f}s")
    print(f"   Results saved in: {output_folder}")

def create_test_image():
    """Create a simple test image if none exists"""
    test_path = "testimg.jpg"
    if Path(test_path).exists():
        return
    
    print("🎨 Creating test image...")
    
    # Create a simple road scene
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray background
    
    # Add sky
    img[:200, :] = [135, 206, 235]  # Sky blue
    
    # Add road
    img[300:, :] = [64, 64, 64]  # Dark gray road
    
    # Add lane markings
    cv2.line(img, (320, 300), (320, 480), (255, 255, 255), 3)
    
    # Add some colored rectangles to simulate objects
    # Car (blue)
    cv2.rectangle(img, (100, 250), (180, 320), (255, 0, 0), -1)
    cv2.putText(img, "CAR", (110, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Person (green)
    cv2.rectangle(img, (400, 220), (430, 300), (0, 255, 0), -1)
    cv2.putText(img, "PERSON", (380, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Truck (red)
    cv2.rectangle(img, (200, 240), (300, 310), (0, 0, 255), -1)
    cv2.putText(img, "TRUCK", (220, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imwrite(test_path, img)
    print(f"✅ Test image created: {test_path}")

def main():
    """Main function to run different inference modes"""
    print("🚗 Vehicle Detection Inference Script")
    print("=" * 40)
    
    # Create test image if it doesn't exist
    create_test_image()
    
    print("\nChoose inference mode:")
    print("1. Single Image")
    print("2. Video File") 
    print("3. Real-time Webcam")
    print("4. Batch Images")
    print("5. Run All Tests")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            run_image_inference()
        elif choice == "2":
            run_video_inference()
        elif choice == "3":
            run_webcam_inference()
        elif choice == "4":
            run_batch_inference()
        elif choice == "5":
            run_image_inference()
            run_video_inference()
            run_batch_inference()
        else:
            print("❌ Invalid choice. Running image inference...")
            run_image_inference()
            
    except KeyboardInterrupt:
        print("\n👋 Inference stopped by user")
    except Exception as e:
        print(f"❌ Error during inference: {e}")

if __name__ == "__main__":
    main()