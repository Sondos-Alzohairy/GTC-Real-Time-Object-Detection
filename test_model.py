"""Simple test script to verify model functionality"""
import sys
from pathlib import Path
import cv2
import numpy as np
import time

sys.path.append(".")
from models.detector import YoloDetector

def test_model_loading():
    """Test if the model loads correctly"""
    print("🧪 Testing model loading...")
    
    try:
        detector = YoloDetector()
        print(f"   ✅ Model loaded: {detector.model is not None}")
        print(f"   📊 Model type: {type(detector.model)}")
        
        # Check if custom model exists
        custom_model_path = Path("models/best.pt")
        if custom_model_path.exists():
            print(f"   🎯 Using custom model: {custom_model_path}")
        else:
            print(f"   🔄 Using default YOLOv8n model")
            
        return detector
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return None

def test_detection():
    """Test detection on a simple image"""
    print("\n🧪 Testing detection...")
    
    detector = test_model_loading()
    if not detector:
        return False
    
    # Create a simple test image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    test_img[:] = (128, 128, 128)  # Gray background
    
    # Add some colored shapes
    cv2.rectangle(test_img, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue square
    cv2.rectangle(test_img, (300, 150), (400, 250), (0, 255, 0), -1)  # Green square
    
    try:
        start_time = time.time()
        result = detector.detect(test_img)
        end_time = time.time()
        
        print(f"   ✅ Detection completed in {end_time - start_time:.3f}s")
        print(f"   🎯 Objects detected: {len(result['boxes'])}")
        print(f"   📊 Classes found: {result['class_names']}")
        
        # Save test result
        cv2.imwrite("test_detection_result.jpg", result['image'])
        print(f"   💾 Test result saved: test_detection_result.jpg")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Detection failed: {e}")
        return False

def test_model_info():
    """Display model information"""
    print("\n🧪 Testing model info...")
    
    detector = test_model_loading()
    if not detector:
        return
    
    try:
        # Get model info
        model = detector.model
        print(f"   📋 Model classes: {len(model.names)}")
        print(f"   🏷️ First 10 classes: {list(model.names.values())[:10]}")
        
        # Check if model has weights
        if hasattr(model.model, 'state_dict'):
            params = sum(p.numel() for p in model.model.parameters())
            print(f"   🔧 Model parameters: {params:,}")
        
    except Exception as e:
        print(f"   ❌ Model info failed: {e}")

def run_performance_test():
    """Test model performance"""
    print("\n🧪 Testing performance...")
    
    detector = test_model_loading()
    if not detector:
        return
    
    # Create test images of different sizes
    test_sizes = [(320, 240), (640, 480), (1280, 720)]
    
    for width, height in test_sizes:
        print(f"\n   📐 Testing {width}x{height} image...")
        
        # Create test image
        test_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        try:
            # Run multiple inferences to get average time
            times = []
            for _ in range(5):
                start_time = time.time()
                result = detector.detect(test_img)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            print(f"      ⏱️ Average time: {avg_time:.3f}s")
            print(f"      🚀 FPS: {fps:.1f}")
            print(f"      🎯 Objects detected: {len(result['boxes'])}")
            
        except Exception as e:
            print(f"      ❌ Performance test failed: {e}")

def main():
    """Run all tests"""
    print("🚗 Model Testing Suite")
    print("=" * 30)
    
    # Run tests
    model_loaded = test_model_loading() is not None
    detection_works = test_detection() if model_loaded else False
    
    if model_loaded:
        test_model_info()
        run_performance_test()
    
    print("\n" + "=" * 30)
    print("📋 Test Summary:")
    print(f"   Model Loading: {'✅ PASS' if model_loaded else '❌ FAIL'}")
    print(f"   Detection: {'✅ PASS' if detection_works else '❌ FAIL'}")
    
    if model_loaded and detection_works:
        print("🎉 All tests passed! Your model is ready for inference.")
        print("🚀 Run: python inference.py")
    else:
        print("⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()