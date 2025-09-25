from ultralytics import YOLO
import sys
import cv2
from utils.dataset import BDD100Dataset
sys.path.append(".")
from models.detector import YoloDetector    

model = YoloDetector()
    

print(F"get data for testing...")

# dataset = BDD100Dataset("bdd100k_labels_release/bdd100k", split="train")
# print(f"dataset loaded : {len(dataset)} images")


test_image_path = "testimg.jpg"

image = cv2.imread(test_image_path)
result = model.detect(image)

cv2.imwrite("inference_with_our_model.jpg", result["image"])