from ultralytics import YOLO
import sys
import cv2
from utils.dataset import BDD100Dataset
sys.path.append(".")
from models.detector import YoloDetector    

model = YoloDetector()
    

print(F"get data for testing...")

dataset = BDD100Dataset("bdd100k_labels_release/bdd100k", split="train")
print(f"dataset loaded : {len(dataset)} images")


test_image_path = "bdd100k/bdd100k/images/100k/test/cb2fe290-8786cd14.jpg"

image = cv2.imread(test_image_path)
result = model.detect(image)

cv2.imwrite("simple_inference_for_yolo.jpg", result["image"])