from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import os 
import numpy as np
from models.detector import YoloDetector
from ultralytics import YOLO
import torchvision.transforms as T
import time
import json
import sys
sys.path.append(".")
sys.path.append("../")
from models import YoloDetector

detector = YoloDetector()

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')

@app.route("/detect", methods=["POST"])
def detect():
    try : 
        data = request.json
        image_data = data.get("image")

        if not image_data : 
            return jsonify({"error" : "No image uploded"}),400
        
        result = detector.detect(image_data)
    except:
        pass


if __name__ == "__main__":
    app.run(debug=True)