from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
import io
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import your existing detector
from models.detector import YoloDetector

# Initialize Flask app
app = Flask(__name__, template_folder='../templates')
CORS(app)

# Initialize detector
detector = YoloDetector()

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Simple detection endpoint"""
    try:
        # Get image data from request
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array (RGB format)
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_np
        
        # Run detection using your existing detector
        result = detector.detect(image_bgr)
        
        # Format response for frontend
        detections = []
        for box, score, class_name in zip(result['boxes'], result['scores'], result['class_names']):
            x1, y1, x2, y2 = box
            
            # Calculate width and height from coordinates
            width = x2 - x1
            height = y2 - y1
            
            detections.append({
                'bbox': [float(x1), float(y1), float(width), float(height)],  # [x, y, w, h] format
                'confidence': float(score),
                'class': class_name
            })
        
        return jsonify({
            'success': True,
            'detections': detections,
            'count': len(detections),
            'inference_time': result['inference_time']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect', methods=['POST'])
def api_detect():
    """Alternative endpoint for compatibility"""
    return detect()

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector.model is not None
    })

if __name__ == '__main__':
    print("🚀 Starting Vehicle Detection Server...")
    print("📍 Access at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)