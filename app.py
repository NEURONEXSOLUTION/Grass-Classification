import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, render_template, request, jsonify
import os
import time

# ----------------- Flask App Setup -----------------
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# ----------------- Model Setup -----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['Dry_Grass', 'Green_Grass']

# Define the same model architecture used during training
model = models.resnet18(weights=None)
num_features = model.fc.in_features

# Must match your training architecture
model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, len(class_names))
)

# Load trained weights
state_dict = torch.load('best_grass_model.pth', map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)

model.eval()
model.to(DEVICE)

IMG_SIZE = 224
def repeat_channels(x):
    return x.repeat(3, 1, 1) if x.shape[0] == 1 else x

# Image preprocessing (same as training transforms)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.Lambda(repeat_channels)
])

# ----------------- Prediction Function -----------------
def predict_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        img_t = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_t)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        label = class_names[predicted.item()]
        return label, confidence.item() * 100
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0

# ----------------- Routes -----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Generate unique filename
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        label, confidence = predict_image(file_path)
        
        if label is None:
            return jsonify({'error': 'Error processing image'}), 500

        return render_template('result.html', 
                             label=label, 
                             confidence=round(confidence, 2), 
                             image_path=file_path)
    
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'error': 'Server error processing image'}), 500

# ----------------- Run App -----------------
if __name__ == '__main__':
    app.run(debug=True)