from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image
import os
import io
import torch.optim as optim
import torch.nn as nn
from transformers import AutoModelForImageClassification
import torch
import torch.nn.functional as F


#labels of the class the predict function returns
class_labels = [
    "Accessories",  # Class 0 
    "Apparel",    # Class 1
    "Footwear",    # Class 2
    "Free Items",    # Class 3
    "Home",   # Class 4 
    "Personal Care",  # Class 5
    "Sporting Goods",    # Class 6
]

#initialize Flask app
app = Flask(__name__)

#loading the model
model_name = "google/vit-base-patch16-224-in21k"
model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=7)

model.load_state_dict(torch.load('/Users/clarina-manuel/Documents/image_classification_webapp/fashion_classifier.pth'))

for name, param in model.named_parameters():
    print(name, param.requires_grad, param.shape)

model.eval()

#image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

#check file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image):
    #image pre-processing
    image = image.convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)

    logits = output.logits

    #softmax -- get probabilities
    probabilities = F.softmax(logits, dim=-1)
    _, predicted_class = torch.max(probabilities, 1) #highest probability
    confidence = probabilities[0, predicted_class].item()

    return predicted_class.item(), confidence


#home route
@app.route('/')
def home():
    return render_template('index.html')

#upload and predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'}), 400
    
    #process image
    image = Image.open(file.stream)
    predicted_class, confidence = predict_image(image)
    return jsonify({'prediction': class_labels[predicted_class], 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
