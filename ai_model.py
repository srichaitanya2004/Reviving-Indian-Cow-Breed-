from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from transformers import AutoModelForImageClassification, AutoProcessor
import torch
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define breed names (Make sure these are in the correct order from training)
breed_names = ["Gir", "Hariana", "Kankrej", "Ongole", "Rathi", "Red Sindhi", "Sahiwal"]

# Load the trained model from cow_breed_model folder
model_path = "D:/Projects/cow_breed_model"
model = AutoModelForImageClassification.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)

# Define function to predict breed
def predict(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    
    print(f"DEBUG: Logits = {logits}")  # Print logits for debugging
    
    predicted_class = torch.argmax(logits, dim=-1).item()
    breed_name = breed_names[predicted_class] if 0 <= predicted_class < len(breed_names) else "Unknown"
    
    return breed_name

@app.route('/predict', methods=['POST'])
def predict_cow_breed():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image = Image.open(request.files['image'])
    prediction = predict(image)

    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)