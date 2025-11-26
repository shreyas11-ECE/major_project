from flask import Flask, render_template, request
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io

app = Flask(__name__)

# --- Define class names for corn seed quality ---
classes = ['Average', 'Bad', 'Excellent', 'Good', 'Worst']

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Corn/Not Corn Model ---
def load_corn_detector():
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load("corn_classifier.pt", map_location=device))
    model.to(device)
    model.eval()
    return model

corn_detector = load_corn_detector()

# --- Load Corn Quality Classification Model ---
def load_quality_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(torch.load("Corn Seed Classification Model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

quality_model = load_quality_model()

# --- Define transforms ---
# For corn/not-corn model
corn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# For quality model (no normalization)
quality_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No file part', 400

        file = request.files['image']
        if file.filename == '':
            return 'No selected file', 400

        img = Image.open(file.stream).convert("RGB")

        # --- Step 1: Check if it's corn or not ---
        input_tensor = corn_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = corn_detector(input_tensor).item()
        is_corn = prob < 0.5  # Adjust threshold if needed

        if not is_corn:
            prediction = "Not Corn"
        else:
            # --- Step 2: Predict corn quality ---
            input_tensor = quality_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = quality_model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                prediction = f"Corn - {classes[predicted.item()]}"

        print(f"[INFO] File: {file.filename} | Predicted Class: {prediction}")

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
