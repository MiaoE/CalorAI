import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models  
from PIL import Image
import numpy as np
import json
import os

# Define paths
DATA_PATH = "data"
MODEL_PATH = "model"
CALORIE_DB_FILE = os.path.join(DATA_PATH, "calories_database.json")
TRAIN_FILE = os.path.join(DATA_PATH, "train.json")
NUM_CLASSES = 26  

# Load calorie per gram database
with open(CALORIE_DB_FILE, "r") as f:
    calorie_db = json.load(f)

# Load train data (for real-world portion sizes)
with open(TRAIN_FILE, "r") as f:
    train_data = json.load(f)

# Define all food labels based on database keys
FOOD_LABELS = sorted(list(calorie_db.keys()))
NUM_CLASSES = len(FOOD_LABELS)

# Load models
class FoodClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FoodClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return torch.sigmoid(self.model(x))  # Multi-label classification

class CalorieRegressor(nn.Module):
    def __init__(self, input_size):
        super(CalorieRegressor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)

# Load model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

food_model = FoodClassifier(NUM_CLASSES).to(device)
food_model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "food_classifier.pth"), map_location="cpu"))
food_model.eval()

# Ensure input size matches saved model
checkpoint = torch.load(os.path.join(MODEL_PATH, "calorie_regressor.pth"), map_location="cpu")
first_layer_shape = checkpoint["fc.0.weight"].shape  # Get input feature size
input_size = first_layer_shape[1]  # Ensure we use the correct input size

calorie_model = CalorieRegressor(input_size).to(device)
calorie_model.load_state_dict(checkpoint)
calorie_model.eval()

# Preprocessing function
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(img_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

# Predict function
def predict_calories(img_path, portion_sizes=None):
    # Step 1: Predict Food Types from Image
    img = preprocess_image(img_path)
    food_probs = food_model(img).detach().cpu().numpy()[0]
    food_types = (food_probs > 0.5).astype(int)  # Multi-label binary classification

    # Step 2: If portion sizes are missing, get from `train.json`
    if portion_sizes is None:
        portion_sizes = {}
        image_name = os.path.basename(img_path).split(".")[0]  # Extract image name
        for item in train_data:
            if item["name"] == image_name:  # Match filename with train.json entry
                for food, portion in zip(item["food type"], item["portion"]):
                    portion_sizes[food] = float(portion)
                break  # Stop searching once we find the match

    # Step 3: Prepare features (food presence + portion sizes)
    portion_vector = np.zeros(NUM_CLASSES)
    for food, portion in portion_sizes.items():
        if food in FOOD_LABELS:
            idx = FOOD_LABELS.index(food)
            portion_vector[idx] = float(portion)  # Assign portion size

    feature_vector = np.concatenate([food_types, portion_vector])
    feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(device)

    # Step 4: Predict calories
    predicted_calories = calorie_model(feature_tensor).item()
    return predicted_calories

# Example Usage
img_path = "test_image.jpg"

# Option 1: Manually provide portion sizes
portion_sizes = {
    "Clementine": 35, "Broccoli": 38, "Cauliflower": 35, "Chives": 5, "Jujube": 16
}

# Option 2: Use portion sizes from `train.json` (if available)
predicted_calories = predict_calories(img_path, portion_sizes=None)

print(f"Predicted Calories: {predicted_calories}")
