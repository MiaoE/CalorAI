import json
import os

import torch
import torchvision.transforms as transforms
from PIL import Image

# Define paths
DATA_PATH = "data"
MODEL_PATH = "model"
CALORIE_DB_FILE = os.path.join(DATA_PATH, "calories_database.json")
FOOD_CLASSIFIER_MODEL = os.path.join(MODEL_PATH, "food_classifier.pth")
PORTION_REGRESSOR_MODEL = os.path.join(MODEL_PATH, "portion_regressor.pth")

# Load food labels
with open(CALORIE_DB_FILE, "r") as f:
    calorie_db = json.load(f)
FOOD_LABELS = sorted(list(calorie_db.keys()))
NUM_CLASSES = len(FOOD_LABELS)

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
])

# Load models
food_classifier = torch.load(FOOD_CLASSIFIER_MODEL, map_location=torch.device('cpu'))
food_classifier.eval()

portion_regressor = torch.load(PORTION_REGRESSOR_MODEL, map_location=torch.device('cpu'))
portion_regressor.eval()

def predict_calories(image_path):
    """Returns the predicted total portion and calorie content given an image"""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict food categories
    with torch.no_grad():
        outputs = food_classifier(image)
        probs = torch.softmax(outputs, dim=1)[0]
        top_indices = (probs > 0.1).nonzero(as_tuple=True)[0]  # Threshold for detected foods
    
    detected_foods = [FOOD_LABELS[i] for i in top_indices]
    detected_probs = [probs[i].item() for i in top_indices]

    # Predict portion sizes
    with torch.no_grad():
        portion_outputs = portion_regressor(image)
        portion_sizes = portion_outputs.squeeze().tolist()

    # Compute total calories
    total_calories = 0.0
    calorie_details = {}
    for food, prob, portion in zip(detected_foods, detected_probs, portion_sizes):
        grams = max(0, portion)  # Ensure non-negative portion size
        calories_per_gram = calorie_db.get(food, 0)
        total_calories += grams * calories_per_gram
        calorie_details[food] = {"grams": grams, "calories": grams * calories_per_gram}
    
    return {"total_calories": total_calories, "details": calorie_details}

# Example usage
if __name__ == "__main__":
    test_image = os.path.join(DATA_PATH, "images", "e002.png")
    result = predict_calories(test_image)
    print(json.dumps(result, indent=4))
