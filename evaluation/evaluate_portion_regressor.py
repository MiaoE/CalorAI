import sys
import os
import json

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from tqdm import tqdm  # Progress bar
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add 'model/' to Python's path for importing the PortionRegressor
sys.path.append(os.path.abspath("model"))
# Import PortionRegressor
from model.portion_regressor import PortionRegressor

# Define paths
DATA_PATH = "data"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "portion_regressor.pth")
TEST_FILE = os.path.join(DATA_PATH, "test.json")
CALORIE_DB_FILE = os.path.join(DATA_PATH, "calories_database.json")
IMG_DIR = os.path.join(DATA_PATH, "images/")

# Load food labels
with open(CALORIE_DB_FILE, "r") as f:
    calorie_db = json.load(f)

FOOD_LABELS = sorted(list(calorie_db.keys()))
NUM_CLASSES = len(FOOD_LABELS)

# Define Dataset Class for Evaluation
class FoodPortionDataset(Dataset):
    """Custom dataset class used for portion regression"""
    def __init__(self, json_path, img_dir, transform=None):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item["name"] + ".png")
        image = Image.open(img_path).convert("RGB")

        # Convert food types to one-hot encoded vector
        food_vector = torch.zeros(NUM_CLASSES)
        portion_vector = torch.zeros(NUM_CLASSES)

        for food, portion in zip(item["food type"], item["portion"]):
            if food in FOOD_LABELS:
                food_idx = FOOD_LABELS.index(food)
                food_vector[food_idx] = 1
                portion_vector[food_idx] = float(portion)

        if self.transform:
            image = self.transform(image)

        return image, food_vector, portion_vector

# Define Data Transform
transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load Dataset
test_dataset = FoodPortionDataset(TEST_FILE, IMG_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PortionRegressor(NUM_CLASSES).to(device)

# Load saved model weights
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"Successfully loaded portion regressor model from {MODEL_PATH}")
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train and save the model first.")

# Evaluation
all_preds = []
all_labels = []

print("\n🔍 Evaluating portion estimation model...")

with torch.no_grad():
    for batch_idx, (images, food_vectors, portions) in enumerate(tqdm(test_loader, desc="Processing", unit="batch")):
        images, food_vectors, portions = images.to(device), food_vectors.to(device), portions.to(device)

        outputs = model(images, food_vectors)
        outputs = outputs.cpu().numpy()
        portions = portions.cpu().numpy()

        all_preds.append(outputs)
        all_labels.append(portions)

# Convert to numpy arrays
all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

# Compute MAE and RMSE per food category
mae_per_food = mean_absolute_error(all_labels, all_preds, multioutput='raw_values')
rmse_per_food = np.sqrt(mean_squared_error(all_labels, all_preds, multioutput='raw_values'))

# Overall metrics
overall_mae = np.mean(mae_per_food)
overall_rmse = np.mean(rmse_per_food)

# Define tolerance range (±10%)
TOLERANCE = 0.10  
success_count = 0
total_count = 0

for i in range(all_labels.shape[0]):
    for j in range(NUM_CLASSES):
        if all_labels[i, j] > 0:  # Only evaluate actual food items
            lower_bound = all_labels[i, j] * (1 - TOLERANCE)
            upper_bound = all_labels[i, j] * (1 + TOLERANCE)
            if lower_bound <= all_preds[i, j] <= upper_bound:
                success_count += 1
            total_count += 1

accuracy = (success_count / total_count) * 100 if total_count > 0 else 0

# Print Results
print("\n=== Portion Estimation Evaluation ===")
for i, food in enumerate(FOOD_LABELS):
    print(f"{food}: MAE = {mae_per_food[i]:.2f}g, RMSE = {rmse_per_food[i]:.2f}g")

print(f"\nOverall MAE: {overall_mae:.2f}g")
print(f"Overall RMSE: {overall_rmse:.2f}g")
print(f"Accuracy within ±10% range: {accuracy:.2f}%")
