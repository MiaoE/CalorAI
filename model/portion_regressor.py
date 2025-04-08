import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import timm
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from datetime import datetime

# Define paths
DATA_PATH = "data"
MODEL_PATH = "model"
TRAIN_FILE = os.path.join(DATA_PATH, "train.json")
CALORIE_DB_FILE = os.path.join(DATA_PATH, "calories_database.json")

# Ensure model directory exists
os.makedirs(MODEL_PATH, exist_ok=True)

# Load calorie database (for food labels)
with open(CALORIE_DB_FILE, "r") as f:
    calorie_db = json.load(f)

FOOD_LABELS = sorted(list(calorie_db.keys()))
NUM_CLASSES = len(FOOD_LABELS)

# Define Dataset
class FoodPortionDataset(Dataset):
    """Custom dataset that iterates over the custom data"""
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

        # Convert food types to a one-hot encoded vector
        food_vector = torch.zeros(NUM_CLASSES)
        portion_vector = torch.zeros(NUM_CLASSES)

        for food, portion in zip(item["food type"], item["portion"]):
            if food in FOOD_LABELS:
                food_idx = FOOD_LABELS.index(food)
                food_vector[food_idx] = 1  # Detected food category
                portion_vector[food_idx] = float(portion)  # Ground truth portion size

        if self.transform:
            image = self.transform(image)

        return image, food_vector, portion_vector  # Return image, detected foods, and portion sizes

# Define Data Transforms
transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define Portion Size Regressor Model
class PortionRegressor(nn.Module):
    """Regression model that uses the ResNet architecture and FC layers for regression"""
    def __init__(self, num_classes):
        super(PortionRegressor, self).__init__()
        self.backbone = timm.create_model('resnet34', pretrained=True, num_classes=16)  # models.resnet34(pretrained=True)
        self.vector_embed = nn.Linear(num_classes, 16)
        
        # Fully connected layers for portion prediction
        self.fc = nn.Sequential(
            nn.Linear(32, 256),  # Image features + food class vector
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, NUM_CLASSES),  # Predict portion sizes for each food type
            nn.ReLU()
        )

    def forward(self, img, food_vector):
        features = self.backbone(img)  # Extract image features
        food_vec_embedding = self.vector_embed(food_vector)
        x = torch.cat((features, food_vec_embedding), dim=1)  # Concatenate with food category vector
        return self.fc(x)  # Predict portion sizes

# Ensure the model definition is accessible when imported
if __name__ == "__main__":
    # Training mode - This will NOT run when imported
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PortionRegressor(NUM_CLASSES).to(device)

    # Loads saved data if it exists
    checkpoint_path = os.path.join(MODEL_PATH, "portion_regressor.pth")
    if os.path.exists(checkpoint_path):
        checkpoint_data = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        print(f"Loaded previous checkpoint saved in { checkpoint_path }")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print("Training portion regressor...")
    start_time = datetime.now()

    # Load Datasets
    train_dataset = FoodPortionDataset(json_path=os.path.join(DATA_PATH, "train.json"), 
                                       img_dir=os.path.join(DATA_PATH, "images"),
                                       transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    EPOCHS = 10

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, food_vectors, portions in train_loader:
            images, food_vectors, portions = images.to(device), food_vectors.to(device), portions.to(device)

            optimizer.zero_grad()
            outputs = model(images, food_vectors)  # Pass both image & detected food types
            loss = criterion(outputs, portions)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader)}")

    # Save Model with Metadata
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_classes": NUM_CLASSES  # Store number of classes
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Portion regressor model saved successfully with NUM_CLASSES = {NUM_CLASSES}!")

    end_time = datetime.now()
    print(f"Time Elapsed: { end_time - start_time }")
