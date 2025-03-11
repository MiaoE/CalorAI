import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import os
import json
import numpy as np
from PIL import Image

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
    def __init__(self, num_classes):
        super(PortionRegressor, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove ResNet FC layer, use features
        
        # Fully connected layers for portion prediction
        self.fc = nn.Sequential(
            nn.Linear(512 + num_classes, 128),  # Image features + food class vector
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)  # Predict portion sizes for each food type
        )

    def forward(self, img, food_vector):
        features = self.backbone(img)  # Extract image features
        x = torch.cat((features, food_vector), dim=1)  # Concatenate with food category vector
        return self.fc(x)  # Predict portion sizes

# Ensure the model definition is accessible when imported
if __name__ == "__main__":
    # Training mode - This will NOT run when imported
    print("Training portion regressor...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PortionRegressor(NUM_CLASSES).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load Datasets
    train_dataset = FoodPortionDataset(json_path=os.path.join(DATA_PATH, "train.json"), 
                                       img_dir=os.path.join(DATA_PATH, "images"),
                                       transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    EPOCHS = 100

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

    torch.save(checkpoint, os.path.join(MODEL_PATH, "portion_regressor.pth"))
    print(f"Portion regressor model saved successfully with NUM_CLASSES = {NUM_CLASSES}!")
