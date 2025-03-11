import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import os
import json
from PIL import Image

# Define constants
IMG_SIZE = 400
BATCH_SIZE = 32
EPOCHS = 20
DATA_PATH = "data"
MODEL_PATH = "model"

# Ensure model directory exists
os.makedirs(MODEL_PATH, exist_ok=True)

# Load food labels from `calories_database.json`
CALORIE_DB_FILE = os.path.join(DATA_PATH, "calories_database.json")
with open(CALORIE_DB_FILE, "r") as f:
    calorie_db = json.load(f)

FOOD_LABELS = sorted(list(calorie_db.keys()))
NUM_CLASSES = len(FOOD_LABELS)  # Auto-detect number of classes

# Define Custom Dataset
class FoodDataset(Dataset):
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

        # Convert food labels to a one-hot vector
        labels = torch.zeros(NUM_CLASSES)
        for food in item["food type"]:
            if food in FOOD_LABELS:
                labels[FOOD_LABELS.index(food)] = 1  # Set corresponding index to 1

        if self.transform:
            image = self.transform(image)

        return image, labels

# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load Datasets
train_dataset = FoodDataset(json_path=os.path.join(DATA_PATH, "train.json"), 
                            img_dir=os.path.join(DATA_PATH, "images"),
                            transform=transform)
val_dataset = FoodDataset(json_path=os.path.join(DATA_PATH, "val.json"), 
                          img_dir=os.path.join(DATA_PATH, "images"),
                          transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define Model
class FoodClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FoodClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Multi-label classification

    def forward(self, x):
        return torch.sigmoid(self.model(x))  # Multi-label sigmoid activation

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FoodClassifier(NUM_CLASSES).to(device)
criterion = nn.BCEWithLogitsLoss()  # Multi-label loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader)}")

# Save Model with Metadata
checkpoint = {
    "model_state_dict": model.state_dict(),
    "num_classes": NUM_CLASSES  # Store number of classes to prevent mismatches
}

torch.save(checkpoint, os.path.join(MODEL_PATH, "food_classifier.pth"))
print(f"Model saved successfully with NUM_CLASSES = {NUM_CLASSES}!")
