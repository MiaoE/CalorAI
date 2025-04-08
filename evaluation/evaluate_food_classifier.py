import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, hamming_loss
import numpy as np
import os
import json
from PIL import Image

# Define constants
IMG_SIZE = 400
BATCH_SIZE = 32
DATA_PATH = "data"
MODEL_PATH = "model"

# Define the complete food label list (26 food types)
FOOD_LABELS = [
    "Bread", "Grapes", "Cherry Tomato", "Chicken Breast", "Cantaloupe", "Strawberries",
    "Blueberries", "Sweet Potato", "Egg", "Broccoli", "Apple", "Carrot", "Honeydew",
    "Clementine", "Pineapple", "Garlic", "Pear", "Chives", "Cauliflower", "Jujube",
    "Orange", "Banana", "Potato", "Raisins", "Mushrooms", "Onion"
]
NUM_CLASSES = len(FOOD_LABELS)

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

# Load Test Dataset
test_dataset = FoodDataset(json_path=os.path.join(DATA_PATH, "test.json"), 
                           img_dir=os.path.join(DATA_PATH, "images"),
                           transform=transform)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FoodClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FoodClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return torch.sigmoid(self.model(x))  # Sigmoid for multi-label classification

model = FoodClassifier(NUM_CLASSES).to(device)
model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "food_classifier.pth")))
model.eval()

# Evaluation
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).float()
        outputs = model(images)
        predictions = (outputs > 0.5).cpu().numpy()  # Convert logits to binary labels
        
        all_preds.append(predictions)
        all_labels.append(labels.cpu().numpy())

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

# Compute Metrics
f1 = f1_score(all_labels, all_preds, average='samples')  # Sample-wise F1 Score
hamming = hamming_loss(all_labels, all_preds)
exact_match = np.mean(np.all(all_preds == all_labels, axis=1))  # Exact Match Ratio

# Print Results
print(f"F1 Score: {f1:.4f}")
print(f"Hamming Loss: {hamming:.4f}")
print(f"Exact Match Ratio: {exact_match:.4f}")

'''
F1 Score: 0.7815
Hamming Loss: 0.0243
Exact Match Ratio: 0.5439
'''

# Compute F1 score per class
per_class_f1 = f1_score(all_labels, all_preds, average=None)

print("\nPer-Class F1 Scores:")
for i, score in enumerate(per_class_f1):
    print(f"{FOOD_LABELS[i]:<15} : {score:.4f}")
