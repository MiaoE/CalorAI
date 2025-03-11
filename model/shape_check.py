import torch

checkpoint = torch.load("model/food_classifier.pth", map_location="cpu")

print(checkpoint.keys())  
print(checkpoint["model.fc.weight"].shape)  # expected number of classes

# checkpoint = torch.load("model/calorie_regressor.pth", map_location="cpu")
# print(checkpoint.keys())  
# print(checkpoint["fc.0.weight"].shape)  # first layer weight shape
