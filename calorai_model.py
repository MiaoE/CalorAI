import torch
import torch.nn as nn
import torchvision
import torch.utils.data as torch_data
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

import collections
import kaggle

from preprocessing import convert_image, label_conversion, get_unique_ingredient_list

import json

def get_device():
    print(torch.__version__)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(f"Using device: {device}")
    return device


class Model(nn.Module):
    def __init__(self, out_dim):
        super(Model, self).__init__()
        print(out_dim)
        # [3,400,400] -> [6,195,195] -> [12, 37, 37]
        self.cnn_layer = nn.Sequential(nn.Conv2d(3, 6, kernel_size=10, stride=2), nn.Conv2d(6, 12, kernel_size=10, stride=5), nn.ReLU(), nn.Flatten())
        self.mlp_layer = nn.Sequential(nn.Linear(37 * 37 * 12, out_dim * 4), nn.Sigmoid(), nn.Linear(out_dim * 4, out_dim * 2), nn.ReLU(), nn.Linear(out_dim * 2, out_dim))

    def forward(self, x):
        h = self.cnn_layer(x)
        return self.mlp_layer(h)


class ResNet50:
    def __init__(self):
        pass

def get_data_list(data_path, image_dir):
    """ 
    image_dir must have / at the end of the string
    returns a tuple of (list of training images tensors, list of labels which in itself is a list)
    """
    all_ingredients = get_unique_ingredient_list()
    with open(data_path, 'r', encoding='utf-8') as training_file:
        training_data = json.load(training_file)
    training_image_tensor = []
    labels = []
    for image in training_data:
        training_image_tensor.append(convert_image(image_dir + image['name'] + '.png'))
        labels.append(label_conversion(image['food type'], image['calorie'], all_ingredients))
    return training_image_tensor, labels


class CalorAI():
    def __init__(self):
        self.device = get_device()
        self.init_custom_model()

    def init_custom_model(self, learning_rate=0.01):
        self.training_data, self.training_label = get_data_list('train.json', 'images_resized/')
        self.val_data, self.val_label = get_data_list('val.json', 'images_resized/')
        self.test_data, self.test_label = get_data_list('test.json', 'images_resized/')
        length_all_ingredients = len(self.training_data[1][0])
        self.model = Model(length_all_ingredients)
        self.loss_fcn = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def resnet(self):
        self.model = ResNet50()

    def train(self, epochs=10):
        for epoch in range(1, epochs+1):
            self.model.train()
            training_loss = 0
            for img, label in zip(self.training_data, self.training_label):
                img, label = img.to(self.device), label.to(self.device)
                self.optimizer.zero_grad()
                prediction = self.model.forward(img)
                loss = self.loss_fcn(prediction, label)
                loss.backward()
                training_loss += loss.item()
                self.optimizer.step()
            print("Epoch {} Average Loss: {:.4f}".format(epoch, training_loss / len(self.training_label)))

            

if __name__ == '__main__':
    device = get_device()
    model = CalorAI()

    model.train()