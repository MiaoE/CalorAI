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

def get_device():
    print(torch.__version__)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(f"Using device: {device}")
    return device


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn_layer = nn.Sequential(nn.Conv2d(3, 30, kernel_size=4, stride=2, padding=1), nn.Conv2d(30, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(), nn.Flatten())
        self.mlp_layer = nn.Sequential(nn.Linear(7 * 7 * 64, 63), nn.Sigmoid(), nn.Linear(63, 128), nn.ReLU(), nn.Linear(128, 64), nn.Linear(64, 32), nn.ReLU())


def plot_result(x, y):
    print('hi')
    pass


class CalorAI():
    def __init__(self, learning_rate=0.01):
        self.model = Model()
        self.loss_fcn = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train(self, epoch=10):
        self.model.train()
        training_loss = 0

if __name__ == '__main__':
    model = Model()
    learning_rate = 0.01
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    results = []
    # Training loop (fixed epochs = 10)
    pbar = tqdm.tqdm(range(1, 11))
    for epoch in pbar:
        model.train()
        train_loss = 0
        correct_predictions = 0
        total_samples = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_samples += target.size(0)
            correct_predictions += (predicted == target).sum().item()

        train_accuracy = 100.0 * correct_predictions / total_samples
        info = {'loss': train_loss / len(train_loader), 'accuracy': train_accuracy}
        pbar.set_postfix(info)
        results.append(info)

    train_df = pd.DataFrame(results)
    ipy_display.display(train_df)

    test_acc, n_params = evaluate(model)
    print(f"Final Test Accuracy after 10 Epochs: {test_acc:.2f}%")
    print(f"Number of trainable parameters: {n_params}")