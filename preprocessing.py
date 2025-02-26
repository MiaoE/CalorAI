import torch
import torch.nn as nn
import torchvision
import torch.utils.data as torch_data
import torch.nn.functional as F
import torchmetrics

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

import collections

print(torch.__version__)
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
print(f"Using device: {device}")

def data_grabber(path):
    if 'http' in path:
        website = path
    else:
        do_stuff = path


def display_image(image, w, h):
    plt.imshow(image.reshape(w, h))