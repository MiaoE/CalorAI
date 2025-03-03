#!pip install kaggle

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
import kaggle

def get_device():
    print(torch.__version__)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(f"Using device: {device}")
    return device


def data_grabber(device, path):
    if 'http' in path:
        website = path
    else:
        do_stuff = path


def image_scaling(images):  # TODO
    '''scale the images to a constant width and height'''
    pass

def label_conversion(input) -> list:  # TODO
    '''
    one hot encoding of the ingredients and their calories
    e.g. if the ingredients in data are [apple, banana, cucumber, egg], 
    then the label [0, 50, 20, 0] means there's 50 kcal worth of bananas, and 20 kcal of cucumbers
    R_type: nested list
    '''
    pass

def display_image(image, w, h):  # debugging
    plt.imshow(image.reshape(w, h))


if __name__ == "__main__":
    # device = get_device()
    pass
