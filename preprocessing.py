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

#import collections
#import kaggle

#Image processing inputs 
from PIL import Image, ImageOps
import os

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


def image_scaling(input_folder, target_size = (400,400)):  # TODO
    """
    Scale image data to a constant width and height (400px by 400px).
    Padding is added to the sides if necessary. (Don't want to lose data)

    :param input_folder: Path to the folder containing the input images.
    :param output_folder: Path to the folder where the scaled images will be saved.
    :param target_size: Tuple (width, height) for the target size. Default is (400, 400).
    """

    #Create a location to store the images!
    output_folder = input_folder + "_resized"
    os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        # Get the image
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        # Scaling factor calculations
        og_width, og_height = img.size 
        scaling_fac = min(target_size[0]/og_width, target_size[1]/og_height) # pick minimum size/400px 

        # Resize 
        new_width = int(og_width*scaling_fac) 
        new_height = int(og_height*scaling_fac)
        new_size = (new_width, new_height)
        resized_img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Make blank image with target size, paste resized image onto the middle of it
        padded_img = Image.new("RGB", target_size, (0, 0, 0)) #black 400px by 400px square 
        offset = ((int)((target_size[0] - new_width)/2), (int)((target_size[1] - new_height)/2)) # (target_dim - resize_dim) /2 for centering 
        padded_img.paste(resized_img, offset)

        # Save image into new folder
        out_path = os.path.join(output_folder, filename)
        padded_img.save(out_path)

    print("All images resized and saved to {output_folder}")

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
    inpt = "" #! Add path to images folder here to resize all images inside 
    image_scaling(inpt)
