#!pip install kaggle

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2

import torch
import torchvision
import torchvision.transforms as transforms

#import collections
#import kaggle

#Image processing inputs 
from PIL import Image, ImageOps
import os
import json

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


def create_unique_ingredient_list(file_name='ingredients_sorted.json'):
    # checks if cached all ordered ingredients exists
    path = './cache/' + file_name
    if not os.path.exists('./cache/'): 
        os.mkdir('cache')
    if not os.path.exists(path):
        ingredients = set()
        with open('data.json', mode='r', encoding='utf-8') as data_file:
            data = json.load(data_file)
        for image in data:
            ingredients.update(image['food type'])
        ingredients = list(ingredients)
        ingredients.sort()
        with open(path, mode='w', encoding='utf-8') as json_file:
            json.dump(ingredients, json_file)
    # print(all_ingredients)

def label_conversion(ingredient_list:list, calorie_list:list, all_ingredients) -> list:  # TODO
    '''
    one hot encoding of the ingredients and their calories
    e.g. if the ingredients in data are [apple, banana, cucumber, egg], 
    then the label [0, 50, 20, 0] means there's 50 kcal worth of bananas, and 20 kcal of cucumbers
    R_type: nested list
    '''
    # convert given ingredient and calorie list to one-hot encoded list
    one_hot_calories = [0] * len(all_ingredients)
    assert(len(ingredient_list) == len(calorie_list))
    for ingredient, calorie in zip(ingredient_list, calorie_list):
        idx = all_ingredients.index(ingredient)
        one_hot_calories[idx] = calorie
    return one_hot_calories

def convert_image(image_path):
    """
    Converts image given a file to a Tensor (C x H x W)
    NOTE: CV2 imread converts the image to BGR
    """
    img = torchvision.io.read_image(image_path)
    print(img)
    return img

def prepare_data():
    pass

def display_image(image, w, h):  # debugging
    plt.imshow(image.reshape(w, h, 3))
    plt.show()
    # to display BGR image converted by CV2, use the following code
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    # inpt = "" #! Add path to images folder here to resize all images inside 
    # image_scaling(inpt)
    # out = label_conversion(['Strawberries', 'Onion', 'Egg'], [45.9, 76, 39.45])
    # print(out)
    im = convert_image('images_resized/v127.png')
    # display_image(im, 400, 400)
    pass