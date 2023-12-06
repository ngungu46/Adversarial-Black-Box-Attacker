import cv2
import random
import tensorflow as tf
import torch
import torchvision
from torchvision import transforms
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import math

import time as time

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

IMAGESIZE = 299

decode_predictions = tf.keras.applications.inception_v3.decode_predictions

def get_imagenet_label(probs):
    if len(probs) > 1:
        return decode_predictions(probs, top=3)
    return decode_predictions(probs, top=3)[0]

def torch_transform(image):
    """
    input: numpy images of shape (B, H, W, C), normalized to (0, 1)
    output: tensor of images of shape (B, C, H, W), normalized to mean [.485, .456, .406], std [.229, .224, .225]
    """

    # print(type(image))
    # print(image.shape)
    if not isinstance(image, np.ndarray):
        image = image.numpy()
    image = torch.tensor(image, dtype=torch.float32)
    if len(image.shape) <= 4:
        image = torch.unsqueeze(image, 1)
    # B, 1, H, W, C
    assert image.shape[-1] == 3
    image = torch.transpose(image, 1, 4)
    # B, C, H, W, 1
    assert image.shape[1] == 3 and image.shape[3] == 299
    image = torch.squeeze(image, dim=4)
    # B, C, H, W
    assert image.shape[1] == 3 and image.shape[3] == 299 and len(image.shape) == 4
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = transform(image)
    # print(image.shape)
    return image


def transforming(image):
    if np.shape(image) == (1, IMAGESIZE, IMAGESIZE, 1):
        image = tf.image.grayscale_to_rgb(image)
    elif np.shape(image) == (1, IMAGESIZE, IMAGESIZE, 4):
        image = tf.image.grayscale_to_rgb(image)

    return torch_transform(image)