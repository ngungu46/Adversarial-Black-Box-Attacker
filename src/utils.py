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

# mpl.rcParams['figure.figsize'] = (8, 8)
# mpl.rcParams['axes.grid'] = False

IMAGESIZE = 299

decode_predictions = tf.keras.applications.inception_v3.decode_predictions

def get_label(probs):
    if len(probs) > 1:
        return decode_predictions(probs, top=3)
    return decode_predictions(probs, top=3)[0]

def torch_transform(image, is_torch):
    """
    input: numpy images of shape (B, H, W, C), normalized to (0, 1)
    output: tensor of images of shape (B, C, H, W), normalized to mean [.485, .456, .406], std [.229, .224, .225]
    """

    # print(type(image))
    # print(image.shape)
    if not isinstance(image, np.ndarray):
        image = image.detach().cpu().numpy()
    image = torch.tensor(image, dtype=torch.float32)
    if len(image.shape) <= 4:
        image = torch.unsqueeze(image, 1)
    # B, 1, H, W, C
    assert image.shape[-1] == 3
    image = torch.transpose(image, 1, 4)
    # B, C, H, W, 1
    assert image.shape[1] == 3 
    image = torch.squeeze(image, dim=4)
    # B, C, H, W
    assert image.shape[1] == 3 and len(image.shape) == 4
    
    if is_torch: 
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = transform(image)
    else: 
        pass 
    
    return image   

def transforming(image, is_torch=True):
    if np.shape(image) == (1, IMAGESIZE, IMAGESIZE, 1):
        image = tf.image.grayscale_to_rgb(image)
    elif np.shape(image) == (1, IMAGESIZE, IMAGESIZE): 
        image = image.squeeze(0).unsqueeze(-1) 
        image = image.repeat(1, 1, 3)
        image = image.unsqueeze(0)
    elif np.shape(image) == (1, IMAGESIZE, IMAGESIZE, 4):
        image = tf.image.grayscale_to_rgb(image)

    return torch_transform(image, is_torch)

def predict(image, model, is_torch=True, device='cuda'):
    """
    input: normalized tensor of shape (B, C, H, W)
    output: numpy array of predictions
    """
    # print("predicting")
    if is_torch: 
        with torch.no_grad():
            preds = model(transforming(image).to(device))
    else: 
        preds = model.predict(tf.cast(image * 255, tf.float32), steps=1)
        preds = torch.tensor(np.array(preds))
    return preds



def save_tensor_as_image(tensor, filename):
    """
    Saves a PyTorch tensor as an image file using OpenCV.

    :param tensor: PyTorch tensor of size (1, 299, 299, 3)
    :param filename: Filename to save the image
    """
    # Check if the input tensor is of the expected shape
    if tensor.shape != (1, 299, 299, 3) and tensor.shape != (1, 224, 224, 3): 
        raise ValueError("Tensor shape is not (1, 299/224, 299/224, 3)")

    # Remove the batch dimension and rearrange to HxWxC format
    tensor = tensor.squeeze(0)

    # Convert to numpy array
    numpy_image = tensor.cpu().numpy()

    # Convert from float tensors (0-1) to 0-255 uint8 format if necessary
    if np.max(numpy_image) <= 1.0:
        numpy_image = (numpy_image * 255).astype(np.uint8)

    # Save the image
    cv2.imwrite(filename, cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB))