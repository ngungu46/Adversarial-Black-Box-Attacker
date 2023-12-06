import cv2
import random
import tensorflow as tf
from tensorflow.python.ops.gen_nn_ops import LRN
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
from utils import *

import time as time

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
pretrained_model = pretrained_model.to("cuda")
pretrained_model.eval()

decode_predictions = tf.keras.applications.inception_v3.decode_predictions

TARGET_EPS = 0.05
LR = 0.01
N = 100
SIGMA = 0.001
MAX_QUERIES = 20000

def NES(
    x_orig,
    y_adv,
    sigma = SIGMA,
    n_samples = N,
):
    """
    x: np.ndarray
    y_class: str
    sigma: float
    n_samples: int
    img_dim: tuple
    classifier: function
    model: tf.keras.Model
    k: int
    """
    _, r, d, _ = x_orig.shape
    noise = torch.normal(mean = 0, std = 1, size = (n_samples//2, 3, r, d))
    noise = torch.cat([noise, -noise], axis = 0).cuda()
    x_orig = transforming(x_orig)
    x = x_orig.repeat((n_samples, 1, 1, 1)).cuda()
    x += noise * sigma
    predictions = pretrained_model(x)
    prob = torch.nn.functional.softmax(predictions).detach()[:, y_adv]
    prob = prob[:, None, None, None]
    g = prob * noise
    return g.sum(dim = 0).unsqueeze(0) / (n_samples)

class NESAttack:
    def __init__(
        self, 
        target_eps = TARGET_EPS,
        lr = LR, 
        n_samples = N, 
        sigma = SIGMA, 
        max_queries = MAX_QUERIES,
        momentum = 0.9
    ):
        self.lr = lr
        self.target_eps = target_eps
        self.n_samples = n_samples
        self.sigma = sigma
        self.max_queries = max_queries
        self.momentum = momentum
        

    def attack(
        self,
        x_orig,
        y_adv,
    ):
        count = 0 
        x_adv = x_orig
        upper = x_orig + self.target_eps
        lower = x_orig - self.target_eps
        grad = torch.zeros((1, 3, 299, 299))
        while count < self.max_queries:
            prev_grad = grad.cuda()
            grad = NES(
                x_adv,
                y_adv,
                self.sigma,
                self.n_samples,
                1
            )
            # grad = self.momentum * prev_grad + (1 - self.momentum) * grad
            count += self.n_samples
            x_adv = x_adv + self.lr * torch.sign(grad).cpu().detach().numpy().transpose(0, 2, 3, 1)
            self.lr *= 0.99
            x_adv = np.clip(x_adv, lower, upper)
            cls = torch.argmax(pretrained_model(transforming(x_adv).cuda())).detach().cpu().numpy().item()
            probs = torch.nn.functional.softmax(pretrained_model(transforming(x_adv).cuda())).cpu().detach().numpy()
            print(decode_predictions(probs, top = 4))
            if(cls == y_adv):
                return x_adv, cls, probs[:, y_adv], count, True
            # print(probs[:, y_adv])

        return x_adv, 0, 0, 0, False

