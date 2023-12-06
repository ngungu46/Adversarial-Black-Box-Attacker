import cv2
import random
import tensorflow as tf
from tensorflow.python.eager.backprop import _extract_tensors_and_variables
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

pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
pretrained_model = pretrained_model.to("cuda")
pretrained_model.eval()

decode_predictions = tf.keras.applications.inception_v3.decode_predictions

SIGMA = 0.001
EPS_DECAY = 0.001
EPS_0 = 0.5
N = 100
K = 1
E_ADV = 0.05
MAX_QUERIES = 20000
MAX_LR = 0.01
MIN_LR = 0.001

def NESPO(
    x_orig,
    y_adv,
    sigma = SIGMA,
    n_samples = N,
    k = K
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

class PartialInfoAttack:
    def __init__(
        self,
        e_adv = E_ADV,
        e_0 = EPS_0,
        sigma = SIGMA,
        n_samples = N,
        eps_decay = EPS_DECAY,
        max_lr = MAX_LR,
        min_lr = MIN_LR,
        k = K,
        max_queries = MAX_QUERIES
    ):
        self.e_adv = e_adv
        self.e_0 = e_0
        self.sigma = sigma
        self.n_samples = n_samples
        self.eps_decay = eps_decay
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.k = k
        self.max_queries = max_queries

    def attack(
        self,
        x_adv,
        y_adv,
        x_orig
    ):
        """
        x_adv: np.ndarray
        y_adv: str
        classifier: function
        x_orig: np.ndarray
        """
        epsilon = self.e_0

        lower = np.clip(x_orig - epsilon, 0, 1)
        upper = np.clip(x_orig + epsilon, 0, 1)
        x_adv = np.clip(x_adv, lower, upper)
        
        count = 0
        while count < self.max_queries and (epsilon > self.e_adv or y_adv != torch.argmax(pretrained_model(transforming(x_adv).cuda())).detach().cpu().numpy().item()):
            g = NESPO(
                x_adv,
                y_adv,
                self.sigma,
                self.n_samples,
                self.k
            )
            count += self.n_samples
            lr = self.max_lr
            g = torch.sign(g).cpu().detach().numpy().transpose(0, 2, 3, 1)
            hat_x_adv = x_adv + lr * g

            # print("hey: ", get_top_k_labels(self.model, hat_x_adv, 1)[0])
            probs = pretrained_model(transforming(hat_x_adv).cuda())
            topk = torch.topk(probs, self.k) # .detach().numpy()
            while y_adv not in topk:
                count += 1
                if count > self.max_queries:
                    return x_adv
                if lr < self.min_lr:
                    epsilon += self.eps_decay
                    self.eps_decay /= 2
                    hat_x_adv = x_adv
                    break

                proposed_eps = max(epsilon - self.eps_decay, self.e_adv)
                print(proposed_eps)
                lower = np.clip(x_orig - proposed_eps, 0, 1)
                upper = np.clip(x_orig + proposed_eps, 0, 1)
                lr /= 2
                hat_x_adv = np.clip(x_adv + lr * g, lower, upper)
            proposed_eps = max(epsilon - self.eps_decay, self.e_adv)

            lower = np.clip(x_orig - proposed_eps, 0, 1)
            upper = np.clip(x_orig + proposed_eps, 0, 1)
            hat_x_adv = np.clip(hat_x_adv, lower, upper)
            x_adv = hat_x_adv
            epsilon -= self.eps_decay

            probs = torch.nn.functional.softmax(pretrained_model(transforming(x_adv).cuda())).cpu().detach().numpy()
            print(decode_predictions(probs, top = 1), epsilon)

        cls = torch.argmax(pretrained_model(transforming(x_adv).cuda())).detach().cpu().numpy().item()
        if cls == y_adv:
            return x_adv, y_adv, True, count
        return x_adv, y_adv, False, count