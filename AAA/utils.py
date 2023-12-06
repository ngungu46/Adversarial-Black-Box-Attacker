import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import cv2
import time
import shutil

import argparse
import PIL.Image
import matplotlib.pyplot as plt
import random

def softmax(x, axis=1):
    x = x - x.max(axis=axis, keepdims=True).values
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def ece_score(y_pred, y_test, n_bins=15):
    py = softmax(y_pred, axis=1) if y_pred.max() > 1 else y_pred

    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)
