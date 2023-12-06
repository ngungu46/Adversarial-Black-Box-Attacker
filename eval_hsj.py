import cv2
import random
import tensorflow as tf
import torch

import numpy as np
import matplotlib as mpl
import os

import sys
import random

from HSJattack.utils import load_image, display_images, hsja
from HSJattack.models import Model

from AAA.models import AAALinear
from AAA.utils import loss

random.seed(685)

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
pretrained_model = pretrained_model.to("cuda")
pretrained_model.eval()

def predict(image):
    """
    input: normalized tensor of shape (B, C, H, W)
    output: numpy array of predictions
    """
    preds = pretrained_model(image)
    return preds

def attack_image(model, image, num_iterations, image_file, save_dir, suffix = ''):
    if suffix != '':
        suffix = f'_{suffix}'

    with torch.no_grad():
        output_image, dist = hsja(model, image, constraint = 'linf', num_iterations=num_iterations)

        success = model.decision(output_image)
        if success:
            print("Success")

        label = model.predict(image)

        folder_name = image_file.replace(".JPEG", "")
        os.makedirs(f"output/{save_dir}/{folder_name}", exist_ok=True)
        np.savetxt(f"output/{save_dir}/{folder_name}/distance{suffix}.txt", np.array([dist]))
        np.savetxt(f"output/{save_dir}/{folder_name}/queries{suffix}.txt", np.array([model.count]))
        np.savetxt(f"output/{save_dir}/{folder_name}/label{suffix}.txt", np.array(label), fmt='%s')
        
        display_images(
            model, 
            output_image, 
            save_path = f"output/{save_dir}/{folder_name}",
            image_name = folder_name, 
            suffix = suffix[1:]
        )
        
    return dist, success

# main_path = os.path.dirname(os.path.abspath("/eval.py")) #path of main folder
data_path = './HSJattack/imagenet_val'
# os.path.join(main_path, "/HSJattack/imagenet_val") #path of validation data

images = {}
class_files = os.listdir(data_path)
for clf in class_files:
    images[clf] = os.listdir(os.path.join(data_path, clf))

# image_size = 299
image_size = 224
num_iterations = 3

count = 0
total_success = 0
total_dist = 0
# save_dir = "output_images_3_iter"
save_dir = "defender_trash"
visited = set()

defender = AAALinear(
    pretrained_model=predict,
    loss=loss,
    device='cuda', 
    batch_size=1000, 
    attractor_interval=6, 
    reverse_step=1, 
    num_iter=100, 
    calibration_loss_weight=5, 
    optimizer_lr=0.1, 
    do_softmax=False,
    temperature=1,
    verbose=False,
)

def predict_defender(image):
    preds = defender(image)
    return preds

while count < 100:
    cls = random.choice(class_files)
    image_file = random.choice(images[cls])
    while image_file in visited:
        cls = random.choice(class_files)
        image_file = random.choice(images[cls])
    visited.add(image_file)
    image_path = os.path.join(data_path, cls, image_file)

    image, label = load_image(image_path, image_size)

    model = Model(predict, label)
    model_defense = Model(predict_defender, label)

    image_probs = model.predict(image)
    correct = image_probs[0][0] == label

    if correct:
        dist, success = attack_image(model, image, num_iterations, image_file, save_dir)

        total_dist += dist
        if success:
            total_success += 1

        dist, success = attack_image(model_defense, image, num_iterations, image_file, save_dir, suffix = 'defense')

        total_dist += dist #probably want to add a separate tally for defense
        if success:
            total_success += 1

        count += 1
        

np.savetxt(f"output/{save_dir}/success.txt", np.array([success, count, total_dist / success]))

# print(count)
# print(total_count)

# img = randomimg()
# hsja = hsja(img, constraint = 'linf', num_iterations=30)
# display_images(hsja)

#########

# count = 0
# total_count = 0
# for cls in images:
#     print(cls)
#     for imgfile in images[cls]:
#         total_count += 1
#         imgpath = os.path.join(data_path, cls, imgfile)

#         if img(imgpath).usable:
#             count += 1
#     print(count)