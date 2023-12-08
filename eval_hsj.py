import random
import torch
from torchvision import transforms

import tensorflow as tf
import keras

import numpy as np
import matplotlib as mpl
import os

from HSJattack.utils_new import load_image, display_images
from HSJattack.utils import hsja
from HSJattack.models import Model

from AAA.models import AAALinear, AAASine
from AAA.utils import loss

random.seed(685)
torch.manual_seed(686)
np.random.seed(687)

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

def predict_imagenet(image, pretrained_model):
    if not torch.is_tensor(image): 
        image = torch.tensor(image, dtype=torch.float32)
    image = image.to(dtype=torch.float32, device='cuda')    

    if len(image.shape) == 3:
        image = image.unsqueeze(dim=0)

    image = torch.clamp(image, 0, 1)
    
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = transform(image)
    
    with torch.no_grad():
        preds = pretrained_model(image)

    preds = preds.cpu().numpy()

    return preds


def predict_butterfly(image, pretrained_model):
    if not torch.is_tensor(image): 
        image = torch.tensor(image, dtype=torch.float32)
    image = image.to(dtype=torch.float32, device='cuda')    

    if len(image.shape) == 3:
        image = image.unsqueeze(dim=0)

    image = torch.clamp(image, 0, 1)
    image = image.permute((0,2,3,1))
    image = image * 255

    image = tf.convert_to_tensor(image.cpu().numpy(), dtype=tf.float32)
    
    probs = pretrained_model(image)
    preds = np.log(probs)

    return preds


def attack_image(model, image, num_iterations, constraint, image_file, save_dir, suffix = ''):
    if suffix != '':
        suffix = f'_{suffix}'

    with torch.no_grad():
        output_image, dists = hsja(model, image, constraint=constraint, num_iterations=num_iterations)

        label = model.predict(image)[0]

        folder_name = image_file.replace(".JPEG", "")
        os.makedirs(f"output/{save_dir}/{folder_name}", exist_ok=True)
        np.savetxt(f"output/{save_dir}/{folder_name}/distance{suffix}.txt", np.array(dists))
        np.savetxt(f"output/{save_dir}/{folder_name}/queries{suffix}.txt", np.array([model.count]))
        np.savetxt(f"output/{save_dir}/{folder_name}/label{suffix}.txt", np.array(label), fmt='%s')
        
        display_images(
            model, 
            output_image, 
            save_path = f"output/{save_dir}/{folder_name}",
            image_name = folder_name, 
            suffix = suffix[1:]
        )

if __name__ == '__main__':

    num_iterations = 5
    # constraint = 'l2'
    constraint = 'linf'

    dataset = 'imagenet'
    # dataset = 'butterfly'

    if dataset == 'imagenet':
        pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        pretrained_model = pretrained_model.to("cuda")
        pretrained_model.eval()

        image_size = 224
        attractor_interval = 4
        predict = lambda x: predict_imagenet(x, pretrained_model)

        data_path = './data_val/imagenet_val_100'
    elif dataset == 'butterfly':
        model_path = './data/butterfly/EfficientNetB0-100-(224 X 224)- 97.59.h5'
        pretrained_model = keras.models.load_model(model_path, custom_objects={'F1_score':'F1_score'})

        image_size = 224
        attractor_interval = 3
        predict = lambda x: predict_butterfly(x, pretrained_model)

        data_path = './data_val/butterfly_val_100'


    images = {}
    class_files = os.listdir(data_path)
    for clf in class_files:
        images[clf] = os.listdir(os.path.join(data_path, clf))


    count = 0
    # save_dir = "output_images_3_iter"
    save_dir = "defender_trash"
    visited = set()

    # defender = AAALinear(
    #     pretrained_model=predict,
    #     loss=loss,
    #     device='cuda', 
    #     batch_size=1000, 
    #     attractor_interval=attractor_interval, 
    #     reverse_step=1, 
    #     num_iter=100, 
    #     calibration_loss_weight=5, 
    #     optimizer_lr=0.1, 
    #     do_softmax=False,
    #     temperature=1,
    #     verbose=False,
    # )

    defender = AAASine(
        pretrained_model=predict,
        loss=loss,
        device='cuda', 
        batch_size=1000, 
        attractor_interval=attractor_interval, 
        reverse_step=0.7, 
        num_iter=100, 
        calibration_loss_weight=5, 
        optimizer_lr=0.1, 
        do_softmax=False,
        temperature=1.2,
        verbose=False,
    )

    def predict_defender(image):
        preds = defender(image)
        return preds

    for cls in os.listdir(data_path):
        for image_file in os.listdir(os.path.join(data_path, cls)):
            image_path = os.path.join(data_path, cls, image_file)

            image, label = load_image(image_path, image_size)

            # print(image.shape)

            model = Model(predict, label, dataset = dataset)
            model_defense = Model(predict_defender, label, dataset = dataset)

            image_probs = model.predict(image)
            correct = image_probs[0][0][0] == label

            if correct:
                attack_image(model, image, num_iterations, constraint, image_file, save_dir)
                attack_image(model_defense, image, num_iterations, constraint, image_file, save_dir, suffix = 'defense')

                count += 1
        
