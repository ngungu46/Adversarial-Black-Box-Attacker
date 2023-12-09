import random
import torch
from torchvision import transforms

import tensorflow as tf
import keras

import numpy as np
import matplotlib as mpl
import os
import sys

import json

from HSJattack.utils_new import load_image, display_images
from HSJattack.utils import hsja
from HSJattack.models import Model

from AAA.models import AAALinear, AAASine

random.seed(685)
torch.manual_seed(686)
np.random.seed(687)

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

def predict_imagenet(image, pretrained_model, device):
    if not torch.is_tensor(image): 
        image = torch.tensor(image, dtype=torch.float32)
    image = image.to(dtype=torch.float32, device=device)    

    if len(image.shape) == 3:
        image = image.unsqueeze(dim=0)

    image = torch.clamp(image, 0, 1)
    
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = transform(image)
    
    with torch.no_grad():
        preds = pretrained_model(image)

    preds = preds.cpu().numpy()

    return preds


def predict_butterfly(image, pretrained_model, device):
    if not torch.is_tensor(image): 
        image = torch.tensor(image, dtype=torch.float32)
    image = image.to(dtype=torch.float32)    

    if len(image.shape) == 3:
        image = image.unsqueeze(dim=0)

    image = torch.clamp(image, 0, 1)
    image = image.permute((0,2,3,1))
    image = image * 255

    # with tf.device(device):
    # image = tf.convert_to_tensor(image.cpu().numpy(), dtype=tf.float32)
    image = image.cpu().numpy()
    probs = pretrained_model(image)

    preds = np.log(probs)

    return preds


def attack_image(model, image, num_iterations, constraint, image_file, save_dir, cls, params, suffix = ''):
    if suffix != '':
        suffix = f'_{suffix}'

    with torch.no_grad():
        output_image, dists, counts = hsja(model, image, constraint=constraint, num_iterations=num_iterations)

        label = model.predict(image)[0]

        folder_name = image_file.replace(".JPEG", "")
        folder_name = f'{cls}_{folder_name}'
        os.makedirs(f"output/{save_dir}/{folder_name}", exist_ok=True)
        np.savetxt(f"output/{save_dir}/{folder_name}/distance{suffix}.txt", np.array(dists))
        np.savetxt(f"output/{save_dir}/{folder_name}/queries{suffix}.txt", np.array(counts))
        np.savetxt(f"output/{save_dir}/{folder_name}/label{suffix}.txt", np.array(label), fmt='%s')

        with open(f"output/{save_dir}/{folder_name}/params{suffix}.json", "w") as f: 
            json.dump(params, f, indent=4) 
        
        display_images(
            model, 
            output_image, 
            save_path = f"output/{save_dir}/{folder_name}",
            image_name = folder_name, 
            suffix = suffix[1:]
        )

if __name__ == '__main__':
    args = sys.argv[1:]
    device_ind = int(args[0])
    device = f'cuda:{device_ind}'

    modulo_ind = int(args[1])

    params = {
        'save_dir': "hsja_linf_butterfly_none",
        'num_iterations': 30,
        # 'constraint': 'l2',
        'constraint': 'linf',
        # 'dataset': 'imagenet',
        'dataset': 'butterfly',
        'image_size': 224,
        'defender': None,
        # 'defender': 'linear',
        # 'defender': 'sine',
        'reverse_step': 0.7,
        'defender_iter': 100,
        'calibration_weight': 5,
        'learning_rate': 0.1,
        'temperature': 1.2,
    }

    if params['dataset'] == 'imagenet':
        pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        pretrained_model = pretrained_model.to(device)
        pretrained_model.eval()

        params['attractor_interval'] = 4
        predict = lambda x: predict_imagenet(x, pretrained_model, device)

        data_path = './data_val/imagenet_val_100'
    elif params['dataset'] == 'butterfly':
        gpus = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(gpus[device_ind], 'GPU')

        model_path = './data/butterfly/EfficientNetB0-100-(224 X 224)- 97.59.h5'
        pretrained_model = keras.models.load_model(model_path, custom_objects={'F1_score':'F1_score'})

        params['attractor_interval'] = 4
        predict = lambda x: predict_butterfly(x, pretrained_model, f'/GPU:{device_ind}')

        data_path = './data_val/butterfly_val_100'


    images = {}
    class_files = os.listdir(data_path)
    for clf in class_files:
        images[clf] = os.listdir(os.path.join(data_path, clf))


    count = 0
    visited = set()

    if params['defender'] == 'linear':
        defender = AAALinear(
            pretrained_model=predict,
            device=device, 
            batch_size=1000, 
            attractor_interval=params['attractor_interval'], 
            reverse_step=params['reverse_step'], 
            num_iter=params['defender_iter'], 
            calibration_loss_weight=params['calibration_weight'], 
            optimizer_lr=params['learning_rate'], 
            do_softmax=False,
            temperature=params['temperature'],
            verbose=False,
        )
    else:
        defender = AAASine(
            pretrained_model=predict,
            device=device, 
            batch_size=1000, 
            attractor_interval=params['attractor_interval'], 
            reverse_step=params['reverse_step'], 
            num_iter=params['defender_iter'], 
            calibration_loss_weight=params['calibration_weight'], 
            optimizer_lr=params['learning_rate'], 
            do_softmax=False,
            temperature=params['temperature'],
            verbose=False,
        )

    def predict_defender(image):
        preds = defender(image)
        return preds

    i = -1
    for cls in os.listdir(data_path):
        for image_file in os.listdir(os.path.join(data_path, cls)):
            i += 1
            if i % 4 != modulo_ind:
                continue

            print(i)
            image_path = os.path.join(data_path, cls, image_file)

            image, label = load_image(image_path, params['image_size'])

            # print(image.shape)

            model = Model(predict, label, dataset = params['dataset'])
            model_defense = Model(predict_defender, label, dataset = params['dataset'])

            image_probs = model.predict(image)
            correct = image_probs[0][0][0] == label

            if correct:
                if params['defender'] == None:
                    attack_image(
                        model, 
                        image,
                        params['num_iterations'], 
                        params['constraint'], 
                        image_file, 
                        params['save_dir'],
                        label,
                        params,
                    )
                else:
                    attack_image(
                        model_defense, 
                        image, 
                        params['num_iterations'], 
                        params['constraint'], 
                        image_file, 
                        params['save_dir'], 
                        params,
                        label,
                        suffix = 'defense',
                    )

                count += 1
        
