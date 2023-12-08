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

from AAA.models import AAALinear
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
        output_image, dist = hsja(model, image, constraint=constraint, num_iterations=num_iterations)

        success = model.decision(output_image)
        if success:
            print("Success")

        label = model.predict(image)[0]

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


num_iterations = 30
constraint = 'l2'
# constraint = 'linf'

# dataset = 'imagenet'
dataset = 'butterfly'

if dataset == 'imagenet':
    pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    pretrained_model = pretrained_model.to("cuda")
    pretrained_model.eval()

    image_size = 224
    predict = lambda x: predict_imagenet(x, pretrained_model)

    data_path = './data/imagenet_val'
elif dataset == 'butterfly':
    model_path = './data/butterfly/EfficientNetB0-100-(224 X 224)- 97.59.h5'
    pretrained_model = keras.models.load_model(model_path, custom_objects={'F1_score':'F1_score'})

    image_size = 224
    predict = lambda x: predict_butterfly(x, pretrained_model)

    data_path = './data/butterfly/test'


images = {}
class_files = os.listdir(data_path)
for clf in class_files:
    images[clf] = os.listdir(os.path.join(data_path, clf))


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

while count < 1:
    cls = random.choice(class_files)
    image_file = random.choice(images[cls])
    print(cls, image_file)
    while image_file in visited:
        cls = random.choice(class_files)
        image_file = random.choice(images[cls])
    visited.add(image_file)
    image_path = os.path.join(data_path, cls, image_file)

    image, label = load_image(image_path, image_size)

    # print(image.shape)

    model = Model(predict, label, dataset = dataset)
    model_defense = Model(predict_defender, label, dataset = dataset)

    image_probs = model.predict(image)
    correct = image_probs[0][0][0] == label

    if correct:
        dist, success = attack_image(model, image, num_iterations, constraint, image_file, save_dir)

        total_dist += dist
        if success:
            total_success += 1

        dist, success = attack_image(model_defense, image, num_iterations, constraint, image_file, save_dir, suffix = 'defense')

        total_dist += dist #probably want to add a separate tally for defense
        if success:
            total_success += 1

        count += 1
        

np.savetxt(f"output/{save_dir}/success.txt", np.array([total_success, count, total_dist / total_success]))

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