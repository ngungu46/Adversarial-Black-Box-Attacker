import os
import shutil
import xml.etree.ElementTree as ET
import random
import shutil
from HSJattack.models import Model
from HSJattack.utils_new import load_image
import torch
import keras
from eval_hsj import predict_butterfly, predict_imagenet

# for img in os.listdir("ILSVRC/Data/CLS-LOC/val"):
#     annotation = img.replace(".JPEG", ".xml")

#     path = f"ILSVRC/Annotations/CLS-LOC/val/{annotation}"
#     tree = ET.parse(path)
#     root = tree.getroot()
#     for object in root.findall("object"):
#         # print(object.find("name").text)
#         synset = object.find("name").text
#         # print(synset)

#     # print(f"ILSVRC/Data/CLS-LOC/val/{img}", f"imagenet_val/{synset}/{img}")
#     if not os.path.isdir(f"imagenet_val/{synset}"):
#         os.makedirs(f"imagenet_val/{synset}")
#     shutil.copyfile(f"ILSVRC/Data/CLS-LOC/val/{img}", f"imagenet_val/{synset}/{img}")

# create 100 image val set

# data_path = './data/imagenet_val'
# new_path = './data/imagenet_val_100'
# pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
# pretrained_model = pretrained_model.to("cuda")
# pretrained_model.eval()
# predict = lambda x: predict_imagenet(x, pretrained_model)
# dataset = 'imagenet'

data_path = './data/butterfly/test'
new_path = './data/butterfly_val_100'
model_path = './data/butterfly/EfficientNetB0-100-(224 X 224)- 97.59.h5'
pretrained_model = keras.models.load_model(model_path, custom_objects={'F1_score':'F1_score'})
predict = lambda x: predict_butterfly(x, pretrained_model)
dataset = 'butterfly'


images = {}
class_files = os.listdir(data_path)
for clf in class_files:
    images[clf] = os.listdir(os.path.join(data_path, clf))
image_size=224

visited_images = set()
visited_classes = set()

count = 0
while count < 100:
    cls = random.choice(class_files)
    image_file = random.choice(images[cls])
    # print(cls, image_file)
    while image_file in visited_images or cls in visited_classes:
        cls = random.choice(class_files)
        image_file = random.choice(images[cls])
    image_path = os.path.join(data_path, cls, image_file)
    image, label = load_image(image_path, image_size)
    model = Model(predict, label, dataset = dataset)

    image_probs = model.predict(image)
    correct = image_probs[0][0][0] == label

    if correct:
        visited_images.add(image_path)
        visited_classes.add(cls)

        if not os.path.isdir(os.path.join(new_path, cls)):
            os.makedirs(os.path.join(new_path, cls))
        shutil.copyfile(image_path, os.path.join(new_path, cls, image_file))
        count += 1

