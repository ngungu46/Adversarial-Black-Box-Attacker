import os
import shutil
import xml.etree.ElementTree as ET
import random
import shutil

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

# data_path = '../data/imagenet_val'
# new_path = '../data/imagenet_val_100'
data_path = '../data/butterfly/test'
new_path = '../data/butterfly_val_100'

images = {}
class_files = os.listdir(data_path)
for clf in class_files:
    images[clf] = os.listdir(os.path.join(data_path, clf))

print(class_files)

visited_images = set()
visited_classes = set()

for i in range(100):
    cls = random.choice(class_files)
    image_file = random.choice(images[cls])
    print(cls, image_file)
    while image_file in visited_images or cls in visited_classes:
        cls = random.choice(class_files)
        image_file = random.choice(images[cls])
    image_path = os.path.join(data_path, cls, image_file)
    visited_images.add(image_path)
    visited_classes.add(cls)

    if not os.path.isdir(os.path.join(new_path, cls)):
        os.makedirs(os.path.join(new_path, cls))
    shutil.copyfile(image_path, os.path.join(new_path, cls, image_file))

