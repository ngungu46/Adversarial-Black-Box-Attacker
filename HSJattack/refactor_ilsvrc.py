import os
import shutil
import xml.etree.ElementTree as ET

for img in os.listdir("ILSVRC/Data/CLS-LOC/val"):
    annotation = img.replace(".JPEG", ".xml")

    path = f"ILSVRC/Annotations/CLS-LOC/val/{annotation}"
    tree = ET.parse(path)
    root = tree.getroot()
    for object in root.findall("object"):
        # print(object.find("name").text)
        synset = object.find("name").text
        # print(synset)

    # print(f"ILSVRC/Data/CLS-LOC/val/{img}", f"imagenet_val/{synset}/{img}")
    if not os.path.isdir(f"imagenet_val/{synset}"):
        os.makedirs(f"imagenet_val/{synset}")
    shutil.copyfile(f"ILSVRC/Data/CLS-LOC/val/{img}", f"imagenet_val/{synset}/{img}")