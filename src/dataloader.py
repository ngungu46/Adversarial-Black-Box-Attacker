import os 
from random import shuffle

def get_random_image_path(directory, orig_path):
    if directory == "imagenet64": 
        directory = os.path.join("data", directory, "val")
    elif directory == "butterflies_and_moths": 
        directory = os.path.join("data", directory, "valid")

    cls = orig_path.split('/')[-2]
    classfiles = os.listdir(directory)
    shuffle(classfiles)
    if classfiles[0] == cls:
        adv_cls = classfiles[1]
    else:
        adv_cls = classfiles[0]

    data_path = os.path.join(directory, adv_cls)
    img_paths = os.listdir(data_path)
    shuffle(img_paths)
    img_path = img_paths[0]
    f = os.path.join(data_path, img_path)

    return f

def find_jpeg_files(directory, num_samples, shuff=True): 
    
    subset_image_filepaths = []
    
    if directory == "imagenet64": 
        directory = os.path.join("data", directory, "val")
    elif directory == "butterflies_and_moths": 
        directory = os.path.join("data", directory, "valid")
    
    classfiles = os.listdir(directory)
    shuffle(classfiles)
    for i in range(num_samples):
        cls = classfiles[i]
        data_path = os.path.join(directory, cls)
        img_paths = os.listdir(data_path)
        shuffle(img_paths)
        img_path = img_paths[0]
        f = os.path.join(data_path, img_path)
        if f.lower().endswith('.jpeg') or f.lower().endswith('.jpg'):
            subset_image_filepaths.append(f) 
                
    return subset_image_filepaths
            

def getSubset(dataset, num_samples, shuff=True): 
    # gets a subset of num_samples from the specified dataset 
    

    if dataset == "imagenet64": 
        
        filepaths = find_jpeg_files(dataset, num_samples, shuff)
        
        pass 
    
    elif dataset == "imagenet_val": 
        
        filepaths = find_jpeg_files(dataset, num_samples, shuff)
        
        pass 
    
    elif dataset == "butterflies_and_moths": 
        
        filepaths = find_jpeg_files(dataset, num_samples, shuff)
        pass 
    
    else: 
        Exception(f"Dataset not available. ")
        
    return filepaths


def class2index(dataset): 
    if dataset == "imagenet64": 
        with open(os.path.join('NESattack', 'imagenet_classes.txt')) as f:
            classes = [line.strip() for line in f.readlines()]

        classes = {cls: i for i, cls in enumerate(classes)}
        
        return classes 

    elif dataset == "butterflies_and_moths": 
        class_names = sorted(os.listdir(os.path.join("data", "butterflies_and_moths", "train"))) 
        
        classes = {cls: i for i, cls in enumerate(class_names)}
        
        return classes 
    


        
