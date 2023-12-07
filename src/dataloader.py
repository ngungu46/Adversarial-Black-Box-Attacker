import os 
from random import shuffle

def find_jpeg_files(directory, num_samples, shuff=True): 
    
    subset_image_filepaths = []
    
    if directory == "imagenet64": 
        directory = os.path.join("data", directory, "val")
    elif directory == "butterflies_and_moths": 
        directory = os.path.join("data", directory, "valid")
    
    all_image_filepaths = os.walk(directory) if shuff is True else os.walk(directory)
    
    for root, dirs, files in all_image_filepaths:
        for file in files:
            if file.lower().endswith('.jpeg') or file.lower().endswith('.jpg'):
                subset_image_filepaths.append(os.path.join(root, file)) 
                if len(subset_image_filepaths) >= num_samples: 
                    return subset_image_filepaths
                
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
    


        
