import argparse, os, sys
import torch 
from pprint import pprint
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import time 
import json 
import onnx
import keras 


# import utility and helper functions/classes 
from src.model import *
from src.dataloader import * 
from src.utils import * 


cudnn.benchmark = True 

parser = argparse.ArgumentParser(description='cfg')

# add parser hyperparameter arguments for running in CLI 
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--target_eps', default=0.05, type=float)
parser.add_argument('--n_samples', default=100, type=int)
parser.add_argument('--sigma', default=0.001, type=float)
parser.add_argument('--max_queries', default=20000, type=int)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--dataset', default="imagenet_val", type=str)
parser.add_argument('--dataset_size', default=100, type=int)
parser.add_argument('--verbose', default=False, type=bool)


args = vars(parser.parse_args())

def set_device(): 
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    
    if device == "cuda": 
        print("Running on GPU")
    else: 
        print("Running on CPU")
    
    return device

def getModel(dataset, device): 
    if dataset == "imagenet64": 
        pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).to(device) 
        pretrained_model.eval()
        return pretrained_model 
    
    elif dataset == "butterflies_and_moths": 
        pretrained_model = keras.models.load_model(os.path.join("HSJattack", "EfficientNetB0_butterfly.h5"), custom_objects={'F1_score':'F1_score'}) 
        pretrained_model.trainable = False 
        return pretrained_model 
    
    else: 
        raise Exception("Not a valid dataset name.")

def main(): 
    
    now = time.time() 
    
    success = 0 
    
    # scan for output directory and create a new run name 
    curr_run_dirs = [int(name.split("_")[-1]) for name in os.listdir(os.path.join("NESattack", "outputs"))] 
    new_run_dir_name = os.path.join("NESattack", "outputs", f"run_{str(max(curr_run_dirs) + 1)}")
    os.mkdir(new_run_dir_name)
    
    print(f"Saving run to {new_run_dir_name}...")
    
    device = set_device() 
    
    # get a random subset of "dataset_size" validation dataset image filepaths and store them in a list 
    image_filepaths = getSubset(args["dataset"], args["dataset_size"], True)
    
    # get class2index and index2 class dictionaries for easy querying 
    image_cls2idx = class2index(args["dataset"]) 
    image_idx2cls = {v: k for k, v in image_cls2idx.items()}
    
    for i, imgpath in enumerate(image_filepaths): 
        
        # load pretrained inception v3 model
        # this must be called inside the loop since we end up changing the weights of the model somehow within this loop 
        pretrained_model = getModel(args["dataset"], device)
        
        
        # instantiate NESattack class 
        ex = NESAttack(
            target_eps = args["target_eps"],
            lr = args["lr"],
            n_samples = args["n_samples"],
            sigma = args["sigma"],
            max_queries = args["max_queries"], 
            momentum=args["momentum"], 
            model = pretrained_model, 
            verbose = bool(args["verbose"]), 
            is_torch = (args["dataset"] == "imagenet64")
        )
        
        sys.stdout.write('\r')
        j = (i+1)/args["dataset_size"]
        sys.stdout.write("[%-20s] %d%% \n" % ('='*int(20*j), 100*j))
        sys.stdout.flush()
        time.sleep(0.25)
        
        # prepare image with pixel values in [0, 1] and shape (1, 299, 299, 3)
        rawimage = Image.open(imgpath)
        rawimage = tf.keras.preprocessing.image.img_to_array(rawimage) / 255
        rawimage = cv2.resize(rawimage, dsize=(299, 299))
        
        
        # now depending on model, convert to torch.tensor or tf.tensor
        if args["dataset"] == "imagenet64": 
            image = torch.tensor(rawimage, dtype=torch.float32).to(device)
            image = torch.unsqueeze(image, 0)
            orig_prediction = predict(image, pretrained_model)
        elif args["dataset"] == "butterflies_and_moths": 
            image = tf.cast(rawimage, tf.float32)
            image = image[None, ...]
            pass 
        
        # calculate the original class index and choose a random adversarial index 
        y_adv_idx = random.randrange(0, len(image_cls2idx))
        y_orig_idx = torch.argmax(orig_prediction).detach().cpu().numpy().item()

        
        print(f"Original/Adversarial Class : {image_idx2cls[y_orig_idx]}/{image_idx2cls[y_adv_idx]}") 
        
        res, _, prob, count, succ, top_k_predictions = ex.attack(image, y_adv_idx) 
        
        
        # save the original and adversarial image into output files 
        save_tensor_as_image(image, os.path.join(new_run_dir_name, f"{i+1}_original_{y_orig_idx}.jpg"))
        save_tensor_as_image(res, os.path.join(new_run_dir_name, f"{i+1}_adversarial_{y_adv_idx}.jpg"))
        
        # save the L-infinity norm, probabilities of outputs, and number of queries into files
        
        # forward the pass over the two images once more to check their top probabilities 
        orig_pred = torch.argmax(predict(image, pretrained_model)).detach().cpu().numpy().item()
        adv_pred = torch.argmax(predict(res, pretrained_model)).detach().cpu().numpy().item()
        
        print(f"Original vs Adversarial Prediction : {image_idx2cls[orig_pred]} - {image_idx2cls[adv_pred]} ", end = "") 
        
        info = {
            "status" : "SUCCESS" if succ else "FAILED", 
            "filepath" : imgpath, 
            "original_label" : f"{y_orig_idx} - {image_idx2cls[y_orig_idx]}", 
            "adversarial_target" : f"{y_adv_idx} - {image_idx2cls[y_adv_idx]}", 
            "adversarial prediction" : f"{adv_pred} - {image_idx2cls[adv_pred]}", 
            "args" : args, 
            "num_queries_needed" : count, 
            "top_k_predictions" : {key:str(val) for (_, key, val) in top_k_predictions}, 
            "runtime" : f"{time.time() - now}", 
            "L_inf_distance1" : torch.norm(image.squeeze().detach().cpu() - res.squeeze().detach().cpu(), p=torch.inf).item(), 
            
            "L_inf_distance2" : torch.norm(torch.norm(image.squeeze().detach().cpu() - res.squeeze().detach().cpu(), p=2, dim=(-1)), p=torch.inf).item(), 
            
            "L_inf_distance3" : torch.norm(torch.norm(image.squeeze().detach().cpu() - res.squeeze().detach().cpu(), p=1, dim=(-1)), p=torch.inf).item()
        }
        
        with open(os.path.join(new_run_dir_name, f"{i+1}_stats.json"), "w") as f: 
            json.dump(info, f, indent=4) 
        
        # increment the success_count 
        if succ: 
            success += 1 
            print("SUCCESS")
        else: 
            print("FAILED")
            
    print(f"SUCCESS RATE = {success/len(image_filepaths)}")
    print(f"TOTAL RUNTIME: {time.time() - now}")

main() 