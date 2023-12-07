import argparse, os, sys
import torch 
from pprint import pprint
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import time 
import json 

# import utility and helper functions/classes 
from src.model import *
from src.dataloader import * 
from src.utils import * 


cudnn.benchmark = True 

parser = argparse.ArgumentParser(description='cfg')

# add parser hyperparameter arguments for running in CLI 
parser.add_argument('--e_adv', default=0.05, type=float)
parser.add_argument('--e_0', default=0.5, type=float)
parser.add_argument('--sigma', default=0.001, type=float)
parser.add_argument('--n_samples', default=50, type=int)
parser.add_argument('--eps_decay', default=0.001, type=float)
parser.add_argument('--max_lr', default=0.01, type=float)
parser.add_argument('--min_lr', default=0.001, type=float)
parser.add_argument('--k', default=1, type=int)
parser.add_argument('--max_queries', default=20000, type=int)
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
        pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).to(device) 
        pretrained_model.eval()
        
        # instantiate NESattack class 
        ex = PartialInfoAttack(
            e_adv = args["e_adv"],
            e_0 = args["e_0"],
            sigma = args["sigma"],
            n_samples = args["n_samples"],
            eps_decay = args["eps_decay"],
            max_lr = args["max_lr"],
            min_lr = args["min_lr"],
            k = args["k"],
            max_queries = args["max_queries"], 
            model = pretrained_model, 
            verbose = args["verbose"]
        )
            
        sys.stdout.write('\r')
        j = (i+1)/args["dataset_size"]
        sys.stdout.write("[%-20s] %d%% \n" % ('='*int(20*j), 100*j))
        sys.stdout.flush()
        time.sleep(0.25)
        
        # prepare image with pixel values in [0, 1] and shape (1, 299, 299, 3)
        rawimage = np.array(Image.open(imgpath)) / 255
        rawimage = cv2.resize(rawimage, dsize=(299, 299))
        
        image = torch.tensor(rawimage, dtype=torch.float32).to(device)
        if image.size() == (299, 299): 
            # if image is grayscale, then repeat the dimensions 
            image = image.unsqueeze(-1) 
            image = image.repeat(1, 1, 3)
        image = torch.unsqueeze(image, 0)
        
        # calculate the original class index and choose a random adversarial index 
        y_adv_idx = random.randrange(0, len(image_cls2idx))
        y_orig_idx = torch.argmax(predict(image, pretrained_model)).detach().cpu().numpy().item()

        
        print(f"Original/Adversarial Class : {image_idx2cls[y_orig_idx]}/{image_idx2cls[y_adv_idx]}") 
        
        res, y_adv, succ, count, top_k_predictions = ex.attack(image, y_adv_idx, y_orig_idx) 
        
        
        # save the original and adversarial image into output files 
        save_tensor_as_image(image, os.path.join(new_run_dir_name, f"{i+1}_original_{y_orig_idx}.jpg"))
        save_tensor_as_image(res, os.path.join(new_run_dir_name, f"{i+1}_adversarial_{y_adv_idx}.jpg"))
        
        
        # forward the pass over the two images once more to check their top probabilities 
        orig_pred = torch.argmax(predict(image, pretrained_model)).detach().cpu().numpy().item()
        adv_pred = torch.argmax(predict(res, pretrained_model)).detach().cpu().numpy().item()
        
        print(f"Original vs Adversarial Prediction : {image_idx2cls[orig_pred]} - {image_idx2cls[adv_pred]} ", end = "") 
        
        # save the L-infinity norm, probabilities of outputs, and number of queries into files
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