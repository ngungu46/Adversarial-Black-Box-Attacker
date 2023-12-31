import argparse, os, sys
import torch 
from pprint import pprint
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import time 
import json 
import keras

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
    
    img_size = 299 if args["dataset"] == "imagenet64" else 224
    
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
        is_torch = (args["dataset"] == "imagenet64")
        # pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).to(device) 
        # pretrained_model.eval()

        adv_path = get_random_image_path(args["dataset"], imgpath)
        
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
            verbose = args["verbose"], 
            is_torch = is_torch 
        )
            
        sys.stdout.write('\r')
        j = (i+1)/args["dataset_size"]
        sys.stdout.write("[%-20s] %d%% \n" % ('='*int(20*j), 100*j))
        sys.stdout.flush()
        time.sleep(0.25)
        
        # prepare image with pixel values in [0, 1] and shape (1, 299, 299, 3)
        raw_orig_image = Image.open(imgpath)
        raw_orig_image = tf.keras.preprocessing.image.img_to_array(raw_orig_image) / 255
        raw_orig_image = cv2.resize(raw_orig_image, dsize=(img_size, img_size))
        
        raw_adv_image = Image.open(adv_path)
        raw_adv_image = tf.keras.preprocessing.image.img_to_array(raw_adv_image) / 255
        raw_adv_image = cv2.resize(raw_adv_image, dsize=(img_size, img_size))
        
        
        if args["dataset"] == "imagenet64": 
            orig_image = torch.tensor(raw_orig_image, dtype=torch.float32).to(device)
            orig_image = torch.unsqueeze(orig_image, 0)
            
            adv_image = torch.tensor(raw_adv_image, dtype=torch.float32).to(device)
            adv_image = torch.unsqueeze(adv_image, 0)
            
            
        elif args["dataset"] == "butterflies_and_moths": 
            orig_image = tf.cast(raw_orig_image, tf.float32)
            orig_image = orig_image[None, ...]
            
            adv_image = tf.cast(raw_adv_image, tf.float32)
            adv_image = adv_image[None, ...]
        
        # calculate the original class index and choose a random adversarial index 
        y_orig_idx = torch.argmax(predict(orig_image, pretrained_model, is_torch)).detach().cpu().numpy().item()
        y_adv_idx = torch.argmax(predict(adv_image, pretrained_model, is_torch)).detach().cpu().numpy().item()
        
        print(f"Original/Adversarial Class : {image_idx2cls[y_orig_idx]}/{image_idx2cls[y_adv_idx]}") 
        
        if is_torch: 
            res, y_adv, succ, count, top_k_predictions = ex.attack(adv_image, y_adv_idx, orig_image) 
            save_tensor_as_image(orig_image, os.path.join(new_run_dir_name, f"{i+1}_original_{y_orig_idx}.jpg"))
            save_tensor_as_image(res, os.path.join(new_run_dir_name, f"{i+1}_adversarial_{y_adv_idx}.jpg"))
            
            orig_pred = torch.argmax(predict(orig_image, pretrained_model, is_torch)).detach().cpu().numpy().item()
            adv_pred = torch.argmax(predict(res, pretrained_model, is_torch)).detach().cpu().numpy().item()
            
            print(f"Original vs Adversarial Prediction : {image_idx2cls[orig_pred]} - {image_idx2cls[adv_pred]} ", end = "") 
            
            print(top_k_predictions)
            
            
            
            info = {
                "status" : "SUCCESS" if succ else "FAILED", 
                "filepath" : imgpath, 
                "original_label" : f"{y_orig_idx} - {image_idx2cls[y_orig_idx]}", 
                "adversarial_target" : f"{y_adv_idx} - {image_idx2cls[y_adv_idx]}", 
                "adversarial prediction" : f"{adv_pred} - {image_idx2cls[adv_pred]}", 
                "args" : args, 
                "num_queries_needed" : count, 
                "top_k_predictions" : {str(k):str(v) for (_, k, v) in top_k_predictions}, 
                "runtime" : f"{time.time() - now}", 
                
                "L_inf_distance1" : torch.norm(orig_image.squeeze().detach().cpu() - res.squeeze().detach().cpu(), p=torch.inf).item(), 
                
                "L_inf_distance2" : torch.norm(torch.norm(orig_image.squeeze().detach().cpu() - res.squeeze().detach().cpu(), p=2, dim=(-1)), p=torch.inf).item(), 
                
                "L_inf_distance3" : torch.norm(torch.norm(orig_image.squeeze().detach().cpu() - res.squeeze().detach().cpu(), p=1, dim=(-1)), p=torch.inf).item()
            }
        else: 
            res, y_adv, succ, count, top_k_predictions = ex.attack(torch.tensor(np.array(adv_image)), y_adv_idx, torch.tensor(np.array(orig_image))) 
            save_tensor_as_image(torch.tensor(np.array(orig_image)), os.path.join(new_run_dir_name, f"{i+1}_original_{y_orig_idx}.jpg"))
            save_tensor_as_image(res, os.path.join(new_run_dir_name, f"{i+1}_adversarial_{y_adv_idx}.jpg"))
            
            orig_pred = torch.argmax(predict(orig_image, pretrained_model, is_torch)).detach().cpu().numpy().item()
            adv_pred = torch.argmax(predict(res, pretrained_model, is_torch)).detach().cpu().numpy().item()
            
            print(f"Original vs Adversarial Prediction : {image_idx2cls[orig_pred]} - {image_idx2cls[adv_pred]} ", end = "") 
            
            print(top_k_predictions)
            
        
            info = {
                "status" : "SUCCESS" if succ else "FAILED", 
                "filepath" : imgpath, 
                "original_label" : f"{y_orig_idx} - {image_idx2cls[y_orig_idx]}", 
                "adversarial_target" : f"{y_adv_idx} - {image_idx2cls[y_adv_idx]}", 
                "adversarial prediction" : f"{adv_pred} - {image_idx2cls[adv_pred]}", 
                "args" : args, 
                "num_queries_needed" : count, 
                "top_k_predictions" : {str(image_idx2cls[k]):str(v) for (_, k, v) in top_k_predictions}, 
                "runtime" : f"{time.time() - now}", 
                
                "L_inf_distance1" : torch.norm(torch.tensor(np.array(orig_image)).squeeze().detach().cpu() - res.squeeze().detach().cpu(), p=torch.inf).item(), 
                
                "L_inf_distance2" : torch.norm(torch.norm(torch.tensor(np.array(orig_image)).squeeze().detach().cpu() - res.squeeze().detach().cpu(), p=2, dim=(-1)), p=torch.inf).item(), 
                
                "L_inf_distance3" : torch.norm(torch.norm(torch.tensor(np.array(orig_image)).squeeze().detach().cpu() - res.squeeze().detach().cpu(), p=1, dim=(-1)), p=torch.inf).item()
            }
        
        # res, y_adv, succ, count, top_k_predictions = ex.attack(adv_image, y_adv_idx, orig_image) 
        
        
        # # save the original and adversarial image into output files 
        # save_tensor_as_image(orig_image, os.path.join(new_run_dir_name, f"{i+1}_original_{y_orig_idx}.jpg"))
        # save_tensor_as_image(res, os.path.join(new_run_dir_name, f"{i+1}_adversarial_{y_adv_idx}.jpg"))
        
        
        # # forward the pass over the two images once more to check their top probabilities 
        # orig_pred = torch.argmax(predict(orig_image, pretrained_model)).detach().cpu().numpy().item()
        # adv_pred = torch.argmax(predict(res, pretrained_model)).detach().cpu().numpy().item()
        
        # print(f"Original vs Adversarial Prediction : {image_idx2cls[orig_pred]} - {image_idx2cls[adv_pred]} ", end = "") 
        
        # # save the L-infinity norm, probabilities of outputs, and number of queries into files
        # info = {
        #     "status" : "SUCCESS" if succ else "FAILED", 
        #     "filepath" : imgpath, 
        #     "original_label" : f"{y_orig_idx} - {image_idx2cls[y_orig_idx]}", 
        #     "adversarial_target" : f"{y_adv_idx} - {image_idx2cls[y_adv_idx]}", 
        #     "adversarial prediction" : f"{adv_pred} - {image_idx2cls[adv_pred]}", 
        #     "args" : args, 
        #     "num_queries_needed" : count, 
        #     "top_k_predictions" : {key:str(val) for (_, key, val) in top_k_predictions}, 
        #     "runtime" : f"{time.time() - now}", 
        #     "L_inf_distance1" : torch.norm(image.squeeze().detach().cpu() - res.squeeze().detach().cpu(), p=torch.inf).item(), 
            
        #     "L_inf_distance2" : torch.norm(torch.norm(image.squeeze().detach().cpu() - res.squeeze().detach().cpu(), p=2, dim=(-1)), p=torch.inf).item(), 
            
        #     "L_inf_distance3" : torch.norm(torch.norm(image.squeeze().detach().cpu() - res.squeeze().detach().cpu(), p=1, dim=(-1)), p=torch.inf).item()
        # }
        
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