import torch
from torch import nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import math
from copy import deepcopy
import matplotlib.pyplot as plt
device = torch.device('cuda:0')

from AAA.utils import loss
from AAA.models import AAALinear

import os

from PIL import Image
import tensorflow as tf

# pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)

# from torchvision.models import resnet50, ResNet50_Weights
# weights = ResNet50_Weights.DEFAULT
# pretrained_model = resnet50(weights=weights)
pretrained_model = getattr(torchvision.models, 'resnet50')(pretrained=True).to(device).eval()

pretrained_model = pretrained_model.to("cuda")
pretrained_model.eval()

def torch_transform(image):
    """
    input: numpy images of shape (B, H, W, C), normalized to (0, 1)
    output: tensor of images of shape (B, C, H, W), normalized to mean [.485, .456, .406], std [.229, .224, .225]
    """

    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = transform(image)
    
    return image

def predict(image):
    """
    input: normalized tensor of shape (B, C, H, W)
    output: numpy array of predictions
    """
    
    with torch.no_grad():
        preds = pretrained_model(image)
    return preds


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
    temperature=1.1236,
    verbose=False,
)

data_path = "./HSJattack/imagenet_val_subset" #path of validation data

images = []
class_files = os.listdir(data_path)

verbose = False
image_size = 224

for clf in class_files:
    class_path = os.path.join(data_path,clf)
    for file_name in os.listdir(class_path):
        image_path = f'{class_path}/{file_name}'

        raw_image = Image.open(image_path)

        transform = transforms.PILToTensor()
        image = transform(raw_image)

        image = image.to(dtype=torch.float32)
        image = image / 255
        transform = transforms.Resize(size=(image_size, image_size))
        image = transform(image)

        # # Add batch dimension
        # # 1, 299, 299, 3
        # image = image[None, ...]

        images.append(image)

images = torch.stack(images, dim=0)
images = torch_transform(images)

print(images.shape)

logits, logits_ori = defender(images, return_original=True)

top, ind = torch.topk(logits_ori, k=6)
top_new, ind_new = torch.topk(logits, k=6)
# print(logits[:,:70])
# print(logits_out[:,:70])
print(top, ind)
# print(top_new, ind_new)
print(np.array([logits[i,ind[i]].tolist() for i in range(len(class_files))]))

# class AAASine(nn.Module):
#     def __init__(self, dataset, arch, norm, model_dir, 
#         device=device, batch_size=1000, attractor_interval=6, reverse_step=1, num_iter=100, calibration_loss_weight=5, optimizer_lr=0.1, do_softmax=False, **kwargs):
#         super(AAASine, self).__init__()
#         self.dataset = dataset
#         try: 
#             self.cnn = getattr(torchvision.models, arch)(pretrained=True).to(device).eval()
#             self.mean = [0.485, 0.456, 0.406]
#             self.std = [0.229, 0.224, 0.225]
#         except AttributeError: 
#             self.cnn = load_model(model_name=arch, dataset=dataset, threat_model=norm, model_dir=model_dir).to(device).eval()
#             self.mean = [0] #if dataset != 'imagenet' else [0, 0, 0]
#             self.std = [1] #if dataset != 'imagenet' else [1, 1, 1]
#             self.cnn.to(device)
        
#         self.loss = loss
#         self.batch_size = batch_size
#         self.device = device

#         self.attractor_interval = attractor_interval
#         self.reverse_step = reverse_step
#         self.dev = 0.5
#         self.optimizer_lr = optimizer_lr
#         self.calibration_loss_weight = calibration_loss_weight
#         self.num_iter = num_iter
#         self.arch_ori = arch
#         self.arch = '%s_AAAsine-Lr-%.1f-Ai-%d-Cw-%d' % (self.arch_ori, self.reverse_step, self.attractor_interval, self.calibration_loss_weight)
#         self.temperature = 1 # 2.08333 #
#         self.do_softmax = do_softmax

#     def set_hp(self, reverse_step, attractor_interval=6, calibration_loss_weight=5):
#         self.attractor_interval = attractor_interval
#         self.reverse_step = reverse_step
#         self.calibration_loss_weight = calibration_loss_weight
#         self.arch = '%s_AAAsine-Lr-%.1f-Ai-%d-Cw-%d' % (self.arch_ori, self.reverse_step, self.attractor_interval, self.calibration_loss_weight)

#     def forward_undefended(self, x): return predict(x, self.cnn, self.batch_size, self.device, self.mean, self.std)
    
#     def get_tuned_temperature(self):
#         t_dict = {
#             'Standard': 2.08333,
#             'resnet50': 1.1236,
#             'resnext101_32x8d': 1.26582,
#             'vit_b_16': 0.94,
#             'wide_resnet50_2': 1.20482,
#             'Rebuffi2021Fixing_28_10_cutmix_ddpm': 0.607,
#             'Salman2020Do_50_2': 0.83,
#             'Dai2021Parameterizing': 0.431,
#             'Rade2021Helper_extra': 0.58
#         }
#         return t_dict.get(self.arch_ori, None)

#     def temperature_rescaling(self, x_val, y_val, step_size=0.001):
#         ts, eces = [], []
#         ece_best, y_best = 100, 1
#         y_pred = self.forward_undefended(x_val)
#         for t in np.arange(0, 1, step_size):
#             y_pred1 = y_pred / t
#             y_pred2 = y_pred * t

#             ts += [t, 1/t]
#             ece1, ece2 = ece_score(y_pred1, y_val), ece_score(y_pred2, y_val)
#             eces += [ece1, ece2]
#             if ece1 < ece_best: 
#                 ece_best = ece1
#                 t_best = t
#             if ece2 < ece_best: 
#                 ece_best = ece2
#                 t_best = 1/t
#             print('t-curr=%.3f, acc=%.2f, %.2f, ece=%.4f, %.4f, t-best=%.5f, ece-best=%.4f' % 
#             (t, (y_pred1.argmax(1) == y_val.argmax(1)).mean() * 100, (y_pred2.argmax(1) == y_val.argmax(1)).mean() * 100, 
#             ece1 * 100, ece2 * 100,
#             t_best, ece_best * 100))
#         self.temperature = t_best

#         plt.rcParams["figure.dpi"] = 500
#         plt.rcParams["font.family"] = "times new roman"
#         plt.scatter(ts, eces, color='#9467bd')
#         plt.xscale('log')
#         plt.xlabel('temperature')
#         plt.ylabel('ece on validation set')
#         plt.savefig('demo/t-%s-%.4f.png' % (self.arch, self.temperature))
#         plt.close()

#     def temperature_rescaling_with_aaa(self, x_val, y_val, step_size=0.001):
#         self.temperature = self.get_tuned_temperature()
#         if self.temperature is not None: return

#         ts, eces = [], []
#         ece_best, y_best = 100, 1
#         for t in np.arange(0, 1, step_size):
#             self.temperature = t
#             y_pred1 = self.forward(x_val)
#             self.temperature = 1/t
#             y_pred2 = self.forward(x_val)

#             ts += [t, 1/t]
#             ece1, ece2 = ece_score(y_pred1, y_val), ece_score(y_pred2, y_val)
#             eces += [ece1, ece2]
#             if ece1 < ece_best: 
#                 ece_best = ece1
#                 t_best = t
#             if ece2 < ece_best: 
#                 ece_best = ece2
#                 t_best = 1/t
#             print('t-curr=%.3f, acc=%.2f, %.2f, ece=%.4f, %.4f, t-best=%.5f, ece-best=%.4f' % 
#             (t, (y_pred1.argmax(1) == y_val.argmax(1)).mean() * 100, (y_pred2.argmax(1) == y_val.argmax(1)).mean() * 100, 
#             ece1 * 100, ece2 * 100,
#             t_best, ece_best * 100))
#         self.temperature = t_best

#         plt.rcParams["figure.dpi"] = 500
#         plt.rcParams["font.family"] = "times new roman"
#         plt.scatter(ts, eces, color='#9467bd')
#         plt.xscale('log')
#         plt.xlabel('temperature')
#         plt.ylabel('ece on validation set')
#         plt.savefig('demo/taaa-%s-%.4f.png' % (self.arch, self.temperature))
#         plt.close()

#     def forward(self, x):
#         if isinstance(x, np.ndarray): 
#             x = np.floor(x * 255.0) / 255.0
#             x = ((x - np.array(self.mean)[np.newaxis, :, np.newaxis, np.newaxis]) / np.array(self.std)[np.newaxis, :, np.newaxis, np.newaxis]).astype(np.float32)
#         else: 
#             x = torch.floor(x * 255.0) / 255.0
#             x = ((x - torch.as_tensor(self.mean, device=self.device)[None, :, None, None]) / torch.as_tensor(self.std, device=self.device)[None, :, None, None])
#         n_batches = math.ceil(x.shape[0] / self.batch_size)
#         logits_list = []

#         for counter in range(n_batches):
#             with torch.no_grad():
#                 if verbose: print('predicting', counter, '/', n_batches, end='\r')
#                 x_curr = x[counter * self.batch_size:(counter + 1) * self.batch_size]
#                 if isinstance(x, np.ndarray): x_curr = torch.as_tensor(x_curr, device=self.device) 
#                 logits = self.cnn(x_curr)
            

#             logits_ori = logits.detach()
#             prob_ori = F.softmax(logits_ori / self.temperature, dim=1)
#             prob_max_ori = prob_ori.max(1)[0] ###
#             value, index_ori = torch.topk(logits_ori, k=2, dim=1)
#             #"""
#             mask_first = torch.zeros(logits.shape, device=self.device)
#             mask_first[torch.arange(logits.shape[0]), index_ori[:, 0]] = 1
#             mask_second = torch.zeros(logits.shape, device=self.device)
#             mask_second[torch.arange(logits.shape[0]), index_ori[:, 1]] = 1
#             #"""
            
#             margin_ori = value[:, 0] - value[:, 1]
#             attractor = ((margin_ori / self.attractor_interval + self.dev).round() - self.dev) * self.attractor_interval
#             #target = attractor - self.reverse_step * (margin_ori - attractor)

#             target = margin_ori - 0.7 * self.attractor_interval * torch.sin(
#                 (1 - 2 / self.attractor_interval * (margin_ori - attractor)) * torch.pi)
#             diff_ori = (margin_ori - target)
#             real_diff_ori = margin_ori - attractor

#             with torch.enable_grad():
#                 logits.requires_grad = True
#                 optimizer = torch.optim.Adam([logits], lr=self.optimizer_lr)
#                 i = 0 
#                 los_reverse_rate = 0
#                 prd_maintain_rate = 0
#                 for i in range(self.num_iter):
#                 #while i < self.num_iter or los_reverse_rate != 1 or prd_maintain_rate != 1:
#                     prob = F.softmax(logits, dim=1)
#                     #loss_calibration = (prob.max(1)[0] - prob_max_ori).abs().mean()
#                     loss_calibration = ((prob * mask_first).max(1)[0] - prob_max_ori).abs().mean() # better
#                     #loss_calibration = (prob - prob_ori).abs().mean()

#                     value, index = torch.topk(logits, k=2, dim=1) 
#                     margin = value[:, 0] - value[:, 1]
#                     #margin = (logits * mask_first).max(1)[0] - (logits * mask_second).max(1)[0]

#                     diff = (margin - target)
#                     real_diff = margin - attractor
#                     loss_defense = diff.abs().mean()
                    
#                     loss = loss_defense + loss_calibration * self.calibration_loss_weight
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()

#                     #i += 1
#                     los_reverse_rate = ((real_diff * real_diff_ori) < 0).float().mean()
#                     prd_maintain_rate = (index_ori[:, 0] == index[:, 0]).float().mean()
#                     #print('%d, %.2f, %.2f' % (i, los_reverse_rate * 100, prd_maintain_rate * 100), end='\r')
#                     #print('%d, %.4f, %.4f, %.4f' % (itre, loss_calibration, loss_defense, loss))
#                 logits_list.append(logits.detach().cpu())

#         logits = torch.vstack(logits_list)
#         if isinstance(x, np.ndarray): logits = logits.numpy()
#         if self.do_softmax: logits = softmax(logits)
#         return logits
    

