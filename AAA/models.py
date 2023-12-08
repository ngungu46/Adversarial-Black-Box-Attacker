import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import math

class AAALinear(nn.Module):
    def __init__(self, 
            pretrained_model,
            loss,
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
        ):
        super(AAALinear, self).__init__()
        self.pretrained_model = pretrained_model
        self.loss = loss
        self.batch_size = batch_size
        self.device = device

        self.attractor_interval = attractor_interval
        self.reverse_step = reverse_step
        self.dev = 0.5
        self.optimizer_lr = optimizer_lr
        self.calibration_loss_weight = calibration_loss_weight
        self.num_iter = num_iter
        self.temperature = temperature
        self.do_softmax = do_softmax
        self.verbose = verbose

    def forward_undefended(self, x): 
        return self.pretrained_model(x)

    def forward(self, x, return_original = False):
        n_batches = math.ceil(x.shape[0] / self.batch_size)
        logits_list = []

        for counter in range(n_batches):
            with torch.no_grad():
                if self.verbose: 
                    print('predicting', counter, '/', n_batches, end='\r')

                x_curr = x[counter * self.batch_size:(counter + 1) * self.batch_size]
                logits = self.pretrained_model(x_curr)

            if type(logits) == np.ndarray:
                logits = torch.from_numpy(logits)
            
            logits = logits.to(self.device)

            # print(torch.topk(F.softmax(logits), k=4, dim=1))

            if return_original:
                logits_out = logits.detach()

            logits_ori = logits.detach()
            prob_ori = F.softmax(logits_ori / self.temperature, dim=1)
            prob_max_ori = prob_ori.max(1)[0] ###
            value, index_ori = torch.topk(logits_ori, k=2, dim=1)
            #"""
            mask_first = torch.zeros(logits.shape, device=self.device)
            mask_first[torch.arange(logits.shape[0]), index_ori[:, 0]] = 1
            #"""
            
            margin_ori = value[:, 0] - value[:, 1]
            attractor = ((margin_ori / self.attractor_interval + self.dev).round() - self.dev) * self.attractor_interval
            target = attractor - self.reverse_step * (margin_ori - attractor)
            #"""
            with torch.enable_grad():
                logits.requires_grad = True
                optimizer = torch.optim.Adam([logits], lr=self.optimizer_lr)
                for i in range(self.num_iter):
                    prob = F.softmax(logits, dim=1)
                    loss_calibration = ((prob * mask_first).max(1)[0] - prob_max_ori).abs().mean()

                    # value_ori = logits[torch.arange(logits.shape[0]),index_ori[:,0]]

                    # subtract = torch.zeros_like(logits)
                    # subtract[torch.arange(logits.shape[0]),index_ori[:,0]] = -1000000
                    # value, index = torch.max(logits + subtract, dim=1)
                    
                    # margin = value_ori - value

                    value, index = torch.topk(logits, k=2, dim=1) 
                    margin = value[:, 0] - value[:, 1]

                    # # print(margin.shape, index.shape, index_ori.shape)
                    # if torch.any(index != index_ori):
                    #     print(index, index_ori)

                    # #maybe this works?
                    # margin = torch.where(index[:,0] == index_ori[:,0], margin, -margin)

                    diff = (margin - target)
                    loss_defense = diff.abs().mean()
                    
                    loss = loss_defense + loss_calibration * self.calibration_loss_weight
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                logits_list.append(logits.detach().cpu())

        logits = torch.vstack(logits_list)
        # print(torch.topk(F.softmax(logits), k=4, dim=1))

        if self.do_softmax: 
            logits = F.softmax(logits)

            if return_original:
                logits_out = F.softmax(logits_out)

        logits = logits.cpu().numpy()
        if return_original:
            logits_out = logits_out.cpu().numpy()

        if return_original:
            return logits, logits_out
        else:
            return logits
        

class AAASine(nn.Module):
    def __init__(self, 
            pretrained_model,
            loss,
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
        ):
        super(AAASine, self).__init__()
        self.pretrained_model = pretrained_model
        self.loss = loss
        self.batch_size = batch_size
        self.device = device

        self.attractor_interval = attractor_interval
        self.reverse_step = reverse_step
        self.dev = 0.5
        self.optimizer_lr = optimizer_lr
        self.calibration_loss_weight = calibration_loss_weight
        self.num_iter = num_iter
        self.temperature = temperature
        self.do_softmax = do_softmax
        self.verbose = verbose

    def forward_undefended(self, x): 
        return self.pretrained_model(x)

    def forward(self, x, return_original = False):
        n_batches = math.ceil(x.shape[0] / self.batch_size)
        logits_list = []

        for counter in range(n_batches):
            with torch.no_grad():
                if self.verbose: 
                    print('predicting', counter, '/', n_batches, end='\r')

                x_curr = x[counter * self.batch_size:(counter + 1) * self.batch_size]
                logits = self.pretrained_model(x_curr)

            if type(logits) == np.ndarray:
                logits = torch.from_numpy(logits)
            
            logits = logits.to(self.device)

            # print(torch.topk(F.softmax(logits), k=4, dim=1))

            if return_original:
                logits_out = logits.detach()

            logits_ori = logits.detach()
            prob_ori = F.softmax(logits_ori / self.temperature, dim=1)
            prob_max_ori = prob_ori.max(1)[0] ###
            value, index_ori = torch.topk(logits_ori, k=2, dim=1)
            #"""
            mask_first = torch.zeros(logits.shape, device=self.device)
            mask_first[torch.arange(logits.shape[0]), index_ori[:, 0]] = 1
            #"""
            
            margin_ori = value[:, 0] - value[:, 1]
            attractor = ((margin_ori / self.attractor_interval + self.dev).round() - self.dev) * self.attractor_interval
            target = margin_ori - self.reverse_step * self.attractor_interval * torch.sin(
                (1 - 2 / self.attractor_interval * (margin_ori - attractor)) * torch.pi)
            
            #"""
            with torch.enable_grad():
                logits.requires_grad = True
                optimizer = torch.optim.Adam([logits], lr=self.optimizer_lr)
                for i in range(self.num_iter):
                    prob = F.softmax(logits, dim=1)
                    loss_calibration = ((prob * mask_first).max(1)[0] - prob_max_ori).abs().mean()

                    # value_ori = logits[torch.arange(logits.shape[0]),index_ori[:,0]]

                    # subtract = torch.zeros_like(logits)
                    # subtract[torch.arange(logits.shape[0]),index_ori[:,0]] = -1000000
                    # value, index = torch.max(logits + subtract, dim=1)
                    
                    # margin = value_ori - value

                    value, index = torch.topk(logits, k=2, dim=1) 
                    margin = value[:, 0] - value[:, 1]

                    # # print(margin.shape, index.shape, index_ori.shape)
                    # if torch.any(index != index_ori):
                    #     print(index, index_ori)

                    # #maybe this works?
                    # margin = torch.where(index[:,0] == index_ori[:,0], margin, -margin)

                    diff = (margin - target)
                    loss_defense = diff.abs().mean()
                    
                    loss = loss_defense + loss_calibration * self.calibration_loss_weight
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                logits_list.append(logits.detach().cpu())

        logits = torch.vstack(logits_list)
        # print(torch.topk(F.softmax(logits), k=4, dim=1))

        if self.do_softmax: 
            logits = F.softmax(logits)

            if return_original:
                logits_out = F.softmax(logits_out)

        logits = logits.cpu().numpy()
        if return_original:
            logits_out = logits_out.cpu().numpy()

        if return_original:
            return logits, logits_out
        else:
            return logits