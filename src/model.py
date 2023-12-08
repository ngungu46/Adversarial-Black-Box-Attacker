import torch
import numpy as np
from src.utils import *
import time as time
from torchvision import transforms


class NESAttack:
    def __init__(
        self,
        target_eps,
        lr,
        n_samples,
        sigma,
        max_queries,
        momentum,
        model,
        verbose, 
        is_torch
    ):
        self.lr = lr
        self.target_eps = target_eps
        self.n_samples = n_samples
        self.sigma = sigma
        self.max_queries = max_queries
        self.momentum = momentum
        self.model = model
        self.verbose = verbose
        self.is_torch = is_torch

    def NES(self, x_orig, y_adv): 
        """
        x: np.ndarray
        y_class: str
        sigma: float
        n_samples: int
        img_dim: tuple
        classifier: function
        model: tf.keras.Model
        k: int
        """

        if self.is_torch: 
            _, r, d, _ = x_orig.shape
            noise = torch.normal(mean = 0, std = 1, size = (self.n_samples//2, 3, r, d))
            noise = torch.cat([noise, -noise], axis = 0).cuda()
            x_orig = transforming(x_orig, self.is_torch)
            x = x_orig.repeat((self.n_samples, 1, 1, 1)).cuda()
            x += noise * self.sigma

            predictions = self.model(x)
            prob = torch.nn.functional.softmax(predictions).detach()[:, y_adv]
            prob = prob[:, None, None, None]
            g = prob * noise
            return g.sum(dim = 0).unsqueeze(0) / (self.n_samples)
        else: 
            _, r, d, _ = x_orig.shape
            noise = torch.normal(mean = 0, std = 1, size = (self.n_samples//2, 3, r, d))
            noise = torch.cat([noise, -noise], axis = 0).cuda()
            x_orig = transforming(x_orig, self.is_torch)
            x = x_orig.repeat((self.n_samples, 1, 1, 1)).cuda()
            x += noise * self.sigma
            x = x.detach().cpu().numpy() 
            x = x.transpose(0, 2, 3, 1)
            x = tf.cast(x, tf.float32)
            
            predictions = self.model.predict(x * 255, steps=1) 
            predictions = torch.tensor(np.array(predictions)).cuda()
                
            prob = torch.nn.functional.softmax(predictions).detach()[:, y_adv]
            prob = prob[:, None, None, None]
            g = prob * noise
            return g.sum(dim = 0).unsqueeze(0) / (self.n_samples)
        

    def attack(self, x_orig, y_adv): 
        x_orig = x_orig.cpu().detach()
        count = 0
        x_adv = x_orig
        upper = x_orig + self.target_eps
        lower = x_orig - self.target_eps
        grad = torch.zeros((1, 3, 299, 299))
        top_k_predictions = None 
        while count < self.max_queries:
            prev_grad = grad.cuda()
            grad = self.NES(x_adv, y_adv) 
            count += self.n_samples
            x_adv = x_adv + self.lr * torch.sign(grad).cpu().detach().numpy().transpose(0, 2, 3, 1)
            self.lr *= 0.99
            x_adv = np.clip(x_adv, lower, upper)
            log_probability_predictions = predict(x_adv, self.model, self.is_torch)
            cls = torch.argmax(log_probability_predictions).detach().cpu().numpy().item()
            
            
            if self.is_torch: 
                probs = torch.nn.functional.softmax(log_probability_predictions).cpu().detach().numpy()
                top_k_predictions = decode_predictions(probs, top = 10)[0]
            else: 
                probs = log_probability_predictions.cpu().detach().numpy()
                top_k_labels = list((np.argsort(probs*-1))[0])
                
                top_k_predictions = [("dummy", label, probs[0][label]) for label in top_k_labels][:5]
            
            if self.verbose: 
                print([x for x in top_k_predictions])
                print(y_adv, probs[0][y_adv])
            
            if(cls == y_adv):
                return x_adv, cls, probs[:, y_adv], count, True, top_k_predictions, torch.abs(x_orig - x_adv).max()
            # print(probs[:, y_adv])
            

        return x_adv, -1, probs[:, y_adv], -1, False, top_k_predictions


class PartialInfoAttack:
    def __init__(
        self,
        e_adv,
        e_0,
        sigma,
        n_samples ,
        eps_decay,
        max_lr,
        min_lr,
        k,
        max_queries, 
        model, 
        verbose, 
        is_torch
    ):
        self.e_adv = e_adv
        self.e_0 = e_0
        self.sigma = sigma
        self.n_samples = n_samples
        self.eps_decay = eps_decay
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.k = k
        self.max_queries = max_queries
        self.model = model 
        self.verbose = verbose
        self.is_torch = is_torch
        
    def NES(self, x_orig, y_adv): 
        """
        x: np.ndarray
        y_class: str
        sigma: float
        n_samples: int
        img_dim: tuple
        classifier: function
        model: tf.keras.Model
        k: int
        """
        if self.is_torch: 
            _, r, d, _ = x_orig.shape
            noise = torch.normal(mean = 0, std = 1, size = (self.n_samples//2, 3, r, d))
            noise = torch.cat([noise, -noise], axis = 0).cuda()
            x_orig = transforming(x_orig, self.is_torch)
            x = x_orig.repeat((self.n_samples, 1, 1, 1)).cuda()
            x += noise * self.sigma
            
            predictions = self.model(x)
            prob = torch.nn.functional.softmax(predictions).detach()[:, y_adv]
            prob = prob[:, None, None, None]
            g = prob * noise
            return g.sum(dim = 0).unsqueeze(0) / (self.n_samples)

        else: 
            _, r, d, _ = x_orig.shape
            noise = torch.normal(mean = 0, std = 1, size = (self.n_samples//2, 3, r, d))
            noise = torch.cat([noise, -noise], axis = 0).cuda()
            x_orig = transforming(x_orig, self.is_torch)
            x = x_orig.repeat((self.n_samples, 1, 1, 1)).cuda()
            x += noise * self.sigma
            x = x.detach().cpu().numpy() 
            x = x.transpose(0, 2, 3, 1)
            x = tf.cast(x, tf.float32)
            
            predictions = self.model.predict(x * 255, steps=1) 
            predictions = torch.tensor(np.array(predictions)).cuda()
                
            prob = torch.nn.functional.softmax(predictions).detach()[:, y_adv]
            prob = prob[:, None, None, None]
            g = prob * noise
            return g.sum(dim = 0).unsqueeze(0) / (self.n_samples)

    def attack(
        self,
        x_adv,
        y_adv,
        x_orig
    ):
        """
        x_adv: np.ndarray
        y_adv: str
        classifier: function
        x_orig: np.ndarray
        """
        x_orig = x_orig.cpu().detach()
        epsilon = self.e_0

        lower = torch.clamp(x_orig - epsilon, 0, 1)
        upper = torch.clamp(x_orig + epsilon, 0, 1)
        x_adv = torch.clamp(x_adv, lower, upper)
        count = 0
        for i in range(self.max_queries//self.n_samples):
           # print(count, epsilon, torch.max(x_adv - x_orig))
            g = self.NES(
                x_adv,
                y_adv
            )
            count += self.n_samples
            lr = self.max_lr
            g = torch.sign(g).transpose(1, 2).transpose(2, 3)
            # print(g.shape, x_adv.shape)
            hat_x_adv = x_adv + lr * g
            prop_de = self.eps_decay
            # print(x_adv.shape)
            while epsilon > self.e_adv:
                proposed_epsilon = max(epsilon - prop_de, self.e_adv)
                lower = torch.clamp(x_orig - proposed_epsilon, 0, 1)
                upper = torch.clamp(x_orig + proposed_epsilon, 0, 1)
                hat_x_adv = torch.clamp(x_adv + lr * g, lower, upper)
                
                
                # probs = self.model(transforming(hat_x_adv, self.is_torch).cuda())
                log_probs = predict(hat_x_adv, self.model, self.is_torch)
                topk = torch.topk(log_probs, self.k) # .detach().numpy()

                count += 1
                if y_adv in topk:

                    if prop_de > 0:
                        self.eps_decay = max(prop_de, 0.1)
                    prev_adv = x_adv
                    x_adv = hat_x_adv
                    epsilon = max(epsilon - prop_de/2, self.e_adv)
                    break

                elif lr >= self.min_lr:
                    lr = lr/2

                else:
                    prop_de /= 2
                    if prop_de == 0:
                        return x_adv, y_adv, False, count, decode_predictions(log_probs, top = 10)[0]
                        raise(ValueError("Not converge"))
                    if prop_de < 2e-3:
                        prop_de = 0
                    lr = self.max_lr
                    print("[log] backtracking eps to %3f" % (epsilon-prop_de,))
            
            if self.is_torch: 
                probs = torch.nn.functional.softmax(self.model(transforming(x_adv, self.is_torch).cuda())).cpu().detach().numpy()
                top_k_predictions = decode_predictions(probs, top = 10)[0]
            else: 
                probs = log_probs.cpu().detach().numpy() 
                top_k_labels = list((np.argsort(probs*-1))[0])
                
                top_k_predictions = [("dummy", label, probs[0][label]) for label in top_k_labels][:10]
                
            print(epsilon)
            
            if self.verbose: 
                print(top_k_predictions, epsilon)

            if(epsilon <= self.e_adv):
                return x_adv, y_adv, True, count, top_k_predictions

        return x_adv, y_adv, False, count, top_k_predictions

    # def attack(self, x_adv, y_adv, x_orig): 

    #     epsilon = self.e_0

    #     lower = np.clip(x_orig - epsilon, 0, 1)
    #     upper = np.clip(x_orig + epsilon, 0, 1)
    #     x_adv = np.clip(x_adv.detach().cpu(), lower, upper)

    #     count = 0
    #     while count < self.max_queries and (epsilon > self.e_adv or y_adv != torch.argmax(self.model(transforming(x_adv).cuda())).detach().cpu().numpy().item()):
    #         # print(count, epsilon, torch.max(x_adv - x_orig))
    #         g = self.NES(x_adv, y_adv)
    #         count += self.n_samples
    #         lr = self.max_lr
    #         g = torch.sign(g).cpu().detach().numpy().transpose(0, 2, 3, 1)
    #         hat_x_adv = x_adv + lr * g

    #         # print("hey: ", get_top_k_labels(self.model, hat_x_adv, 1)[0])
    #         probs = self.model(transforming(hat_x_adv).cuda())
    #         topk = torch.topk(probs, self.k) # .detach().numpy()
    #         while y_adv not in topk:
    #             count += 1
    #             if count > self.max_queries:
    #                 return x_adv, y_adv, False, -1, top_k_predictions
    #             if lr < self.min_lr:
    #                 epsilon += self.eps_decay
    #                 self.eps_decay /= 2
    #                 hat_x_adv = x_adv
    #                 break

    #             proposed_eps = max(epsilon - self.eps_decay, self.e_adv)
    #             # print(proposed_eps)
    #             lower = np.clip(x_orig - proposed_eps, 0, 1)
    #             upper = np.clip(x_orig + proposed_eps, 0, 1)
    #             lr /= 2
    #             hat_x_adv = np.clip(x_adv + lr * g, lower, upper)
                
    #             probs = self.model(transforming(hat_x_adv).cuda())
    #             topk = torch.topk(probs, self.k) # .detach().numpy()
    #         proposed_eps = max(epsilon - self.eps_decay, self.e_adv)

    #         lower = np.clip(x_orig - proposed_eps, 0, 1)
    #         upper = np.clip(x_orig + proposed_eps, 0, 1)
    #         hat_x_adv = np.clip(hat_x_adv, lower, upper)
    #         x_adv = hat_x_adv
    #         epsilon -= self.eps_decay

    #         probs = torch.nn.functional.softmax(self.model(transforming(x_adv).cuda())).cpu().detach().numpy()
            
    #         top_k_predictions = decode_predictions(probs, top = 10)[0]
    #         if self.verbose: 
    #             print([x[1:] for x in decode_predictions(probs, top = 1)[0]], epsilon)
    #         # print(epsilon)

    #     return x_adv, y_adv, True, count, top_k_predictions