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
        verbose
    ):
        self.lr = lr
        self.target_eps = target_eps
        self.n_samples = n_samples
        self.sigma = sigma
        self.max_queries = max_queries
        self.momentum = momentum, 
        self.model = model
        self.verbose = verbose

    def NES(self, x_orig, y_adv, y_orig, k): 
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

        _, r, d, _ = x_orig.shape
        noise = torch.normal(mean = 0, std = 1, size = (self.n_samples//2, 3, r, d))
        noise = torch.cat([noise, -noise], axis = 0).cuda()
        x_orig = transforming(x_orig)
        x = x_orig.repeat((self.n_samples, 1, 1, 1)).cuda()
        x += noise * self.sigma
        predictions = self.model(x)
        prob = torch.nn.functional.softmax(predictions).detach()[:, y_adv]
        prob = prob[:, None, None, None]
        g = prob * noise
        return g.sum(dim = 0).unsqueeze(0) / (self.n_samples)

    def attack(self, x_orig, y_adv, y_orig): 
        x_orig = x_orig.cpu().detach()
        count = 0
        x_adv = x_orig
        upper = x_orig + self.target_eps
        lower = x_orig - self.target_eps
        grad = torch.zeros((1, 3, 299, 299))
        top_k_predictions = None 
        while count < self.max_queries:
            prev_grad = grad.cuda()
            grad = self.NES(x_adv, y_adv, y_orig, 1) 
            # grad = self.momentum * prev_grad + (1 - self.momentum) * grad
            count += self.n_samples
            x_adv = x_adv + self.lr * torch.sign(grad).cpu().detach().numpy().transpose(0, 2, 3, 1)
            self.lr *= 0.99
            x_adv = np.clip(x_adv, lower, upper)
            log_probability_predictions = self.model(transforming(x_adv).cuda())
            cls = torch.argmax(log_probability_predictions).detach().cpu().numpy().item()
            probs = torch.nn.functional.softmax(log_probability_predictions).cpu().detach().numpy()
            
            top_k_predictions = decode_predictions(probs, top = 10)[0]
            
            if self.verbose: 
                print([x[1:] for x in top_k_predictions])
            
            if(cls == y_adv):
                return x_adv, cls, probs[:, y_adv], count, True, top_k_predictions
            # print(probs[:, y_adv])
            

        return x_adv, -1, probs[:, y_adv], -1, False, top_k_predictions

torch_transform = transforms.Compose([
    transforms.
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])    

def predict(image, model, device='cuda'):
    """
    input: normalized tensor of shape (B, C, H, W)
    output: numpy array of predictions
    """
    # print("predicting")
    with torch.no_grad():
        preds = model(transforming(image).to(device))
    return preds