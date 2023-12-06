import numpy as np

import torch
from torchvision import transforms

import tensorflow as tf
decode_predictions = tf.keras.applications.inception_v3.decode_predictions

class Model:
    def __init__(self, pretrained_model, label):
        self.count = 0
        self.pretrained_model = pretrained_model
        self.label = label


    def get_imagenet_label(self, probs):
        if len(probs) > 1:
            return decode_predictions(probs, top=6)
        return decode_predictions(probs, top=6)[0]


    def torch_transform(self, image):
        """
        input: numpy images of shape (B, H, W, C), normalized to (0, 1)
        output: tensor of images of shape (B, C, H, W), normalized to mean [.485, .456, .406], std [.229, .224, .225]
        """

        if not isinstance(image, np.ndarray):
            image = image.numpy()

        image = torch.tensor(image, dtype=torch.float32)
        if len(image.shape) <= 4:
            image = torch.unsqueeze(image, 1)
        # B, 1, H, W, C
        assert image.shape[-1] == 3

        image = torch.transpose(image, 1, 4)
        # B, C, H, W, 1
        # assert image.shape[1] == 3 and image.shape[3] == 299

        image = torch.squeeze(image, dim=4)
        # B, C, H, W
        # assert image.shape[1] == 3 and image.shape[3] == 299 and len(image.shape) == 4

        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = transform(image)
        
        return image


    def get_logits(self, image):
        """
        input: normalized tensor of shape (B, H, W, C)
        output: numpy array of logits
        """

        self.count += 1

        with torch.no_grad():
            preds = self.pretrained_model(self.torch_transform(image).to("cuda"))
        
        return preds.cpu().detach().numpy()
    

    def predict(self, image):
        return self.get_imagenet_label(self.get_logits(image))


    def decision(self, image):
        check = self.predict(image)
        if check[0][0] != self.label:
            return True
        else:
            return False