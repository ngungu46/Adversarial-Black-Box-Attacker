import torch.nn.functional as F
import numpy as np

def loss(y, logits, targeted=False, loss_type='margin_loss'):
    if loss_type == 'margin_loss':
        preds_correct_class = (logits * y).sum(1, keepdims=True)
        diff = preds_correct_class - logits
        diff[y] = np.inf
        margin = diff.min(1, keepdims=True)
        loss = margin * -1 if targeted else margin
    elif loss_type == 'cross_entropy':
        probs = F.softmax(logits)
        loss = -np.log(probs[y])
        loss = loss * -1 if not targeted else loss
    else:
        raise ValueError('Wrong loss.')
    return loss.flatten()