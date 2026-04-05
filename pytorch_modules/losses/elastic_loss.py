import torch
import torch.nn as nn 

def elastic_loss(model):
    
    l1_loss, l2_loss = 0, 0 
    
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            if layer.weight is not None:
                l1_loss += torch.sum(torch.abs(layer.weight))
                l2_loss += torch.sum(layer.weight ** 2)

    return l1_loss, l2_loss