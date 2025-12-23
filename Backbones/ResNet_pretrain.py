"""
Pretrained ResNet models for federated learning.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet18Pretrained(nn.Module):
    """
    Pretrained ResNet18 model adapted for federated learning.
    """
    
    def __init__(self, cfg):
        super(ResNet18Pretrained, self).__init__()
        
        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=True)
        
        # Modify the final fully connected layer for the target number of classes
        num_classes = cfg.DATASET.n_classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


def resnet18_pretrained(cfg):
    """
    Create a pretrained ResNet18 model.
    
    Args:
        cfg: Configuration object with DATASET.n_classes
    
    Returns:
        ResNet18Pretrained model
    """
    return ResNet18Pretrained(cfg)
