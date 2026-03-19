import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34

def get_resnet18(num_classes=10):
    model = resnet18(num_classes=num_classes)
    # Adapt for 32x32 CIFAR images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

def get_resnet34(num_classes=10):
    model = resnet34(num_classes=num_classes)
    # Adapt for 32x32 CIFAR images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
