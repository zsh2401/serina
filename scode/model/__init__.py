import torchvision
from torch import nn as nn
from .model import *

def create_model(num_classes):
    model = torchvision.models.resnet18(pretrained=True)
    # model = torchvision.models.resnet50(pretrained=True)
    # model.conv1 = nn.Conv2d(, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
    # return Serina(num_classes)
