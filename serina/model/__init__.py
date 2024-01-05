import torchvision
from torch import nn as nn
from .model import *

def create_model(num_classes):
    model = torchvision.models.resnet18(pretrained="imagenet")
    # model = torchvision.models.resnet50(pretrained=True)
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # 使用自适应平均池化以确保全连接层的输入尺寸是固定的
    # model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
    # return Serina(num_classes)
