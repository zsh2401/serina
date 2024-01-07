import torchvision
from torch import nn as nn
from .model import *
from .. import conf


def create_model(num_classes) -> nn.Module:
    # torchvision.models.resnet
    if conf["model"] == "densenet121":
        model = torchvision.models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif conf["model"] == "serina":
        return Serina(num_classes)
    else:
        model = torchvision.models.resnet18(pretrained="imagenet")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
