import torch
from torch import nn
from typing import Union


class ModelWrapper:
    def __init__(self, model: nn.Module, transformation):
        self.model = model
        self.transformation = transformation

    def learn(self, waveforms, labels: list[int]):
        pass

    def classify(self, waveforms):
        pass
