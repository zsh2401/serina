#!/usr/bin/env python
import torch
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from config import PTH_NAME


def draw_for_model(checkpoint):
    losses = checkpoint["loss_curve"]
    accuracy = checkpoint["accuracy_curve"]

    _, ax1 = plt.subplots()
    _, ax2 = plt.subplots()

    # Drawing losses curve
    ax1.plot(losses)
    ax1.set_title('Training Loss Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    # Drawing accuracy curve
    ax2.plot(accuracy)
    ax2.set_title('Accuracy Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')


draw_for_model(torch.load(PTH_NAME))

plt.show()
