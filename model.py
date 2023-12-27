import torch.nn as nn
import torch.nn.functional as F
import torch

from label import get_categories


# Create the model
def create_model():
    return Serina()


# The Serina model can identify
# specified sound effect.
class Serina(nn.Module):

    def __init__(self):
        super(Serina, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(160_000, 64)
        self.fc2 = nn.Linear(64, get_categories())  # 假设有10个类别

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
