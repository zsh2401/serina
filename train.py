#!/usr/bin/env python
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from config import DEVICE, PTH_NAME, LEARN_RATE
from dataset import TrainSet, ValidationSet
from label import get_categories
from model import Serina, create_model

import os

batch_size = 64
if "BATCH_SIZE" in os.environ:
    batch_size = int(os.environ["BATCH_SIZE"])
# 玄学
torch.manual_seed(3407)

data_loader = DataLoader(TrainSet(), shuffle=True, batch_size=batch_size)
val_loader = DataLoader(ValidationSet(), shuffle=True, batch_size=batch_size)

model = create_model(get_categories())
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

model.to(DEVICE)
criterion = criterion.to(DEVICE)

epoch = 0


def resume_state():
    if os.path.isfile(PTH_NAME) is False:
        return
    state = torch.load(PTH_NAME)
    model.load_state_dict(state["model"])
    global epoch
    epoch = state["epoch"]
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    print(f"State resumed")


def validate():
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(val_loader):
            labels = labels.to(DEVICE)
            inputs = inputs.to(DEVICE)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # 获取每个样本的最大logit值索引作为预测结果

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        # print(f'***Validation Set Accuracy: {accuracy * 100:.2f}% ***')
        return accuracy


def train_one_epoch():
    for i, (inputs, labels) in enumerate(data_loader):
        labels = labels.to(DEVICE)
        inputs = inputs.to(DEVICE)

        # 梯度清零
        optimizer.zero_grad()

        # 前向 + 反向 + 优化
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {loss.item()}')
    return loss


resume_state()
print(f"Running on {DEVICE}")
while True:
    epoch += 1
    print(f"====Epoch {epoch}====")
    loss = train_one_epoch()
    scheduler.step()
    print(f'Epoch: {epoch}, Loss: {loss.item()}.')

    accuracy = validate()
    print(f"Validation Accuracy {accuracy * 100:.2f}%")

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "accuracy": accuracy
    }, "serina.pth")

    print("Saved")
