#!/usr/bin/env python
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from serina.config import DEVICE, PTH_NAME, LEARN_RATE, EPOCH
from serina.dataset import TrainSet, ValidationSet
from serina.dataset.label import get_categories
from serina.model import create_model

import os
from progress.bar import Bar

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
loss_curve = []
accuracy_curve = []


def resume_state():
    if os.path.isfile(PTH_NAME) is False:
        return
    state = torch.load(PTH_NAME)
    model.load_state_dict(state["model"])
    global epoch
    global accuracy_curve
    global loss_curve
    epoch = state["epoch"]
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    loss_curve = state["loss_curve"]
    accuracy_curve = state["accuracy_curve"]
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


def train_one_epoch(epoch_str):
    with Bar(f'Epoch {epoch_str} Training ', max=len(data_loader), suffix='%(percent)d%%') as bar:
        for i, (inputs, labels) in enumerate(data_loader):
            start = time.time()
            labels = labels.to(DEVICE)
            inputs = inputs.to(DEVICE)
            print(f"moving to {DEVICE} costs {time.time() - start}s")

            # 梯度清零
            optimizer.zero_grad()

            bar.next(1)

            # 前向 + 反向 + 优化
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {loss.item()}')
        return loss



resume_state()
print(f"Running on {DEVICE}")

while EPOCH < 0 or epoch < EPOCH:
    epoch += 1
    epoch_str = epoch
    if EPOCH > 0:
        epoch_str = f"[{epoch}/{EPOCH}]"

    print(f"====Epoch {epoch_str}====")
    start = time.time()
    loss = train_one_epoch(epoch_str)
    scheduler.step()

    loss_curve.append(loss.item())
    print(f'loss: {loss.item()}.')
    print(f"costs {time.time() - start:.2f}s")
    accuracy = validate()
    print(f"validation accuracy {accuracy * 100:.2f}%")
    accuracy_curve.append(accuracy * 100)

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "accuracy": accuracy,
        "accuracy_curve": accuracy_curve,
        "loss_curve": loss_curve
    }, PTH_NAME)

    print("Saved")
