#!/usr/bin/env python
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from config import device
from dataset import TrainSet, ValidationSet
from model import Serina

# 玄学
torch.manual_seed(3407)

data_loader = DataLoader(TrainSet(), shuffle=True, batch_size=200)
val_loader = DataLoader(ValidationSet(), shuffle=True, batch_size=200)

model = Serina()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

model.to(device)
criterion = criterion.to(device)

epoch = 0


def validate():
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(val_loader):
            labels = labels.to(device)
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # 获取每个样本的最大logit值索引作为预测结果

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'***Validation Set Accuracy: {accuracy * 100:.2f}% ***')
        return accuracy

def train_one_epoch():
    for i, (inputs, labels) in enumerate(data_loader):
        labels = labels.to(device)
        inputs = inputs.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向 + 反向 + 优化
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {loss.item()}')

    return loss

while True:
    loss =  train_one_epoch()
    accuracy = validate()
    scheduler.step()
    epoch += 1
    print("saving model")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "accuracy": accuracy
    }, "serina.pth")
    print(f'Epoch: {epoch}, Loss: {loss.item()}')
