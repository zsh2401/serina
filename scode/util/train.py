import torch
from progress.bar import Bar

from scode.config import DEVICE


def train_one_epoch(epoch_str, data_loader, model, optimizer, criterion, scheduler=None):
    with Bar(f'Epoch {epoch_str} Training ', max=len(data_loader), suffix='%(percent)d%%') as bar:
        for i, (inputs, labels) in enumerate(data_loader):
            labels = labels.to(DEVICE)
            inputs = inputs.to(DEVICE)

            # 梯度清零
            optimizer.zero_grad()

            bar.next(1)

            # 前向 + 反向 + 优化
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        return loss


def validate(val_loader, model):
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
        return accuracy
