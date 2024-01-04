#!/usr/bin/env python
import torch
from torch.utils.data import DataLoader

from serina.dataset import TestSet
from serina.dataset.label import get_categories
from serina.model import create_model
from serina.config import DEVICE, PTH_NAME

with torch.no_grad():
    model = create_model(get_categories())

    pth = torch.load(PTH_NAME)
    model.load_state_dict(pth["model"])

    model.to(DEVICE)
    model.eval()

    test_dataset = TestSet()
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=64)

    # 正确的预测数和总的预测数
    correct = 0
    total = 0

    with torch.no_grad():  # 在这个块中，不跟踪梯度，节省内存和计算
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)  # 获取每个样本的最大logit值索引作为预测结果

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 打印准确率
    accuracy = correct / total
    print(f'*** Accuracy: {accuracy * 100:.2f}% ***')
