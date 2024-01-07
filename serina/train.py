import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from serina import conf, get_pth_name
from serina.dataset import TrainSet, ValidationSet
from serina.dataset.label import get_num_classes
from serina.model import create_model

import os
from progress.bar import Bar


def train():
    torch.manual_seed(42)
    batch_size = conf["batch_size"]

    data_loader = DataLoader(TrainSet(), shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(ValidationSet(), shuffle=True, batch_size=batch_size)

    model = create_model(get_num_classes())
    criterion = nn.CrossEntropyLoss()

    if conf["optimizer"].lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=conf["learn_rate"],weight_decay=1e-5)
    else:
        optimizer = optim.SGD(model.parameters(), lr=conf["learn_rate"])
    print(f"Using {optimizer} as optimizer")
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    model.to(conf["device"])
    model = torch.nn.DataParallel(model)
    criterion = criterion.to(conf["device"])

    epoch = 0
    loss_curve = []
    accuracy_curve = []

    def resume_state():
        if os.path.isfile(get_pth_name()) is False:
            return
        state = torch.load(get_pth_name())
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
            val_loss = 0
            for i, (inputs, labels) in enumerate(val_loader):
                labels = labels.to(conf["device"])
                inputs = inputs.to(conf["device"])

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)  # 获取每个样本的最大logit值索引作为预测结果

                val_loss += criterion(outputs, labels).item()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            _accuracy = correct / total
            # print(f'***Validation Set Accuracy: {accuracy * 100:.2f}% ***')
            val_loss /= len(val_loader.dataset)
            return _accuracy, val_loss

    def train_one_epoch():
        with Bar(f'Epoch {epoch} Training ', max=len(data_loader), suffix='%(percent)d%%') as bar:
            for i, (inputs, labels) in enumerate(data_loader):
                # start = time.time()
                labels = labels.to(conf["device"])
                inputs = inputs.to(conf["device"])
                # print(f"moving to {DEVICE} costs {time.time() - start}s")

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
    print(f"Running on {conf['device']}")

    while conf["epoch"] < 0 or epoch < conf["epoch"]:
        epoch += 1
        epoch_str = epoch
        if conf["epoch"] > 0:
            epoch_str = f"[{epoch}/{conf['epoch']}]"

        print(f"====Epoch {epoch_str}====")
        current_learning_rate = optimizer.param_groups[0]['lr']
        print("Current learning rate:", current_learning_rate)
        start = time.time()
        loss = train_one_epoch()

        loss_curve.append(loss.item())
        print(f'training loss: {loss.item()}.')
        print(f"costs {time.time() - start:.2f}s")
        accuracy, loss = validate()
        scheduler.step()

        print(f"validation accuracy {accuracy * 100:.2f}% loss {loss}")
        accuracy_curve.append(accuracy * 100)

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "accuracy": accuracy,
            "accuracy_curve": accuracy_curve,
            "loss_curve": loss_curve
        }, get_pth_name())

        print(f"Model saved as {get_pth_name()} size: {os.path.getsize(get_pth_name())}")
