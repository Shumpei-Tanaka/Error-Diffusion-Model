# -*- coding: utf-8 -*-
from datetime import datetime
from enum import Enum
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import Monoamine

torch.manual_seed(0)


def train(
    model,
    train_loader: torch.utils.data.DataLoader,
    criterion,
    optimizer,
    device: torch.device = "cpu",
):

    loss_total = 0.0
    num_train = 0
    acc_total = 0

    model.train()

    for images, labels in train_loader:
        num_train += len(labels)

        images, labels = images.view(-1, 28 * 28).to(device), labels.to(device)
        optimizer.zero_grad()

        outputs: torch.Tensor = model((images, images))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()

        labels_inf = outputs.argmax(dim=-1).detach()
        acc_total += (labels_inf == labels).sum().item()

    loss = loss_total / num_train
    acc = acc_total / num_train

    return loss, acc


def eval(model, eval_loader, criterion, device: torch.device = "cpu"):

    loss_total = 0.0
    num_test = 0
    acc_total = 0

    model.eval()

    with torch.no_grad():
        for images, labels in eval_loader:
            num_test += len(labels)
            images, labels = images.view(-1, 28 * 28).to(device), labels.to(device)
            outputs = model((images, images))
            loss = criterion(outputs, labels)
            loss_total += loss.item()

            labels_inf = outputs.argmax(dim=-1).detach()
            acc_total += (labels_inf == labels).sum().item()
        loss = loss_total / num_test
        acc = acc_total / num_test
    return loss, acc


def learn(
    model,
    train_loader,
    eval_loader,
    criterion,
    optimizer,
    num_epochs,
    logpath: Path,
    writer: SummaryWriter,
    device: torch.device = "cpu",
):

    logheader = (
        f"{'epoch':>12}"
        f"{'train_loss':>12}"
        f"{'eval_loss':>12}"
        f"{'train_acc':>12}"
        f"{'eval_acc':>12}"
    )

    with open(logpath, "w") as f:
        f.write(logheader)

    for epoch in range(1, num_epochs + 1, 1):

        train_loss, train_acc = train(
            model, train_loader, criterion, optimizer, device=device
        )
        eval_loss, eval_acc = eval(model, eval_loader, criterion, device=device)

        print(
            "epoch : {}, t_loss : {:.5f}, e_loss : {:.5f}, t_acc : {:.5f}, e_acc : {:.5f}".format(
                epoch, train_loss, eval_loss, train_acc, eval_acc
            )
        )

        with open(logpath, "a") as f:
            f.write(
                f"\n{epoch:12d}{train_loss:12.5f}{eval_loss:12.5f}{train_acc:12.5f}{eval_acc:12.5f}"
            )

        step = epoch * len(train_loader)

        writer.add_scalar("Loss/Train", train_loss, step)
        writer.add_scalar("Loss/Eval", eval_loss, step)
        writer.add_scalar("ACC/Train", train_acc, step)
        writer.add_scalar("ACC/Eval", eval_acc, step)

    return


class SwModel(Enum):
    m_1layer = 0
    m_3layer = 1
    m_5layer = 2


def main():

    swmodel = SwModel.m_1layer

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    savebasename = "Result"
    savename = "MonoamineResult"
    logname = "training-log.txt"
    strdate = datetime.now().strftime("%y%m%dT%H%M%S")
    savefoldername = savename + "-" + strdate

    savepath = Path(savebasename) / savefoldername
    logpath = savepath / logname
    savepath.mkdir(parents=True, exist_ok=True)

    print(f"save to {savepath}")

    writer = SummaryWriter(savepath)

    print(f"traning on {device=}")

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )
    # 検証データ
    eval_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor(), download=True
    )
    print(f"dataset: {train_dataset}")

    fig, label = train_dataset[0]

    print(f"raw input shape={fig.shape}, label={label}")

    print(f"input shape = {fig.view(1, -1).shape}")

    n_input = fig.view(1, -1).shape[1]
    n_out = 10

    print(f"create model: {swmodel.name}")

    if swmodel == SwModel.m_1layer:
        n_hidden = n_input * n_out
        model = nn.Sequential(
            Monoamine.MonoamineFixedDual2RepInputLayer(
                n_input, n_hidden, activate=nn.ReLU()
            ),
            Monoamine.MonoamineDual2MultiOutLayer(
                n_hidden, n_out, activate=nn.Softmax(dim=-1)
            ),
        )
    elif swmodel == SwModel.m_3layer:
        n_hidden = n_input * n_out * 2
        model = nn.Sequential(
            Monoamine.MonoamineFixedDual2RepInputLayer(
                n_input, n_hidden, activate=nn.ReLU()
            ),
            Monoamine.MonoamineFixedDual2MidLayer(n_hidden, activate=nn.ReLU()),
            Monoamine.MonoamineDual2MultiOutLayer(
                n_hidden // 2, n_out, activate=nn.Softmax(dim=-1)
            ),
        )
    elif swmodel == SwModel.m_5layer:
        n_hidden = n_input * n_out * 2**3
        model = nn.Sequential(
            Monoamine.MonoamineFixedDual2RepInputLayer(
                n_input, n_hidden, activate=nn.ReLU()
            ),
            Monoamine.MonoamineFixedDual2MidLayer(n_hidden, activate=nn.ReLU()),
            Monoamine.MonoamineFixedDual2MidLayer(n_hidden // 2, activate=nn.ReLU()),
            Monoamine.MonoamineFixedDual2MidLayer(
                n_hidden // 2 // 2, activate=nn.ReLU()
            ),
            Monoamine.MonoamineDual2MultiOutLayer(
                n_hidden // 2 // 2 // 2, n_out, activate=nn.Softmax(dim=-1)
            ),
        )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.8)

    batch_size = 256

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset, batch_size=batch_size, shuffle=True
    )
    num_epochs = 1000
    learn(
        model,
        train_loader,
        eval_loader,
        criterion,
        optimizer,
        num_epochs,
        device=device,
        logpath=logpath,
        writer=writer,
    )


if __name__ == "__main__":
    main()
