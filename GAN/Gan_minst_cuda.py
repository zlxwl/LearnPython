# -*- coding: utf-8 -*-
# @Time    : 2021/6/23 下午4:17
# @Author  : Zhong Lei
# @FileName: Gan_minst_cuda.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pandas as pd


class MinstData(Dataset):
    def __init__(self, path):
        self.data_df = pd.read_csv(path, header=None)[1:]

    def __getitem__(self, item):
        label = int(self.data_df.iloc[item, 0])
        target = torch.zeros((10), dtype=torch.float32)
        target[label] = 1.0
        images = torch.from_numpy(np.float32(np.array(self.data_df.iloc[item, 1:].values).astype(float)))
        return label, target, images

    def __len__(self):
        return len(self.data_df)

    def plot_imgae(self, index):
        array = np.array(self.data_df.iloc[index, 1:].values).astype(float).reshape(28, 28)
        plt.title("label = " + str(self.data_df.iloc[index, 0]))
        plt.imshow(array, interpolation="none", cmap="Blues")
        plt.show()


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.loss_function = nn.BCELoss()
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)

        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss)

        if self.counter % 10000 == 0:
            print("counter =", self.counter)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def plot_progress(self):
        x = np.arange(0, len(self.progress), 1)
        y = np.array(self.progress)
        plt.scatter(x, y)
        plt.show()


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 784),
            nn.Sigmoid()
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, D: Discriminator, inputs, targets):
        g_output = self.forward(inputs)
        d_output = D.forward(g_output)

        self.optimizer.zero_grad()
        loss = D.loss_function(d_output, targets)
        loss.backward()

        self.counter += 1
        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss)

        if self.counter % 10000 == 0:
            print("counter =", self.counter)
        self.optimizer.step()

    def plot_progress(self):
        x = np.arange(0, len(self.progress), 1)
        y = np.array(self.progress)
        plt.scatter(x, y)
        plt.show()


class Gan(object):
    def __init__(self, D: Discriminator, G: Generator, minst_train: Dataset):
        super(Gan, self).__init__()
        self.device = torch.device("cuda: 1" if torch.cuda.is_available() else "cpu")
        self.dis = D.to(self.device)
        self.gen = G.to(self.device)
        self.train_data = minst_train

    def train(self):
        for label, target, image in self.train_data:
            self.dis.train(image.cuda(device=self.device),
                           torch.FloatTensor([1.0]).cuda(device=self.device))
            self.dis.train(self.gen.forward(torch.randn(100, device=self.device)).detach(),
                           torch.FloatTensor([0.0]).cuda(device=self.device))
            self.gen.train(self.dis, torch.randn(100).cuda(device=self.device),
                           torch.FloatTensor([1.0]).cuda(device=self.device))

        self.dis.plot_progress()
        self.gen.plot_progress()

        print(self.gen.forward(torch.randn(100).cuda(device=self.device)).cpu())
        print(self.dis.forward(minst_train[1][2].cuda(device=self.device)).cpu().item())
        print(self.dis.forward(torch.rand(784).cuda(device=self.device)).cpu().item())

        f, axarr = plt.subplots(2, 3, figsize=(16, 8))
        for i in range(2):
            for j in range(3):
                output = self.gen.forward(torch.randn(100).cuda(device=self.device))
                img = output.cpu().detach().numpy().reshape(28, 28)
                axarr[i, j].imshow(img, interpolation="none", cmap="Blues")
        plt.show()
        torch.save(self.dis, "dis.pth")
        torch.save(self.gen, "gen.pth")


if __name__ == '__main__':
    minst_train = MinstData("mnist_csv/mnist_train.csv")
    # print(minst_train)
    D = Discriminator()
    G = Generator()
    minst_gan = Gan(D, G, minst_train)
    minst_gan.train()
