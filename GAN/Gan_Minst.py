# -*- coding: utf-8 -*-
# @Time    : 2021/6/22 上午10:12
# @Author  : Zhong Lei
# @FileName: Gan_Minst.py
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            # nn.Sigmoid(),

            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),

            nn.Linear(200, 1),
            nn.Sigmoid()
        )
        # self.loss_function = nn.MSELoss()
        self.loss_function = nn.BCELoss()
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
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
            # 这里直接保存会保存最开始训练的模型
            # torch.save(self.model.state_dict(), "minst_dis.pkl")

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
            # nn.Sigmoid(),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),

            nn.Linear(200, 784),
            nn.Sigmoid()
        )
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, D: Discriminator, inputs, targets):
        g_outputs = self.forward(inputs)
        d_outputs = D.forward(g_outputs)

        self.optimizer.zero_grad()
        loss = D.loss_function(d_outputs, targets)
        loss.backward()

        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
        if self.counter % 10000 == 0:
            torch.save(self.model.state_dict(), "generator_min.pkl")
        self.optimizer.step()

    def plot_progress(self):
        x = np.arange(0, len(self.progress), 1)
        y = np.array(self.progress)
        plt.scatter(x, y)
        plt.show()


class MinstData(Dataset):
    def __init__(self, path: str):
        self.data_df = pd.read_csv(path, header=None)[1:]

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, item):
        label = int(self.data_df.iloc[item, 0])
        target = torch.zeros((10))
        target[label] = 1.0
        images = torch.from_numpy(np.array(self.data_df.iloc[item, 1:].values).astype(float))/255.0
        return label, target.float(), images.float()

    def plot_imgae(self, index):
        array = np.array(self.data_df.iloc[index, 1:].values).astype(float).reshape(28, 28)
        plt.title("label = " + str(self.data_df.iloc[index, 0]))
        plt.imshow(array, interpolation="none", cmap="Blues")
        plt.show()


def train_minst(minst_train: Dataset):
    D = Discriminator()
    G = Generator()
    for label, target, image in minst_train:
        # pos
        D.train(image, torch.FloatTensor([1.0]))
        # neg
        D.train(G.forward(torch.randn(100)).detach(), torch.FloatTensor([0.0]))

        G.train(D, torch.randn(100), torch.FloatTensor([1.0]))

    D.plot_progress()
    G.plot_progress()

    # print(G.forward(torch.rand(1)))
    print(G.forward(torch.randn(100)))
    print(D.forward(minst_train[1][2]).item())
    print(D.forward(torch.rand(784)).item())

    f, axarr = plt.subplots(2, 3, figsize=(16, 8))
    for i in range(2):
        for j in range(3):
            output = G.forward(torch.randn(100))
            img = output.detach().numpy().reshape(28, 28)
            axarr[i, j].imshow(img, interpolation="none", cmap="Blues")
    plt.show()
    torch.save(D, "dis.pth")
    torch.save(G, "gen.pth")


def plot_random_seed(G: Generator):
    seed1 = torch.randn(100)
    seed2 = torch.randn(100)

    img1 = G.forward(seed1).detach().numpy().reshape(28, 28)
    img2 = G.forward(seed2).detach().numpy().reshape(28, 28)
    plt.imshow(img1, cmap="Blues")
    plt.show()
    plt.imshow(img2, cmap="Blues")
    plt.show()

    count = 0
    f, axarr = plt.subplots(3, 4, figsize=(16, 8))
    for i in range(3):
        for j in range(4):
            seed = seed1 + (seed2 - seed1)/11 * count
            img = G.forward(seed).detach().numpy().reshape(28, 28)
            axarr[i, j].imshow(img, interpolation="none", cmap="Blues")
            count += 1
    plt.show()


if __name__ == '__main__':
    minst_train = MinstData("mnist_csv/mnist_train.csv")
    # print(minst_train[33])

    # train_discriminator
    # D = Discriminator()
    # for label, target, images in minst_train:
    #     D.train(images, torch.FloatTensor([1.0]))
    #     D.train(torch.rand(784), torch.FloatTensor([0.0]))
    # D.plot_progress()
    # # save
    # torch.save(D.model.state_dict(), "minst_dis_1.pkl")

    # test_discriminator
    # D = Discriminator()
    # for i in range(4):
    #     image = minst_train[random.randint(0, 60000)][2]
    #     print(D.forward(image))
    #
    # for i in range(4):
    #     image = torch.rand(784)
    #     print(D.forward(image))

    # test_generator
    # G = Generator()
    # imge = G.forward(torch.rand(1)).detach().numpy().reshape(28, 28)
    # plt.imshow(imge, interpolation="none", cmap="Blues")
    # plt.show()

    train_minst(minst_train)
    # gen = torch.load("gen.pth")
    # imge = gen.forward(torch.randn(100)).detach().numpy().reshape(28, 28)
    # plt.imshow(imge, interpolation="none", cmap="Blues")
    # plt.show()
    # plot_random_seed(gen)

