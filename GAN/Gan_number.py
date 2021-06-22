# -*- coding: utf-8 -*-
# @Time    : 2021/6/21 上午10:37
# @Author  : Zhong Lei
# @FileName: Gan_number.py
import torch
import torch.nn as nn
import pandas
import matplotlib.pyplot as plt
import random
import numpy as np


def generate_real():
    # real_data = torch.FloatTensor([1, 0, 1, 0])
    real_data = torch.FloatTensor([
        random.uniform(0.8, 1),
        random.uniform(0.0, 0.2),
        random.uniform(0.8, 1.0),
        random.uniform(0.0, 0.2)
    ])
    return real_data


def generate_noise(size):
    random_data = torch.rand(size)
    return random_data


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 3),
            nn.Sigmoid(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.loss_function(outputs, targets)

        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())

        if self.counter % 10000 == 0:
            print("counter =", self.counter)
            torch.save(self.model.state_dict(), "discriminator.pkl")

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
            nn.Linear(1, 3),
            nn.Sigmoid(),

            nn.Linear(3, 4),
            nn.Sigmoid()
        )
        # self.loss_function = nn.MSELoss()直接使用discriminator中更新的梯度来反generator
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
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
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
        if self.counter % 10000 == 0:
            torch.save(self.model.state_dict(), "generator.pkl")
        self.optimizer.step()

    def plot_progress(self):
        x = np.arange(0, len(self.progress), 1)
        y = np.array(self.progress)
        plt.scatter(x, y)
        plt.show()


def train_gan():
    image_list = []
    D = Discriminator()
    G = Generator()
    for i in range(30000):

        # 先更新discriminator中的梯度，对generator使用detach()方法防止更新梯度
        # pos
        D.train(generate_real(), torch.FloatTensor([1.0]))
        # neg
        D.train(G.forward(torch.FloatTensor([0.5])), torch.FloatTensor([0.0]))

        # 再训练generator
        G.train(D, torch.FloatTensor([0.5]), torch.FloatTensor([1.0]))
        if i % 1000 == 0:
            image_list.append(np.array(G.forward(torch.FloatTensor([0.5])).detach()))

    D.plot_progress()
    G.plot_progress()
    plt.figure(figsize=(16, 8))
    plt.imshow(np.array(image_list).T, interpolation="none", cmap="Blues")
    plt.show()
    print(G.forward(torch.FloatTensor([0.2])))
    print(D.forward(generate_real()).item())
    print(D.forward(generate_noise(4)).item())


if __name__ == '__main__':
    # real = generate_real()
    # noise = generate_noise(4)
    # dis = Discriminator()

    # train_discriminator code
    # for i in range(60000):
    #     dis.train(generate_real(), torch.FloatTensor([1.0]))
    #     dis.train(generate_noise(4), torch.FloatTensor([0.0]))
    # dis.plot_progress()
    # # dis.load_state_dict(torch.load("discriminator.pkl"), False)
    # print(dis.forward(generate_real()).item())
    # print(dis.forward(generate_noise(4)).item())

    #generator check
    # gen = Generator()
    # print(gen.forward(torch.FloatTensor([0.5])))

    # train_gan
    # train_gan()
    G = Generator()
    G.load_state_dict(torch.load("generator.pkl"), False)
    with torch.no_grad():
        print(G.forward(torch.FloatTensor([0.5])))
    #
    # dis = Discriminator()
    # dis.load_state_dict(torch.load("discriminator.pkl"), False)
    # print(dis.forward(generate_real()).item())
    # print(dis.forward(generate_noise(4)).item())
