import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import random

def generate_random_one_hot(size):
    label = torch.zeros((size), dtype=torch.float32)
    random_idx = random.randint(0, size-1)
    label[random_idx] = 1.0
    return label


class Minst(Dataset):
    def __init__(self, path):
        super().__init__()
        self.df = pd.read_csv(path, header=None)[1:]

    def __getitem__(self, item):
        label = int(self.df.iloc[item, 0])
        target = torch.zeros((10), dtype=torch.float32)
        target[label] = 1.0
        images = torch.from_numpy(np.float32(np.array(self.df.iloc[item, 1:].values).astype(float)))
        return label, target, images

    def __len__(self):
        return len(self.df)

    def plot_image(self, index):
        array = np.array(self.df.iloc[index, 1:].values).astype(float).reshape(28, 28)
        plt.title("lable = " + str(self.df.iloc[index, 0]))
        plt.imshow(array, interpolation="none", cmap="Blues")
        plt.show()


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784+10, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),

            nn.Linear(200, 1),
            nn.Sigmoid()
        )
        self.loss_function = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.progress = []
        self.counter = 0

    def forward(self, inputs, labels):
        inputs = torch.cat([inputs, labels])
        return self.model(inputs)

    def train(self, inputs, labels, targets):
        outputs = self.forward(inputs, labels)
        loss = self.loss_function(outputs, targets)

        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())

        if self.counter % 10000 == 0:
            print("counter dis = ", self.counter)

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
            nn.Linear(100+10, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 784),
            nn.Sigmoid()
        )
        self.counter = 0
        self.progress = []
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, seed_tensor, label):
        inputs = torch.cat([seed_tensor, label])
        return self.model(inputs)

    def train(self, D: Discriminator, inputs, label_tensor, targets):
        g_output = self.forward(inputs, label_tensor)
        d_output = D.forward(g_output, label_tensor)

        self.optimizer.zero_grad()
        loss = D.loss_function(d_output, targets)
        loss.backward()
        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss)

        if self.counter % 10000 == 0:
            print("counter gen = ", self.counter)
        self.optimizer.step()

    def plot_progress(self):
        x = np.arange(0, len(self.progress), 1)
        y = np.array(self.progress)
        plt.scatter(x, y)
        plt.show()


class ConditionGan(object):
    def __init__(self, D: Discriminator, G: Generator, minst_train:Dataset, epoch):
        super(ConditionGan, self).__init__()
        self.dis = D
        self.gen = G
        self.train_data = minst_train
        self.epoch = epoch

    def train(self):
        for i in range(self.epoch):
            for label, target, image in self.train_data:
                self.dis.train(image, target, torch.FloatTensor([1.0]))
                self.dis.train(self.gen.forward(torch.randn(100), generate_random_one_hot(10)).detach(), generate_random_one_hot(10), torch.FloatTensor([0.0]))
                self.gen.train(self.dis, torch.randn(100), generate_random_one_hot(10), torch.FloatTensor([1.0]))

    def plot_image(self, label):
        label_tensor = torch.zeros((10))
        label_tensor[label] = 1.0
        f, axarr = plt.subplots(2, 3, figsize=(16, 8))
        for i in range(2):
            for j in range(3):
                output = self.gen.forward(torch.randn(100), label_tensor)
                img = output.cpu().detach().numpy().reshape(28, 28)
                axarr[i, j].imshow(img, interpolation="none", cmap="Blues")
        plt.show()
        torch.save(self.dis, "dis.pth")
        torch.save(self.gen, "gen.pth")

if __name__ == '__main__':
    minst = Minst("mnist_csv\mnist_train.csv")
    # print(minst[0][1])
    # print(minst[0][2])
    # print(minst[0][0])
    # minst.plot_image(0)
    D = Discriminator()
    G = Generator()
    con = ConditionGan(D, G, minst, 15)
    con.train()
    con.plot_image(7)
