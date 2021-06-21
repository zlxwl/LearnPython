# -*- coding: utf-8 -*-
# @Time    : 2021/6/3 下午2:27
# @Author  : Zhong Lei
# @FileName: VAE.py
import os
import time
from typing import List, TypeVar, Any

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST

Tensor = TypeVar("torch.tensor")

if not os.path.exists("./vae_img"):
    os.mkdir("./vae_img")


def to_img(x: Tensor):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

num_epoch = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([transforms.ToTensor()])
dataset = MNIST("../data", transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2_1 = nn.Linear(400, 20)
        self.fc2_2 = nn.Linear(400, 20)
        self.fc_3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        result = F.relu(self.fc1(x))
        return self.fc2_1(result), self.fc2_2(result)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return eps * logvar + mu

    def decode(self, z):
        result = F.relu(self.fc_3(z))
        return torch.sigmoid(self.fc4(result))

    def forward(self, input: Tensor, **kwargs: Any) -> Tensor:
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        result, _, _ = self.forward(x)
        return result


def loss_function(recon_x, x, mu, log_var):
    BCE = F.mse_loss(recon_x, x, size_average=False)
    KL = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 -
                                     log_var.exp(), dim=1), dim=0)
    return BCE + KL


if __name__ == '__main__':
    start = time.time()
    model = VAE()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if torch.cuda.is_available():
        model.to("cuda:0")

    for epoch in range(num_epoch):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(dataloader):
            img, _ = data
            img = img.view(img.size(0), -1)
            img = img.cuda() if torch.cuda.is_available() else img
            optimizer.zero_grad()

            recon_batch, mu, log_var = model(img)
            loss = loss_function(recon_batch, img, mu, log_var)
            loss.backward()
            train_loss += loss.item()

            optimizer.step()
            if batch_idx % 100 == 0:
                end = time.time()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                      'Loss: {:.6f} '
                      'time:{:.2f}s'.format(
                        epoch,
                        batch_idx * len(img),
                        len(dataloader.dataset),
                        100. * batch_idx / len(dataloader),
                        loss.item() / len(img),
                        (end-start)
                    )
                )
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(dataloader.dataset)))

        if epoch % 10 == 0:
            save = to_img(recon_batch.cpu().data)
            save_image(save, './vae_img/image_{}.png'.format(epoch))

    torch.save(model.state_dict(), './vae.pth')





