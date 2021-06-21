import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
# pytorch 提供的导入数据方法
from torch.utils.data import Dataset

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # model
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            # nn.Sigmoid(),
            nn.LeakyReLU(),
            nn.Linear(200, 10),
            nn.Sigmoid()
        )
        # loss_func
        self.loss_function = nn.BCELoss()
        # optimizer, 用于优化参数
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        # monitor 监控运行的次数，并记录下中间结果。
        self.counter = 0
        self.progress = []

    # foward_pass
    def forward(self, inputs):
        return self.model(inputs)

    # train 训练方法
    def train(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)

        # 梯度更新过程
        # 清零， 计算， 更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 监控
        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
        if self.counter % 10000 == 0:
            print("counter = ", self.counter)
            print("loss = ", str(loss.item()))

    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=["loss"])
        df.plot(ylim=(0, 0.1), figsize=(16, 8), alpha=0.1, marker=".", grid=True, yticks=(0, 0.25, 0.5))


class MinstDataset(Dataset):
    def __init__(self, csv_file):
        self.data_df = pd.read_csv(csv_file, header=None)[1:]

    def __len__(self):
        return len(self.data_df - 1)

    def __getitem__(self, item):
        label = self.data_df.iloc[item, 0]
        target = torch.zeros((10))
        target[int(label)] = 1.0
        # print(self.data_df.iloc[item, 1:].values.dtype)
        images = torch.from_numpy(np.array(self.data_df.iloc[item,1:].values).astype(float))/255.0
        return label, target.float(), images.float()

    def plot_imgae(self, index):
        array = np.array(self.data_df.iloc[index,1:].values).astype(float).reshape(28, 28)
        plt.title("label = " + str(self.data_df.iloc[index, 0]))
        plt.imshow(array, interpolation="none", cmap="Blues")
        plt.show()


if __name__ == '__main__':
    # train_set
    minst = MinstDataset("mnist_csv/mnist_test.csv")
    # test_set
    minst_test = MinstDataset("mnist_csv/mnist_test.csv")
    print(minst_test[33])
    minst_test.plot_imgae(33)


    # train_code
    # classifier = Classifier()
    # epochs = 3
    # for i in range(epochs):
    #     print("training epoch", str(i+1),  "of", str(epochs))
    #     for label, target, image in minst:
    #         classifier.train(image, target)
    #     torch.save(classifier.state_dict(),"iter.pkl")
    # classifier.plot_progress()

    # test_output 并画出对应结果的直方图，注意输入模型时单个样本输入要进行unsqueeze(0)增加一个维度，变成batch形式输入模型，
    # 在绘制直方图时需要squeeze(0)把增加的维度去掉才能画出图形。
    # model = Classifier()
    # model.load_state_dict(torch.load("iter.pkl"))
    # test_data = minst_test[34][2]
    # print(test_data)
    # output = model.forward(test_data.unsqueeze(0))
    # print(output)
    # x = list(range(10))
    # y = output.squeeze(0).detach().numpy()
    # print(x, y)
    # plt.bar(x=x, height=y)
    # plt.show()

    # eval评估模型。
    model = Classifier()
    model.load_state_dict(torch.load("iter.pkl"))
    score, item = 0, 0
    for label, target, image in minst_test:
        predict = model.forward(image.unsqueeze(0)).squeeze(0).detach().numpy()
        if predict.argmax() == int(label):
            score += 1
        item += 1
    print(score/item)


