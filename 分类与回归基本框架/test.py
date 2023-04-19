from torch import nn  # 导入神经网络模块
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # 输入大小 （1， 28， 28）
            nn.Conv2d(
            in_channels=1,                  # 输入的通道数  (pyporch中是channel first）
            out_channels=16,                # 要得到多少个特征图，卷积核的个数
                kernel_size=5,              # 卷积核大小
                stride=1,                   # 步长
                padding=2,                  # 如果希望输出之后的图像和卷积之前一样，需要计算设置padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    # 输出大小 （16， 14， 14）
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),     # 输入大小 （16， 14， 14）
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, 1, 2),     # 输出大小 （32， 14， 14）
            nn.ReLU(),
            nn.MaxPool2d(2),                # 输出大小 （32， 7， 7）
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),     # 输入大小 （32， 7， 7）
            nn.ReLU(),
        )

        self.out = nn.Linear(64 * 7 * 7, 10)   # 输出大小 （64 * 7 * 7， 10）

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)           # 将（batch， 64， 7， 7）展开成（batch， 64 * 7 * 7）
        output = self.out(x)
        return output

def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    right = pred.eq(labels.data.view_as(pred)).sum()
    return right, len(labels)


if __name__ == '__main__':
    # 超参数
    num_epochs = 5

    # 实例化
    net= CNN()

    # 损失函数
    loss_func = nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # 数据集
    train_dataset = torchvision.datasets.MNIST(root='./data/mnist/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data/mnist/', train=False, transform=transforms.ToTensor())

    # 数据加载器
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # 训练
    for epoch in range(num_epochs):
        print('\n', f'{"||进度":<44s}'
                    f'{"||epoch":<30s} '
                    f'{"||Train Accuracy":>35s}'
                    f'{"||Test Accuracy":>34s}'
              )
        print('\033[34m——\033[0m'*90)
        train_rights = []  # 保存每个batch的正确结果

        for batch_idx, datas in enumerate(tqdm(train_loader)):
            data, target = datas
            net.train()
            output = net(data)
            loss = loss_func(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_rights.append(accuracy(output, target))

            if batch_idx% 200 == 199:
                tqdm.write('||Train Epoch: {} [{:<4d}/{:<2d} ({:.1f}%)] Loss: {:.6f}'.format(
                    epoch+1,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()
                ),
                    end='\033[31m||\033[0m')

                net.eval()
                val_rights = []

                for data, traget in test_loader:
                    output = net(data)
                    val_rights.append(accuracy(output, traget))

                # 计算正确率
                train_right = (sum([s[0] for s in train_rights]), sum([s[1] for s in train_rights]))
                val_right = (sum([s[0] for s in val_rights]), sum([s[1] for s in val_rights]))

                print('Train Accuracy: {}/{:>4d} ({:.4f}%)'.format(
                    train_right[0], train_right[1], 100. * train_right[0] / train_right[1]), end='\033[31m||\033[0m')
                print('Test Accuracy: {}/{:>4d} ({:.4f}%)'.format(
                    val_right[0], val_right[1], 100. * val_right[0] / val_right[1]))














