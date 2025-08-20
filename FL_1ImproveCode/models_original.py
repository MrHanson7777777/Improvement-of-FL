#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn # 从 PyTorch 中导入 nn 模块，它是构建神经网络的核心
import torch.nn.functional as F # 导入 PyTorch 中的函数式 API，包含激活函数、池化等
import torch
import math

class MLP(nn.Module): # 定义一个多层感知机模型，继承自 nn.Module
    def __init__(self, dim_in, dim_hidden, dim_out): # 构造函数
        super(MLP, self).__init__() # 调用父类的构造函数
        self.layer_input = nn.Linear(dim_in, dim_hidden) # 定义输入层到隐藏层的全连接层
        self.relu = nn.ReLU() # 定义 ReLU 激活函数
        self.dropout = nn.Dropout() # 定义 Dropout 层，用于防止过拟合
        self.layer_hidden = nn.Linear(dim_hidden, dim_out) # 定义隐藏层到输出层的全连接层

    def forward(self, x): # 定义模型的前向传播逻辑
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1]) # 将输入 x 展平 (flatten)
        x = self.layer_input(x) # 通过输入层
        x = self.dropout(x) # 应用 Dropout
        x = self.relu(x) # 应用 ReLU 激活
        x = self.layer_hidden(x) # 通过隐藏层
        return F.log_softmax(x, dim=1) # 返回 log softmax 输出


class CNNMnist(nn.Module): # 原始简单的MNIST CNN模型
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(nn.Module): # 原始简单的Fashion-MNIST CNN模型
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)


class CNNCifar(nn.Module): # 原始简单的CIFAR-10 CNN模型
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class CNNCifar100(nn.Module): # 原始简单的CIFAR-100 CNN模型
    def __init__(self, args):
        super(CNNCifar100, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 100)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
