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
        self.log_softmax = nn.LogSoftmax(dim=1) # 定义 LogSoftmax 层，用于输出概率分布，dim=1 表示对每一行进行 softmax 操作

    def forward(self, x): # 定义模型的前向传播逻辑
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1]) # 将输入 x 展平 (flatten)
                                                          # -1 表示该维度大小由其他维度和总元素数自动推断
                                                          # x.shape[1]*x.shape[-2]*x.shape[-1] 计算图像的总像素数
        x = self.layer_input(x) # 通过输入层
        x = self.dropout(x) # 应用 Dropout
        x = self.relu(x) # 应用 ReLU 激活
        x = self.layer_hidden(x) # 通过隐藏层
        return self.log_softmax(x) # 返回 Softmax 输出


class CNNMnist(nn.Module): # 定义一个针对 MNIST 数据集的 CNN 模型
    def __init__(self, args): # 构造函数，接收命令行参数作为输入
        super(CNNMnist, self).__init__()
        # 第一个卷积层：输入通道数为 args.num_channels (MNIST 为 1)，输出通道数为 10，卷积核大小为 5x5
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        # 第二个卷积层：输入通道数为 10，输出通道数为 20，卷积核大小为 5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # Dropout 层，作用于2D特征图
        self.conv2_drop = nn.Dropout2d()
        # 第一个全连接层：输入特征数为 320 (这个值取决于前面卷积和池化后的特征图大小)，输出特征数为 50
        self.fc1 = nn.Linear(320, 50)
        # 第二个全连接层 (输出层)：输入特征数为 50，输出特征数为 args.num_classes (MNIST 为 10)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x): # 定义前向传播逻辑
        # x -> conv1 -> max_pool2d (2x2) -> relu
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x -> conv2 -> conv2_drop -> max_pool2d (2x2) -> relu
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # 将特征图展平
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x)) # 通过第一个全连接层并应用 ReLU
        x = F.dropout(x, training=self.training) # 应用 Dropout，training=self.training 确保只在训练时生效
        x = self.fc2(x) # 通过输出层
        return F.log_softmax(x, dim=1) # 返回 log_softmax 输出，常用于 NLLLoss 损失函数


class CNNFashion_Mnist(nn.Module): # 定义一个针对 Fashion-MNIST 数据集的 CNN 模型
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        # 第一个卷积块：包含 Conv2d, BatchNorm2d, ReLU, MaxPool2d
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2), # 输入通道 1, 输出通道 16, 5x5 核, padding 2 保持尺寸
            nn.BatchNorm2d(16), # 批归一化
            nn.ReLU(),
            nn.MaxPool2d(2)) # 2x2 最大池化
        # 第二个卷积块
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2), # 输入通道 16, 输出通道 32, 5x5 核, padding 2
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        # 全连接层：输入特征数 7*7*32 (Fashion-MNIST 图像大小 28x28，经过两次 2x2 池化后变为 7x7)
        self.fc = nn.Linear(7*7*32, 10) # 输出 10 个类别

    def forward(self, x):
        out = self.layer1(x) # 通过第一个卷积块
        out = self.layer2(out) # 通过第二个卷积块
        out = out.view(out.size(0), -1) # 展平特征图
        out = self.fc(out) # 通过全连接层
        return out # 直接返回 logits (通常与 CrossEntropyLoss 配合使用)


class CNNCifar(nn.Module): # 定义一个针对 CIFAR-10 数据集的 CNN 模型
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        # CIFAR-10 图像通道数为 3
        self.conv1 = nn.Conv2d(3, 6, 5) # 输入通道 3, 输出通道 6, 5x5 核
        self.pool = nn.MaxPool2d(2, 2) # 2x2 最大池化，步长为 2
        self.conv2 = nn.Conv2d(6, 16, 5) # 输入通道 6, 输出通道 16, 5x5 核
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 输入特征数取决于卷积和池化后的尺寸
                                             # CIFAR-10 图像 32x32 -> conv1 (28x28) -> pool (14x14) -> conv2 (10x10) -> pool (5x5)
                                             # 所以是 16 (通道数) * 5 * 5
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes) # 输出层，类别数为 args.num_classes (CIFAR-10 为 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # conv1 -> relu -> pool
        x = self.pool(F.relu(self.conv2(x))) # conv2 -> relu -> pool
        x = x.view(-1, 16 * 5 * 5) # 展平
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1) # 返回 log_softmax 输出


class CNNCifar100(nn.Module): # 定义一个针对 CIFAR-100 数据集的 CNN 模型
    def __init__(self, args):
        super(CNNCifar100, self).__init__()
        # CIFAR-100 图像通道数为 3，类别数为 100
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1) # 输入通道 3, 输出通道 32, 3x3 核
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 输入通道 32, 输出通道 64, 3x3 核
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 输入通道 64, 输出通道 128, 3x3 核
        self.pool = nn.MaxPool2d(2, 2) # 2x2 最大池化，步长为 2
        self.dropout = nn.Dropout(0.5) # Dropout层防止过拟合
        
        # 全连接层
        # CIFAR-100 图像 32x32 -> conv1+pool (16x16) -> conv2+pool (8x8) -> conv3+pool (4x4)
        # 所以是 128 (通道数) * 4 * 4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, args.num_classes) # 输出层，CIFAR-100 有 100 个类别

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # conv1 -> relu -> pool
        x = self.pool(F.relu(self.conv2(x))) # conv2 -> relu -> pool
        x = self.pool(F.relu(self.conv3(x))) # conv3 -> relu -> pool
        x = x.view(-1, 128 * 4 * 4) # 展平
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1) # 返回 log_softmax 输出

# 注意：下面的 modelC 类定义中 super(AllConvNet, self).__init__() 似乎是一个笔误，
# 应该是 super(modelC, self).__init__()。
# 并且 AllConvNet 这个名字也没有在类名中使用。
class modelC(nn.Module): # 定义一个更复杂的 CNN 模型 (可能参考了 AllConvNet)
    def __init__(self, input_size, n_classes=10, **kwargs): # input_size 应该是输入通道数
        super(modelC, self).__init__() # 修正：应该是 super(modelC, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2) # stride=2 实现下采样
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1) # 1x1 卷积

        self.class_conv = nn.Conv2d(192, n_classes, 1) # 使用 1x1 卷积作为分类层

    def forward(self, x):
        x_drop = F.dropout(x, .2) # 输入层 Dropout
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5) # 中间层 Dropout
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5) # 中间层 Dropout
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out)) # 分类卷积层
        pool_out = F.adaptive_avg_pool2d(class_out, 1) # 自适应平均池化到 1x1
        pool_out.squeeze_(-1) # 移除大小为 1 的维度
        pool_out.squeeze_(-1) # 再次移除大小为 1 的维度，得到 (batch_size, n_classes) 的形状
        return pool_out # 返回 logits
