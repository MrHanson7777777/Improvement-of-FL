#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
'''
该文件主要用于定义和解析命令行参数。这些参数允许用户在运行脚本时配置联邦学习实验的各种超参数,例如学习率、epoch数量、客户端数量、模型类型等。
'''

def args_parser():
    parser = argparse.ArgumentParser() # 创建一个 ArgumentParser 对象

    # federated arguments (Notation for the arguments followed from paper)
    # 定义联邦学习相关的参数
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training") # 全局训练的轮数
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K") # 参与联邦学习的客户端总数
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C') # 每轮选择参与训练的客户端比例
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E") # 每个客户端本地训练的 epoch 数
    parser.add_argument('--local_bs', type=int, default=32,
                        help="local batch size: B") # 客户端本地训练的批量大小，增加到32
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate') # 学习率，降低到0.001
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)') # SGD 优化器的动量
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (L2 regularization)') # 权重衰减
    parser.add_argument('--lr_scheduler', type=str, default='none',
                        help='learning rate scheduler: none, step, exp, cosine') # 学习率调度策略
    parser.add_argument('--lr_step_size', type=int, default=20,
                        help='step size for StepLR scheduler') # StepLR调度器的步长
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='gamma for StepLR and ExponentialLR scheduler') # 学习率衰减因子
    parser.add_argument('--cosine_t_max', type=int, default=50,
                        help='T_max for CosineAnnealingLR scheduler') # 余弦退火调度器的最大周期
    parser.add_argument('--adam_beta1', type=float, default=0.9,
                        help='beta1 for Adam and AdamW optimizer') # Adam优化器的beta1参数
    parser.add_argument('--adam_beta2', type=float, default=0.999,
                        help='beta2 for Adam and AdamW optimizer') # Adam优化器的beta2参数
    parser.add_argument('--adam_eps', type=float, default=1e-8,
                        help='eps for Adam and AdamW optimizer') # Adam优化器的eps参数

    # model arguments
    # 定义模型相关的参数
    parser.add_argument('--model', type=str, default='mlp', 
                       help='model name: 原有模型(mlp, cnn, resnet) | 优化模型(resnet18, efficientnet, densenet, cnn_optimized)') # 使用的模型名称，支持新的优化模型
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel') # (特定于某些CNN模型) 每种卷积核的数量
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution') # (特定于某些CNN模型) 卷积核大小，逗号分隔
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs") # 图像的通道数 (例如 MNIST 为 1, CIFAR-10 为 3)
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None") # 归一化类型
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.") # (特定于某些CNN模型) 卷积核数量/滤波器数量
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions") # 是否使用最大池化

    # other arguments
    # 定义其他参数
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset") # 使用的数据集名称 (例如 'mnist', 'fmnist', 'cifar', 'cifar100')
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes") # 数据集的类别数量 (MNIST/CIFAR-10: 10, CIFAR-100: 100)
    parser.add_argument('--gpu', type=int, default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.") # 指定使用的 GPU ID，默认为 CPU
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer: sgd, adam, adamw") # 优化器类型
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.') # 数据是否独立同分布 (IID)。1 表示 IID，0 表示 Non-IID
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)') # 在 Non-IID 设置下，数据划分是否不均衡
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping') # 早停机制的轮数
    parser.add_argument('--verbose', type=int, default=1, help='verbose') # 是否打印详细日志
    parser.add_argument('--seed', type=int, default=1, help='random seed') # 随机种子，用于实验复现
    args = parser.parse_args() # 解析命令行传入的参数
    return args # 返回解析后的参数对象

'''
关键函数和语法:
●	import argparse: 导入Python标准库中的argparse模块,它是专门用来处理命令行参数的。
●	parser = argparse.ArgumentParser(): 创建一个ArgumentParser对象。这个对象包含了所有必要的参数信息,并能从sys.argv中解析出这些参数。
●	parser.add_argument(): 这个方法用于向ArgumentParser对象添加一个你期望程序接受的命令行选项。
○	'--epochs': 参数的名称，通常以--开头表示可选参数。
○	type=int: 指定参数的类型。argparse会尝试将输入转换为这个类型。
○	default=10: 如果命令行中没有提供这个参数，则使用这个默认值。
○	help="...": 参数的描述信息，当用户使用-h或--help选项时会显示。
●	args = parser.parse_args(): 这个方法会检查命令行,将每个参数转换成适当的类型,然后调用相应的动作。它返回一个包含所有参数及其值的对象（通常是一个Namespace对象）。
●	return args: 函数返回这个包含所有已解析参数的对象，以便在其他脚本中使用。

'''