#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy # 用于深拷贝对象，特别是模型权重
import torch
from torchvision import datasets, transforms
# 从 sampling.py 导入数据划分函数
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


def get_dataset(args): # 根据参数加载并划分数据集
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    返回:
        train_dataset: 原始的完整训练数据集
        test_dataset: 原始的完整测试数据集
        user_groups: 一个字典,键是用户ID,值是分配给该用户的数据索引
    """

    if args.dataset == 'cifar': # 如果是 CIFAR-10 数据集
        data_dir = '../data/cifar/' # 数据存储目录
        # 定义 CIFAR-10 的图像预处理操作
        apply_transform = transforms.Compose(
            [transforms.ToTensor(), #转为张量并归一化到 [0,1],把PIL图片或numpy数组转换为PyTorch的张量（Tensor），并且把像素值从0<del>255缩放到0</del>1之间（除以255）
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 标准化，三个通道的均值和标准差都设为0.5
            #让数据分布更适合神经网络训练（均值为0，方差为1），有助于加快收敛速度和提升模型性能。保证不同通道的数值分布一致，避免某一通道对训练产生过大影响。

        # 加载 CIFAR-10 训练集
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)
        # 加载 CIFAR-10 测试集
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users (在用户间划分训练数据)
        if args.iid: # 如果是 IID 设置
            # Sample IID user data from Cifar
            user_groups = cifar_iid(train_dataset, args.num_users)
        else: # 如果是 Non-IID 设置
            # Sample Non-IID user data from Cifar
            if args.unequal: # 如果数据划分不均衡
                # Chose uneuqal splits for every user
                raise NotImplementedError() # 此处代码表示不均衡的 CIFAR Non-IID 划分未实现
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'cifar100': # 如果是 CIFAR-100 数据集
        data_dir = '../data/cifar100/' # 数据存储目录
        # 定义 CIFAR-100 的图像预处理操作
        apply_transform = transforms.Compose(
            [transforms.ToTensor(), 
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # 加载 CIFAR-100 训练集
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                          transform=apply_transform)
        # 加载 CIFAR-100 测试集
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                         transform=apply_transform)

        # sample training data amongst users (在用户间划分训练数据)
        if args.iid: # 如果是 IID 设置
            # Sample IID user data from CIFAR-100 (复用CIFAR-10的IID函数)
            user_groups = cifar_iid(train_dataset, args.num_users)
        else: # 如果是 Non-IID 设置
            # Sample Non-IID user data from CIFAR-100
            if args.unequal: # 如果数据划分不均衡
                # Chose uneuqal splits for every user
                raise NotImplementedError() # 此处代码表示不均衡的 CIFAR-100 Non-IID 划分未实现
            else: # 如果数据划分均衡
                # Chose euqal splits for every user (复用CIFAR-10的Non-IID函数)
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset in ['mnist', 'fmnist']: # 如果是 MNIST 或 Fashion-MNIST 数据集
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else: # args.dataset == 'fmnist'
            data_dir = '../data/fmnist/'

        # 定义 MNIST/Fashion-MNIST 的图像预处理操作
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]) # MNIST 的均值和标准差 (单通道)

        # 加载训练集和测试集
        if args.dataset == 'mnist':
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)
        elif args.dataset == 'fmnist':
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                                  transform=apply_transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                                 transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)
    else: # 如果数据集名称无法识别
        exit(f"Error: unrecognized dataset {args.dataset}")


    return train_dataset, test_dataset, user_groups


def average_weights(w,lens): # 计算模型权重的平均值 (FedAvg 算法的核心)
    """
    Returns the average of the weights.
    :param w: 一个列表，其中每个元素是一个客户端的模型权重 (state_dict)
    :return: 平均后的模型权重 (state_dict)
    """
    total = sum(lens)
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w[0][key] * (lens[0] / total)
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * (lens[i] / total)
    return w_avg
'''
 average_weights 实现是简单平均（每个客户端权重相同），而标准的 FedAvg 算法应该是加权平均，每个客户端的权重应与其本地数据量成正比
'''


def exp_details(args): # 打印实验的配置参数
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
