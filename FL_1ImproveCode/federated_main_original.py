#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os # 用于与操作系统交互，例如路径操作
import copy # 用于深拷贝
import time # 用于计时
import pickle # 用于序列化和反序列化 Python 对象 (例如保存训练结果)
import numpy as np
from tqdm import tqdm # 进度条

import torch
from tensorboardX import SummaryWriter # 用于 TensorBoard 日志记录，可视化训练过程

from options import args_parser
from update import LocalUpdate, test_inference # 从 update.py 导入本地更新类和测试函数
from models_original import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNCifar100
from utils import get_dataset, average_weights, exp_details # 从 utils.py 导入数据加载、权重平均和打印实验细节的函数

if __name__ == '__main__':
    start_time = time.time() # 记录开始时间
    print(start_time)
    print('原始版本（简单CNN架构）')

    # define paths
    path_project = os.path.abspath('..') # 获取项目上级目录的绝对路径 (可能用于保存文件)
    logger = SummaryWriter('../logs') # 初始化 TensorBoard 的 SummaryWriter，日志保存在 ../logs 目录下

    args = args_parser() # 解析命令行参数
    exp_details(args) # 打印实验配置详情

    if args.gpu is not None and torch.cuda.is_available(): # 检查GPU是否可用
        try:
            torch.cuda.set_device(int(args.gpu)) # 设置 GPU
            device = 'cuda'
            print(f"Using GPU: {args.gpu}")
        except:
            print(f"GPU {args.gpu} not available, using CPU instead")
            device = 'cpu'
    else:
        device = 'cpu' # 确定设备
        print("Using CPU")

    # load dataset and user groups (加载数据集和用户数据划分)
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL (构建全局模型，使用原始简单架构)
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
        elif args.dataset == 'cifar100':
            global_model = CNNCifar100(args=args)
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64,
                           dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device) # 模型移至设备
    global_model.train() # 设置为训练模式
    print(global_model) # 打印模型结构

    # copy weights (获取全局模型的初始权重)
    global_weights = global_model.state_dict() # state_dict() 返回包含模型所有参数的字典

    # Training (联邦学习训练过程)
    train_loss, train_accuracy = [], [] # 记录每轮的平均训练损失和准确率
    val_acc_list, net_list = [], [] # (未使用)
    cv_loss, cv_acc = [], [] # (未使用)
    print_every = 2 # 每隔多少轮打印一次训练统计信息

    for epoch in tqdm(range(args.epochs)): # 外层循环：全局通信轮次
        local_weights, local_losses = [], [] # 存储本轮所有参与客户端上传的本地模型权重和损失
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train() # 确保全局模型在分发给客户端前处于训练模式
        
        # 选择参与本轮训练的客户端
        m = max(int(args.frac * args.num_users), 1) # 计算参与客户端数量 (至少为1)
        # 从所有客户端中随机选择 m 个不重复的客户端索引
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users: # 内层循环：遍历被选中的客户端
            # 创建 LocalUpdate 实例，传入客户端的本地数据 (user_groups[idx])
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            # 客户端进行本地训练
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w)) # 收集本地更新后的模型权重
            local_losses.append(copy.deepcopy(loss)) # 收集本地训练的损失

        # update global weights (聚合客户端权重，更新全局模型)
        lens = [len(user_groups[idx]) for idx in idxs_users]  # 计算每个客户端的数据量
        global_weights = average_weights(local_weights, lens) # 调用 utils.py 中的 average_weights 函数

        # update global model with new weights
        global_model.load_state_dict(global_weights) # 将聚合后的平均权重加载到全局模型中

        loss_avg = sum(local_losses) / len(local_losses) # 计算本轮所有参与客户端的平均本地损失
        train_loss.append(loss_avg) # 记录

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval() # 将全局模型设置为评估模式
        for c in range(args.num_users): # 遍历所有客户端
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger)
            acc, loss = local_model.inference(model=global_model) # 在该客户端的本地测试集上评估全局模型
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc)) # 计算所有客户端上的平均准确率

        # print global training loss after every 'print_every' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training (训练完成后，在全局测试集上评估最终的全局模型)
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {len(train_loss)} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1])) # 最终的平均训练准确率
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc)) # 最终的测试准确率

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time)) # 打印总运行时间
