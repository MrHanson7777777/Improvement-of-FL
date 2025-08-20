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
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNCifar100
from utils import get_dataset, average_weights, exp_details # 从 utils.py 导入数据加载、权重平均和打印实验细节的函数

if __name__ == '__main__':
    start_time = time.time() # 记录开始时间
    print(start_time)
    print('尝试')

    # define paths
    path_project = os.path.abspath('..') # 获取项目上级目录的绝对路径 (可能用于保存文件)
    logger = SummaryWriter('../logs') # 初始化 TensorBoard 的 SummaryWriter，日志保存在 ../logs 目录下

    args = args_parser() # 解析命令行参数
    exp_details(args) # 打印实验配置详情
    '''
    功能：调用 utils.py 文件中的 exp_details(args) 函数，把刚刚解析到的所有实验参数详细打印出来。
    意义：方便你在控制台/日志中确认本次实验的所有配置，避免参数设置错误，便于后续复现实验和调试。
    '''

    if args.gpu: # 如果指定了 GPU
        torch.cuda.set_device(int(args.gpu)) # 将 GPU ID 转换为整数并设置设备
    device = 'cuda' if args.gpu else 'cpu' # 确定设备

    # load dataset and user groups (加载数据集和用户数据划分)
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL (构建全局模型，与 baseline_main.py 类似)
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
    global_model.train() # 设置为训练模式 (尽管全局模型主要通过聚合更新，但初始状态和分发给客户端时应为训练模式)
    print(global_model) # 打印模型结构

    # copy weights (获取全局模型的初始权重)
    global_weights = global_model.state_dict() # state_dict() 返回包含模型所有参数的字典
    '''
    获取当前全局模型的所有参数（权重和偏置）并保存为一个字典
    global_model.state_dict() 会返回一个包含模型所有可学习参数(如权重、偏置)的有序字典(OrderedDict)。
    这样做的目的是保存全局模型的当前参数状态，后续可以用这些参数分发给各个客户端，或者在聚合后更新全局模型。
    在联邦学习中，每一轮开始时，客户端会拿到全局模型的参数进行本地训练，训练后再上传参数，最后服务器端聚合这些参数，更新全局模型。
    例如
    OrderedDict([
    ('fc1.weight', tensor([[ 0.01, -0.02, ...], [...], ...])),   # 第一层全连接的权重
    ('fc1.bias', tensor([0.0, 0.0, ...])),                      # 第一层全连接的偏置
    ('fc2.weight', tensor([[...], [...], ...])),                # 第二层全连接的权重
    ('fc2.bias', tensor([0.0, 0.0, ...]))                       # 第二层全连接的偏置
    ])
    '''

    # Training (联邦学习训练过程)
    train_loss, train_accuracy = [], [] # 记录每轮的平均训练损失和准确率
    val_acc_list, net_list = [], [] # (未使用)
    cv_loss, cv_acc = [], [] # (未使用)
    print_every = 2 # 每隔多少轮打印一次训练统计信息
    val_loss_pre, counter = 0, 0 # (未使用，可能与早停相关但未实现)

    for epoch in tqdm(range(args.epochs)): # 外层循环：全局通信轮次
        local_weights, local_losses = [], [] # 存储本轮所有参与客户端上传的本地模型权重和损失
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train() # 确保全局模型在分发给客户端前处于训练模式
        '''
        如果你选择了 CNNMnist,那么:
        全局的 global_model 就是一个 CNNMnist 实例。
        每个用户本地训练时，也是用 copy.deepcopy(global_model)，即每个客户端拿到的模型结构和参数，都是和全局模型一模一样的 CNNMnist
        每轮通信时，客户端会在自己的数据上训练这个模型，然后把更新后的参数上传，最后服务器端聚合这些参数，更新全局的 CNNMnist 模型。
        整个联邦学习过程中，模型结构始终保持一致，只是参数在不断更新。
        '''
        # 选择参与本轮训练的客户端
        m = max(int(args.frac * args.num_users), 1) # 计算参与客户端数量 (至少为1)
                                                    # args.frac 是参与比例，args.num_users 是总客户端数
        # 从所有客户端中随机选择 m 个不重复的客户端索引
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        '''idxs_users:在所有用户中随机选出5个用户的编号,比如 [2, 5, 7, 1, 9]'''

        for idx in idxs_users: # 内层循环：遍历被选中的客户端
            '''假设 idxs_users = [2, 5, 7, 1, 9],即本轮选中了第2、5、7、1、9号用户。'''
            # 创建 LocalUpdate 实例，传入客户端的本地数据 (user_groups[idx])
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            # 客户端进行本地训练：
            # copy.deepcopy(global_model) 确保每个客户端拿到的是当前全局模型的一个副本
            # global_round=epoch 传递当前全局轮次信息
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w)) # 收集本地更新后的模型权重
            local_losses.append(copy.deepcopy(loss)) # 收集本地训练的损失
            '''
            第一次循环,idx=2,用第2号用户的数据训练模型,得到本地权重w2和损失loss2,加入列表。
            第二次循环,idx=5,用第5号用户的数据训练模型,得到w5和loss5,加入列表。
            依次类推，直到所有被选中的用户都训练完毕。
            local_weights = [w2, w5, w7, w1, w9]
            local_losses = [loss2, loss5, loss7, loss1, loss9]
            '''

        # update global weights (聚合客户端权重，更新全局模型)
        lens = [len(user_groups[idx]) for idx in idxs_users]
        global_weights = average_weights(local_weights, lens) # 调用 utils.py 中的 average_weights 函数
        '''这一步会把所有本地权重(如w2, w5, w7, w1, w9)做平均,得到新的全局权重'''

        # update global model with new weights
        global_model.load_state_dict(global_weights) # 将聚合后的平均权重加载到全局模型中

        loss_avg = sum(local_losses) / len(local_losses) # 计算本轮所有参与客户端的平均本地损失
        train_loss.append(loss_avg) # 记录

        # Calculate avg training accuracy over all users at every epoch
        # (在每个全局轮次结束后，在所有客户端的本地数据上评估当前全局模型的平均准确率)
        # 注意：这部分计算开销较大，因为它涉及到在所有客户端上进行推断。
        # 在实际大规模联邦学习中，可能只在部分客户端或一个代理验证集上进行。
        list_acc, list_loss = [], []
        global_model.eval() # 将全局模型设置为评估模式
        for c in range(args.num_users): # 遍历所有客户端 (不仅仅是本轮参与训练的)
            # 为每个客户端创建一个 LocalUpdate 实例 (这里主要用它的 inference 方法和测试数据加载器)
            # user_groups[idx] 应该是 user_groups[c] 来获取第 c 个用户的数据
            # BUG: 这里使用了 idxs_users 中的最后一个 idx，而不是当前循环的 c。
            # 应该改为： local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[c], logger=logger)
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger) # BUG in user_groups index
            acc, loss = local_model.inference(model=global_model) # 在该客户端的本地测试集上评估全局模型
            list_acc.append(acc)
            list_loss.append(loss) # (这个 loss 是本地测试集上的，与 train_loss 不同)
        train_accuracy.append(sum(list_acc)/len(list_acc)) # 计算所有客户端上的平均准确率

        # print global training loss after every 'print_every' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}') # 打印的是到目前为止所有轮次的平均训练损失
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1])) # 打印当前轮次的平均训练准确率

    # Test inference after completion of training (训练完成后，在全局测试集上评估最终的全局模型)
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1])) # 最终的平均训练准确率
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc)) # 最终的测试准确率

    # Saving the objects train_loss and train_accuracy: (保存训练过程中的损失和准确率数据)
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs) # 构建保存文件名

    with open(file_name, 'wb') as f: # 以二进制写模式打开文件
        pickle.dump([train_loss, train_accuracy], f) # 使用 pickle 将数据序列化并保存到文件

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time)) # 打印总运行时间

    # PLOTTING (optional) (可选的绘图部分，被注释掉了)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg') # 设置 matplotlib 后端为 'Agg'，适用于无图形界面的服务器环境

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
