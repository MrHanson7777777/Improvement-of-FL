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
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNCifar100, ResNet18Fed, replace_bn_with_gn
from model_factory import get_model, get_recommended_model, list_available_models  # 新的模型工厂
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

    if args.gpu is not None and torch.cuda.is_available(): # 检查GPU是否可用
        try:
            torch.cuda.set_device(int(args.gpu)) # 设置 GPU
            device = 'cuda'
        except:
            print(f"GPU {args.gpu} not available, using CPU instead")
            device = 'cpu'
    else:
        device = 'cpu' # 确定设备

    # load dataset and user groups (加载数据集和用户数据划分)
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL (构建全局模型，支持原有模型和新的优化模型)
    print(f"正在构建模型: {args.model} for dataset: {args.dataset}")
    
    # 首先尝试使用新的优化模型
    try:
        if args.model in ['resnet18', 'resnet18_fed']:
            # 使用新的ResNet18Fed优化模型
            if args.dataset == 'mnist':
                global_model = get_model('mnist', 'optimized')  # CNN_MNIST_Optimized
            elif args.dataset == 'cifar':
                global_model = get_model('cifar10', 'resnet18')  # ResNet18_CIFAR10_Fed 
            elif args.dataset == 'cifar100':
                global_model = get_model('cifar100', 'resnet18')  # ResNet18_CIFAR100_Fed
            else:
                raise ValueError(f"ResNet18不支持数据集: {args.dataset}")
                
        elif args.model in ['efficientnet', 'efficient']:
            # 使用新的EfficientNet优化模型
            if args.dataset == 'cifar':
                global_model = get_model('cifar10', 'efficientnet')  # EfficientNet_CIFAR10
            elif args.dataset == 'cifar100':
                global_model = get_model('cifar100', 'efficientnet')  # EfficientNet_CIFAR100
            else:
                raise ValueError(f"EfficientNet不支持数据集: {args.dataset}")
                
        elif args.model == 'densenet':
            # 使用新的DenseNet模型 (仅CIFAR-100)
            if args.dataset == 'cifar100':
                global_model = get_model('cifar100', 'densenet')  # DenseNet_CIFAR100
            else:
                raise ValueError(f"DenseNet不支持数据集: {args.dataset}")
                
        elif args.model in ['cnn_optimized', 'cnn_opt']:
            # 使用新的优化CNN模型
            if args.dataset == 'mnist':
                global_model = get_model('mnist', 'optimized')  # CNN_MNIST_Optimized
            else:
                raise ValueError(f"优化CNN目前仅支持MNIST数据集")
                
        else:
            # 使用推荐模型或抛出异常尝试原有模型
            raise ValueError("尝试原有模型")
            
        print(f"✅ 成功加载优化模型: {global_model.__class__.__name__}")
        
    except Exception as e:
        print(f"⚠️  优化模型加载失败: {e}")
        print(f"🔄 回退到原有模型...")
        
        # 回退到原有模型构建逻辑
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
        elif args.model == 'resnet':
            # ResNet for federated learning
            if args.dataset == 'cifar':
                global_model = ResNet18Fed(num_classes=args.num_classes)
                # 使用GroupNorm替换BatchNorm以提高联邦学习性能
                global_model = replace_bn_with_gn(global_model)
            elif args.dataset == 'cifar100':
                global_model = ResNet18Fed(num_classes=100)
                global_model = replace_bn_with_gn(global_model)
            else:
                print(f"ResNet not implemented for dataset {args.dataset}, using CNN instead")
                if args.dataset == 'mnist':
                    global_model = CNNMnist(args=args)
                elif args.dataset == 'fmnist':
                    global_model = CNNFashion_Mnist(args=args)
        elif args.model == 'mlp':
            # Multi-layer preceptron
            img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
        else:
            print(f"❌ 错误: 不支持的模型 '{args.model}'")
            print("📋 支持的模型:")
            print("   原有模型: mlp, cnn, resnet")
            print("   优化模型: resnet18, efficientnet, densenet, cnn_optimized")
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
    
    # 早停机制参数
    best_val_acc = 0.0
    patience = args.stopping_rounds  # 使用参数中的early stopping rounds
    patience_counter = 0
    best_global_weights = None

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
        global_weights = average_weights(local_weights) # 调用 utils.py 中的 average_weights 函数
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
            # 修复BUG: 使用正确的客户端索引
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger)
            acc, loss = local_model.inference(model=global_model) # 在该客户端的本地测试集上评估全局模型
            list_acc.append(acc)
            list_loss.append(loss) # (这个 loss 是本地测试集上的，与 train_loss 不同)
        train_accuracy.append(sum(list_acc)/len(list_acc)) # 计算所有客户端上的平均准确率
        
        # 早停机制检查
        current_val_acc = train_accuracy[-1]
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            patience_counter = 0
            best_global_weights = copy.deepcopy(global_model.state_dict())
            print(f'New best validation accuracy: {100*best_val_acc:.2f}%')
        else:
            patience_counter += 1
            print(f'Validation accuracy did not improve. Patience: {patience_counter}/{patience}')

        # print global training loss after every 'print_every' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}') # 打印的是到目前为止所有轮次的平均训练损失
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1])) # 打印当前轮次的平均训练准确率
            
        # 早停检查
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} global rounds')
            if best_global_weights is not None:
                global_model.load_state_dict(best_global_weights)
                print("Loaded best model weights for final testing.")
            break

    # Test inference after completion of training (训练完成后，在全局测试集上评估最终的全局模型)
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {len(train_loss)} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1])) # 最终的平均训练准确率
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc)) # 最终的测试准确率

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time)) # 打印总运行时间
