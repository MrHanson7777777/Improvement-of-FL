#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
'''
优化版本的参数配置文件
新增了学习率调度、数据增强、正则化等优化参数
'''

def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    # 定义联邦学习相关的参数
    parser.add_argument('--epochs', type=int, default=150,
                        help="number of rounds of training") # 全局训练的轮数，增加到150
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K") # 参与联邦学习的客户端总数
    parser.add_argument('--frac', type=float, default=0.2,
                        help='the fraction of clients: C') # 每轮选择更多客户端参与训练
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E") # 每个客户端本地训练的 epoch 数
    parser.add_argument('--local_bs', type=int, default=64,
                        help="local batch size: B") # 客户端本地训练的批量大小，增加到64
    parser.add_argument('--lr', type=float, default=0.03,
                        help='learning rate') # 学习率，调整到0.03
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)') # SGD 优化器的动量，提高到0.9

    # model arguments
    # 定义模型相关的参数
    parser.add_argument('--model', type=str, default='mlp', 
                        help='model name (mlp, cnn, enhanced_cnn, resnet50, wide_resnet)') # 使用的模型名称
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel') # (特定于某些CNN模型) 每种卷积核的数量
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution') # (特定于某些CNN模型) 卷积核大小，逗号分隔
    parser.add_argument('--num_channels', type=int, default=1, 
                        help="number of channels of imgs") # 图像的通道数 (例如 MNIST 为 1, CIFAR-10 为 3)
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None") # 归一化类型
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for mini-imagenet, 64 for omiglot.") # (特定于某些CNN模型) 卷积核数量/滤波器数量
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions") # 是否使用最大池化

    # =============== 新增优化参数 ===============
    
    # 学习率调度参数
    parser.add_argument('--scheduler', type=str, default='cosine',
                        help='learning rate scheduler (cosine, step, multistep, none)')
    parser.add_argument('--lr_decay', type=float, default=0.1,
                        help='learning rate decay factor for step scheduler')
    parser.add_argument('--lr_decay_step', type=int, default=50,
                        help='step size for learning rate decay')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='minimum learning rate for cosine annealing')
    
    # 正则化参数
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout rate for regularization')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (L2 regularization)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='label smoothing factor (0.0 means no smoothing)')
    
    # 数据增强参数
    parser.add_argument('--use_cutmix', action='store_true',
                        help='whether to use CutMix data augmentation')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                        help='CutMix alpha parameter')
    parser.add_argument('--use_mixup', action='store_true',
                        help='whether to use MixUp data augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='MixUp alpha parameter')
    parser.add_argument('--use_autoaugment', action='store_true',
                        help='whether to use AutoAugment')
    parser.add_argument('--randaug_n', type=int, default=2,
                        help='number of augmentation transformations in RandAugment')
    parser.add_argument('--randaug_m', type=int, default=14,
                        help='magnitude of augmentation transformations in RandAugment')
    
    # 训练策略参数
    parser.add_argument('--use_scheduler', action='store_true',
                        help='whether to use learning rate scheduler')
    parser.add_argument('--use_label_smoothing', action='store_true',
                        help='whether to use label smoothing')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='number of warmup epochs')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='gradient clipping value (0 means no clipping)')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='exponential moving average decay for model weights')
    parser.add_argument('--use_ema', action='store_true',
                        help='whether to use exponential moving average')
    
    # 模型特定参数
    parser.add_argument('--drop_path', type=float, default=0.2,
                        help='drop path rate for stochastic depth')
    parser.add_argument('--se_ratio', type=float, default=0.25,
                        help='squeeze-and-excitation ratio')
    
    # =============== 原有参数 ===============
    
    # other arguments
    # 定义其他参数
    parser.add_argument('--dataset', type=str, default='mnist', 
                        help="name of dataset") # 使用的数据集名称 (例如 'mnist', 'fmnist', 'cifar', 'cifar100')
    parser.add_argument('--num_classes', type=int, default=10, 
                        help="number of classes") # 数据集的类别数量 (MNIST/CIFAR-10: 10, CIFAR-100: 100)
    parser.add_argument('--gpu', type=int, default=None, 
                        help="To use cuda, set to a specific GPU ID. Default set to use CPU.") # 指定使用的 GPU ID，默认为 CPU
    parser.add_argument('--optimizer', type=str, default='sgd', 
                        help="type of optimizer") # 优化器类型 (例如 'sgd', 'adam')
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.') # 数据是否独立同分布 (IID)。1 表示 IID，0 表示 Non-IID
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for non-i.i.d setting (use 0 for equal splits)') # 在 Non-IID 设置下，数据划分是否不均衡
    parser.add_argument('--stopping_rounds', type=int, default=30,
                        help='rounds of early stopping') # 早停机制的轮数，增加到30以给模型更多时间
    parser.add_argument('--verbose', type=int, default=1, 
                        help='verbose') # 是否打印详细日志
    parser.add_argument('--seed', type=int, default=1, 
                        help='random seed') # 随机种子，用于实验复现
    
    args = parser.parse_args() # 解析命令行传入的参数
    return args # 返回解析后的参数对象


def get_optimized_args():
    """
    获取优化后的默认参数配置
    针对不同数据集和模型的推荐设置
    """
    import sys
    args = args_parser()
    
    # 检查是否有命令行参数，如果没有任何参数，则使用优化的默认设置
    if len(sys.argv) == 1:  # 只有脚本名，没有其他参数
        print("⚠️  未检测到命令行参数，使用CIFAR-10 + Enhanced CNN的优化配置")
        args.dataset = 'cifar'
        args.model = 'enhanced_cnn'
    else:
        # 有命令行参数时，严格按照用户输入执行
        print(f"✅ 使用用户指定参数: 数据集={args.dataset}, 模型={args.model}")
    
    # 根据数据集调整参数
    if args.dataset == 'cifar':
        if args.model in ['ultra_cnn', 'ultra_wide_resnet', 'cifar_ultra']:
            # 超高性能CIFAR-10模型配置 - 目标85%+
            print(f"🚀 启用CIFAR-10超高性能配置 for {args.model}")
            args.epochs = 300   # 充分训练
            args.frac = 0.3     # 更多客户端参与
            args.local_bs = 128 # 大批量训练
            args.local_ep = 3   # 减少本地轮数避免过拟合
            args.scheduler = 'cosine'
            args.warmup_epochs = 15  # 长warmup
            args.stopping_rounds = 50 # 更大的耐心
            
            if args.model == 'ultra_cnn':
                args.lr = 0.08
                args.weight_decay = 3e-4
            elif args.model == 'ultra_wide_resnet':
                args.lr = 0.1
                args.weight_decay = 5e-4
            elif args.model == 'cifar_ultra':
                args.lr = 0.06
                args.weight_decay = 2e-4
                
            # 启用所有数据增强
            args.use_cutmix = True
            args.use_mixup = True
            args.use_label_smoothing = True
            args.label_smoothing = 0.1
            args.dropout = 0.3
            
        elif args.model in ['resnet50', 'wide_resnet']:
            # CIFAR-10 + ResNet的高性能参数 - 目标85%+
            args.epochs = 200  # 充分训练
            args.lr = 0.05     # 适中的学习率
            args.local_bs = 64 # 大批量提高稳定性
            args.local_ep = 10 # 增加本地训练轮数
            args.scheduler = 'cosine'  # 余弦退火调度
            args.weight_decay = 1e-3   # 适度正则化
            args.label_smoothing = 0.1 # 标签平滑
            args.dropout = 0.2         # 适度dropout
            args.use_cutmix = True     # 启用数据增强
            args.use_mixup = True
            args.warmup_epochs = 10    # warmup帮助稳定训练
            
        elif args.model == 'enhanced_cnn':
            # CIFAR-10 + EfficientNet的高性能参数 - 目标85%+
            args.epochs = 150  # 充分训练
            args.lr = 0.03     # 适中的学习率
            args.local_bs = 64 # 大批量
            args.local_ep = 10 # 增加本地训练
            args.scheduler = 'cosine'
            args.weight_decay = 5e-4
            args.label_smoothing = 0.1
            args.dropout = 0.15
            args.use_cutmix = True
            args.use_mixup = False  # EfficientNet用CutMix即可
            args.warmup_epochs = 5
            
        else:
            # 基础CNN的高性能参数 - 目标80%+
            args.epochs = 120
            args.lr = 0.01
            args.local_bs = 64
            args.local_ep = 10
            args.scheduler = 'cosine'
            args.weight_decay = 5e-4
            args.label_smoothing = 0.05
            args.dropout = 0.1
            args.use_cutmix = False
            args.use_mixup = False
            args.warmup_epochs = 5
            
    elif args.dataset == 'cifar100':
        if args.model in ['resnet50', 'wide_resnet']:
            # CIFAR-100 + ResNet的优化参数
            args.epochs = 200  # CIFAR-100更复杂，需要更多轮次
            args.lr = 0.1
            args.scheduler = 'cosine'
            args.weight_decay = 5e-4
            args.label_smoothing = 0.2  # 更强的标签平滑
            args.dropout = 0.4
            args.use_cutmix = True
            args.use_mixup = True
            args.warmup_epochs = 10
            
    elif args.dataset in ['mnist', 'fmnist']:
        # MNIST/Fashion-MNIST的优化参数
        args.epochs = 50  # 这些数据集较简单
        args.lr = 0.01
        args.scheduler = 'step'
        args.weight_decay = 1e-4
        args.label_smoothing = 0.05
        args.dropout = 0.2
        args.use_cutmix = False  # 对于简单数据集，可能不需要
        args.use_mixup = False
    # 参数验证
    args = validate_args(args)
    
    return args


def validate_args(args):
    """
    验证参数组合的合理性，防止数据集和模型不匹配
    """
    print(f"\n🔍 验证参数组合...")
    print(f"数据集: {args.dataset}")
    print(f"模型: {args.model}")
    
    # 检查数据集和模型的兼容性
    if args.dataset in ['cifar', 'cifar100']:
        if args.model == 'mlp':
            print("⚠️  警告: MLP模型对图像分类任务效果较差，建议使用CNN系列模型")
            response = input("是否继续使用MLP模型？(y/n): ")
            if response.lower() != 'y':
                print("建议的模型: enhanced_cnn, resnet50, wide_resnet")
                exit(1)
        
        # 设置正确的通道数
        args.num_channels = 3
        args.num_classes = 10 if args.dataset == 'cifar' else 100
        
    elif args.dataset in ['mnist', 'fmnist']:
        # 设置正确的通道数
        args.num_channels = 1
        args.num_classes = 10
        
        if args.model in ['enhanced_cnn', 'resnet50', 'wide_resnet']:
            print("⚠️  警告: 高级模型对于简单数据集可能过于复杂")
    
    # 检查GPU设置
    if args.gpu is not None and args.gpu >= 0:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  警告: 指定了GPU但CUDA不可用，将使用CPU")
            args.gpu = None
        elif args.gpu >= torch.cuda.device_count():
            print(f"⚠️  警告: GPU {args.gpu} 不存在，系统有 {torch.cuda.device_count()} 个GPU，将使用CPU")
            args.gpu = None
    
    print(f"✅ 参数验证完成: {args.dataset} + {args.model}")
    print(f"   图像通道数: {args.num_channels}")
    print(f"   类别数量: {getattr(args, 'num_classes', '未设置')}")
    
    return args


if __name__ == '__main__':
    # 测试参数解析
    args = get_optimized_args()
    print("优化参数配置:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
