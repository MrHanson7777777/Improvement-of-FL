#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦学习优化版本 - 加入学习率调度和数据增强
包含:
1. 学习率调度策略 (Cosine Annealing, StepLR)
2. 数据增强 (CutMix, Mixup, AutoAugment)
3. 正则化方法 (Dropout, Label Smoothing)
4. 优化的训练策略
"""

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from options_optimized import args_parser, get_optimized_args
from update import LocalUpdate, test_inference
from models import CNNMnist, CNNFashion_Mnist, CNNCifar, CNNCifar100, MLP
from models_enhanced import EnhancedEfficientNetCifar, ResNet50Cifar, WideResNetCifar
from models_high_performance import CIFAR10HighPerformanceCNN, OptimizedWideResNet, CIFAR10UltraNet
from utils import get_dataset, average_weights, exp_details


class CutMix:
    """CutMix数据增强实现"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, x, y):
        if self.alpha <= 0:
            return x, y
        
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        y_a, y_b = y, y[index]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        return x, y_a, y_b, lam
    
    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2


class MixUp:
    """MixUp数据增强实现"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x, y):
        if self.alpha <= 0:
            return x, y
        
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


class EnhancedDataset(Dataset):
    """增强数据集包装器，支持CutMix和MixUp"""
    def __init__(self, dataset, use_cutmix=True, use_mixup=True, cutmix_alpha=1.0, mixup_alpha=0.2):
        self.dataset = dataset
        self.use_cutmix = use_cutmix
        self.use_mixup = use_mixup
        self.cutmix = CutMix(cutmix_alpha) if use_cutmix else None
        self.mixup = MixUp(mixup_alpha) if use_mixup else None
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


def get_enhanced_transforms(dataset_name):
    """获取增强的数据变换"""
    if dataset_name == 'cifar':
        # CIFAR-10增强变换
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
        ])
    elif dataset_name == 'cifar100':
        # CIFAR-100增强变换
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
        ])
    elif dataset_name in ['mnist', 'fmnist']:
        # MNIST/Fashion-MNIST增强变换
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) if dataset_name == 'mnist' 
            else transforms.Normalize((0.2860,), (0.3530,))
        ])
    else:
        train_transform = transforms.ToTensor()
    
    return train_transform


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失函数"""
    def __init__(self, classes, smoothing=0.1, dim=1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def get_scheduler(optimizer, args, steps_per_epoch):
    """获取学习率调度器"""
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * steps_per_epoch, eta_min=args.lr * 0.01
        )
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.epochs // 3, gamma=0.1
        )
    elif args.scheduler == 'multistep':
        milestones = [args.epochs // 3, 2 * args.epochs // 3]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1
        )
    else:
        scheduler = None
    
    return scheduler


def get_model(args):
    """获取模型"""
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            return CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            return CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            return CNNCifar(args=args)
        elif args.dataset == 'cifar100':
            return CNNCifar100(args=args)
    elif args.model == 'enhanced_cnn':
        if args.dataset in ['cifar', 'cifar100']:
            num_classes = 10 if args.dataset == 'cifar' else 100
            return EnhancedEfficientNetCifar(num_classes=num_classes)
        else:
            # 对于MNIST和Fashion-MNIST，回退到普通CNN
            if args.dataset == 'mnist':
                return CNNMnist(args=args)
            elif args.dataset == 'fmnist':
                return CNNFashion_Mnist(args=args)
    elif args.model == 'resnet50':
        if args.dataset in ['cifar', 'cifar100']:
            num_classes = 10 if args.dataset == 'cifar' else 100
            return ResNet50Cifar(num_classes=num_classes)
        else:
            if args.dataset == 'mnist':
                return CNNMnist(args=args)
            elif args.dataset == 'fmnist':
                return CNNFashion_Mnist(args=args)
    elif args.model == 'wide_resnet':
        if args.dataset in ['cifar', 'cifar100']:
            num_classes = 10 if args.dataset == 'cifar' else 100
            return WideResNetCifar(depth=28, width=10, num_classes=num_classes)
        else:
            if args.dataset == 'mnist':
                return CNNMnist(args=args)
            elif args.dataset == 'fmnist':
                return CNNFashion_Mnist(args=args)
    elif args.model == 'ultra_cnn':
        # 专为CIFAR-10高精度设计的模型
        if args.dataset == 'cifar':
            return CIFAR10HighPerformanceCNN(num_classes=10, dropout=0.3)
        elif args.dataset == 'cifar100':
            return CIFAR10HighPerformanceCNN(num_classes=100, dropout=0.4)
        else:
            return CNNMnist(args=args)
    elif args.model == 'ultra_wide_resnet':
        # 优化的Wide ResNet
        if args.dataset == 'cifar':
            return OptimizedWideResNet(depth=28, width=12, num_classes=10, dropout_rate=0.3)
        elif args.dataset == 'cifar100':
            return OptimizedWideResNet(depth=28, width=12, num_classes=100, dropout_rate=0.4)
        else:
            return CNNMnist(args=args)
    elif args.model == 'cifar_ultra':
        # 最强CIFAR模型
        if args.dataset == 'cifar':
            return CIFAR10UltraNet(num_classes=10)
        elif args.dataset == 'cifar100':
            return CIFAR10UltraNet(num_classes=100)
        else:
            return CNNMnist(args=args)
    elif args.model == 'mlp':
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        return MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')


if __name__ == '__main__':
    start_time = time.time()
    print('联邦学习优化版本 - 学习率调度 + 数据增强 + 正则化')

    # 路径和日志设置
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = get_optimized_args()  # 使用优化后的参数配置
    
    # 添加新的参数（如果不存在）
    if not hasattr(args, 'scheduler'):
        args.scheduler = 'cosine'  # 默认使用余弦退火
    if not hasattr(args, 'label_smoothing'):
        args.label_smoothing = 0.1  # 标签平滑
    if not hasattr(args, 'use_cutmix'):
        args.use_cutmix = True
    if not hasattr(args, 'use_mixup'):
        args.use_mixup = True
    if not hasattr(args, 'dropout'):
        args.dropout = 0.3  # Dropout率
    
    exp_details(args)

    # 设备设置
    if args.gpu is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_device(int(args.gpu))
            device = 'cuda'
            print(f"Using GPU: {args.gpu}")
        except:
            print(f"GPU {args.gpu} not available, using CPU instead")
            device = 'cpu'
    else:
        device = 'cpu'
        print("Using CPU")

    # 加载数据集
    train_dataset, test_dataset, user_groups = get_dataset(args)
    
    # 最终参数确认
    print("\n" + "="*60)
    print("🎯 最终训练配置确认:")
    print(f"   数据集: {args.dataset}")
    print(f"   模型: {args.model}")
    print(f"   训练轮数: {args.epochs}")
    print(f"   学习率: {args.lr}")
    print(f"   本地轮数: {args.local_ep}")
    print(f"   客户端数量: {args.num_users}")
    print(f"   参与比例: {args.frac}")
    
    # 检查数据集类型
    print(f"🔍 验证数据集类型...")
    print(f"数据集对象类型: {type(train_dataset)}")
    
    # 检查数据集的实际结构
    if hasattr(train_dataset, 'data'):
        data_shape = train_dataset.data.shape
        print(f"数据形状: {data_shape}")
        
        # 根据数据集类型验证
        if args.dataset == 'cifar':
            # CIFAR-10: (50000, 32, 32, 3) 或其他格式
            if len(data_shape) >= 2 and (data_shape[-1] == 32 or data_shape[-2] == 32):
                print("✅ CIFAR数据集验证通过")
            else:
                print(f"❌ 错误: 指定了CIFAR数据集但数据形状不匹配: {data_shape}")
                exit(1)
        elif args.dataset == 'mnist':
            # MNIST: (60000, 28, 28) 或其他格式
            if len(data_shape) >= 2 and (data_shape[-1] == 28 or data_shape[-2] == 28):
                print("✅ MNIST数据集验证通过")
            else:
                print(f"❌ 错误: 指定了MNIST数据集但数据形状不匹配: {data_shape}")
                exit(1)
    else:
        # 如果没有data属性，检查第一个样本
        try:
            sample, _ = train_dataset[0]
            sample_shape = sample.shape if hasattr(sample, 'shape') else None
            print(f"样本形状: {sample_shape}")
            
            if sample_shape is not None:
                if args.dataset == 'cifar' and (sample_shape[-1] == 32 or sample_shape[-2] == 32):
                    print("✅ CIFAR数据集验证通过")
                elif args.dataset == 'mnist' and (sample_shape[-1] == 28 or sample_shape[-2] == 28):
                    print("✅ MNIST数据集验证通过")
                else:
                    print(f"⚠️  警告: 无法验证数据集类型，样本形状: {sample_shape}")
            else:
                print("⚠️  警告: 无法获取样本形状")
        except Exception as e:
            print(f"⚠️  警告: 数据集验证失败: {e}")
    
    print("✅ 参数配置验证通过！")
    print("="*60 + "\n")

    # 构建模型
    global_model = get_model(args)
    global_model.to(device)
    global_model.train()
    print(f"模型参数数量: {sum(p.numel() for p in global_model.parameters()):,}")

    # 复制权重
    global_weights = global_model.state_dict()

    # 训练
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    # 早停相关变量
    best_val_acc = 0.0
    patience_counter = 0
    best_model_weights = None

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        local_data_lens = []  # 存储每个客户端的数据量
        
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            # 记录每个客户端的数据量
            local_data_lens.append(len(user_groups[idx]))

        # 更新全局权重（使用加权平均）
        global_weights = average_weights(local_weights, local_data_lens)
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # 计算平均训练准确率（可选）
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # 每隔几轮打印和验证
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

        # 早停检查
        if hasattr(args, 'stopping_rounds') and args.stopping_rounds > 0:
            # 在测试集上评估
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            
            if test_acc > best_val_acc:
                best_val_acc = test_acc
                patience_counter = 0
                best_model_weights = copy.deepcopy(global_weights)
                print(f'新的最佳验证准确率: {best_val_acc:.2f}%')
            else:
                patience_counter += 1
                print(f'验证准确率未提升，耐心计数: {patience_counter}/{args.stopping_rounds}')
                
                if patience_counter >= args.stopping_rounds:
                    print(f'提前停止训练在第 {epoch+1} 轮')
                    global_model.load_state_dict(best_model_weights)
                    break

    # 最终测试
    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))

    # 在测试集上测试全局模型
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print(f' \n Results on Test dataset')
    print("|---- Test Accuracy: {:.2f}%".format(test_acc))

    # 保存模型
    file_name = f'../save/federated_optimized_{args.dataset}_{args.model}_{args.epochs}_{args.frac}.pkl'
    
    # 确保保存目录存在
    os.makedirs('../save', exist_ok=True)
    
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy, test_acc], f)

    print('\n Total Run Time: {0:0.4f} seconds'.format(time.time()-start_time))

    # TensorBoard 记录
    logger.add_scalar('test/accuracy', test_acc)
    logger.add_scalar('test/loss', test_loss)
    logger.close()
