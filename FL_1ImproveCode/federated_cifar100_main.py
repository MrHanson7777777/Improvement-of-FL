#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CIFAR-100专用联邦学习训练脚本
针对EfficientNet在CIFAR-100上的最优性能调优
"""

import copy
import numpy as np
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import argparse
import time
import os
import random
from collections import defaultdict
import math

from models_cifar100_optimized import (
    get_cifar100_efficientnet, 
    get_cifar100_transforms, 
    LabelSmoothingCrossEntropy,
    CosineAnnealingWarmRestarts,
    enable_mixed_precision
)
from ultra_model_cifar100 import get_ultra_cifar100_model, LabelSmoothingCrossEntropy
from lightweight_ultra_model import get_lightweight_ultra_cifar100_model
from utils import get_dataset, average_weights, exp_details
from update_optimized import LocalUpdate
from sampling import mnist_iid, mnist_noniid, cifar_iid


class CIFAR100AdvancedSampling:
    """CIFAR-100的高级数据分布策略"""
    
    @staticmethod
    def cifar100_advanced_noniid(dataset, num_users, alpha=0.5):
        """使用Dirichlet分布创建更现实的non-IID分布"""
        num_items = int(len(dataset) / num_users)
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        
        # 获取标签
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
        num_classes = 100
        
        # 使用Dirichlet分布
        for i in range(num_users):
            # 每个客户端的类别分布
            proportions = np.random.dirichlet(np.repeat(alpha, num_classes))
            proportions = proportions / proportions.sum()
            
            # 根据比例分配数据
            dict_users[i] = []
            for class_id in range(num_classes):
                class_indices = np.where(targets == class_id)[0]
                class_size = int(proportions[class_id] * num_items)
                if class_size > 0 and len(class_indices) > 0:
                    selected = np.random.choice(class_indices, 
                                              min(class_size, len(class_indices)), 
                                              replace=False)
                    dict_users[i].extend(selected.tolist())
            
            # 确保每个客户端有足够的数据
            while len(dict_users[i]) < num_items // 2:
                remaining_indices = list(set(all_idxs) - set([idx for user_data in dict_users.values() for idx in user_data]))
                if remaining_indices:
                    dict_users[i].append(random.choice(remaining_indices))
                else:
                    break
        
        return dict_users
    
    @staticmethod
    def cifar100_balanced_noniid(dataset, num_users, classes_per_user=10):
        """每个客户端拥有固定数量的类别"""
        num_items = int(len(dataset) / num_users)
        dict_users = {i: [] for i in range(num_users)}
        
        # 获取标签
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
        
        # 为每个用户分配类别
        all_classes = list(range(100))
        for i in range(num_users):
            user_classes = np.random.choice(all_classes, classes_per_user, replace=False)
            
            for class_id in user_classes:
                class_indices = np.where(targets == class_id)[0]
                class_size = num_items // classes_per_user
                if len(class_indices) >= class_size:
                    selected = np.random.choice(class_indices, class_size, replace=False)
                    dict_users[i].extend(selected.tolist())
        
        return dict_users


def test_inference(args, model, test_dataset):
    """测试模型性能"""
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = LabelSmoothingCrossEntropy(epsilon=args.label_smoothing)
    
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
            
            # 预测
            _, pred_labels = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred_labels == labels).sum().item()
    
    accuracy = correct / total
    avg_loss = loss / len(testloader)
    
    return accuracy, avg_loss


def get_cifar100_advanced_dataloader(dataset, idxs, args):
    """获取CIFAR-100的高级数据加载器"""
    idxs_loader = DataSubset(dataset, idxs)
    
    # 使用更大的batch size来提高训练效率
    batch_size = min(64, len(idxs))
    
    return DataLoader(idxs_loader, batch_size=batch_size, shuffle=True, 
                     num_workers=2, pin_memory=True, drop_last=True)


class DataSubset(Dataset):
    """数据子集"""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
    
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # 如果image是张量则克隆，否则直接转换为张量
        if torch.is_tensor(image):
            image = image.clone().detach()
        else:
            image = torch.tensor(image)
        
        # label通常是整数，直接转换为张量
        if torch.is_tensor(label):
            label = label.clone().detach()
        else:
            label = torch.tensor(label)
            
        return image, label


class CIFAR100LocalUpdate:
    """CIFAR-100专用的本地更新"""
    
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader = get_cifar100_advanced_dataloader(dataset, idxs, args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = LabelSmoothingCrossEntropy(epsilon=args.label_smoothing)
        
        # 混合精度训练
        self.scaler = enable_mixed_precision()
    
    def update_weights(self, model, global_round):
        """更新本地模型权重"""
        model.train()
        epoch_loss = []
        
        # 优化器设置
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), 
                                      lr=self.args.lr,
                                      momentum=0.9, 
                                      weight_decay=5e-4,
                                      nesterov=True)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), 
                                        lr=self.args.lr,
                                        weight_decay=1e-4,
                                        eps=1e-8)
        elif self.args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), 
                                        lr=self.args.lr,
                                        weight_decay=1e-4,
                                        eps=1e-8)
        else:
            # 默认使用AdamW
            optimizer = torch.optim.AdamW(model.parameters(), 
                                        lr=self.args.lr,
                                        weight_decay=1e-4,
                                        eps=1e-8)
        
        # 学习率调度器 - 优化参数
        if self.args.optimizer == 'adamw':
            # AdamW使用更温和的学习率调度
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.args.local_ep, eta_min=1e-6
            )
        else:
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                if self.scaler:
                    # 混合精度训练
                    with torch.cuda.amp.autocast():
                        log_probs = model(images)
                        loss = self.criterion(log_probs, labels)
                    
                    self.scaler.scale(loss).backward()
                    
                    # 梯度裁剪
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # 标准训练
                    log_probs = model(images)
                    loss = self.criterion(log_probs, labels)
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                
                batch_loss.append(loss.item())
                
                # 显示训练进度（每10个batch显示一次）
                if self.args.verbose and batch_idx % 10 == 0 and len(self.trainloader) > 10:
                    progress = batch_idx / len(self.trainloader) * 100
                    print(f"| Global Round : {global_round} | Local Epoch : {iter} | [{batch_idx}/{len(self.trainloader)} ({progress:.0f}%)] Loss: {loss.item():.6f}")
            
            # 更新学习率
            scheduler.step()
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            
            # 显示每个本地轮次的完成信息
            if self.args.verbose and len(self.trainloader) > 10:
                print(f"| Global Round : {global_round} | Local Epoch : {iter} | [{len(self.trainloader)}/{len(self.trainloader)} (100%)] Loss: {epoch_loss[-1]:.6f}")
        
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


def test_inference(args, model, test_dataset):
    """在测试集上评估模型"""
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = LabelSmoothingCrossEntropy(epsilon=args.label_smoothing)
    
    testloader = DataLoader(test_dataset, batch_size=128,
                          shuffle=False, num_workers=4, pin_memory=True)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            
            # 预测
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            
            loss += batch_loss.item()
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
    
    accuracy = correct / total
    avg_loss = loss / len(testloader)
    return accuracy, avg_loss


def main():
    start_time = time.time()
    
    # 解析参数
    parser = argparse.ArgumentParser()
    
    # 联邦学习参数
    parser.add_argument('--epochs', type=int, default=500, help="全局轮数")
    parser.add_argument('--num_users', type=int, default=20, help="用户数量")
    parser.add_argument('--frac', type=float, default=0.3, help="每轮参与的用户比例")
    parser.add_argument('--local_ep', type=int, default=10, help="本地训练轮数")
    parser.add_argument('--local_bs', type=int, default=32, help="本地批次大小")
    parser.add_argument('--lr', type=float, default=0.01, help="学习率")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD动量")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="权重衰减")
    
    # 模型参数
    parser.add_argument('--model', type=str, default='cifar100_efficientnet', 
                       choices=['cifar100_efficientnet', 'ultra', 'lightweight_ultra'], 
                       help='模型名称: cifar100_efficientnet, ultra (超强融合模型), lightweight_ultra (轻量级版本)')
    parser.add_argument('--efficientnet_variant', type=str, default='b1', help='EfficientNet变体')
    parser.add_argument('--dataset', type=str, default='cifar100', help="数据集名称")
    parser.add_argument('--optimizer', type=str, default='sgd', help='优化器类型')
    
    # 数据分布
    parser.add_argument('--iid', type=int, default=0, help='是否IID分布')
    parser.add_argument('--unequal', type=int, default=0, help='是否不等分布')
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet分布参数')
    parser.add_argument('--classes_per_user', type=int, default=10, help='每用户类别数')
    
    # 模型正则化参数
    parser.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout率')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='标签平滑率')
    
    # 系统参数
    parser.add_argument('--gpu', default=None, help="使用的GPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help="早停轮数")
    parser.add_argument('--verbose', type=int, default=1, help='详细输出')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 设备配置
    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据集加载
    print("加载CIFAR-100数据集...")
    train_transform = get_cifar100_transforms(is_training=True)
    test_transform = get_cifar100_transforms(is_training=False)
    
    train_dataset = datasets.CIFAR100(
        root='./data/', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(
        root='./data/', train=False, download=True, transform=test_transform)
    
    # 数据分布
    sampling = CIFAR100AdvancedSampling()
    if args.iid:
        user_groups = cifar_iid(train_dataset, args.num_users)
    else:
        if args.alpha > 0:
            user_groups = sampling.cifar100_advanced_noniid(train_dataset, args.num_users, args.alpha)
        else:
            user_groups = sampling.cifar100_balanced_noniid(train_dataset, args.num_users, args.classes_per_user)
    
    # 模型构建
    if args.model == 'ultra':
        print("构建超强融合模型...")
        global_model = get_ultra_cifar100_model(dropout_rate=args.dropout_rate)
    elif args.model == 'lightweight_ultra':
        print("构建轻量级Ultra模型...")
        global_model = get_lightweight_ultra_cifar100_model(dropout_rate=args.dropout_rate)
    else:
        print(f"构建EfficientNet-{args.efficientnet_variant.upper()}模型...")
        global_model = get_cifar100_efficientnet(args.efficientnet_variant, dropout_rate=args.dropout_rate)
    
    global_model.to(device)
    global_model.train()
    
    # 复制初始权重
    global_weights = global_model.state_dict()
    
    # 训练记录
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 10
    val_loss_pre, counter = 0, 0
    
    print("\n开始联邦学习训练...")
    print(f"模型: EfficientNet-{args.efficientnet_variant.upper()}")
    print(f"总轮数: {args.epochs}, 用户数: {args.num_users}, 参与比例: {args.frac}")
    print(f"本地轮数: {args.local_ep}, 学习率: {args.lr}")
    print(f"数据分布: {'IID' if args.iid else 'Non-IID'}")
    print(f"预计训练时间: {'15-20小时' if args.efficientnet_variant=='b2' and args.epochs>=700 else '8-12小时'}")
    print("-" * 80)
    print(f"✨ 提示: 训练已开始，请耐心等待。第一个轮次可能需要较长时间来初始化...")
    
    for epoch in range(args.epochs):
        local_weights, local_losses = [], []
        round_start_time = time.time()
        print(f'\n全局轮次 {epoch+1}/{args.epochs}')
        
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        print(f"  选择 {m} 个用户进行训练")
        
        for i, idx in enumerate(idxs_users):
            if args.verbose:
                print(f"正在训练用户 {idx+1} (第 {i+1}/{m} 个用户)...")
            local_model = CIFAR100LocalUpdate(args=args, dataset=train_dataset,
                                            idxs=user_groups[idx], logger=None)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch+1)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
        
        print(f"  正在聚合权重...")
        
        # 获取每个用户的数据量用于加权平均
        user_data_lens = [len(user_groups[idx]) for idx in idxs_users]
        
        # 更新全局权重（使用加权平均）
        global_weights = average_weights(local_weights, user_data_lens)
        global_model.load_state_dict(global_weights)
        
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        
        # 每轮都计算测试准确率
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        train_accuracy.append(test_acc)
        
        # 显示轮次完成信息
        round_time = time.time() - round_start_time
        elapsed_time = time.time() - start_time
        avg_time_per_round = elapsed_time / (epoch + 1)
        estimated_remaining = avg_time_per_round * (args.epochs - epoch - 1)
        
        print(f"  ✅ 轮次完成 | 用时: {round_time:.1f}s | 训练损失: {loss_avg:.4f}")
        print(f"  📊 测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%) | 测试损失: {test_loss:.4f}")
        print(f"  ⏱️  总用时: {elapsed_time/60:.1f}分钟 | 预计剩余: {estimated_remaining/60:.1f}分钟")
        print(f"  {'='*50}")
        
        # 早停检查
        if test_loss > val_loss_pre:
            counter += 1
        else:
            counter = 0
        val_loss_pre = test_loss
        
        if counter >= args.stopping_rounds:
            print(f'🛑 早停于第 {epoch+1} 轮')
            break
    
    # 最终测试
    print("\n" + "="*80)
    print("训练完成，进行最终测试...")
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    
    print(f"\n最终结果:")
    print(f"测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"测试损失: {test_loss:.4f}")
    print(f"总训练时间: {(time.time() - start_time):.2f} 秒")
    
    # 保存模型
    model_path = f'./save/cifar100_efficientnet_{args.efficientnet_variant}_final.pth'
    os.makedirs('./save', exist_ok=True)
    torch.save(global_model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")
    
    print("\n实验详情:")
    exp_details(args)


if __name__ == '__main__':
    main()
