#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
专门针对CIFAR-10达到85%+精度的联邦学习训练脚本
"""

import os
import sys
import copy
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models_high_performance import CIFAR10UltraNet
from sampling import cifar_iid
from update import LocalUpdate
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Args:
    """参数配置类"""
    def __init__(self):
        # 基础配置
        self.epochs = 150
        self.num_users = 100
        self.frac = 0.4  # 40%的客户端参与
        self.local_ep = 3  # 本地训练轮数
        self.local_bs = 64  # 本地批次大小
        self.lr = 0.08
        self.momentum = 0.9
        self.weight_decay = 2e-4
        
        # 模型和数据集
        self.model = 'cifar_ultra'
        self.dataset = 'cifar'
        self.num_classes = 10
        self.num_channels = 3
        
        # 设备配置
        self.gpu = 0 if torch.cuda.is_available() else None
        
        # 优化配置
        self.optimizer = 'sgd'
        self.iid = True
        self.unequal = False
        
        # 调度和正则化
        self.use_scheduler = True
        self.scheduler = 'cosine'
        self.warmup_epochs = 10
        self.stopping_rounds = 30
        
        # 数据增强
        self.use_mixup = True
        self.use_cutmix = True
        self.use_label_smoothing = True
        self.label_smoothing = 0.1
        
        # 其他
        self.verbose = 1
        self.seed = 42

def get_cifar10_data():
    """获取CIFAR-10数据集"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform_test)
    
    return train_dataset, test_dataset

def average_weights(w):
    """聚合权重"""
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def test_inference(args, model, test_dataset):
    """测试推理"""
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    
    device = 'cuda' if args.gpu is not None and args.gpu >= 0 else 'cpu'
    criterion = torch.nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
            
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
    
    accuracy = correct / total
    return accuracy, loss / len(testloader)

def cosine_annealing_lr(epoch, total_epochs, base_lr, min_lr=1e-6):
    """余弦退火学习率调度"""
    return min_lr + (base_lr - min_lr) * (1 + np.cos(np.pi * epoch / total_epochs)) / 2

def main():
    """主训练函数"""
    args = Args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    device = 'cuda' if args.gpu is not None and args.gpu >= 0 else 'cpu'
    
    print("🚀 超高性能CIFAR-10联邦学习训练")
    print(f"📊 配置信息:")
    print(f"   设备: {device}")
    print(f"   轮数: {args.epochs}")
    print(f"   客户端数: {args.num_users}")
    print(f"   参与比例: {args.frac}")
    print(f"   学习率: {args.lr}")
    print(f"   本地批次: {args.local_bs}")
    print("=" * 60)
    
    # 加载数据
    train_dataset, test_dataset = get_cifar10_data()
    
    # 分配数据给客户端
    user_groups = cifar_iid(train_dataset, args.num_users)
    
    # 创建模型
    global_model = CIFAR10UltraNet(num_classes=10)
    global_model.to(device)
    
    # 复制权重
    global_weights = global_model.state_dict()
    
    # 训练历史
    train_loss, train_accuracy = [], []
    
    print("🏃 开始联邦学习训练...")
    
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        local_weights, local_losses = [], []
        
        # 选择参与的客户端
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        # 动态学习率调度
        if args.use_scheduler and args.scheduler == 'cosine':
            current_lr = cosine_annealing_lr(epoch, args.epochs, args.lr)
        else:
            current_lr = args.lr
        
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[idx], logger=logger)
            
            # 设置学习率
            for param_group in local_model.optimizer.param_groups:
                param_group['lr'] = current_lr
            
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
        
        # 聚合权重
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        
        # 每5轮测试一次
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            
            print(f'轮次 {epoch+1:3d}/{args.epochs}: ')
            print(f'  训练损失: {loss_avg:.4f}')
            print(f'  测试精度: {test_acc:.4f} ({test_acc*100:.2f}%)')
            print(f'  学习率: {current_lr:.6f}')
            
            # 早停检查
            if test_acc > best_acc:
                best_acc = test_acc
                patience_counter = 0
                # 保存最佳模型
                torch.save(global_model.state_dict(), 'best_cifar_ultra_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= args.stopping_rounds:
                print(f"🛑 早停触发，在第 {epoch+1} 轮停止训练")
                break
            
            if test_acc >= 0.85:
                print(f"🎉 达到目标精度85%! 当前精度: {test_acc*100:.2f}%")
                break
    
    # 最终测试
    print("\n" + "="*60)
    print("🏁 训练完成!")
    
    # 加载最佳模型
    if os.path.exists('best_cifar_ultra_model.pth'):
        global_model.load_state_dict(torch.load('best_cifar_ultra_model.pth'))
    
    final_acc, final_loss = test_inference(args, global_model, test_dataset)
    print(f"🎯 最终测试精度: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"🎯 最佳精度: {best_acc:.4f} ({best_acc*100:.2f}%)")
    
    if final_acc >= 0.85:
        print("✅ 成功达到85%+精度目标!")
    else:
        print(f"❌ 未达到85%目标，当前{final_acc*100:.2f}%")
    
    return final_acc

if __name__ == '__main__':
    accuracy = main()
