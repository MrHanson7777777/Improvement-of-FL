#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试超高性能模型的脚本
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models_high_performance import CIFAR10HighPerformanceCNN, CIFAR10UltraNet
import numpy as np
import time

def quick_cifar_test():
    """快速测试CIFAR-10模型性能"""
    
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 加载数据
    print("📦 加载CIFAR-10数据集...")
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)
    
    # 创建模型
    print("🚀 创建超高性能模型...")
    models_to_test = [
        ('HighPerformanceCNN', CIFAR10HighPerformanceCNN()),
        ('UltraNet', CIFAR10UltraNet()),
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    for model_name, model in models_to_test:
        print(f"\n🔥 测试模型: {model_name}")
        model = model.to(device)
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 模型参数量: {total_params:,}")
        
        # 优化器
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, 
                                   momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        # 快速训练几个epoch
        print("🏃 开始快速训练...")
        model.train()
        for epoch in range(3):
            running_loss = 0.0
            correct = 0
            total = 0
            
            start_time = time.time()
            for i, (inputs, labels) in enumerate(trainloader):
                if i >= 50:  # 只训练50个batch进行快速测试
                    break
                    
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            epoch_time = time.time() - start_time
            acc = 100. * correct / total
            print(f"Epoch {epoch+1}: Loss={running_loss/50:.4f}, Acc={acc:.2f}%, Time={epoch_time:.1f}s")
        
        # 测试
        print("🧪 测试模型性能...")
        model.eval()
        correct = 0
        total = 0
        test_loss = 0
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testloader):
                if i >= 20:  # 只测试20个batch
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * correct / total
        print(f"✅ {model_name} 测试精度: {test_acc:.2f}%")
        print(f"   (基于{total}个样本的快速测试)")


if __name__ == '__main__':
    quick_cifar_test()
