#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用示例：如何在四个数据集上运行联邦学习
"""

import os

# 示例命令行参数组合
examples = {
    "MNIST + CNN": {
        "command": "python federated_main.py --dataset mnist --model cnn --epochs 5 --num_users 10 --frac 0.1",
        "description": "在MNIST数据集上使用CNN模型进行联邦学习"
    },
    
    "Fashion-MNIST + CNN": {
        "command": "python federated_main.py --dataset fmnist --model cnn --epochs 5 --num_users 10 --frac 0.1",
        "description": "在Fashion-MNIST数据集上使用CNN模型进行联邦学习"
    },
    
    "CIFAR-10 + CNN": {
        "command": "python federated_main.py --dataset cifar --model cnn --epochs 5 --num_users 10 --frac 0.1",
        "description": "在CIFAR-10数据集上使用CNN模型进行联邦学习"
    },
    
    "CIFAR-100 + CNN": {
        "command": "python federated_main.py --dataset cifar100 --model cnn --epochs 5 --num_users 10 --frac 0.1",
        "description": "在CIFAR-100数据集上使用CNN模型进行联邦学习"
    },
    
    "MNIST + MLP": {
        "command": "python federated_main.py --dataset mnist --model mlp --epochs 5 --num_users 10 --frac 0.1",
        "description": "在MNIST数据集上使用MLP模型进行联邦学习"
    }
}

# 基线学习（非联邦）示例
baseline_examples = {
    "MNIST Baseline": {
        "command": "python baseline_main.py --dataset mnist --model cnn --epochs 5",
        "description": "在MNIST数据集上进行基线（非联邦）学习"
    },
    
    "CIFAR-100 Baseline": {
        "command": "python baseline_main.py --dataset cifar100 --model cnn --epochs 5",
        "description": "在CIFAR-100数据集上进行基线（非联邦）学习"
    }
}

def print_examples():
    print("=== 联邦学习示例 ===")
    for name, info in examples.items():
        print(f"\n{name}:")
        print(f"  描述: {info['description']}")
        print(f"  命令: {info['command']}")
    
    print("\n=== 基线学习示例 ===")
    for name, info in baseline_examples.items():
        print(f"\n{name}:")
        print(f"  描述: {info['description']}")
        print(f"  命令: {info['command']}")
    
    print("\n=== 参数说明 ===")
    print("--dataset: 数据集选择 (mnist, fmnist, cifar, cifar100)")
    print("--model: 模型类型 (cnn, mlp)")
    print("--epochs: 训练轮数")
    print("--num_users: 客户端数量")
    print("--frac: 每轮参与训练的客户端比例")
    print("--iid: 数据分布 (1=IID, 0=Non-IID)")
    print("--lr: 学习率")
    print("--local_ep: 客户端本地训练轮数")
    print("--local_bs: 客户端本地批次大小")

if __name__ == '__main__':
    print_examples()
