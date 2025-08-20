#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行改进后的联邦学习实验的脚本
支持MNIST、Fashion-MNIST、CIFAR-10、CIFAR-100数据集
使用改进的模型架构和早停机制
"""

import subprocess
import sys
import os

def run_baseline_experiment(dataset, model='cnn', epochs=50, gpu=0):
    """运行基准实验（非联邦学习）"""
    print(f"\n=== 运行 {dataset} 基准实验 (模型: {model}) ===")
    
    # 设置类别数
    num_classes = 100 if dataset == 'cifar100' else 10
    num_channels = 3 if dataset in ['cifar', 'cifar100'] else 1
    
    cmd = [
        sys.executable, 'baseline_main.py',
        '--dataset', dataset,
        '--model', model,
        '--epochs', str(epochs),
        '--num_classes', str(num_classes),
        '--num_channels', str(num_channels),
        '--gpu', str(gpu),
        '--optimizer', 'adam',
        '--lr', '0.001'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            print(f"✓ {dataset} 基准实验完成")
            print(result.stdout[-500:])  # 打印最后500个字符
        else:
            print(f"✗ {dataset} 基准实验失败")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print(f"✗ {dataset} 基准实验超时")
    except Exception as e:
        print(f"✗ {dataset} 基准实验出错: {e}")

def run_federated_experiment(dataset, model='cnn', epochs=20, num_users=10, frac=0.3, local_ep=5, gpu=0, iid=1):
    """运行联邦学习实验"""
    print(f"\n=== 运行 {dataset} 联邦学习实验 (模型: {model}) ===")
    
    # 设置类别数
    num_classes = 100 if dataset == 'cifar100' else 10
    num_channels = 3 if dataset in ['cifar', 'cifar100'] else 1
    
    cmd = [
        sys.executable, 'federated_main.py',
        '--dataset', dataset,
        '--model', model,
        '--epochs', str(epochs),
        '--num_users', str(num_users),
        '--frac', str(frac),
        '--local_ep', str(local_ep),
        '--local_bs', '32',
        '--num_classes', str(num_classes),
        '--num_channels', str(num_channels),
        '--gpu', str(gpu),
        '--optimizer', 'adam',
        '--lr', '0.001',
        '--iid', str(iid),
        '--stopping_rounds', '5'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode == 0:
            print(f"✓ {dataset} 联邦学习实验完成")
            print(result.stdout[-500:])  # 打印最后500个字符
        else:
            print(f"✗ {dataset} 联邦学习实验失败")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print(f"✗ {dataset} 联邦学习实验超时")
    except Exception as e:
        print(f"✗ {dataset} 联邦学习实验出错: {e}")

def main():
    """主函数 - 运行所有实验"""
    print("开始运行改进的联邦学习实验...")
    
    # 数据集列表
    datasets = ['mnist', 'fmnist', 'cifar', 'cifar100']
    
    # 1. 运行基准实验（非联邦学习）
    print("\n" + "="*50)
    print("第一阶段：基准实验（中心化训练）")
    print("="*50)
    
    for dataset in datasets:
        run_baseline_experiment(dataset, model='cnn', epochs=30, gpu=0)
    
    # 2. 运行联邦学习实验 (IID)
    print("\n" + "="*50)
    print("第二阶段：联邦学习实验 (IID)")
    print("="*50)
    
    for dataset in datasets:
        run_federated_experiment(dataset, model='cnn', epochs=20, num_users=10, 
                                frac=0.3, local_ep=5, gpu=0, iid=1)
    
    # 3. 运行联邦学习实验 (Non-IID)
    print("\n" + "="*50)
    print("第三阶段：联邦学习实验 (Non-IID)")
    print("="*50)
    
    for dataset in datasets:
        run_federated_experiment(dataset, model='cnn', epochs=20, num_users=10, 
                                frac=0.3, local_ep=5, gpu=0, iid=0)
    
    print("\n" + "="*50)
    print("所有实验完成！")
    print("="*50)

if __name__ == '__main__':
    main()
