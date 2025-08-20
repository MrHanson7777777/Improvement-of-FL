#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：验证四个数据集（MNIST、Fashion-MNIST、CIFAR-10、CIFAR-100）的支持情况
"""

import torch
from options import args_parser
from utils import get_dataset
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNCifar100


def test_dataset_support():
    """测试数据集支持"""
    datasets_to_test = ['mnist', 'fmnist', 'cifar', 'cifar100']
    
    for dataset_name in datasets_to_test:
        print(f"\n=== 测试数据集: {dataset_name} ===")
        
        # 创建一个简单的args对象
        class SimpleArgs:
            def __init__(self, dataset_name):
                self.dataset = dataset_name
                self.iid = 1
                self.num_users = 10
                self.unequal = 0
                self.model = 'cnn'
                
                # 根据数据集设置参数
                if dataset_name in ['mnist', 'fmnist']:
                    self.num_channels = 1
                    self.num_classes = 10
                elif dataset_name == 'cifar':
                    self.num_channels = 3
                    self.num_classes = 10
                elif dataset_name == 'cifar100':
                    self.num_channels = 3
                    self.num_classes = 100
        
        args = SimpleArgs(dataset_name)
        
        try:
            # 测试数据集加载
            train_dataset, test_dataset, user_groups = get_dataset(args)
            print(f"✓ 数据集加载成功")
            print(f"  训练集大小: {len(train_dataset)}")
            print(f"  测试集大小: {len(test_dataset)}")
            print(f"  用户数量: {len(user_groups)}")
            
            # 测试模型创建
            if args.model == 'cnn':
                if args.dataset == 'mnist':
                    model = CNNMnist(args=args)
                elif args.dataset == 'fmnist':
                    model = CNNFashion_Mnist(args=args)
                elif args.dataset == 'cifar':
                    model = CNNCifar(args=args)
                elif args.dataset == 'cifar100':
                    model = CNNCifar100(args=args)
            
            print(f"✓ 模型创建成功")
            
            # 测试一个小批次的前向传播
            sample_data, sample_label = train_dataset[0]
            sample_batch = sample_data.unsqueeze(0)  # 添加batch维度
            
            model.eval()
            with torch.no_grad():
                output = model(sample_batch)
                print(f"✓ 前向传播测试成功，输出形状: {output.shape}")
                print(f"  期望类别数: {args.num_classes}, 实际输出类别数: {output.shape[1]}")
                
            if output.shape[1] == args.num_classes:
                print(f"✓ 输出维度正确")
            else:
                print(f"✗ 输出维度错误")
                
        except Exception as e:
            print(f"✗ 测试失败: {str(e)}")


if __name__ == '__main__':
    print("开始测试四个数据集的支持情况...")
    test_dataset_support()
    print("\n测试完成！")
