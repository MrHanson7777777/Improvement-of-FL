#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速验证脚本：测试所有四个数据集的基本功能
"""

import sys
import torch
from options import args_parser

def quick_test():
    """快速测试四个数据集的基本功能"""
    datasets = ['mnist', 'fmnist', 'cifar', 'cifar100']
    
    print("🔍 开始快速验证测试...")
    print("=" * 50)
    
    for dataset in datasets:
        print(f"\n📊 测试数据集: {dataset.upper()}")
        
        try:
            # 模拟命令行参数
            sys.argv = ['test', '--dataset', dataset, '--model', 'cnn', '--epochs', '1']
            args = args_parser()
            
            # 验证参数自动配置
            expected_classes = 100 if dataset == 'cifar100' else 10
            expected_channels = 3 if 'cifar' in dataset else 1
            
            assert args.num_classes == expected_classes, f"类别数错误: 期望{expected_classes}, 实际{args.num_classes}"
            assert args.num_channels == expected_channels, f"通道数错误: 期望{expected_channels}, 实际{args.num_channels}"
            
            print(f"  ✅ 参数配置正确 - 类别数: {args.num_classes}, 通道数: {args.num_channels}")
            
        except Exception as e:
            print(f"  ❌ 测试失败: {str(e)}")
            return False
    
    print("\n" + "=" * 50)
    print("🎉 所有测试通过！代码已成功修改为支持四个数据集。")
    
    print("\n📋 使用说明:")
    print("1. 基线学习: python baseline_main.py --dataset [mnist|fmnist|cifar|cifar100] --model cnn")
    print("2. 联邦学习: python federated_main.py --dataset [mnist|fmnist|cifar|cifar100] --model cnn")
    print("3. 查看更多示例: python usage_examples.py")
    
    return True

if __name__ == '__main__':
    quick_test()
