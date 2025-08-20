#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试增强模型脚本
用于快速验证新增的模型架构是否正常工作
"""

import torch
import argparse
from options import args_parser

def test_model_creation():
    """测试模型创建是否正常"""
    print("开始测试增强模型...")
    
    # 解析参数
    args = args_parser()
    
    # 测试参数设置
    test_configs = [
        # MNIST测试
        {'dataset': 'mnist', 'model': 'enhanced_cnn', 'num_channels': 1, 'num_classes': 10},
        {'dataset': 'mnist', 'model': 'resnet50', 'num_channels': 1, 'num_classes': 10},
        {'dataset': 'mnist', 'model': 'wide_resnet', 'num_channels': 1, 'num_classes': 10},
        
        # CIFAR-10测试
        {'dataset': 'cifar', 'model': 'enhanced_cnn', 'num_channels': 3, 'num_classes': 10},
        {'dataset': 'cifar', 'model': 'resnet50', 'num_channels': 3, 'num_classes': 10},
        {'dataset': 'cifar', 'model': 'wide_resnet', 'num_channels': 3, 'num_classes': 10},
        
        # Fashion-MNIST测试
        {'dataset': 'fmnist', 'model': 'enhanced_cnn', 'num_channels': 1, 'num_classes': 10},
        {'dataset': 'fmnist', 'model': 'resnet50', 'num_channels': 1, 'num_classes': 10},
        {'dataset': 'fmnist', 'model': 'wide_resnet', 'num_channels': 1, 'num_classes': 10},
    ]
    
    try:
        # 动态导入模型
        from models_enhanced import EnhancedEfficientNetCifar, ResNet50Cifar, WideResNetCifar
        
        print("成功导入增强模型模块!")
        
        for config in test_configs:
            dataset = config['dataset']
            model_name = config['model']
            num_channels = config['num_channels']
            num_classes = config['num_classes']
            
            print(f"\n测试 {dataset.upper()} 数据集上的 {model_name.upper()} 模型:")
            
            try:
                # 创建模型
                if model_name == 'enhanced_cnn':
                    model = EnhancedEfficientNetCifar(num_classes=num_classes, num_channels=num_channels)
                elif model_name == 'resnet50':
                    model = ResNet50Cifar(num_classes=num_classes, num_channels=num_channels)
                elif model_name == 'wide_resnet':
                    model = WideResNetCifar(num_classes=num_classes, num_channels=num_channels)
                else:
                    print(f"未知模型类型: {model_name}")
                    continue
                
                # 计算参数数量
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                print(f"  ✓ 模型创建成功")
                print(f"  ✓ 总参数数量: {total_params:,}")
                print(f"  ✓ 可训练参数数量: {trainable_params:,}")
                
                # 测试前向传播
                if dataset == 'mnist' or dataset == 'fmnist':
                    input_size = (2, num_channels, 28, 28)  # batch_size=2 for testing
                else:  # cifar
                    input_size = (2, num_channels, 32, 32)
                
                test_input = torch.randn(input_size)
                
                model.eval()
                with torch.no_grad():
                    output = model(test_input)
                
                expected_output_shape = (2, num_classes)
                if output.shape == expected_output_shape:
                    print(f"  ✓ 前向传播成功，输出形状: {output.shape}")
                else:
                    print(f"  ✗ 前向传播输出形状错误，期望: {expected_output_shape}, 实际: {output.shape}")
                
            except Exception as e:
                print(f"  ✗ 模型测试失败: {str(e)}")
                import traceback
                traceback.print_exc()
    
    except ImportError as e:
        print(f"✗ 导入模型模块失败: {str(e)}")
        print("请确保 models_enhanced.py 文件存在且语法正确")
        return False
    
    except Exception as e:
        print(f"✗ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 所有模型测试完成!")
    return True

def test_model_integration():
    """测试模型与联邦学习框架的集成"""
    print("\n开始测试模型集成...")
    
    try:
        # 测试是否能正确导入和选择模型
        from federated_main_improved import get_model
        
        # 测试参数
        class TestArgs:
            def __init__(self):
                self.dataset = 'cifar'
                self.model = 'enhanced_cnn'
                self.gpu = -1  # 使用CPU进行测试
                self.num_channels = 3
        
        args = TestArgs()
        
        # 测试模型获取
        try:
            net_glob = get_model(args)
            print("✓ 模型集成测试成功!")
            print(f"✓ 成功获取 {args.model} 模型")
            
            # 测试模型参数数量
            total_params = sum(p.numel() for p in net_glob.parameters())
            print(f"✓ 模型参数数量: {total_params:,}")
            
        except Exception as e:
            print(f"✗ 模型集成测试失败: {str(e)}")
            return False
            
    except ImportError as e:
        print(f"✗ 导入联邦学习模块失败: {str(e)}")
        return False
    
    return True

if __name__ == '__main__':
    print("="*60)
    print("增强模型测试脚本")
    print("="*60)
    
    # 测试模型创建
    model_test_success = test_model_creation()
    
    # 测试模型集成
    integration_test_success = test_model_integration()
    
    print("\n" + "="*60)
    if model_test_success and integration_test_success:
        print("🎉 所有测试通过! 增强模型已准备就绪。")
        print("\n推荐测试命令:")
        print("python federated_main_improved.py --dataset cifar --model enhanced_cnn --epochs 5 --lr 0.01 --local_ep 2 --num_users 10 --frac 1.0")
    else:
        print("❌ 部分测试失败，请检查错误信息并修复。")
    print("="*60)
