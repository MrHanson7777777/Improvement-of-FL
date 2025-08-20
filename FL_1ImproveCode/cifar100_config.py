#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CIFAR-100专用优化配置
包含针对不同EfficientNet变体的最优参数设置
"""

import argparse

def add_cifar100_args():
    """添加CIFAR-100专用参数"""
    parser = argparse.ArgumentParser()
    
    # ====== 基础联邦学习参数 ======
    parser.add_argument('--epochs', type=int, default=600, 
                       help="全局训练轮数 (CIFAR-100需要更多轮数)")
    parser.add_argument('--num_users', type=int, default=30, 
                       help="参与联邦学习的用户数量")
    parser.add_argument('--frac', type=float, default=0.4, 
                       help="每轮参与训练的用户比例")
    parser.add_argument('--local_ep', type=int, default=15, 
                       help="每个用户的本地训练轮数")
    parser.add_argument('--local_bs', type=int, default=32, 
                       help="本地批次大小")
    
    # ====== 优化器参数 ======
    parser.add_argument('--lr', type=float, default=0.05, 
                       help="初始学习率 (较高的学习率配合cosine衰减)")
    parser.add_argument('--momentum', type=float, default=0.9, 
                       help="SGD动量")
    parser.add_argument('--weight_decay', type=float, default=5e-4, 
                       help="权重衰减 (L2正则化)")
    parser.add_argument('--optimizer', type=str, default='sgd', 
                       choices=['sgd', 'adam', 'adamw'],
                       help='优化器类型')
    
    # ====== 模型参数 ======
    parser.add_argument('--model', type=str, default='cifar100_efficientnet',
                       help='模型类型')
    parser.add_argument('--efficientnet_variant', type=str, default='b1',
                       choices=['b0', 'b1', 'b2'],
                       help='EfficientNet变体')
    parser.add_argument('--dataset', type=str, default='cifar100',
                       help="数据集")
    
    # ====== 数据分布策略 ======
    parser.add_argument('--iid', type=int, default=0,
                       help='是否使用IID数据分布 (0=Non-IID, 1=IID)')
    parser.add_argument('--unequal', type=int, default=0,
                       help='是否使用不等数据分布')
    parser.add_argument('--alpha', type=float, default=0.3,
                       help='Dirichlet分布的concentration参数 (越小越不均匀)')
    parser.add_argument('--classes_per_user', type=int, default=15,
                       help='平衡non-IID中每个用户的类别数')
    
    # ====== 训练策略 ======
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='标签平滑参数')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                       help='Mixup数据增强参数')
    parser.add_argument('--cutmix_prob', type=float, default=0.5,
                       help='CutMix数据增强概率')
    parser.add_argument('--autoaugment', type=int, default=1,
                       help='是否使用AutoAugment')
    
    # ====== 正则化参数 ======
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout比率')
    parser.add_argument('--drop_path_rate', type=float, default=0.2,
                       help='DropPath比率 (随机深度)')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                       help='梯度裁剪范数')
    
    # ====== 学习率调度 ======
    parser.add_argument('--lr_scheduler', type=str, default='cosine_warm_restart',
                       choices=['cosine', 'cosine_warm_restart', 'step', 'exponential'],
                       help='学习率调度策略')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                       help='预热轮数')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                       help='最小学习率')
    
    # ====== 系统参数 ======
    parser.add_argument('--gpu', default=None, help="指定GPU设备")
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器线程数')
    parser.add_argument('--pin_memory', type=int, default=1, help='是否使用pin memory')
    
    # ====== 早停和保存 ======
    parser.add_argument('--stopping_rounds', type=int, default=50,
                       help="早停轮数")
    parser.add_argument('--save_every', type=int, default=50,
                       help='每隔多少轮保存一次模型')
    parser.add_argument('--eval_every', type=int, default=10,
                       help='每隔多少轮评估一次')
    
    # ====== 高级功能 ======
    parser.add_argument('--mixed_precision', type=int, default=1,
                       help='是否使用混合精度训练')
    parser.add_argument('--use_ema', type=int, default=1,
                       help='是否使用指数移动平均')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                       help='EMA衰减率')
    
    # ====== 输出控制 ======
    parser.add_argument('--verbose', type=int, default=1, help='详细输出级别')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='日志输出间隔')
    
    return parser


def get_cifar100_optimal_config(variant='b1', strategy='aggressive'):
    """获取CIFAR-100的最优配置
    
    Args:
        variant: EfficientNet变体 ('b0', 'b1', 'b2')
        strategy: 训练策略 ('conservative', 'balanced', 'aggressive')
    """
    configs = {
        'b0': {
            'conservative': {
                'epochs': 400,
                'lr': 0.03,
                'local_ep': 10,
                'frac': 0.3,
                'weight_decay': 1e-4,
                'dropout_rate': 0.2,
                'label_smoothing': 0.05
            },
            'balanced': {
                'epochs': 500,
                'lr': 0.05,
                'local_ep': 12,
                'frac': 0.35,
                'weight_decay': 5e-4,
                'dropout_rate': 0.25,
                'label_smoothing': 0.1
            },
            'aggressive': {
                'epochs': 600,
                'lr': 0.08,
                'local_ep': 15,
                'frac': 0.4,
                'weight_decay': 1e-3,
                'dropout_rate': 0.3,
                'label_smoothing': 0.15
            }
        },
        'b1': {
            'conservative': {
                'epochs': 500,
                'lr': 0.02,
                'local_ep': 12,
                'frac': 0.35,
                'weight_decay': 5e-4,
                'dropout_rate': 0.25,
                'label_smoothing': 0.1
            },
            'balanced': {
                'epochs': 600,
                'lr': 0.04,
                'local_ep': 15,
                'frac': 0.4,
                'weight_decay': 8e-4,
                'dropout_rate': 0.3,
                'label_smoothing': 0.12
            },
            'aggressive': {
                'epochs': 700,
                'lr': 0.06,
                'local_ep': 18,
                'frac': 0.45,
                'weight_decay': 1e-3,
                'dropout_rate': 0.35,
                'label_smoothing': 0.15
            }
        },
        'b2': {
            'conservative': {
                'epochs': 600,
                'lr': 0.015,
                'local_ep': 15,
                'frac': 0.4,
                'weight_decay': 8e-4,
                'dropout_rate': 0.3,
                'label_smoothing': 0.12
            },
            'balanced': {
                'epochs': 700,
                'lr': 0.03,
                'local_ep': 18,
                'frac': 0.45,
                'weight_decay': 1e-3,
                'dropout_rate': 0.35,
                'label_smoothing': 0.15
            },
            'aggressive': {
                'epochs': 800,
                'lr': 0.05,
                'local_ep': 20,
                'frac': 0.5,
                'weight_decay': 1.5e-3,
                'dropout_rate': 0.4,
                'label_smoothing': 0.2
            }
        }
    }
    
    return configs.get(variant, {}).get(strategy, configs['b1']['balanced'])


def get_recommended_commands():
    """获取推荐的训练命令"""
    commands = {
        '快速测试 (EfficientNet-B0)': {
            'command': 'python federated_cifar100_main.py --efficientnet_variant b0 --epochs 250 --lr 0.045 --local_ep 12 --frac 0.4 --weight_decay 8e-4 --dropout_rate 0.2 --label_smoothing 0.1 --alpha 0.25 --num_users 25',
            'description': '快速验证配置，资源友好',
            'expected_time': '约3-5小时',
            'expected_accuracy': '68-74%'
        },
        
        '平衡训练 (EfficientNet-B1)': {
            'command': 'python federated_cifar100_main.py --efficientnet_variant b1 --epochs 300 --lr 0.04 --local_ep 10 --frac 0.45 --weight_decay 1e-3 --dropout_rate 0.25 --label_smoothing 0.12 --alpha 0.2 --num_users 30',
            'description': 'Non-IID优化的平衡配置',
            'expected_time': '约6-8小时',
            'expected_accuracy': '74-80%'
        },
        
        '高性能训练 (EfficientNet-B2)': {
            'command': 'python federated_cifar100_main.py --efficientnet_variant b2 --epochs 350 --lr 0.035 --local_ep 8 --frac 0.5 --weight_decay 1.2e-3 --dropout_rate 0.3 --label_smoothing 0.15 --alpha 0.15 --num_users 40',
            'description': '针对Non-IID优化的最高准确率配置',
            'expected_time': '约8-12小时',
            'expected_accuracy': '78-84%'
        },
        
        '终极训练 (EfficientNet-B2)': {
            'command': 'python federated_cifar100_main.py --efficientnet_variant b2 --epochs 350 --lr 0.035 --local_ep 8 --frac 0.5 --weight_decay 1.2e-3 --dropout_rate 0.3 --label_smoothing 0.15 --alpha 0.15 --num_users 40 --verbose 1',
            'description': '最优准确率配置，Non-IID友好',
            'expected_time': '约8-12小时',
            'expected_accuracy': '78-84%'
        },
        
        'Non-IID强化训练': {
            'command': 'python federated_cifar100_main.py --efficientnet_variant b1 --epochs 600 --lr 0.05 --local_ep 15 --frac 0.4 --alpha 0.1 --classes_per_user 10 --weight_decay 1e-3',
            'description': '针对强non-IID场景的优化训练',
            'expected_time': '约8-10小时',
            'expected_accuracy': '70-75%'
        }
    }
    
    return commands


def print_usage_guide():
    """打印使用指南"""
    print("="*80)
    print("CIFAR-100 EfficientNet 联邦学习训练指南")
    print("="*80)
    
    print("\n🎯 推荐的训练命令:")
    print("-"*50)
    
    commands = get_recommended_commands()
    for name, config in commands.items():
        print(f"\n📊 {name}:")
        print(f"   命令: {config['command']}")
        print(f"   说明: {config['description']}")
        print(f"   预期时间: {config['expected_time']}")
        print(f"   预期准确率: {config['expected_accuracy']}")
    
    print("\n🔧 重要参数说明:")
    print("-"*50)
    print("--efficientnet_variant: 模型变体 (b0=最快, b1=平衡, b2=最准)")
    print("--epochs: 全局轮数 (CIFAR-100需要较多轮数)")
    print("--lr: 学习率 (配合cosine衰减，可以设置较高)")
    print("--local_ep: 本地轮数 (增加可以提高性能)")
    print("--frac: 参与比例 (较高比例有助于收敛)")
    print("--alpha: Dirichlet参数 (越小数据越不均匀)")
    print("--label_smoothing: 标签平滑 (有助于泛化)")
    print("--weight_decay: 权重衰减 (防止过拟合)")
    
    print("\n⚡ 性能优化技巧:")
    print("-"*50)
    print("1. 使用混合精度训练节省内存: --mixed_precision 1")
    print("2. 增加数据增强强度提高泛化")
    print("3. 适当的学习率衰减策略")
    print("4. 梯度裁剪防止梯度爆炸")
    print("5. EMA (指数移动平均) 稳定训练")
    
    print("\n📈 期望性能范围:")
    print("-"*50)
    print("EfficientNet-B0: 65-72% (快速训练)")
    print("EfficientNet-B1: 72-78% (推荐选择)")
    print("EfficientNet-B2: 75-82% (最高性能)")
    
    print("\n⚠️  注意事项:")
    print("-"*50)
    print("1. CIFAR-100比CIFAR-10难度大，需要更多轮数")
    print("2. 数据增强和正则化对小数据集很重要")
    print("3. 非IID设置下性能会下降5-10%")
    print("4. GPU内存不足时可以减小batch size")
    print("5. 训练时间较长，建议使用后台运行")


if __name__ == '__main__':
    print_usage_guide()
