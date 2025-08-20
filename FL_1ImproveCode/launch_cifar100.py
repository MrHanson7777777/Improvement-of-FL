#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CIFAR-100 EfficientNet 联邦学习快速启动脚本
提供预设的训练配置和便捷的启动方式
"""

import subprocess
import sys
import os
import argparse
import time
from datetime import datetime


class CIFAR100Launcher:
    """CIFAR-100训练启动器"""
    
    def __init__(self):
        self.script_path = "federated_cifar100_main.py"
        self.base_commands = {
            'quick': [
                '--efficientnet_variant', 'b0',
                '--epochs', '200',
                '--lr', '0.03',
                '--local_ep', '8',
                '--frac', '0.3',
                '--num_users', '20',
                '--alpha', '0.5'
            ],
            'balanced': [
                '--efficientnet_variant', 'b1',
                '--epochs', '500',
                '--lr', '0.04',
                '--local_ep', '15',
                '--frac', '0.4',
                '--num_users', '30',
                '--weight_decay', '8e-4',
                '--label_smoothing', '0.12',
                '--alpha', '0.3'
            ],
            'aggressive': [
                '--efficientnet_variant', 'b2',
                '--epochs', '700',
                '--lr', '0.06',
                '--local_ep', '18',
                '--frac', '0.45',
                '--num_users', '30',
                '--weight_decay', '1e-3',
                '--dropout_rate', '0.35',
                '--label_smoothing', '0.15',
                '--alpha', '0.2'
            ],
            'ultimate': [
                '--efficientnet_variant', 'b2',
                '--epochs', '800',
                '--lr', '0.05',
                '--local_ep', '20',
                '--frac', '0.5',
                '--num_users', '40',
                '--weight_decay', '1.5e-3',
                '--dropout_rate', '0.4',
                '--label_smoothing', '0.2',
                '--alpha', '0.15'
            ],
            'noniid_challenge': [
                '--efficientnet_variant', 'b1',
                '--epochs', '600',
                '--lr', '0.05',
                '--local_ep', '15',
                '--frac', '0.4',
                '--num_users', '25',
                '--alpha', '0.1',  # 强non-IID
                '--classes_per_user', '8',
                '--weight_decay', '1e-3',
                '--label_smoothing', '0.15'
            ]
        }
    
    def print_banner(self):
        """打印启动横幅"""
        print("=" * 80)
        print("🚀 CIFAR-100 EfficientNet 联邦学习训练启动器")
        print("=" * 80)
        print(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def print_configs(self):
        """打印可用配置"""
        configs_info = {
            'quick': {
                'name': '快速测试',
                'model': 'EfficientNet-B0',
                'time': '2-3小时',
                'accuracy': '65-70%',
                'description': '适合快速验证和调试'
            },
            'balanced': {
                'name': '平衡训练',
                'model': 'EfficientNet-B1',
                'time': '6-8小时',
                'accuracy': '72-77%',
                'description': '推荐的标准配置'
            },
            'aggressive': {
                'name': '高性能训练',
                'model': 'EfficientNet-B2',
                'time': '10-12小时',
                'accuracy': '75-80%',
                'description': '追求高性能的激进配置'
            },
            'ultimate': {
                'name': '终极训练',
                'model': 'EfficientNet-B2',
                'time': '15-20小时',
                'accuracy': '78-82%',
                'description': '最大容量模型，最高性能'
            },
            'noniid_challenge': {
                'name': 'Non-IID挑战',
                'model': 'EfficientNet-B1',
                'time': '8-10小时',
                'accuracy': '70-75%',
                'description': '强non-IID场景优化'
            }
        }
        
        print("📋 可用的训练配置:")
        print("-" * 50)
        for key, info in configs_info.items():
            print(f"🔸 {key:<18} - {info['name']}")
            print(f"   模型: {info['model']:<20} 预期时间: {info['time']}")
            print(f"   预期准确率: {info['accuracy']:<10} {info['description']}")
            print()
    
    def run_training(self, config_name, extra_args=None, background=False):
        """运行训练"""
        if config_name not in self.base_commands:
            print(f"❌ 错误: 未知的配置 '{config_name}'")
            self.print_configs()
            return False
        
        # 构建命令
        cmd = ['python', self.script_path] + self.base_commands[config_name]
        
        # 添加额外参数
        if extra_args:
            cmd.extend(extra_args)
        
        print(f"🎯 启动配置: {config_name}")
        print(f"📝 执行命令: {' '.join(cmd)}")
        print("-" * 80)
        
        # 创建日志文件
        log_filename = f"cifar100_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        try:
            if background:
                # 后台运行
                print(f"🔄 后台运行中... 日志文件: {log_filename}")
                with open(log_filename, 'w') as log_file:
                    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
                print(f"📊 进程ID: {process.pid}")
                print("💡 使用 'tail -f {log_filename}' 查看实时日志")
            else:
                # 前台运行
                print("🔄 开始训练...")
                result = subprocess.run(cmd, capture_output=False)
                if result.returncode == 0:
                    print("✅ 训练完成!")
                else:
                    print("❌ 训练失败!")
                    return False
        
        except KeyboardInterrupt:
            print("\n⚠️ 训练被用户中断")
            return False
        except Exception as e:
            print(f"❌ 运行错误: {e}")
            return False
        
        return True
    
    def interactive_mode(self):
        """交互式模式"""
        self.print_banner()
        self.print_configs()
        
        while True:
            print("🤖 请选择操作:")
            print("1. 选择预设配置训练")
            print("2. 查看配置详情")
            print("3. 退出")
            
            choice = input("\n👉 请输入选择 (1-3): ").strip()
            
            if choice == '1':
                config_name = input("👉 请输入配置名称: ").strip()
                if config_name in self.base_commands:
                    run_bg = input("👉 是否后台运行? (y/N): ").strip().lower() == 'y'
                    self.run_training(config_name, background=run_bg)
                    break
                else:
                    print("❌ 无效的配置名称")
            
            elif choice == '2':
                self.print_configs()
            
            elif choice == '3':
                print("👋 再见!")
                break
            
            else:
                print("❌ 无效选择")


def main():
    parser = argparse.ArgumentParser(description='CIFAR-100 EfficientNet 联邦学习启动器')
    parser.add_argument('--config', type=str, 
                       choices=['quick', 'balanced', 'aggressive', 'ultimate', 'noniid_challenge'],
                       help='预设配置名称')
    parser.add_argument('--background', action='store_true', help='后台运行')
    parser.add_argument('--interactive', action='store_true', help='交互式模式')
    parser.add_argument('--list', action='store_true', help='列出所有可用配置')
    
    # 允许传递额外参数给训练脚本
    parser.add_argument('--gpu', type=str, help='指定GPU')
    parser.add_argument('--seed', type=int, help='随机种子')
    parser.add_argument('--epochs', type=int, help='覆盖全局轮数')
    parser.add_argument('--lr', type=float, help='覆盖学习率')
    
    args, unknown = parser.parse_known_args()
    
    launcher = CIFAR100Launcher()
    
    if args.list:
        launcher.print_banner()
        launcher.print_configs()
        return
    
    if args.interactive:
        launcher.interactive_mode()
        return
    
    if args.config:
        # 构建额外参数
        extra_args = []
        if args.gpu is not None:
            extra_args.extend(['--gpu', str(args.gpu)])
        if args.seed is not None:
            extra_args.extend(['--seed', str(args.seed)])
        if args.epochs is not None:
            extra_args.extend(['--epochs', str(args.epochs)])
        if args.lr is not None:
            extra_args.extend(['--lr', str(args.lr)])
        
        # 添加未知参数
        extra_args.extend(unknown)
        
        launcher.print_banner()
        launcher.run_training(args.config, extra_args, args.background)
    else:
        # 默认交互模式
        launcher.interactive_mode()


if __name__ == '__main__':
    main()
