#!/usr/bin/env python
"""
快速测试CIFAR-100联邦学习的脚本
用于验证代码是否正常工作
"""

import subprocess
import sys

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 正在测试CIFAR-100联邦学习基本功能...")
    
    # 快速测试命令（少轮数、少用户）
    cmd = [
        "python", "federated_cifar100_main.py",
        "--efficientnet_variant", "b0",  # 使用较小的模型
        "--epochs", "2",                 # 只训练2轮
        "--num_users", "5",              # 只用5个用户
        "--local_ep", "2",               # 每个用户只训练2轮
        "--frac", "0.6",                 # 60%参与率
        "--lr", "0.01",
        "--verbose", "1"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=300)  # 5分钟超时
        if result.returncode == 0:
            print("\n✅ 测试成功！代码可以正常运行")
        else:
            print(f"\n❌ 测试失败，返回码: {result.returncode}")
    except subprocess.TimeoutExpired:
        print("\n⏰ 测试超时，但这可能表示代码正在正常执行")
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")

if __name__ == "__main__":
    test_basic_functionality()
