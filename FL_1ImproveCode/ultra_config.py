#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CIFAR-100超强模型配置
集成所有最优参数和超强融合模型
"""

# ===========================================
# 🔥 超强模型命令 (推荐用于最高准确率)
# ===========================================

ULTRA_MODEL_COMMAND = """
python federated_cifar100_main.py --model ultra --epochs 350 --num_users 40 --frac 0.5 --local_ep 8 --local_bs 32 --lr 0.003 --optimizer adamw --dropout_rate 0.4 --label_smoothing 0.1 --alpha 0.3 --gpu 0
"""

# ===========================================
# EfficientNet最优命令 (Non-IID优化)
# ===========================================

# EfficientNet-B0 (快速训练，良好性能)
EFFICIENTNET_B0_NONIID_COMMAND = """
python federated_cifar100_main.py --model cifar100_efficientnet --efficientnet_variant b0 --epochs 350 --num_users 40 --frac 0.5 --local_ep 8 --local_bs 32 --lr 0.003 --optimizer adamw --dropout_rate 0.3 --label_smoothing 0.1 --alpha 0.3 --gpu 0
"""

# EfficientNet-B1 (平衡性能和速度)
EFFICIENTNET_B1_NONIID_COMMAND = """
python federated_cifar100_main.py --model cifar100_efficientnet --efficientnet_variant b1 --epochs 350 --num_users 40 --frac 0.5 --local_ep 8 --local_bs 32 --lr 0.003 --optimizer adamw --dropout_rate 0.35 --label_smoothing 0.1 --alpha 0.3 --gpu 0
"""

# EfficientNet-B2 (最高性能，需要更长时间)
EFFICIENTNET_B2_NONIID_COMMAND = """
python federated_cifar100_main.py --model cifar100_efficientnet --efficientnet_variant b2 --epochs 350 --num_users 40 --frac 0.5 --local_ep 8 --local_bs 28 --lr 0.002 --optimizer adamw --dropout_rate 0.4 --label_smoothing 0.1 --alpha 0.3 --gpu 0
"""

# ===========================================
# 性能预期 (基于Non-IID优化)
# ===========================================

PERFORMANCE_EXPECTATIONS = {
    "ultra_model": {
        "accuracy": "68-72%",
        "training_time": "10-14小时",
        "gpu_memory": "8-12GB",
        "description": "超强融合模型，最高准确率",
        "features": [
            "SE注意力机制",
            "随机深度正则化", 
            "多层分类器",
            "高级数据增强",
            "标签平滑"
        ]
    },
    
    "efficientnet_b0": {
        "accuracy": "62-66%", 
        "training_time": "6-8小时",
        "gpu_memory": "4-6GB",
        "description": "快速训练，良好性能"
    },
    
    "efficientnet_b1": {
        "accuracy": "65-69%",
        "training_time": "8-10小时", 
        "gpu_memory": "6-8GB",
        "description": "平衡性能和效率"
    },
    
    "efficientnet_b2": {
        "accuracy": "67-71%",
        "training_time": "12-16小时",
        "gpu_memory": "8-12GB", 
        "description": "高性能，需要更多资源"
    }
}

# ===========================================
# 快速启动建议
# ===========================================

QUICK_START_RECOMMENDATIONS = {
    "最高准确率": "使用超强模型 (Ultra) - 68-72%准确率",
    "平衡性能": "使用EfficientNet-B1 - 65-69%准确率", 
    "快速验证": "使用EfficientNet-B0 - 62-66%准确率",
    "资源充足": "使用EfficientNet-B2 - 67-71%准确率"
}

# ===========================================
# 命令生成函数
# ===========================================

def get_optimal_command(model_type="ultra", gpu_id=0):
    """获取最优训练命令"""
    commands = {
        "ultra": ULTRA_MODEL_COMMAND,
        "b0": EFFICIENTNET_B0_NONIID_COMMAND,
        "b1": EFFICIENTNET_B1_NONIID_COMMAND,
        "b2": EFFICIENTNET_B2_NONIID_COMMAND
    }
    
    command = commands.get(model_type, ULTRA_MODEL_COMMAND)
    return command.replace("--gpu 0", f"--gpu {gpu_id}").strip()


def print_model_comparison():
    """打印模型对比"""
    print("🏆 CIFAR-100 模型性能对比")
    print("=" * 60)
    
    models = [
        ("Ultra超强模型", "68-72%", "10-14小时", "最高准确率"),
        ("EfficientNet-B2", "67-71%", "12-16小时", "高性能"), 
        ("EfficientNet-B1", "65-69%", "8-10小时", "平衡性能"),
        ("EfficientNet-B0", "62-66%", "6-8小时", "快速训练")
    ]
    
    for model, acc, time, desc in models:
        print(f"📊 {model:<15} | 准确率: {acc:<8} | 时间: {time:<10} | {desc}")


if __name__ == "__main__":
    print("🔥 CIFAR-100 超强模型配置")
    print("=" * 50)
    print("\n💎 推荐命令 (超强模型 - 最高准确率):")
    print(get_optimal_command("ultra"))
    
    print("\n")
    print_model_comparison()
    
    print(f"\n✨ 超强模型特性:")
    for feature in PERFORMANCE_EXPECTATIONS["ultra_model"]["features"]:
        print(f"   • {feature}")
    
    print(f"\n🎯 使用建议:")
    for situation, recommendation in QUICK_START_RECOMMENDATIONS.items():
        print(f"   • {situation}: {recommendation}")
    
    print("\n🚀 注意: 超强模型融合了所有最新优化技术，预期达到最高准确率!")
