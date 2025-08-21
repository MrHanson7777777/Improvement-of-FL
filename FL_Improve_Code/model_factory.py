#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦学习优化模型统一入口
========================

本文件提供了所有数据集专用模型的统一访问接口，包括：

1. MNIST数据集模型 (models_mnist.py)
   - CNN_MNIST_Optimized: 优化CNN模型 (推荐)

2. CIFAR-10数据集模型 (models_cifar10.py)
   - ResNet18_CIFAR10_Fed: 联邦学习ResNet18
   - EfficientNet_CIFAR10: EfficientNet风格模型

3. CIFAR-100数据集模型 (models_cifar100.py)
   - ResNet18_CIFAR100_Fed: 联邦学习ResNet18
   - EfficientNet_CIFAR100: EfficientNet-B3风格模型
   - DenseNet_CIFAR100: DenseNet模型

使用方法：
```python
from model_factory import get_model, list_available_models

# 获取MNIST优化CNN模型
model = get_model('mnist', 'optimized')

# 获取CIFAR-10联邦学习ResNet18
model = get_model('cifar10', 'resnet18_fed', use_groupnorm=True)

# 列出所有可用模型
list_available_models()
```
"""

import torch
import torch.nn as nn

try:
    from .models_mnist import get_mnist_model, MNIST_MODEL_CONFIGS
    from .models_cifar10 import get_cifar10_model, CIFAR10_MODEL_CONFIGS, replace_bn_with_gn
    from .models_cifar100 import get_cifar100_model, CIFAR100_MODEL_CONFIGS
except ImportError:
    # 处理直接运行或不同导入路径的情况
    try:
        from models_mnist import get_mnist_model, MNIST_MODEL_CONFIGS
        from models_cifar10 import get_cifar10_model, CIFAR10_MODEL_CONFIGS, replace_bn_with_gn
        from models_cifar100 import get_cifar100_model, CIFAR100_MODEL_CONFIGS
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保模型文件在正确的路径下")


def get_model(dataset, model_type, **kwargs):
    """
    统一模型获取接口
    ================
    
    根据数据集和模型类型返回相应的模型实例。
    
    参数:
        dataset (str): 数据集名称
            - 'mnist': MNIST手写数字识别
            - 'cifar10': CIFAR-10自然图像分类
            - 'cifar100': CIFAR-100自然图像分类
        model_type (str): 模型类型（取决于具体数据集）
        **kwargs: 模型特定参数
    
    返回:
        nn.Module: 对应的模型实例
    
    示例:
        >>> model = get_model('mnist', 'optimized', dropout_rate=0.5)
        >>> model = get_model('cifar10', 'resnet18_fed', use_groupnorm=True)
        >>> model = get_model('cifar100', 'efficientnet', dropout_rate=0.3)
    """
    dataset = dataset.lower()
    
    if dataset == 'mnist':
        return get_mnist_model(model_type, **kwargs)
    elif dataset == 'cifar10':
        return get_cifar10_model(model_type, **kwargs)
    elif dataset == 'cifar100':
        return get_cifar100_model(model_type, **kwargs)
    else:
        raise ValueError(f"不支持的数据集: {dataset}. 支持的数据集: ['mnist', 'cifar10', 'cifar100']")


def get_model_by_config(dataset, config_name):
    """
    根据预设配置获取模型
    ====================
    
    使用预定义的配置快速获取模型实例。
    
    参数:
        dataset (str): 数据集名称
        config_name (str): 配置名称
    
    返回:
        nn.Module: 对应的模型实例
    """
    dataset = dataset.lower()
    
    if dataset == 'mnist':
        if config_name not in MNIST_MODEL_CONFIGS:
            raise ValueError(f"不支持的MNIST配置: {config_name}")
        config = MNIST_MODEL_CONFIGS[config_name]
        return get_mnist_model(**config)
    elif dataset == 'cifar10':
        if config_name not in CIFAR10_MODEL_CONFIGS:
            raise ValueError(f"不支持的CIFAR-10配置: {config_name}")
        config = CIFAR10_MODEL_CONFIGS[config_name]
        return get_cifar10_model(**config)
    elif dataset == 'cifar100':
        if config_name not in CIFAR100_MODEL_CONFIGS:
            raise ValueError(f"不支持的CIFAR-100配置: {config_name}")
        config = CIFAR100_MODEL_CONFIGS[config_name]
        return get_cifar100_model(**config)
    else:
        raise ValueError(f"不支持的数据集: {dataset}")


def list_available_models():
    """
    列出所有可用的模型和配置
    ========================
    
    打印所有数据集的可用模型类型和预设配置。
    """
    print("可用模型概览")
    print("=" * 80)
    
    print("\n📊 MNIST数据集模型:")
    print("   模型类型:")
    print("   - 'mlp': 多层感知机")
    print("   - 'optimized': 优化CNN模型（ResNet风格）")
    print("   预设配置:")
    for config in MNIST_MODEL_CONFIGS.keys():
        print(f"   - {config}")
    
    print("\n🌅 CIFAR-10数据集模型:")
    print("   模型类型:")
    print("   - 'resnet18_fed': ResNet18联邦学习版本")
    print("   - 'efficientnet': EfficientNet风格模型")
    print("   预设配置:")
    for config in CIFAR10_MODEL_CONFIGS.keys():
        print(f"   - {config}")
    
    print("\n🎯 CIFAR-100数据集模型:")
    print("   模型类型:")
    print("   - 'resnet18_fed': ResNet18联邦学习版本")
    print("   - 'efficientnet': EfficientNet-B3风格模型")
    print("   - 'densenet': DenseNet模型")
    print("   预设配置:")
    for config in CIFAR100_MODEL_CONFIGS.keys():
        print(f"   - {config}")


def get_model_info(model):
    """
    获取模型信息
    ============
    
    返回模型的基本信息，包括参数量、模型大小等。
    
    参数:
        model (nn.Module): 模型实例
    
    返回:
        dict: 包含模型信息的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 估算模型大小（MB）
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb,
        'model_class': model.__class__.__name__
    }


def compare_models(models_dict, input_shape):
    """
    比较多个模型的性能指标
    ======================
    
    对比多个模型的参数量、推理时间等指标。
    
    参数:
        models_dict (dict): {模型名称: 模型实例} 的字典
        input_shape (tuple): 输入张量形状 (batch_size, channels, height, width)
    
    返回:
        dict: 包含比较结果的字典
    """
    import time
    
    results = {}
    test_input = torch.randn(*input_shape)
    
    for name, model in models_dict.items():
        model.eval()
        info = get_model_info(model)
        
        # 测试推理时间
        with torch.no_grad():
            start_time = time.time()
            for _ in range(100):  # 运行100次取平均
                _ = model(test_input)
            avg_time = (time.time() - start_time) / 100
        
        results[name] = {
            **info,
            'inference_time_ms': avg_time * 1000
        }
    
    return results


def print_model_comparison(results):
    """
    打印模型比较结果
    ================
    
    以表格形式展示模型比较结果。
    """
    print("\n模型性能对比")
    print("=" * 100)
    print(f"{'模型名称':<20} {'参数量':<12} {'大小(MB)':<10} {'推理时间(ms)':<15} {'模型类型':<20}")
    print("-" * 100)
    
    for name, info in results.items():
        print(f"{name:<20} {info['total_params']:<12,} {info['model_size_mb']:<10.2f} "
              f"{info['inference_time_ms']:<15.2f} {info['model_class']:<20}")


# 推荐配置
RECOMMENDED_CONFIGS = {
    'mnist': {
        'high_accuracy': 'cnn_optimized',
        'fast_training': 'mlp_large'
    },
    'cifar10': {
        'federated_learning': 'resnet18_fed_default',
        'high_accuracy': 'efficientnet_default'
    },
    'cifar100': {
        'federated_learning': 'resnet18_fed_default',
        'high_accuracy': 'efficientnet_default',
        'research': 'densenet_default'
    }
}


def get_recommended_model(dataset, scenario):
    """
    获取推荐模型配置
    ================
    
    根据使用场景返回推荐的模型配置。
    
    参数:
        dataset (str): 数据集名称
        scenario (str): 使用场景
            - 'high_accuracy': 高精度场景
            - 'federated_learning': 联邦学习场景
            - 'fast_training': 快速训练场景
            - 'research': 研究场景
    
    返回:
        nn.Module: 推荐的模型实例
    """
    dataset = dataset.lower()
    
    if dataset not in RECOMMENDED_CONFIGS:
        raise ValueError(f"不支持的数据集: {dataset}")
    
    if scenario not in RECOMMENDED_CONFIGS[dataset]:
        available_scenarios = list(RECOMMENDED_CONFIGS[dataset].keys())
        raise ValueError(f"数据集 {dataset} 不支持场景 {scenario}. 可用场景: {available_scenarios}")
    
    config_name = RECOMMENDED_CONFIGS[dataset][scenario]
    return get_model_by_config(dataset, config_name)


if __name__ == "__main__":
    """演示模型工厂的使用"""
    print("联邦学习模型工厂演示")
    print("=" * 50)
    
    # 显示所有可用模型
    list_available_models()
    
    # 测试各数据集的推荐模型
    datasets = ['mnist', 'cifar10', 'cifar100']
    input_shapes = {
        'mnist': (1, 1, 28, 28),
        'cifar10': (1, 3, 32, 32),
        'cifar100': (1, 3, 32, 32)
    }
    
    for dataset in datasets:
        print(f"\n测试 {dataset.upper()} 数据集模型:")
        
        # 获取联邦学习推荐模型
        try:
            model = get_recommended_model(dataset, 'federated_learning')
            info = get_model_info(model)
            print(f"  联邦学习推荐模型: {info['model_class']}")
            print(f"  参数量: {info['total_params']:,}")
            print(f"  模型大小: {info['model_size_mb']:.2f} MB")
            
            # 测试前向传播
            test_input = torch.randn(*input_shapes[dataset])
            with torch.no_grad():
                output = model(test_input)
            print(f"  输出形状: {output.shape}")
            
        except Exception as e:
            print(f"  错误: {e}")
    
    print(f"\n模型工厂演示完成！")
