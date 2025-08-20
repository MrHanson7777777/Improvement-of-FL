#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CIFAR-100 EfficientNet 模型测试脚本
验证模型架构和前向传播是否正常工作
"""

import torch
import time
import traceback
from models_cifar100_optimized import get_cifar100_efficientnet, get_cifar100_transforms
from torchvision import datasets


def test_model_architecture():
    """测试模型架构"""
    print("=" * 60)
    print("🔍 测试CIFAR-100 EfficientNet模型架构")
    print("=" * 60)
    
    try:
        # 测试不同变体
        variants = ['b0', 'b1', 'b2']
        
        for variant in variants:
            print(f"\n📊 测试 EfficientNet-{variant.upper()}:")
            
            # 创建模型
            model = get_cifar100_efficientnet(variant)
            
            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"   总参数量: {total_params:,}")
            print(f"   可训练参数: {trainable_params:,}")
            print(f"   模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
            
            # 测试前向传播
            model.eval()
            test_input = torch.randn(4, 3, 32, 32)  # batch_size=4
            
            start_time = time.time()
            with torch.no_grad():
                output = model(test_input)
            inference_time = time.time() - start_time
            
            print(f"   输入形状: {test_input.shape}")
            print(f"   输出形状: {output.shape}")
            print(f"   推理时间: {inference_time*1000:.2f} ms")
            print(f"   输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")
            
            # 验证输出是否为log概率
            probs = torch.exp(output)
            prob_sum = probs.sum(dim=1)
            print(f"   概率和: {prob_sum.mean().item():.6f} (应该接近1.0)")
            
            print("   ✅ 测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        traceback.print_exc()
        return False


def test_data_transforms():
    """测试数据变换"""
    print("\n" + "=" * 60)
    print("🎨 测试数据变换")
    print("=" * 60)
    
    try:
        # 测试训练变换
        train_transform = get_cifar100_transforms(is_training=True)
        test_transform = get_cifar100_transforms(is_training=False)
        
        # 创建模拟数据
        from PIL import Image
        import numpy as np
        
        # 创建随机CIFAR-100图像
        dummy_image = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        
        print("📊 训练时数据变换:")
        train_tensor = train_transform(dummy_image)
        print(f"   输出形状: {train_tensor.shape}")
        print(f"   数值范围: [{train_tensor.min().item():.3f}, {train_tensor.max().item():.3f}]")
        print(f"   数据类型: {train_tensor.dtype}")
        
        print("\n📊 测试时数据变换:")
        test_tensor = test_transform(dummy_image)
        print(f"   输出形状: {test_tensor.shape}")
        print(f"   数值范围: [{test_tensor.min().item():.3f}, {test_tensor.max().item():.3f}]")
        print(f"   数据类型: {test_tensor.dtype}")
        
        print("   ✅ 数据变换测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 数据变换测试失败: {e}")
        traceback.print_exc()
        return False


def test_gpu_compatibility():
    """测试GPU兼容性"""
    print("\n" + "=" * 60)
    print("🖥️  测试GPU兼容性")
    print("=" * 60)
    
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✅ CUDA可用")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            print(f"   当前GPU: {torch.cuda.current_device()}")
            print(f"   GPU名称: {torch.cuda.get_device_name()}")
            print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # 测试模型在GPU上的运行
            model = get_cifar100_efficientnet('b1').to(device)
            test_input = torch.randn(2, 3, 32, 32).to(device)
            
            start_time = time.time()
            with torch.no_grad():
                output = model(test_input)
            gpu_time = time.time() - start_time
            
            print(f"   GPU推理时间: {gpu_time*1000:.2f} ms")
            print(f"   GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            
            # 测试混合精度
            try:
                scaler = torch.cuda.amp.GradScaler()
                with torch.cuda.amp.autocast():
                    output = model(test_input)
                print("   ✅ 混合精度支持")
            except:
                print("   ⚠️  混合精度不支持")
                
        else:
            print("⚠️  CUDA不可用，将使用CPU")
            device = torch.device('cpu')
            
            # CPU测试
            model = get_cifar100_efficientnet('b0').to(device)
            test_input = torch.randn(2, 3, 32, 32).to(device)
            
            start_time = time.time()
            with torch.no_grad():
                output = model(test_input)
            cpu_time = time.time() - start_time
            
            print(f"   CPU推理时间: {cpu_time*1000:.2f} ms")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU兼容性测试失败: {e}")
        traceback.print_exc()
        return False


def test_memory_requirements():
    """测试内存需求"""
    print("\n" + "=" * 60)
    print("💾 测试内存需求")
    print("=" * 60)
    
    try:
        variants = ['b0', 'b1', 'b2']
        batch_sizes = [16, 32, 64]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for variant in variants:
            print(f"\n📊 EfficientNet-{variant.upper()} 内存测试:")
            model = get_cifar100_efficientnet(variant).to(device)
            
            for batch_size in batch_sizes:
                try:
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    
                    test_input = torch.randn(batch_size, 3, 32, 32).to(device)
                    
                    # 前向传播
                    with torch.no_grad():
                        output = model(test_input)
                    
                    if device.type == 'cuda':
                        memory_used = torch.cuda.max_memory_allocated() / 1024**2
                        print(f"   Batch {batch_size:2d}: {memory_used:6.1f} MB")
                    else:
                        print(f"   Batch {batch_size:2d}: CPU模式")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"   Batch {batch_size:2d}: 内存不足")
                    else:
                        raise e
        
        return True
        
    except Exception as e:
        print(f"❌ 内存测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🚀 CIFAR-100 EfficientNet 模型测试套件")
    print("🕒 测试时间:", time.strftime("%Y-%m-%d %H:%M:%S"))
    
    # 运行所有测试
    tests = [
        ("模型架构", test_model_architecture),
        ("数据变换", test_data_transforms),
        ("GPU兼容性", test_gpu_compatibility),
        ("内存需求", test_memory_requirements)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n🔄 开始测试: {test_name}")
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ 测试 {test_name} 出现异常: {e}")
            results.append((test_name, False))
    
    # 测试总结
    print("\n" + "=" * 60)
    print("📋 测试总结")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:<15}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{len(results)} 测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！模型准备就绪！")
        print("\n💡 推荐的下一步:")
        print("   1. 运行快速测试: python launch_cifar100.py --config quick")
        print("   2. 运行平衡训练: python launch_cifar100.py --config balanced")
        print("   3. 查看详细指南: 查看 CIFAR100_使用指南.md")
    else:
        print("⚠️  部分测试失败，请检查环境配置")
    
    return passed == len(results)


if __name__ == '__main__':
    main()
