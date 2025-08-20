import torch
print("✅ PyTorch导入成功")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

try:
    from models_cifar100_optimized import get_cifar100_efficientnet
    print("✅ CIFAR-100模型导入成功")
    
    model = get_cifar100_efficientnet('b1')
    print("✅ EfficientNet-B1创建成功")
    
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"✅ 前向传播成功，输出形状: {y.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型参数量: {total_params:,}")
    
    print("\n🎉 所有基础测试通过！")
    print("💡 可以开始CIFAR-100训练:")
    print("   python launch_cifar100.py --config quick")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
