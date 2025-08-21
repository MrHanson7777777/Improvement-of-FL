# FL模型使用指南

## 🎯 快速开始

### 基线训练 (中心化学习)
```bash
# MNIST
python baseline_main.py --model=mlp --dataset=mnist --epochs=50

# CIFAR-10  
python baseline_main.py --model=resnet18 --dataset=cifar --epochs=100

# CIFAR-100
python baseline_main.py --model=resnet18 --dataset=cifar100 --epochs=150
```

### 联邦学习训练
```bash
# MNIST (优化CNN)
python federated_main.py --model=cnn_optimized --dataset=mnist --epochs=50 --num_users=100 --frac=0.1

# CIFAR-10 (ResNet18Fed优化版)
python federated_main.py --model=resnet18 --dataset=cifar --epochs=100 --num_users=100 --frac=0.1

# CIFAR-100 (ResNet18Fed + SE注意力)
python federated_main.py --model=resnet18 --dataset=cifar100 --epochs=200 --num_users=100 --frac=0.1

# CIFAR-10 (EfficientNet优化版)  
python federated_main.py --model=efficientnet --dataset=cifar --epochs=100 --num_users=100 --frac=0.1

# CIFAR-100 (DenseNet优化版)
python federated_main.py --model=densenet --dataset=cifar100 --epochs=200 --num_users=100 --frac=0.1
```

## ⚙️ 一键修改参数

### 修改训练轮数
```bash
# 短期测试 (10轮)
python federated_main.py --model=resnet18 --dataset=cifar --epochs=10

# 标准训练 (100轮)  
python federated_main.py --model=resnet18 --dataset=cifar --epochs=100

# 长期训练 (500轮)
python federated_main.py --model=resnet18 --dataset=cifar --epochs=500
```

### 修改优化器
```bash  
# SGD优化器
python federated_main.py --model=resnet18 --dataset=cifar --optimizer=sgd --lr=0.01 --momentum=0.9

# Adam优化器
python federated_main.py --model=resnet18 --dataset=cifar --optimizer=adam --lr=0.001

# AdamW优化器 (推荐)
python federated_main.py --model=resnet18 --dataset=cifar --optimizer=adamw --lr=0.001 --weight_decay=0.01
```

### 修改学习率策略
```bash
# 固定学习率
python federated_main.py --model=resnet18 --dataset=cifar --lr_scheduler=none --lr=0.001

# 步长衰减
python federated_main.py --model=resnet18 --dataset=cifar --lr_scheduler=step --lr=0.01 --lr_step_size=30 --lr_gamma=0.1

# 指数衰减
python federated_main.py --model=resnet18 --dataset=cifar --lr_scheduler=exp --lr=0.01 --lr_gamma=0.95

# 余弦退火 (推荐)
python federated_main.py --model=resnet18 --dataset=cifar --lr_scheduler=cosine --lr=0.01 --cosine_t_max=100
```
# SGD优化器 (默认)
python federated_main.py --model=resnet18 --dataset=cifar --optimizer=sgd

# Adam优化器
python federated_main.py --model=resnet18 --dataset=cifar --optimizer=adam
```

### 修改数据分布 (IID vs Non-IID)
```bash
# IID数据分布 (独立同分布)
python federated_main.py --model=resnet18 --dataset=cifar --iid=1

# Non-IID数据分布 (非独立同分布) 
python federated_main.py --model=resnet18 --dataset=cifar --iid=0
```

### 组合参数修改
```bash
# CIFAR-10, ResNet18, 200轮, Adam优化器, Non-IID分布
python federated_main.py --model=resnet18 --dataset=cifar --epochs=200 --optimizer=adam --iid=0 --num_users=50 --frac=0.2

# CIFAR-100, EfficientNet, 300轮, SGD优化器, IID分布
python federated_main.py --model=efficientnet --dataset=cifar100 --epochs=300 --optimizer=sgd --iid=1 --lr=0.01
```

## 📁 模型架构

### 专用模型文件
- `models_mnist.py` - MNIST专用模型 (MLP, CNN优化版, 轻量级版)
- `models_cifar10.py` - CIFAR-10专用模型 (ResNet18Fed, EfficientNet, 轻量级版)  
- `models_cifar100.py` - CIFAR-100专用模型 (ResNet18Fed, EfficientNet, DenseNet)
- `model_factory.py` - 统一模型工厂接口

### 使用模型工厂
```python
from model_factory import get_model, get_recommended_model

# 获取指定模型
model = get_model('mnist', 'cnn_optimized')

# 获取推荐模型
model = get_recommended_model('cifar10', performance='high')
```

## 🔧 支持的模型

### 原有模型 (向后兼容)
- `mlp` - 基础多层感知机 (通过models.py)
- `cnn` - 基础CNN模型 (通过models.py)
- `resnet` - 原版ResNet (会自动应用GroupNorm优化)

### 新增优化模型 ⭐
- `resnet18` - ResNet18Fed联邦学习优化版
- `efficientnet` - EfficientNet系列优化模型
- `densenet` - DenseNet优化模型 (仅CIFAR-100)
- `cnn_optimized` - 优化CNN模型 (仅MNIST, 基于ResNet思想)

### 数据集支持矩阵

| 模型 | MNIST | CIFAR-10 | CIFAR-100 | 特点 |
|------|-------|----------|-----------|------|
| `mlp` | ✅ | ✅ | ✅ | 基础模型 |
| `cnn` | ✅ | ✅ | ✅ | 基础CNN |
| `resnet` | ➡️ CNN | ✅ | ✅ | 原版ResNet |
| `resnet18` | ➡️ 优化CNN | ✅ | ✅ | **推荐** |
| `efficientnet` | ❌ | ✅ | ✅ | 高精度 |
| `densenet` | ❌ | ❌ | ✅ | CIFAR-100专用 |
| `cnn_optimized` | ✅ | ❌ | ❌ | MNIST专用ResNet风格 |

## ⚙️ 关键特性

### ResNet18Fed
- ✅ GroupNorm替代BatchNorm (适合FL小批量)
- ✅ 残差连接缓解梯度消失
- ✅ 轻量级架构减少通信开销

### EfficientNet系列
- ✅ 深度-宽度-分辨率复合缩放
- ✅ MBConv模块高效卷积
- ✅ Squeeze-and-Excitation注意力

### 优化技术
- ✅ 标签平滑 (Label Smoothing)
- ✅ 权重衰减正则化
- ✅ 学习率衰减策略
- ✅ 梯度裁剪

## 📊 推荐配置

### 快速测试
```bash
python federated_main.py --model=cnn --dataset=mnist --epochs=10 --num_users=10 --frac=0.5
```

### 高精度训练
```bash
python federated_main.py --model=resnet18 --dataset=cifar100 --epochs=200 --num_users=100 --frac=0.1 --local_ep=5 --lr=0.01
```

### 高精度训练
```bash  
python federated_main.py --model=efficientnet --dataset=cifar --epochs=50 --num_users=50 --frac=0.2
```

## 🛠️ 故障排除

### 内存不足
- 减少 `--local_bs` (批次大小)
- 减少 `--num_users`

### 训练速度慢
- 增加 `--frac` (参与比例)  
- 减少 `--local_ep` (本地轮数)
- 使用GPU加速 `--gpu=0`

### 精度不高
- 增加 `--epochs` (总轮数)
- 调整 `--lr` (学习率)
- 使用更复杂模型 (`resnet18`, `efficientnet`)

## 📝 文件说明

### 核心文件
- `federated_main.py` - 主要联邦学习训练脚本
- `baseline_main.py` - 基线中心化训练脚本  
- `options.py` - 命令行参数配置
- `update.py` - 本地更新逻辑
- `utils.py` - 工具函数
- `sampling.py` - 数据分布策略

### 测试文件
- `test_resnet18fed.py` - ResNet18Fed模型测试
- `example_usage.py` - 使用示例

### 数据处理
- `download.py` - 数据集下载脚本

---

## ✨ 总结

现在的架构更加模块化和专业化：
- 每个数据集都有专门优化的模型
- 统一的工厂接口便于使用  
- 保持向后兼容性
- 清晰的文档和示例
