# 联邦学习优化代码 (FL_Improve_Code)

这是FL_Primitive_Code的升级版本，包含了基线、普通联邦学习和优化联邦学习的实现，并添加了ResNet18Fed模型支持。

## 主要文件说明

### 核心训练脚本
- `baseline_main.py` - 基线训练（中心化学习）
- `federated_main.py` - 基础联邦学习
- `federated_main_improved.py` - 改进版联邦学习（包含更多模型选择）
- `federated_main_optimized.py` - 优化版联邦学习（包含学习率调度、数据增强等）

### 模型定义
- `models.py` - 主要模型定义文件，包含：
  - MLP - 多层感知机
  - CNNMnist - MNIST专用CNN
  - CNNFashion_Mnist - Fashion-MNIST专用CNN
  - CNNCifar - CIFAR-10专用CNN
  - CNNCifar100 - CIFAR-100专用CNN
  - **ResNet18Fed** - 专为联邦学习优化的ResNet18（新增）
  - **replace_bn_with_gn** - BatchNorm替换为GroupNorm的函数（新增）

### 配置和工具
- `options.py` - 命令行参数配置
- `utils.py` - 工具函数
- `sampling.py` - 数据采样策略
- `update.py` - 本地更新和测试函数
- `download.py` - 数据集下载

## 新增ResNet18Fed模型特性

### 1. ResNet18Fed模型
- 专为联邦学习场景设计的ResNet18变体
- 使用GroupNorm替代BatchNorm，提高联邦学习性能
- 支持CIFAR-10和CIFAR-100数据集

### 2. 使用方法

#### 基线训练（中心化）
```bash
# CIFAR-10 + ResNet18Fed
python baseline_main.py --model=resnet --dataset=cifar --epochs=50 --lr=0.01

# CIFAR-100 + ResNet18Fed  
python baseline_main.py --model=resnet --dataset=cifar100 --epochs=100 --lr=0.01
```

#### 联邦学习训练
```bash
# 基础联邦学习
python federated_main.py --model=resnet --dataset=cifar --epochs=100 --num_users=100 --frac=0.1

# 改进版联邦学习
python federated_main_improved.py --model=resnet --dataset=cifar --epochs=100 --num_users=100 --frac=0.1

# 优化版联邦学习（推荐）
python federated_main_optimized.py --model=resnet --dataset=cifar --epochs=100 --num_users=100 --frac=0.1
```

### 3. 模型优势
- **GroupNorm替代BatchNorm**: 在小批量数据上表现更稳定，适合联邦学习场景
- **残差连接**: 缓解梯度消失问题，提高训练效果
- **适配联邦学习**: 专为分布式训练优化的架构设计

## 支持的数据集和模型组合

| 数据集 | 支持的模型 |
|--------|------------|
| MNIST | cnn, mlp |
| Fashion-MNIST | cnn, mlp |
| CIFAR-10 | cnn, resnet, mlp |
| CIFAR-100 | cnn, resnet, mlp |

## 主要参数说明

- `--model`: 模型类型 (mlp, cnn, resnet)
- `--dataset`: 数据集 (mnist, fmnist, cifar, cifar100)
- `--epochs`: 训练轮数
- `--num_users`: 联邦学习客户端数量
- `--frac`: 每轮参与训练的客户端比例
- `--local_ep`: 本地训练轮数
- `--local_bs`: 本地批量大小
- `--lr`: 学习率

## 文件清理说明

已删除的不必要文件：
- 各种测试脚本（test_*.py, quick_*.py）
- 调试脚本（debug_*.py, deep_debug.py）
- 实验运行脚本（run_experiments.*)
- 多余的模型文件（models_original.py, models_high_performance.py）
- 配置文件（ultra_*.py, lightweight_*.py）
- 空的文档文件
- Python缓存文件夹（__pycache__）

保留的核心文件专注于基线、普通联邦学习和优化联邦学习的实现。

## 代码特点

### ResNet18Fed优势
1. **适配联邦学习**: 使用GroupNorm替代BatchNorm，在小批量数据上表现更稳定
2. **模型压缩**: 相比ResNet50等大型模型，参数量更少，通信开销更小
3. **残差连接**: 有效缓解梯度消失，提升收敛速度
4. **灵活配置**: 支持不同类别数，适配CIFAR-10和CIFAR-100

### 联邦学习优化
- **学习率调度**: 优化版本包含多种学习率调度策略
- **数据增强**: 支持CutMix、Mixup等数据增强技术
- **正则化**: 包含Dropout、Label Smoothing等正则化方法
- **早停机制**: 防止过拟合，提高泛化性能

## 项目结构说明

这是FL_Primitive_Code的升级版本，主要改进：
1. 添加了ResNet18Fed模型支持
2. 清理了冗余和测试文件
3. 优化了代码结构和可读性
4. 增强了联邦学习性能
