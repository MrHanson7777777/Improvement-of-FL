# 多数据集联邦学习框架

本代码已修改为支持四个常用的机器学习数据集：**MNIST**、**Fashion-MNIST**、**CIFAR-10** 和 **CIFAR-100**。

## 📋 功能特性

- ✅ **四数据集支持**: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100
- ✅ **多模型架构**: CNN (针对不同数据集优化) 和 MLP
- ✅ **联邦学习**: 支持IID和Non-IID数据分布
- ✅ **基线学习**: 传统的中心化训练作为对比
- ✅ **自动参数配置**: 根据数据集自动设置通道数和类别数

## 🏗️ 代码结构

```
Code/
├── baseline_main.py      # 基线（中心化）学习主程序
├── federated_main.py     # 联邦学习主程序  
├── models.py            # 模型定义（CNN和MLP）
├── options.py           # 命令行参数解析
├── utils.py             # 数据集加载和工具函数
├── sampling.py          # 数据划分函数（IID/Non-IID）
├── update.py            # 本地更新和测试函数
├── test_datasets.py     # 数据集支持测试脚本
├── usage_examples.py    # 使用示例
└── README_datasets.md   # 本文档
```

## 🚀 快速开始

### 1. 基线学习示例

```bash
# MNIST 数据集
python baseline_main.py --dataset mnist --model cnn --epochs 10

# Fashion-MNIST 数据集  
python baseline_main.py --dataset fmnist --model cnn --epochs 10

# CIFAR-10 数据集
python baseline_main.py --dataset cifar --model cnn --epochs 10

# CIFAR-100 数据集
python baseline_main.py --dataset cifar100 --model cnn --epochs 10
```

### 2. 联邦学习示例

```bash
# MNIST 联邦学习 (IID)
python federated_main.py --dataset mnist --model cnn --epochs 10 --num_users 100 --frac 0.1 --iid 1

# Fashion-MNIST 联邦学习 (Non-IID)
python federated_main.py --dataset fmnist --model cnn --epochs 10 --num_users 100 --frac 0.1 --iid 0

# CIFAR-10 联邦学习
python federated_main.py --dataset cifar --model cnn --epochs 10 --num_users 100 --frac 0.1

# CIFAR-100 联邦学习
python federated_main.py --dataset cifar100 --model cnn --epochs 10 --num_users 100 --frac 0.1
```

## 📊 数据集详情

| 数据集 | 图像尺寸 | 通道数 | 类别数 | 训练集大小 | 测试集大小 |
|--------|----------|--------|--------|-----------|-----------|
| MNIST | 28×28 | 1 (灰度) | 10 | 60,000 | 10,000 |
| Fashion-MNIST | 28×28 | 1 (灰度) | 10 | 60,000 | 10,000 |
| CIFAR-10 | 32×32 | 3 (RGB) | 10 | 50,000 | 10,000 |
| CIFAR-100 | 32×32 | 3 (RGB) | 100 | 50,000 | 10,000 |

## 🏛️ 模型架构

### CNN模型
- **CNNMnist**: 适用于MNIST，2层卷积 + 2层全连接
- **CNNFashion_Mnist**: 适用于Fashion-MNIST，带批归一化的深层CNN
- **CNNCifar**: 适用于CIFAR-10，经典的LeNet风格CNN
- **CNNCifar100**: 适用于CIFAR-100，加深网络适应100类分类

### MLP模型
- 通用的多层感知机，可用于所有数据集
- 自动计算输入维度（图像像素总数）

## ⚙️ 主要参数

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--dataset` | 数据集选择 | mnist | mnist, fmnist, cifar, cifar100 |
| `--model` | 模型类型 | mlp | cnn, mlp |
| `--epochs` | 训练轮数 | 10 | 任意正整数 |
| `--num_users` | 客户端数量 | 100 | 任意正整数 |
| `--frac` | 参与训练的客户端比例 | 0.1 | 0-1之间的浮点数 |
| `--iid` | 数据分布类型 | 1 | 1 (IID), 0 (Non-IID) |
| `--lr` | 学习率 | 0.01 | 任意正浮点数 |
| `--local_ep` | 本地训练轮数 | 10 | 任意正整数 |
| `--local_bs` | 本地批次大小 | 10 | 任意正整数 |

## 🔧 代码修改说明

### 主要修改内容：

1. **修复Fashion-MNIST支持**：
   - 修复了`utils.py`中的bug，现在正确加载Fashion-MNIST数据集
   
2. **添加CIFAR-100支持**：
   - 新增`CNNCifar100`模型架构
   - 在所有主要文件中添加CIFAR-100的支持

3. **自动参数配置**：
   - 在`options.py`中添加了根据数据集自动设置参数的功能
   - 自动设置`num_channels`和`num_classes`

4. **改进的数据处理**：
   - 统一了数据预处理流程
   - 复用现有的数据划分函数支持新数据集

## 🧪 测试验证

运行测试脚本验证所有数据集支持：

```bash
python test_datasets.py
```

查看使用示例：

```bash
python usage_examples.py  
```

## 📈 实验建议

### 不同数据集的训练策略：

1. **MNIST**: 简单数据集，快速验证算法
   - 建议epochs: 5-10
   - 学习率: 0.01
   
2. **Fashion-MNIST**: 比MNIST稍难，图像更复杂
   - 建议epochs: 10-20  
   - 学习率: 0.01
   
3. **CIFAR-10**: 彩色图像，需要更多训练
   - 建议epochs: 20-50
   - 学习率: 0.001-0.01
   
4. **CIFAR-100**: 最具挑战性，100个类别
   - 建议epochs: 50-100
   - 学习率: 0.001
   - 可能需要学习率调度

## 🚨 注意事项

1. 首次运行会自动下载数据集到`../data/`目录
2. CIFAR-100训练需要更多时间和计算资源
3. GPU训练可以使用`--gpu 0`参数
4. 确保有足够的存储空间（所有数据集约2GB）

## 📝 输出文件

- 训练曲线图：保存在`./save/`目录
- TensorBoard日志：保存在`../logs/`目录
- 可视化训练损失变化情况
