#!/bin/bash
# run_experiments.sh
# 联邦学习模型架构改进效果验证实验自动化脚本

echo "开始联邦学习模型架构改进效果验证实验..."

# 创建结果目录
mkdir -p results/original
mkdir -p results/improved

echo "实验配置:"
echo "- 控制组: 传统简单CNN架构"
echo "- 实验组: 深度CNN架构（ResNet/EfficientNet）"
echo ""

# 1. MNIST实验
echo "=== 1. MNIST数据集实验 ==="
echo "运行原始版本（简单CNN）..."
python federated_main_original.py --dataset mnist --model cnn --epochs 50 --lr 0.01 --frac 0.1 --num_users 100 --local_ep 5 --local_bs 64 --optimizer sgd --kernel_num 64 --kernel_sizes 3,4,5 --num_channels 1 --norm batch_norm --gpu 0 --seed 42 > results/original/mnist_cnn.log 2>&1

echo "运行改进版本（ResNet架构）..."
python federated_main_improved.py --dataset mnist --model cnn --epochs 50 --lr 0.01 --frac 0.1 --num_users 100 --local_ep 5 --local_bs 64 --optimizer sgd --kernel_num 64 --kernel_sizes 3,4,5 --num_channels 1 --norm batch_norm --stopping_rounds 10 --gpu 0 --seed 42 > results/improved/mnist_cnn.log 2>&1

echo "MNIST实验完成！"
echo ""

# 2. Fashion-MNIST实验
echo "=== 2. Fashion-MNIST数据集实验 ==="
echo "运行原始版本（基础CNN）..."
python federated_main_original.py --dataset fmnist --model cnn --epochs 80 --lr 0.001 --frac 0.1 --num_users 100 --local_ep 5 --local_bs 32 --optimizer adam --kernel_num 64 --kernel_sizes 3,4,5 --num_channels 1 --norm batch_norm --gpu 0 --seed 42 > results/original/fmnist_cnn.log 2>&1

echo "运行改进版本（EfficientNet + SE注意力）..."
python federated_main_improved.py --dataset fmnist --model cnn --epochs 80 --lr 0.001 --frac 0.1 --num_users 100 --local_ep 5 --local_bs 32 --optimizer adam --kernel_num 64 --kernel_sizes 3,4,5 --num_channels 1 --norm batch_norm --stopping_rounds 15 --gpu 0 --seed 42 > results/improved/fmnist_cnn.log 2>&1

echo "Fashion-MNIST实验完成！"
echo ""

# 3. CIFAR-10实验
echo "=== 3. CIFAR-10数据集实验 ==="
echo "运行原始版本（LeNet风格）..."
python federated_main_original.py --dataset cifar --model cnn --epochs 100 --lr 0.001 --frac 0.05 --num_users 100 --local_ep 5 --local_bs 32 --optimizer adam --kernel_num 64 --kernel_sizes 3,4,5 --num_channels 3 --norm batch_norm --gpu 0 --seed 42 > results/original/cifar_cnn.log 2>&1

echo "运行改进版本（EfficientNet-B0）..."
python federated_main_improved.py --dataset cifar --model cnn --epochs 100 --lr 0.001 --frac 0.05 --num_users 100 --local_ep 5 --local_bs 32 --optimizer adam --kernel_num 64 --kernel_sizes 3,4,5 --num_channels 3 --norm batch_norm --stopping_rounds 20 --gpu 0 --seed 42 > results/improved/cifar_cnn.log 2>&1

echo "CIFAR-10实验完成！"
echo ""

# 4. CIFAR-100实验
echo "=== 4. CIFAR-100数据集实验 ==="
echo "运行原始版本（简单CNN）..."
python federated_main_original.py --dataset cifar100 --model cnn --epochs 150 --lr 0.001 --frac 0.05 --num_users 100 --local_ep 10 --local_bs 16 --optimizer adam --kernel_num 128 --kernel_sizes 3,4,5 --num_channels 3 --norm batch_norm --gpu 0 --seed 42 > results/original/cifar100_cnn.log 2>&1

echo "运行改进版本（EfficientNet-B3）..."
python federated_main_improved.py --dataset cifar100 --model cnn --epochs 150 --lr 0.001 --frac 0.05 --num_users 100 --local_ep 10 --local_bs 16 --optimizer adam --kernel_num 128 --kernel_sizes 3,4,5 --num_channels 3 --norm batch_norm --stopping_rounds 25 --gpu 0 --seed 42 > results/improved/cifar100_cnn.log 2>&1

echo "CIFAR-100实验完成！"
echo ""

# 5. MLP模型对比实验（基准测试）
echo "=== 5. MLP模型对比实验 ==="
echo "运行MNIST-MLP原始版本..."
python federated_main_original.py --dataset mnist --model mlp --epochs 80 --lr 0.01 --frac 0.1 --num_users 100 --local_ep 10 --local_bs 64 --optimizer sgd --gpu 0 --seed 42 > results/original/mnist_mlp.log 2>&1

echo "运行MNIST-MLP改进版本..."
python federated_main_improved.py --dataset mnist --model mlp --epochs 80 --lr 0.01 --frac 0.1 --num_users 100 --local_ep 10 --local_bs 64 --optimizer sgd --stopping_rounds 15 --gpu 0 --seed 42 > results/improved/mnist_mlp.log 2>&1

echo "MLP实验完成！"
echo ""

echo "所有实验完成！"
echo "结果保存在以下目录:"
echo "- 原始版本结果: results/original/"
echo "- 改进版本结果: results/improved/"
echo ""
echo "请查看日志文件分析实验结果，或运行 python analyze_results.py 进行自动分析。"
