#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6  # 指定Python解释器及编码格式（兼容中文注释）

# 导入依赖库
from tqdm import tqdm       # 进度条显示工具
import matplotlib.pyplot as plt  # 可视化绘图库
import torch
from torch.utils.data import DataLoader  # 数据批量加载工具

# 导入自定义模块
from utils import get_dataset    # 数据集加载工具函数
from options import args_parser # 命令行参数解析器
from update import test_inference  # 测试推理函数
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNCifar100, ResNet18Fed, replace_bn_with_gn  # 自定义模型定义

'''
该文件实现了一个传统的、非联邦的（中心化）机器学习模型训练流程。它通常用作联邦学习性能的基准 (baseline)。
代码会加载数据，构建模型，在整个训练集上进行训练，并在测试集上评估模型。
'''

if __name__ == '__main__':
    # 参数解析与设备配置
    args = args_parser()  # 解析命令行参数（如模型类型、数据集、epoch数等）
    if args.gpu:          # 若启用GPU加速
        torch.cuda.set_device(int(args.gpu))  # 指定GPU设备编号
        #加了int之后就不用像之前一样在命令行输入gpu=cuda:0了,只用写gpu=0
    device = 'cuda' if args.gpu else 'cpu'  # 确定计算设备（GPU/CPU）

    # 加载数据集
    train_dataset, test_dataset, _ = get_dataset(args)  # 获取训练集、测试集及可能的额外信息
    
    # 构建模型
    if args.model == 'cnn':  # 卷积神经网络模型分支
        # 根据数据集选择不同CNN结构
        if args.dataset == 'mnist':      # MNIST手写数字识别
            global_model = CNNMnist(args=args)  # 28x28灰度图输入的网络
        elif args.dataset == 'fmnist':   # Fashion-MNIST服装分类
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':    # CIFAR-10图像分类
            global_model = CNNCifar(args=args)  # 适用于32x32彩色图的CNN
        elif args.dataset == 'cifar100': # CIFAR-100图像分类
            global_model = CNNCifar100(args=args)  # 适用于32x32彩色图的深度CNN
    elif args.model == 'resnet':  # ResNet18Fed模型分支
        # ResNet for better performance
        if args.dataset == 'cifar':
            global_model = ResNet18Fed(num_classes=args.num_classes)
            global_model = replace_bn_with_gn(global_model)  # 使用GroupNorm
        elif args.dataset == 'cifar100':
            global_model = ResNet18Fed(num_classes=100)
            global_model = replace_bn_with_gn(global_model)
        else:
            print(f"ResNet not implemented for dataset {args.dataset}, using CNN instead")
            if args.dataset == 'mnist':
                global_model = CNNMnist(args=args)
            elif args.dataset == 'fmnist':
                global_model = CNNFashion_Mnist(args=args)
    elif args.model == 'mlp':  # 多层感知机模型分支[7](@ref)
        img_size = train_dataset[0][0].shape  # 获取输入图像尺寸
        len_in = 1
        for x in img_size:  # 计算输入层维度（展平后的像素总数）
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')  # 模型类型错误处理
    
    # 模型配置
    global_model.to(device)    # 将模型移动到指定设备（GPU/CPU）
    global_model.train()       # 设置为训练模式（启用BN/Dropout等层）
    print(global_model)        # 打印模型结构
    
    # 训练配置[8,6](@ref)
    # 优化器选择（SGD带动量或Adam）
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr, momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 数据加载器（批处理+随机打乱）
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # 验证集数据加载器 - 用于早停
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    trainloader = DataLoader(train_subset, batch_size=64, shuffle=True)
    valloader = DataLoader(val_subset, batch_size=64, shuffle=False)
    
    # 损失函数：负对数似然损失（需配合LogSoftmax输出层）
    criterion = torch.nn.NLLLoss().to(device)
    epoch_loss = []  # 记录每轮平均损失
    val_losses = []  # 记录验证损失
    
    # 早停机制参数
    best_val_loss = float('inf')
    patience = 5  # 连续多少个epoch验证损失不下降就停止
    patience_counter = 0

    # 训练循环[6,8](@ref)
    for epoch in tqdm(range(args.epochs)):  # 进度条显示epoch进度
        '''
        tqdm(...) 包裹这个序列后，每次循环时会在控制台显示当前进度条
        包括已完成的轮数、总轮数、预计剩余时间等信息
        '''
        batch_loss = []
        
        # 训练阶段
        global_model.train()
        # 遍历训练数据批次
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)  # 数据送设备
            
            # 前向传播
            optimizer.zero_grad()       # 清空梯度（避免累积）
            outputs = global_model(images)  # 模型推理
            loss = criterion(outputs, labels)  # 计算损失
            
            # 反向传播与优化
            loss.backward()             # 反向传播计算梯度
            optimizer.step()            # 更新模型参数
            
            # 每50个批次打印训练状态
            if batch_idx % 50 == 0:
                print(f'Train Epoch: {epoch+1} [{batch_idx * len(images)}/{len(trainloader.dataset)}'
                      f' ({100. * batch_idx / len(trainloader):.0f}%)]\tLoss: {loss.item():.6f}')
            batch_loss.append(loss.item())  # 记录当前批次损失
        
        # 计算并记录当前epoch平均训练损失
        loss_avg = sum(batch_loss) / len(batch_loss)
        print(f'\nTrain loss: {loss_avg}')
        epoch_loss.append(loss_avg)
        
        # 验证阶段
        global_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = global_model(images)
                val_loss += criterion(outputs, labels).item()
        
        val_loss /= len(valloader)
        val_losses.append(val_loss)
        print(f'Validation loss: {val_loss:.6f}')
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f'New best validation loss: {val_loss:.6f}')
        else:
            patience_counter += 1
            print(f'Validation loss did not improve. Patience: {patience_counter}/{patience}')
            
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break

    # 可视化训练损失曲线
    print(f'\nTraining completed after {len(epoch_loss)} epochs')
    print(f'Final training loss: {epoch_loss[-1]:.6f}')
    print(f'Final validation loss: {val_losses[-1]:.6f}')
    print(f'Best validation loss: {best_val_loss:.6f}')
    
    # 模型测试评估
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print(f'Test on {len(test_dataset)} samples')
    print(f"Test Accuracy: {100*test_acc:.2f}%")  # 输出测试准确率（百分比形式）

