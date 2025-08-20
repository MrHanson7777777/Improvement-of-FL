#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn # 导入 PyTorch 神经网络模块
from torch.utils.data import DataLoader, Dataset # 导入数据加载工具

class DatasetSplit(Dataset): # 自定义数据集类，用于从大数据集中抽取一部分作为某个客户端的数据
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs): # 构造函数
        self.dataset = dataset # 原始的完整数据集
        self.idxs = [int(i) for i in idxs] # 分配给该客户端的数据样本的索引列表

    def __len__(self): # 返回该客户端拥有的数据样本数量
        return len(self.idxs)

    def __getitem__(self, item): # 根据索引获取单个数据样本
        image, label = self.dataset[self.idxs[item]] # 从原始数据集中获取图像和标签
        # 返回张量形式的图像和标签 (尽管原始 dataset 通常已返回张量，这里再次转换确保类型)
        # 注意：如果 self.dataset[self.idxs[item]] 返回的已经是 tensor，
        # 再次调用 torch.tensor() 可能会创建副本或发出警告，取决于 PyTorch 版本和数据类型。
        # 通常，如果原始 dataset 的 __getitem__ 返回的是 tensor，这里可以直接返回 image, label。
        return torch.tensor(image), torch.tensor(label)

class LocalUpdate(object): # 定义客户端本地更新过程的类
    def __init__(self, args, dataset, idxs, logger): # 构造函数
        self.args = args # 命令行参数
        self.logger = logger # 用于记录日志 (例如 TensorBoard)
        # 调用 train_val_test 方法，将分配给该客户端的数据 (由 idxs 指定) 划分为本地训练集、验证集和测试集
        # 并创建对应的 DataLoader
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        # 健壮的GPU设备检查
        try:
            gpu_id = int(args.gpu) if args.gpu is not None else -1
            self.device = 'cuda' if gpu_id >= 0 else 'cpu'
        except (ValueError, TypeError):
            self.device = 'cpu'
        # Default criterion set to NLL loss function (默认损失函数为负对数似然损失)
        self.criterion = nn.NLLLoss().to(self.device) # 将损失函数也移动到相应的设备

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        为给定的数据集和用户索引返回训练、验证和测试的 DataLoader。
        """
        # split indexes for train, validation, and test (80, 10, 10)
        # 将客户端数据按 80% 训练，10% 验证，10% 测试的比例划分
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        # 使用 DatasetSplit 和 DataLoader 创建对应的数据加载器
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True) # 本地训练批量大小，打乱数据
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False) # 验证集批量大小，不打乱
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False) # 测试集批量大小，不打乱
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round): # 执行本地模型更新 (训练)
        # 确保模型在正确的设备上
        model = model.to(self.device)
        # Set mode to train model
        model.train() # 将模型设置为训练模式 (这会启用 Dropout, BatchNorm 等层的训练行为)
        epoch_loss = [] # 记录每个本地 epoch 的平均损失

        # Set optimizer for the local updates (根据参数选择优化器)
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum) # 使用 SGD 优化器
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4) # 使用 Adam 优化器

        for iter_epoch in range(self.args.local_ep): # 进行指定次数的本地 epoch 训练
            batch_loss = [] # 记录当前 epoch 内每个 batch 的损失
            for batch_idx, (images, labels) in enumerate(self.trainloader): # 遍历本地训练数据加载器
                images, labels = images.to(self.device), labels.to(self.device) # 将数据移到设备 (CPU/GPU)

                model.zero_grad() # 清空之前的梯度
                log_probs = model(images) # 模型前向传播，得到 log_softmax 输出
                loss = self.criterion(log_probs, labels) # 计算损失
                loss.backward() # 反向传播，计算梯度
                optimizer.step() # 更新模型参数

                if self.args.verbose and (batch_idx % 10 == 0): # 如果 verbose 为 True 且每 10 个 batch
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter_epoch, batch_idx * len(images), # 当前处理的图片数量
                        len(self.trainloader.dataset), # 训练集总图片数
                        100. * batch_idx / len(self.trainloader), loss.item())) # 打印训练信息
                self.logger.add_scalar('loss', loss.item()) # 使用 logger 记录损失值
                batch_loss.append(loss.item()) # 记录当前 batch 的损失
            epoch_loss.append(sum(batch_loss)/len(batch_loss)) # 计算当前 epoch 的平均损失

        # 返回更新后的模型权重 (state_dict) 和所有本地 epoch 的平均损失
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model): # 在本地测试集上进行模型推断 (评估)
        """ Returns the inference accuracy and loss.
        """
        '''
        本地评估
        '''
        # 确保模型在正确的设备上
        model = model.to(self.device)
        model.eval() # 将模型设置为评估模式 (这会禁用 Dropout, BatchNorm 等层的训练行为)
        loss, total, correct = 0.0, 0.0, 0.0 # 初始化损失、总样本数、正确预测数

        for batch_idx, (images, labels) in enumerate(self.testloader): # 遍历本地测试数据加载器
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images) # 模型前向传播
            batch_loss = self.criterion(outputs, labels) # 计算当前 batch 的损失
            loss += batch_loss.item() # 累加损失

            # Prediction
            _, pred_labels = torch.max(outputs, 1) # 获取预测概率最高的类别索引
            pred_labels = pred_labels.view(-1) # 展平预测标签
            correct += torch.sum(torch.eq(pred_labels, labels)).item() # 计算正确预测的数量
            total += len(labels) # 累加总样本数

        accuracy = correct/total # 计算准确率
        return accuracy, loss # 返回准确率和总损失


def test_inference(args, model, test_dataset): # 在全局测试集上进行模型推断 (评估全局模型性能)
    """ Returns the test accuracy and loss.
    """
    '''
    server评估
    '''
    # 健壮的GPU设备检查
    try:
        gpu_id = int(args.gpu) if args.gpu is not None else -1
        device = 'cuda' if gpu_id >= 0 else 'cpu'
    except (ValueError, TypeError):
        device = 'cpu'
    # 确保模型在正确的设备上
    model = model.to(device)
    model.eval() # 设置为评估模式
    loss, total, correct = 0.0, 0.0, 0.0

    criterion = nn.NLLLoss().to(device)
    # 使用完整的测试数据集创建 DataLoader
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    with torch.no_grad(): # 在评估时，不需要计算梯度，可以节省内存和计算
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

    accuracy = correct/total
    return accuracy, loss # 返回在整个测试集上的准确率和总损失
