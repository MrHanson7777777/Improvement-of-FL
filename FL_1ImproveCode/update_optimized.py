#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化版本的本地更新模块
包含学习率调度、数据增强、正则化等优化策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import copy


class CutMixLoss:
    """CutMix损失函数"""
    def __init__(self, criterion):
        self.criterion = criterion
    
    def __call__(self, output, y_a, y_b, lam):
        return lam * self.criterion(output, y_a) + (1 - lam) * self.criterion(output, y_b)


class MixUpLoss:
    """MixUp损失函数"""
    def __init__(self, criterion):
        self.criterion = criterion
    
    def __call__(self, output, y_a, y_b, lam):
        return lam * self.criterion(output, y_a) + (1 - lam) * self.criterion(output, y_b)


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失函数"""
    def __init__(self, classes, smoothing=0.1, dim=1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class EnhancedLocalUpdate(object):
    """优化版本的本地更新类，包含学习率调度和数据增强"""
    
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu is not None and args.gpu >= 0 else 'cpu'
        
        # 损失函数设置
        if hasattr(args, 'label_smoothing') and args.label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(args.num_classes, args.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # CutMix和MixUp设置
        self.cutmix_loss = CutMixLoss(self.criterion)
        self.mixup_loss = MixUpLoss(self.criterion)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def cutmix(self, x, y, alpha=1.0):
        """CutMix数据增强"""
        if alpha <= 0:
            return x, y, y, 1.0
        
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        y_a, y_b = y, y[index]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        return x, y_a, y_b, lam

    def mixup(self, x, y, alpha=0.2):
        """MixUp数据增强"""
        if alpha <= 0:
            return x, y, y, 1.0
        
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def rand_bbox(self, size, lam):
        """随机生成边界框for CutMix"""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

    def get_scheduler(self, optimizer, steps_per_epoch):
        """获取学习率调度器"""
        if hasattr(self.args, 'scheduler'):
            if self.args.scheduler == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.args.local_ep * steps_per_epoch, 
                    eta_min=getattr(self.args, 'min_lr', self.args.lr * 0.01)
                )
            elif self.args.scheduler == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=getattr(self.args, 'lr_decay_step', self.args.local_ep // 2), 
                    gamma=getattr(self.args, 'lr_decay', 0.1)
                )
            elif self.args.scheduler == 'multistep':
                milestones = [self.args.local_ep // 3, 2 * self.args.local_ep // 3]
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=milestones, 
                    gamma=getattr(self.args, 'lr_decay', 0.1)
                )
            else:
                scheduler = None
        else:
            scheduler = None
        
        return scheduler

    def update_weights(self, model, global_round):
        """优化版本的权重更新，包含学习率调度和数据增强"""
        # 确保模型在正确的设备上
        model = model.to(self.device)
        print(f"🔧 本地更新设备: {self.device}, 模型设备: {next(model.parameters()).device}")
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum, 
                                        weight_decay=getattr(self.args, 'weight_decay', 0))
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=getattr(self.args, 'weight_decay', 0))

        # 获取学习率调度器
        steps_per_epoch = len(self.trainloader)
        scheduler = self.get_scheduler(optimizer, steps_per_epoch)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # 应用数据增强
                use_cutmix = (hasattr(self.args, 'use_cutmix') and self.args.use_cutmix and 
                             self.args.dataset in ['cifar', 'cifar100'] and random.random() < 0.5)
                use_mixup = (hasattr(self.args, 'use_mixup') and self.args.use_mixup and 
                            self.args.dataset in ['cifar', 'cifar100'] and random.random() < 0.5 and not use_cutmix)

                model.zero_grad()
                
                if use_cutmix:
                    # 应用CutMix
                    cutmix_alpha = getattr(self.args, 'cutmix_alpha', 1.0)
                    images, y_a, y_b, lam = self.cutmix(images, labels, cutmix_alpha)
                    log_probs = model(images)
                    loss = self.cutmix_loss(log_probs, y_a, y_b, lam)
                elif use_mixup:
                    # 应用MixUp
                    mixup_alpha = getattr(self.args, 'mixup_alpha', 0.2)
                    images, y_a, y_b, lam = self.mixup(images, labels, mixup_alpha)
                    log_probs = model(images)
                    loss = self.mixup_loss(log_probs, y_a, y_b, lam)
                else:
                    # 正常训练
                    log_probs = model(images)
                    loss = self.criterion(log_probs, labels)

                loss.backward()
                
                # 梯度裁剪
                if hasattr(self.args, 'gradient_clip') and self.args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.gradient_clip)
                
                optimizer.step()
                
                # 更新学习率
                if scheduler is not None:
                    scheduler.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        # 确保模型在正确的设备上
        model = model.to(self.device)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu is not None and args.gpu >= 0 else 'cpu'
    
    # 确保模型在正确的设备上
    model = model.to(device)
    print(f"🧪 测试设备: {device}, 模型设备: {next(model.parameters()).device}")
    
    # 损失函数
    if hasattr(args, 'label_smoothing') and args.label_smoothing > 0:
        criterion = LabelSmoothingLoss(args.num_classes, args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

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
    return accuracy, loss


# 为了向后兼容，保留原始的LocalUpdate类
class LocalUpdate(EnhancedLocalUpdate):
    """向后兼容的LocalUpdate类"""
    pass
