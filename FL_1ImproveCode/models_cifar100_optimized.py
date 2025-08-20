#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
专门针对CIFAR-100优化的EfficientNet模型
包含完整的数据增强、正则化和模型优化策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CIFAR100EfficientNet(nn.Module):
    """专门为CIFAR-100优化的EfficientNet"""
    
    def __init__(self, num_classes=100, width_mult=1.0, depth_mult=1.0, dropout_rate=0.4):
        super(CIFAR100EfficientNet, self).__init__()
        
        self.dropout_rate = dropout_rate
        
        # 适配CIFAR-100的stem层
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # EfficientNet-B0的配置，适配CIFAR-100
        # [expand_ratio, channels, num_blocks, stride, kernel_size]
        settings = [
            [1, 16, 1, 1, 3],   # stage 1
            [6, 24, 2, 2, 3],   # stage 2
            [6, 40, 2, 2, 5],   # stage 3
            [6, 80, 3, 2, 3],   # stage 4
            [6, 112, 3, 1, 5],  # stage 5
            [6, 192, 4, 2, 5],  # stage 6
            [6, 320, 1, 1, 3],  # stage 7
        ]
        
        # 构建MBConv块
        self.blocks = nn.ModuleList()
        in_channels = 32
        
        for expand_ratio, channels, num_blocks, stride, kernel_size in settings:
            out_channels = int(channels * width_mult)
            num_blocks = int(math.ceil(num_blocks * depth_mult))
            
            for i in range(num_blocks):
                block_stride = stride if i == 0 else 1
                self.blocks.append(
                    MBConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        expand_ratio=expand_ratio,
                        stride=block_stride,
                        kernel_size=kernel_size,
                        se_ratio=0.25,
                        drop_path_rate=0.2 * len(self.blocks) / sum([int(math.ceil(n * depth_mult)) for _, _, n, _, _ in settings])
                    )
                )
                in_channels = out_channels
        
        # Head层，专门为CIFAR-100优化
        self.head_conv = nn.Conv2d(in_channels, 1280, kernel_size=1, bias=False)
        self.head_bn = nn.BatchNorm2d(1280)
        self.head_act = nn.SiLU(inplace=True)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # 分类器，增强正则化
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(1280, 512),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier/Kaiming初始化，针对SiLU激活优化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.head_act(self.head_bn(self.head_conv(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block with SE and DropPath"""
    
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size, se_ratio=0.25, drop_path_rate=0.0):
        super(MBConvBlock, self).__init__()
        
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        self.drop_path_rate = drop_path_rate
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True)
            )
        else:
            self.expand_conv = nn.Identity()
            expanded_channels = in_channels
        
        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel_size, 
                     stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        )
        
        # Squeeze-and-Excitation
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded_channels, se_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_channels, expanded_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Projection phase
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # DropPath for regularization
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
    
    def forward(self, x):
        identity = x
        
        # Expansion
        x = self.expand_conv(x)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        
        # Squeeze-and-Excitation
        se_weight = self.se(x)
        x = x * se_weight
        
        # Projection
        x = self.project_conv(x)
        
        # Residual connection with DropPath
        if self.use_residual:
            x = identity + self.drop_path(x)
        
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class CIFAR100EfficientNetB0(CIFAR100EfficientNet):
    """EfficientNet-B0 for CIFAR-100"""
    def __init__(self, num_classes=100):
        super().__init__(num_classes=num_classes, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2)


class CIFAR100EfficientNetB1(CIFAR100EfficientNet):
    """EfficientNet-B1 for CIFAR-100 with more capacity"""
    def __init__(self, num_classes=100):
        super().__init__(num_classes=num_classes, width_mult=1.1, depth_mult=1.1, dropout_rate=0.3)


class CIFAR100EfficientNetB2(CIFAR100EfficientNet):
    """EfficientNet-B2 for CIFAR-100 with even more capacity"""
    def __init__(self, num_classes=100):
        super().__init__(num_classes=num_classes, width_mult=1.2, depth_mult=1.2, dropout_rate=0.4)


def get_cifar100_efficientnet(variant='b0', dropout_rate=None):
    """获取CIFAR-100专用的EfficientNet模型"""
    if variant == 'b0':
        model = CIFAR100EfficientNetB0()
    elif variant == 'b1':
        model = CIFAR100EfficientNetB1()
    elif variant == 'b2':
        model = CIFAR100EfficientNetB2()
    else:
        raise ValueError(f"Unsupported variant: {variant}")
    
    # 如果指定了dropout_rate，更新模型的dropout率
    if dropout_rate is not None:
        model.dropout_rate = dropout_rate
    
    return model


# 专门的CIFAR-100数据增强
import torchvision.transforms as transforms

def get_cifar100_transforms(is_training=True):
    """获取CIFAR-100的数据变换"""
    if is_training:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])


# 标签平滑损失函数
class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss for better generalization"""
    
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        
        return loss * self.epsilon / c + (1 - self.epsilon) * F.nll_loss(log_preds, target, reduction=self.reduction)


# 混合精度训练辅助函数
def enable_mixed_precision():
    """启用混合精度训练以节省内存并加速训练"""
    return torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None


# Cosine Annealing with Warm Restarts
class CosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with warm restarts for better convergence"""
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.T_cur = epoch
        if epoch >= self.T_i:
            self.T_cur = epoch - self.T_i
            self.T_i *= self.T_mult
        # 更新 last_epoch 并触发学习率更新
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
