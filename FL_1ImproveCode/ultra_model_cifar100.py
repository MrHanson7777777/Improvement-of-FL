#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
超强CIFAR-100模型：融合EfficientNet和CNNCifar100的优点
集成所有最先进的优化技术
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import transforms


class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力机制"""
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class StochasticDepth(nn.Module):
    """随机深度，提升泛化能力"""
    def __init__(self, drop_rate):
        super(StochasticDepth, self).__init__()
        self.drop_rate = drop_rate

    def forward(self, x, skip_connection):
        if not self.training or self.drop_rate == 0.:
            return x + skip_connection
        
        keep_prob = 1 - self.drop_rate
        random_tensor = keep_prob + torch.rand((x.shape[0], 1, 1, 1), 
                                             dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return (x / keep_prob) * binary_tensor + skip_connection


class EnhancedMBConvBlock(nn.Module):
    """增强的MBConv块，融合多种优化技术"""
    
    def __init__(self, in_channels, out_channels, stride, expand_ratio, 
                 kernel_size=3, se_ratio=0.25, drop_rate=0.0):
        super(EnhancedMBConvBlock, self).__init__()
        
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # 扩展阶段
        expanded_channels = in_channels * expand_ratio
        self.expand_conv = None
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True)
            )
        
        # 深度卷积
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size,
                     stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        )
        
        # SE注意力机制
        self.se_block = SEBlock(expanded_channels, reduction=int(1/se_ratio)) if se_ratio > 0 else None
        
        # 投影阶段
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 随机深度
        self.stochastic_depth = StochasticDepth(drop_rate) if drop_rate > 0 else None
        
    def forward(self, x):
        identity = x
        
        # 扩展
        if self.expand_conv is not None:
            x = self.expand_conv(x)
        
        # 深度卷积
        x = self.depthwise_conv(x)
        
        # SE注意力
        if self.se_block is not None:
            x = self.se_block(x)
        
        # 投影
        x = self.project_conv(x)
        
        # 残差连接（带随机深度）
        if self.use_residual:
            if self.stochastic_depth is not None:
                x = self.stochastic_depth(x, identity)
            else:
                x = x + identity
                
        return x


class UltraEfficientNetCIFAR100(nn.Module):
    """超强CIFAR-100模型，融合所有优化技术"""
    
    def __init__(self, num_classes=100, dropout_rate=0.4, stochastic_depth_rate=0.2):
        super(UltraEfficientNetCIFAR100, self).__init__()
        
        self.dropout_rate = dropout_rate
        
        # 更强的Stem层
        self.stem = nn.Sequential(
            nn.Conv2d(3, 40, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.SiLU(inplace=True)
        )
        
        # 深层架构配置 (参考您的CNNCifar100)
        # [expand_ratio, channels, num_blocks, stride, kernel_size, se_ratio]
        settings = [
            [1,  24,  1, 1, 3, 0.25],  # stage 1: 32x32 -> 32x32
            [6,  32,  2, 2, 3, 0.25],  # stage 2: 32x32 -> 16x16
            [6,  48,  3, 2, 5, 0.25],  # stage 3: 16x16 -> 8x8
            [6,  96,  4, 2, 3, 0.25],  # stage 4: 8x8 -> 4x4
            [6, 136,  4, 1, 5, 0.25],  # stage 5: 4x4 -> 4x4
            [6, 232,  5, 2, 5, 0.25],  # stage 6: 4x4 -> 2x2
            [6, 384,  2, 1, 3, 0.25],  # stage 7: 2x2 -> 2x2
        ]
        
        # 构建网络层
        self.stages = nn.ModuleList()
        in_channels = 40
        total_blocks = sum([blocks for _, _, blocks, _, _, _ in settings])
        block_id = 0
        
        for expand_ratio, channels, num_blocks, stride, kernel_size, se_ratio in settings:
            stage = nn.ModuleList()
            
            for i in range(num_blocks):
                block_stride = stride if i == 0 else 1
                # 计算随机深度率
                drop_rate = stochastic_depth_rate * block_id / total_blocks
                
                stage.append(EnhancedMBConvBlock(
                    in_channels, channels, block_stride, expand_ratio,
                    kernel_size, se_ratio, drop_rate
                ))
                in_channels = channels
                block_id += 1
                
            self.stages.append(stage)
        
        # 更强的头部
        self.head_conv = nn.Conv2d(384, 1536, kernel_size=1, bias=False)
        self.head_bn = nn.BatchNorm2d(1536)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # 多层分类器（参考您的设计）
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(1536, 768),
            nn.BatchNorm1d(768),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(384, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # 所有stage
        for stage in self.stages:
            for block in stage:
                x = block(x)
        
        # Head
        x = F.silu(self.head_bn(self.head_conv(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 分类器
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)


def get_ultra_cifar100_model(dropout_rate=0.4):
    """获取超强CIFAR-100模型"""
    return UltraEfficientNetCIFAR100(
        num_classes=100,
        dropout_rate=dropout_rate,
        stochastic_depth_rate=0.2
    )


# 改进的数据增强策略
def get_ultra_transforms(is_training=True):
    """超强数据增强策略"""
    if is_training:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            ),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])


# 改进的损失函数
class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失"""
    def __init__(self, epsilon=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.epsilon / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.epsilon))
        loss = (-weight * log_prob).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


if __name__ == "__main__":
    # 测试模型
    model = get_ultra_cifar100_model()
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    print(f"Model output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
