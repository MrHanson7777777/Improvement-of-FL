#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轻量级Ultra CIFAR-100模型：优化版本，减少复杂度，提升训练速度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightSEBlock(nn.Module):
    """轻量级SE注意力机制"""
    def __init__(self, in_channels, reduction=16):
        super(LightweightSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, max(in_channels // reduction, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(in_channels // reduction, 8), in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class LightweightMBConvBlock(nn.Module):
    """轻量级MBConv块"""
    
    def __init__(self, in_channels, out_channels, stride, expand_ratio, 
                 kernel_size=3, se_ratio=0.25):
        super(LightweightMBConvBlock, self).__init__()
        
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # 扩展阶段
        expanded_channels = in_channels * expand_ratio
        self.expand_conv = None
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.ReLU6(inplace=True)  # 使用ReLU6替代SiLU，速度更快
            )
        
        # 深度卷积
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size,
                     stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True)
        )
        
        # 轻量级SE注意力
        self.se_block = LightweightSEBlock(expanded_channels, reduction=16) if se_ratio > 0 else None
        
        # 投影阶段
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
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
        
        # 残差连接
        if self.use_residual:
            x = x + identity
                
        return x


class LightweightUltraCIFAR100(nn.Module):
    """轻量级Ultra CIFAR-100模型，优化训练速度"""
    
    def __init__(self, num_classes=100, dropout_rate=0.3):
        super(LightweightUltraCIFAR100, self).__init__()
        
        self.dropout_rate = dropout_rate
        
        # 简化的Stem层
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        # 简化的架构配置 - 减少块数量
        # [expand_ratio, channels, num_blocks, stride, kernel_size, se_ratio]
        settings = [
            [1,  16,  1, 1, 3, 0.25],  # stage 1: 32x32 -> 32x32
            [6,  24,  2, 2, 3, 0.25],  # stage 2: 32x32 -> 16x16
            [6,  40,  2, 2, 5, 0.25],  # stage 3: 16x16 -> 8x8
            [6,  80,  3, 2, 3, 0.25],  # stage 4: 8x8 -> 4x4
            [6, 112,  2, 1, 5, 0.25],  # stage 5: 4x4 -> 4x4
            [6, 192,  2, 2, 5, 0.25],  # stage 6: 4x4 -> 2x2
        ]
        
        # 构建网络层
        self.stages = nn.ModuleList()
        in_channels = 32
        
        for expand_ratio, channels, num_blocks, stride, kernel_size, se_ratio in settings:
            stage = nn.ModuleList()
            
            for i in range(num_blocks):
                block_stride = stride if i == 0 else 1
                
                stage.append(LightweightMBConvBlock(
                    in_channels, channels, block_stride, expand_ratio,
                    kernel_size, se_ratio
                ))
                in_channels = channels
                
            self.stages.append(stage)
        
        # 简化的头部
        self.head_conv = nn.Conv2d(192, 1280, kernel_size=1, bias=False)
        self.head_bn = nn.BatchNorm2d(1280)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # 简化的分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(1280, 512),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
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
        x = F.relu6(self.head_bn(self.head_conv(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 分类器
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)


def get_lightweight_ultra_cifar100_model(dropout_rate=0.3):
    """获取轻量级Ultra CIFAR-100模型"""
    return LightweightUltraCIFAR100(
        num_classes=100,
        dropout_rate=dropout_rate
    )


if __name__ == "__main__":
    # 测试模型
    model = get_lightweight_ultra_cifar100_model()
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    print(f"Model output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
