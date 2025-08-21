#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CIFAR-100数据集专用深度学习模型库
===============================

本文件包含针对CIFAR-100自然图像分类数据集（32x32像素，3通道彩色图像，100个类别）
优化设计的三种主流深度学习模型架构：

1. ResNet18联邦学习版本 - 基于残差连接的深度网络，使用GroupNorm适配联邦学习
2. EfficientNet-B3风格模型 - 基于移动端倒置残差块的高效深度网络
3. DenseNet模型 - 基于密集连接的特征复用网络架构

所有模型都针对CIFAR-100数据集的特点进行了优化，考虑到100个类别
需要更强的特征提取能力和更复杂的分类器。

数据集特点：
- 图像尺寸: 32x32
- 通道数: 3 (RGB彩色图像)
- 类别数: 100 (分为20个超类，每个超类5个细粒度类别)
- 训练样本: 50,000 (每类500张)
- 测试样本: 10,000 (每类100张)
- 挑战: 类别多，每类样本少，类间相似性高

模型特色：
- ResNet18: 残差连接 + 注意力机制 + 联邦学习友好
- EfficientNet: 移动端高效 + 深度可分离卷积 + 复合缩放
- DenseNet: 密集连接 + 特征复用 + 参数效率高
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BasicBlock_CIFAR100(nn.Module):
    """
    CIFAR-100专用ResNet基础块
    =========================
    
    针对100分类任务优化的ResNet基础残差块。
    使用更强的特征提取能力，适应更细粒度的分类需求。
    """
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.1):
        super(BasicBlock_CIFAR100, self).__init__()
        # 第一个3x3卷积层，可能改变空间分辨率
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 第二个3x3卷积层，保持空间分辨率不变
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # 添加Dropout提高泛化能力，防止过拟合
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # 残差连接的快捷路径，当输入输出维度不匹配时需要投影
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            # 使用1x1卷积进行维度匹配
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        # 第一个卷积 -> BN -> ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # 应用Dropout正则化
        out = self.dropout(out)
        # 第二个卷积 -> BN（不加激活函数）
        out = self.bn2(self.conv2(out))
        # 残差连接：输出 = F(x) + x
        out += self.shortcut(x)
        # 最终激活
        out = F.relu(out)
        return out


class SEBlock(nn.Module):
    """
    挤压激励注意力模块 (Squeeze-and-Excitation Block)
    ================================================
    
    通过学习通道间的依赖关系来重新校准特征图。
    对于CIFAR-100这样的细粒度分类任务特别有效。
    
    工作原理：
    1. Squeeze: 全局平均池化压缩空间维度
    2. Excitation: 两个全连接层学习通道重要性
    3. Scale: 用学到的权重重新标定原特征图
    """
    
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        # 全局平均池化，将H×W压缩为1×1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 两层全连接网络学习通道重要性
        self.fc = nn.Sequential(
            # 降维层，减少参数量
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            # 升维层，恢复到原通道数
            nn.Linear(channels // reduction, channels, bias=False),
            # Sigmoid确保输出在[0,1]范围
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: 全局平均池化 [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        y = self.avg_pool(x).view(b, c)
        # Excitation: 学习通道重要性 [B, C] -> [B, C]
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: 重新标定特征图 [B, C, H, W] * [B, C, 1, 1]
        return x * y.expand_as(x)


class ResNet18_CIFAR100_Fed(nn.Module):
    """
    CIFAR-100专用联邦学习ResNet18模型
    =================================
    
    专门为CIFAR-100设计的ResNet18变体，支持100个类别的分类。
    使用GroupNorm替代BatchNorm，增加注意力机制，提高细粒度分类能力。
    
    网络架构：
    - 输入：32x32x3的RGB图像
    - 4个残差层：[2, 2, 2, 2]个基础块
    - 挤压激励注意力模块
    - 输出：100个类别的概率分布
    
    针对CIFAR-100优化：
    1. 更宽的网络通道数
    2. 注意力机制增强特征表示
    3. 更复杂的分类器
    4. 适当的正则化防止过拟合
    """
    
    def __init__(self, num_classes=100, use_groupnorm=True, num_groups=8, 
                 use_se=True, dropout_rate=0.3):
        super(ResNet18_CIFAR100_Fed, self).__init__()
        self.in_planes = 64  # 当前层的输入通道数
        self.use_groupnorm = use_groupnorm
        self.num_groups = num_groups
        self.use_se = use_se
        
        # 初始卷积层：3通道RGB -> 64通道特征图
        # 使用较小的kernel避免CIFAR-100小图像信息丢失
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 4个残差层（比CIFAR-10使用更多通道适应100类）
        # layer1: 64通道，空间尺寸32x32
        self.layer1 = self._make_layer(64, 2, stride=1, dropout_rate=0.1)
        # layer2: 128通道，空间尺寸16x16
        self.layer2 = self._make_layer(128, 2, stride=2, dropout_rate=0.1)
        # layer3: 256通道，空间尺寸8x8
        self.layer3 = self._make_layer(256, 2, stride=2, dropout_rate=0.2)
        # layer4: 512通道，空间尺寸4x4
        self.layer4 = self._make_layer(512, 2, stride=2, dropout_rate=0.2)
        
        # 注意力模块，增强特征表示能力
        if use_se:
            self.se = SEBlock(512)
        
        # 全局平均池化，将4x4特征图压缩为1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器 - 为100类设计的更复杂分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            # 第一层：512 -> 256，增加非线性表达能力
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            # 第二层：256 -> 100类输出
            nn.Linear(256, num_classes)
        )
        
        # 如果使用GroupNorm，则替换所有BatchNorm（联邦学习友好）
        if use_groupnorm:
            self._replace_bn_with_gn()
        
        # 权重初始化
        self._initialize_weights()

    def _make_layer(self, planes, blocks, stride, dropout_rate=0.1):
        """
        构建残差层
        
        Args:
            planes: 输出通道数
            blocks: 残差块数量
            stride: 第一个块的步长
            dropout_rate: Dropout比率
        """
        strides = [stride] + [1] * (blocks - 1)  # 只有第一个块可能改变空间尺寸
        layers = []
        for s in strides:
            layers.append(BasicBlock_CIFAR100(self.in_planes, planes, s, dropout_rate))
            self.in_planes = planes  # 更新输入通道数
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始特征提取：[B, 3, 32, 32] -> [B, 64, 32, 32]
        out = F.relu(self.bn1(self.conv1(x)))
        
        # 4个残差阶段，逐步提取更高级特征
        out = self.layer1(out)  # [B, 64, 32, 32]
        out = self.layer2(out)  # [B, 128, 16, 16]
        out = self.layer3(out)  # [B, 256, 8, 8]
        out = self.layer4(out)  # [B, 512, 4, 4]
        
        # 应用注意力机制增强特征表示
        if self.use_se:
            out = self.se(out)  # [B, 512, 4, 4] -> [B, 512, 4, 4]
        
        # 全局平均池化：[B, 512, 4, 4] -> [B, 512, 1, 1]
        out = self.avgpool(out)
        # 展平：[B, 512, 1, 1] -> [B, 512]
        out = torch.flatten(out, 1)
        # 分类器：[B, 512] -> [B, 100]
        out = self.classifier(out)
        # 返回log概率分布
        return F.log_softmax(out, dim=1)
    
    def _replace_bn_with_gn(self):
        """
        将BatchNorm替换为GroupNorm
        GroupNorm在联邦学习中更稳定，不依赖batch统计
        """
        def replace_bn(module):
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    num_channels = child.num_features
                    # 确保groups数不超过通道数
                    groups = min(self.num_groups, num_channels)
                    gn = nn.GroupNorm(groups, num_channels)
                    setattr(module, name, gn)
                else:
                    # 递归处理子模块
                    replace_bn(child)
        
        replace_bn(self)
    
    def _initialize_weights(self):
        """
        权重初始化，使用He初始化等标准方法
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层使用Kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # 归一化层权重初始化为1，偏置为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层使用正态分布初始化
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class MBConvBlock_CIFAR100(nn.Module):
    """
    CIFAR-100专用增强MBConv块 (Mobile Inverted Bottleneck Convolution)
    ================================================================
    
    针对100分类任务增强的MobileNet倒置残差块。
    结合了深度可分离卷积和倒置残差连接的高效设计。
    
    工作流程：
    1. Expansion: 1x1卷积扩展通道数
    2. Depthwise: 3x3深度可分离卷积提取空间特征
    3. SE: 挤压激励注意力机制
    4. Projection: 1x1卷积压缩通道数
    5. Residual: 残差连接（如果适用）
    """
    
    def __init__(self, in_channels, out_channels, stride, expand_ratio, 
                 se_ratio=0.25, dropout_rate=0.1):
        super(MBConvBlock_CIFAR100, self).__init__()
        self.stride = stride
        # 只有当步长为1且输入输出通道相同时才使用残差连接
        self.use_residual = stride == 1 and in_channels == out_channels
        self.dropout_rate = dropout_rate
        
        # 扩展阶段：增加通道数以提高表达能力
        if expand_ratio != 1:
            expanded_channels = in_channels * expand_ratio
            self.expand_conv = nn.Sequential(
                # 1x1卷积扩展通道
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.ReLU(inplace=True)
            )
            self.has_expansion = True
        else:
            # 扩展比为1时不需要扩展
            expanded_channels = in_channels
            self.expand_conv = nn.Identity()
            self.has_expansion = False
            
        self.expanded_channels = expanded_channels
            
        # 深度可分离卷积：每个通道独立进行空间卷积
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, 3, stride, 1, 
                     groups=expanded_channels, bias=False),  # groups=输入通道数实现深度可分离
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU(inplace=True)
        )
        
        # 增强的挤压激励注意力机制
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # 全局平均池化
                nn.Conv2d(expanded_channels, se_channels, 1),  # 降维
                nn.ReLU(inplace=True),
                nn.Conv2d(se_channels, expanded_channels, 1),  # 升维
                nn.Sigmoid()  # 输出[0,1]权重
            )
        else:
            self.se = nn.Identity()
            
        # 点卷积投影：压缩通道数到目标维度
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Dropout正则化防止过拟合
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(dropout_rate)
        else:
            self.dropout = nn.Identity()
        
    def forward(self, x):
        identity = x  # 保存输入用于残差连接
        
        # 1. 扩展阶段
        if self.has_expansion:
            x = self.expand_conv(x)
        
        # 2. 深度可分离卷积
        x = self.depthwise_conv(x)
        
        # 3. 挤压激励注意力
        if hasattr(self.se, 'weight') or len(list(self.se.modules())) > 1:
            se_weight = self.se(x)
            x = x * se_weight
            
        # 4. 投影到输出维度
        x = self.project_conv(x)
        
        # 5. Dropout正则化
        x = self.dropout(x)
        
        # 6. 残差连接（如果适用）
        if self.use_residual:
            x = x + identity
            
        return x


class EfficientNet_CIFAR100(nn.Module):
    """
    CIFAR-100专用EfficientNet-B3风格模型
    ===================================
    
    基于EfficientNet-B3的深度网络，专门为CIFAR-100的100分类任务设计。
    使用复合缩放方法平衡网络深度、宽度和分辨率。
    
    网络特点：
    - 深度可分离卷积：减少参数量和计算量
    - 倒置残差结构：提高特征表达能力
    - 挤压激励注意力：增强重要特征
    - 复合缩放：平衡各维度的扩展
    
    适用场景：
    - 高精度CIFAR-100分类
    - 资源相对充足的场景
    - 需要高效推理的任务
    """
    
    def __init__(self, num_classes=100, dropout_rate=0.3):
        super(EfficientNet_CIFAR100, self).__init__()
        
        # Stem层：初始特征提取
        # 将3通道RGB图像转换为40通道特征图
        self.stem = nn.Sequential(
            nn.Conv2d(3, 40, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True)
        )
        
        # 7个MBConv阶段，逐步提取更高级特征
        # 每个阶段使用不同的通道数和重复次数
        self.stage1 = self._make_stage(40, 24, 1, 1, expand_ratio=1)     # 32x32，基础特征
        self.stage2 = self._make_stage(24, 32, 2, 2, expand_ratio=6)     # 16x16，浅层特征
        self.stage3 = self._make_stage(32, 48, 3, 2, expand_ratio=6)     # 8x8，中层特征  
        self.stage4 = self._make_stage(48, 96, 4, 2, expand_ratio=6)     # 4x4，深层特征
        self.stage5 = self._make_stage(96, 136, 4, 1, expand_ratio=6)    # 4x4，更深特征
        self.stage6 = self._make_stage(136, 232, 5, 2, expand_ratio=6)   # 2x2，高级特征
        self.stage7 = self._make_stage(232, 384, 2, 1, expand_ratio=6)   # 2x2，最高级特征
        
        # Head层：最终特征转换
        # 将384通道扩展到1536通道以增强表达能力
        self.head_conv = nn.Conv2d(384, 1536, kernel_size=1, bias=False)
        self.head_bn = nn.BatchNorm2d(1536)
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        
        # 更复杂的分类器适应100个类别
        # 使用多层结构逐步降维到类别数
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(1536, 512),  # 第一层降维
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, 256),   # 第二层降维
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 4),
            nn.Linear(256, num_classes)  # 输出100个类别
        )
        
        # 权重初始化
        self._initialize_weights()
        
    def _make_stage(self, in_channels, out_channels, num_blocks, stride, expand_ratio):
        """
        构建MBConv阶段
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数  
            num_blocks: 该阶段的MBConv块数量
            stride: 第一个块的步长
            expand_ratio: 通道扩展比例
        """
        layers = []
        # 第一个块可能改变空间分辨率
        layers.append(MBConvBlock_CIFAR100(in_channels, out_channels, stride, expand_ratio))
        # 后续块保持空间分辨率不变
        for _ in range(1, num_blocks):
            layers.append(MBConvBlock_CIFAR100(out_channels, out_channels, 1, expand_ratio))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Stem特征提取：[B, 3, 32, 32] -> [B, 40, 32, 32]
        x = self.stem(x)
        
        # 7个MBConv阶段，逐步提取层次化特征
        x = self.stage1(x)  # [B, 40, 32, 32] -> [B, 24, 32, 32]
        x = self.stage2(x)  # [B, 24, 32, 32] -> [B, 32, 16, 16]
        x = self.stage3(x)  # [B, 32, 16, 16] -> [B, 48, 8, 8]
        x = self.stage4(x)  # [B, 48, 8, 8] -> [B, 96, 4, 4]
        x = self.stage5(x)  # [B, 96, 4, 4] -> [B, 136, 4, 4]
        x = self.stage6(x)  # [B, 136, 4, 4] -> [B, 232, 2, 2]
        x = self.stage7(x)  # [B, 232, 2, 2] -> [B, 384, 2, 2]
        
        # Head层特征增强：[B, 384, 2, 2] -> [B, 1536, 2, 2]
        x = F.relu(self.head_bn(self.head_conv(x)))
        
        # 全局平均池化：[B, 1536, 2, 2] -> [B, 1536, 1, 1]
        x = self.avgpool(x)
        # 展平：[B, 1536, 1, 1] -> [B, 1536]
        x = torch.flatten(x, 1)
        # 分类器：[B, 1536] -> [B, 100]
        x = self.classifier(x)
        
        # 返回log概率分布
        return F.log_softmax(x, dim=1)
    
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
                nn.init.constant_(m.bias, 0)


class DenseBlock(nn.Module):
    """
    DenseNet风格密集连接块
    ======================
    
    实现密集连接模式：每一层都与前面所有层直接连接。
    这种连接方式能够最大化信息流，缓解梯度消失问题。
    
    特点：
    - 特征复用：每层都能访问前面所有层的特征
    - 隐式深度监督：梯度能直接传播到早期层
    - 参数效率：通过特征复用减少冗余参数
    """
    
    def __init__(self, in_channels, growth_rate, num_layers, dropout_rate=0.1):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        
        # 构建密集连接的卷积层
        for i in range(num_layers):
            # 每一层的输入是前面所有层的特征拼接
            layer_input_channels = in_channels + i * growth_rate
            
            # DenseNet的标准层结构：BN-ReLU-Conv1x1-BN-ReLU-Conv3x3
            layer = nn.Sequential(
                # 第一个BN-ReLU-Conv：压缩特征维度
                nn.BatchNorm2d(layer_input_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(layer_input_channels, growth_rate * 4, 1, bias=False),  # 1x1卷积
                
                # 第二个BN-ReLU-Conv：提取空间特征
                nn.BatchNorm2d(growth_rate * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(growth_rate * 4, growth_rate, 3, padding=1, bias=False),  # 3x3卷积
                
                # Dropout正则化
                nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
            )
            self.layers.append(layer)
    
    def forward(self, x):
        # 特征列表，初始包含输入特征
        features = [x]
        
        # 逐层前向传播，每层输入是所有前层特征的拼接
        for layer in self.layers:
            # 拼接所有已有特征作为当前层输入
            new_feature = layer(torch.cat(features, 1))
            # 将新特征加入特征列表
            features.append(new_feature)
        
        # 返回所有特征的拼接（包括输入）
        return torch.cat(features, 1)


class DenseNet_CIFAR100(nn.Module):
    """
    CIFAR-100专用DenseNet模型
    =========================
    
    使用密集连接的深度网络，特别适合细粒度分类任务。
    通过特征复用提高参数效率和分类性能。
    
    网络架构：
    - 初始卷积：提取基础特征
    - 4个密集块：逐步加深网络并复用特征
    - 3个过渡层：在密集块间降维和下采样
    - 分类器：全连接层输出100类预测
    
    优势：
    - 参数效率高：特征复用减少参数冗余
    - 梯度流畅：密集连接缓解梯度消失
    - 特征丰富：多尺度特征融合
    """
    
    def __init__(self, num_classes=100, growth_rate=32, num_layers=[6, 12, 24, 16]):
        super(DenseNet_CIFAR100, self).__init__()
        
        # 初始特征提取层
        # 将3通道RGB图像转换为64通道特征图
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        in_channels = 64  # 当前通道数
        
        # 构建4个密集块和3个过渡层
        for i, num_layer in enumerate(num_layers):
            # 添加密集块
            block = DenseBlock(in_channels, growth_rate, num_layer)
            self.add_module(f'denseblock{i+1}', block)
            # 更新通道数：原有通道 + 新增通道
            in_channels += num_layer * growth_rate
            
            # 除了最后一个密集块，都要添加过渡层
            if i != len(num_layers) - 1:
                # 过渡层：降维 + 下采样
                trans = nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    # 1x1卷积降维：减少一半通道数
                    nn.Conv2d(in_channels, in_channels // 2, 1, bias=False),
                    # 2x2平均池化下采样：空间尺寸减半
                    nn.AvgPool2d(2, 2)
                )
                self.add_module(f'transition{i+1}', trans)
                in_channels = in_channels // 2  # 更新通道数
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(in_channels),    # 最终特征归一化
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),   # 全局平均池化
            nn.Flatten(),                   # 展平为向量
            nn.Dropout(0.3),               # Dropout防过拟合
            nn.Linear(in_channels, num_classes)  # 全连接输出100类
        )
    
    def forward(self, x):
        # 初始特征提取：[B, 3, 32, 32] -> [B, 64, 32, 32]
        x = self.features(x)
        
        # 第一个密集块 + 过渡层：[B, 64, 32, 32] -> [B, 256, 16, 16]
        x = self.denseblock1(x)     # [B, 64, 32, 32] -> [B, 256, 32, 32]
        x = self.transition1(x)     # [B, 256, 32, 32] -> [B, 128, 16, 16]
        
        # 第二个密集块 + 过渡层：[B, 128, 16, 16] -> [B, 512, 8, 8]
        x = self.denseblock2(x)     # [B, 128, 16, 16] -> [B, 512, 16, 16]
        x = self.transition2(x)     # [B, 512, 16, 16] -> [B, 256, 8, 8]
        
        # 第三个密集块 + 过渡层：[B, 256, 8, 8] -> [B, 1024, 4, 4]
        x = self.denseblock3(x)     # [B, 256, 8, 8] -> [B, 1024, 8, 8]
        x = self.transition3(x)     # [B, 1024, 8, 8] -> [B, 512, 4, 4]
        
        # 第四个密集块（最后一个）：[B, 512, 4, 4] -> [B, 1024, 4, 4]
        x = self.denseblock4(x)
        
        # 分类器：[B, 1024, 4, 4] -> [B, 100]
        x = self.classifier(x)
        
        # 返回log概率分布
        return F.log_softmax(x, dim=1)


def get_cifar100_model(model_type='resnet18_fed', **kwargs):
    """
    CIFAR-100模型工厂函数
    =====================
    
    根据指定类型返回相应的CIFAR-100模型实例。
    提供统一的模型创建接口。
    
    参数:
        model_type (str): 模型类型
            - 'resnet18_fed': ResNet18联邦学习版本（默认）
            - 'efficientnet': EfficientNet-B3风格模型
            - 'densenet': DenseNet模型
        **kwargs: 模型特定参数
    
    返回:
        nn.Module: 对应的模型实例
        
    使用示例:
        # 创建默认ResNet18模型
        model = get_cifar100_model('resnet18_fed')
        
        # 创建带自定义参数的EfficientNet模型
        model = get_cifar100_model('efficientnet', dropout_rate=0.4)
    """
    if model_type == 'resnet18_fed':
        return ResNet18_CIFAR100_Fed(**kwargs)
    elif model_type == 'efficientnet':
        return EfficientNet_CIFAR100(**kwargs)
    elif model_type == 'densenet':
        return DenseNet_CIFAR100(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# 模型配置预设
CIFAR100_MODEL_CONFIGS = {
    'resnet18_fed_default': {
        'model_type': 'resnet18_fed',
        'use_groupnorm': True,
        'num_groups': 8,
        'use_se': True,
        'dropout_rate': 0.3
    },
    'resnet18_fed_heavy': {
        'model_type': 'resnet18_fed',
        'use_groupnorm': True,
        'num_groups': 16,
        'use_se': True,
        'dropout_rate': 0.4
    },
    'efficientnet_default': {
        'model_type': 'efficientnet',
        'dropout_rate': 0.3
    },
    'efficientnet_heavy': {
        'model_type': 'efficientnet',
        'dropout_rate': 0.4
    },
    'densenet_default': {
        'model_type': 'densenet',
        'growth_rate': 32
    },
    'densenet_compact': {
        'model_type': 'densenet',
        'growth_rate': 16,
        'num_layers': [6, 8, 12, 8]
    }
}


if __name__ == "__main__":
    """测试模型实例化和前向传播"""
    import torch
    
    # 创建测试输入：模拟CIFAR-100数据
    test_input = torch.randn(4, 3, 32, 32)  # batch_size=4, channels=3, height=32, width=32
    
    print("CIFAR-100模型测试")
    print("=" * 50)
    
    # 测试主要模型配置
    test_configs = ['resnet18_fed_default', 'efficientnet_default', 'densenet_default']
    
    for config_name in test_configs:
        print(f"\n测试配置: {config_name}")
        config = CIFAR100_MODEL_CONFIGS[config_name]
        
        # 创建模型实例
        model = get_cifar100_model(**config)
        
        # 计算模型参数量
        param_count = sum(p.numel() for p in model.parameters())
        
        # 前向传播测试
        with torch.no_grad():
            output = model(test_input)
        
        print(f"  参数量: {param_count:,}")
        print(f"  输出形状: {output.shape}")  # 应该是 [4, 100]
        print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")
        
        # 验证输出是log概率分布
        prob_sum = torch.exp(output).sum(dim=1)
        print(f"  概率和: {prob_sum.mean():.4f} (应该接近1.0)")
    
    print(f"\n所有CIFAR-100模型测试完成！")
