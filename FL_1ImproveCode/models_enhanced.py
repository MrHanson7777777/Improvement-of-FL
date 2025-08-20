"""
Enhanced models for improved CIFAR performance in federated learning.
These models aim to achieve higher accuracy while maintaining efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedEfficientNetCifar(nn.Module):
    """Enhanced EfficientNet for CIFAR with improved architecture for higher accuracy"""
    def __init__(self, num_classes=10):
        super(EnhancedEfficientNetCifar, self).__init__()
        
        # Enhanced stem with more capacity
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # More sophisticated stage design
        self.stage1 = self._make_stage(64, 80, 2, 1, 2)    # 32x32
        self.stage2 = self._make_stage(80, 112, 3, 2, 4)   # 16x16
        self.stage3 = self._make_stage(112, 160, 3, 2, 4)  # 8x8
        self.stage4 = self._make_stage(160, 224, 4, 2, 6)  # 4x4
        self.stage5 = self._make_stage(224, 320, 2, 1, 6)  # 4x4
        
        # Enhanced head
        self.head_conv = nn.Conv2d(320, 512, 1, bias=False)
        self.head_bn = nn.BatchNorm2d(512)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Enhanced classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_stage(self, in_channels, out_channels, num_blocks, stride, expand_ratio):
        layers = []
        layers.append(EnhancedMBConvBlock(in_channels, out_channels, stride, expand_ratio, se_ratio=0.25))
        for _ in range(1, num_blocks):
            layers.append(EnhancedMBConvBlock(out_channels, out_channels, 1, expand_ratio, se_ratio=0.25))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
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
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        
        x = F.relu(self.head_bn(self.head_conv(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)


class EnhancedMBConvBlock(nn.Module):
    """Enhanced Mobile Inverted Bottleneck Convolution Block with improvements"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio, se_ratio=0.25):
        super(EnhancedMBConvBlock, self).__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expansion with better activation
        if expand_ratio != 1:
            expanded_channels = in_channels * expand_ratio
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True)  # Use SiLU instead of ReLU
            )
            self.has_expansion = True
        else:
            expanded_channels = in_channels
            self.expand_conv = nn.Identity()
            self.has_expansion = False
            
        self.expanded_channels = expanded_channels
            
        # Enhanced depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, 3, stride, 1, 
                     groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        )
        
        # Enhanced Squeeze-and-Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expanded_channels, se_channels, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(se_channels, expanded_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.se = nn.Identity()
            
        # Pointwise convolution
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Stochastic depth for regularization
        self.drop_path = nn.Dropout2d(0.1) if self.use_residual else nn.Identity()
        
    def forward(self, x):
        identity = x
        
        # Expansion
        if self.has_expansion:
            x = self.expand_conv(x)
        
        # Depthwise
        x = self.depthwise_conv(x)
        
        # SE
        if hasattr(self.se, 'weight') or len(list(self.se.children())) > 0:
            se_weight = self.se(x)
            x = x * se_weight
            
        # Projection
        x = self.project_conv(x)
        
        # Residual connection with stochastic depth
        if self.use_residual:
            x = self.drop_path(x) + identity
            
        return x


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet"""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet50Cifar(nn.Module):
    """ResNet-50 adapted for CIFAR datasets"""
    def __init__(self, num_classes=10):
        super(ResNet50Cifar, self).__init__()
        
        # Initial layers adapted for CIFAR
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet layers
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 256, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 512, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 1024, 512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, in_planes, planes, num_blocks, stride):
        layers = []
        layers.append(block(in_planes, planes, stride))
        in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(in_planes, planes, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)


class WideResNetCifar(nn.Module):
    """Wide ResNet for CIFAR with improved performance"""
    def __init__(self, depth=28, width=10, num_classes=10, dropout_rate=0.3):
        super(WideResNetCifar, self).__init__()
        
        self.in_planes = 16
        
        n = (depth - 4) // 6
        k = width
        
        nStages = [16, 16*k, 32*k, 64*k]
        
        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._wide_layer(WideBasicBlock, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasicBlock, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasicBlock, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.linear = nn.Linear(nStages[3], num_classes)
        
    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
            
        return nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return F.log_softmax(out, dim=1)


class WideBasicBlock(nn.Module):
    """Wide Basic Block for Wide ResNet"""
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            )
            
    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        
        return out
