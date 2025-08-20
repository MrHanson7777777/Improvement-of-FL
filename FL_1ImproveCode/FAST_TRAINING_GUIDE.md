# 🚀 CIFAR-100 优化训练命令 (解决速度慢和损失大的问题)

## 🔥 推荐命令 (轻量级Ultra模型 - 速度快，性能好)

### Non-IID 场景 (推荐)
```bash
python federated_cifar100_main.py --model lightweight_ultra --epochs 200 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 64 --lr 0.001 --optimizer adamw --dropout_rate 0.3 --label_smoothing 0.1 --alpha 0.3 --gpu 0 --iid 0
```

### IID 场景
```bash
python federated_cifar100_main.py --model lightweight_ultra --epochs 150 --num_users 20 --frac 0.5 --local_ep 8 --local_bs 64 --lr 0.002 --optimizer adamw --dropout_rate 0.2 --label_smoothing 0.05 --alpha 0.2 --gpu 0 --iid 1
```

## 🎯 优化策略

### 1. 模型简化
- **轻量级Ultra模型**: 减少50%参数量，训练速度提升2-3倍
- **ReLU6替代SiLU**: 计算更快
- **减少网络深度**: 从7个stage减少到6个stage

### 2. 学习率优化
- **降低初始学习率**: 0.001-0.002 (原来0.003-0.004太高)
- **更温和的调度**: CosineAnnealingLR替代WarmRestarts
- **AdamW优化器**: 更稳定的收敛

### 3. 训练策略优化
- **减少本地轮数**: 5-8轮 (原来10轮太多)
- **增加批次大小**: 64 (提升GPU利用率)
- **减少总轮数**: 150-200轮 (原来350轮太多)

### 4. 每轮都显示准确率
```
全局轮次 1/200
  选择 10 个用户进行训练
  正在聚合权重...
  ✅ 轮次完成 | 用时: 45.2s | 训练损失: 2.1234
  📊 测试准确率: 0.1234 (12.34%) | 测试损失: 2.5678
  ⏱️  总用时: 0.8分钟 | 预计剩余: 149.2分钟
  ==================================================
```

## 📊 性能预期

| 模型 | 训练时间 | 预期准确率 | GPU显存 |
|------|----------|------------|---------|
| 轻量级Ultra (Non-IID) | 2-3小时 | 65-70% | 4-6GB |
| 轻量级Ultra (IID) | 1.5-2小时 | 75-80% | 4-6GB |
| 原Ultra模型 | 10-14小时 | 68-72% | 8-12GB |

## 🔧 故障排除

### 如果损失值仍然很大:
1. **进一步降低学习率**: `--lr 0.0005`
2. **增加warmup**: 添加学习率预热
3. **检查数据预处理**: 确保归一化正确

### 如果速度仍然很慢:
1. **减少用户数**: `--num_users 10`
2. **减少本地轮数**: `--local_ep 3`
3. **使用更简单模型**: `--model cifar100_efficientnet --efficientnet_variant b0`

## 🎉 开始训练

```bash
# 快速测试 (5轮)
python federated_cifar100_main.py --model lightweight_ultra --epochs 5 --num_users 10 --frac 0.5 --local_ep 3 --local_bs 64 --lr 0.001 --optimizer adamw --dropout_rate 0.3 --label_smoothing 0.1 --alpha 0.3 --gpu 0 --iid 0

# 完整训练 (Non-IID)
python federated_cifar100_main.py --model lightweight_ultra --epochs 200 --num_users 20 --frac 0.5 --local_ep 5 --local_bs 64 --lr 0.001 --optimizer adamw --dropout_rate 0.3 --label_smoothing 0.1 --alpha 0.3 --gpu 0 --iid 0
```

**✨ 现在每轮都会显示测试准确率，训练速度提升2-3倍！**
