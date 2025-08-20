#!/usr/bin/env python
# -*- coding: utf-8 -*-
# analyze_results.py
# 联邦学习实验结果分析脚本

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def extract_results_from_log(log_file_path):
    """从日志文件中提取实验结果"""
    results = {
        'test_accuracy': None,
        'train_accuracy': None,
        'total_time': None,
        'actual_rounds': None,
        'early_stopped': False,
        'best_val_acc': None
    }
    
    if not os.path.exists(log_file_path):
        print(f"文件不存在: {log_file_path}")
        return results
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取测试准确率
    test_acc_match = re.search(r'\|\-\-\-\- Test Accuracy: ([\d\.]+)%', content)
    if test_acc_match:
        results['test_accuracy'] = float(test_acc_match.group(1))
    
    # 提取训练准确率
    train_acc_match = re.search(r'\|\-\-\-\- Avg Train Accuracy: ([\d\.]+)%', content)
    if train_acc_match:
        results['train_accuracy'] = float(train_acc_match.group(1))
    
    # 提取总运行时间
    time_match = re.search(r'Total Run Time: ([\d\.]+)', content)
    if time_match:
        results['total_time'] = float(time_match.group(1))
    
    # 提取实际训练轮次
    rounds_match = re.search(r'Results after (\d+) global rounds', content)
    if rounds_match:
        results['actual_rounds'] = int(rounds_match.group(1))
    
    # 检查是否早停
    if 'Early stopping triggered' in content:
        results['early_stopped'] = True
        early_stop_match = re.search(r'Early stopping triggered after (\d+) global rounds', content)
        if early_stop_match:
            results['actual_rounds'] = int(early_stop_match.group(1))
    
    # 提取最佳验证准确率
    best_val_matches = re.findall(r'New best validation accuracy: ([\d\.]+)%', content)
    if best_val_matches:
        results['best_val_acc'] = max([float(acc) for acc in best_val_matches])
    
    return results

def analyze_experiment_results():
    """分析实验结果"""
    
    # 实验配置
    experiments = [
        ('mnist', 'cnn'),
        ('fmnist', 'cnn'),
        ('cifar', 'cnn'),
        ('cifar100', 'cnn'),
        ('mnist', 'mlp')
    ]
    
    results_data = []
    
    for dataset, model in experiments:
        # 原始版本结果
        original_log = f'results/original/{dataset}_{model}.log'
        original_results = extract_results_from_log(original_log)
        
        # 改进版本结果
        improved_log = f'results/improved/{dataset}_{model}.log'
        improved_results = extract_results_from_log(improved_log)
        
        # 计算改进效果
        improvement = {}
        if original_results['test_accuracy'] and improved_results['test_accuracy']:
            improvement['test_acc_gain'] = improved_results['test_accuracy'] - original_results['test_accuracy']
            improvement['test_acc_gain_pct'] = (improvement['test_acc_gain'] / original_results['test_accuracy']) * 100
        
        if original_results['total_time'] and improved_results['total_time']:
            improvement['time_saved'] = original_results['total_time'] - improved_results['total_time']
            improvement['time_saved_pct'] = (improvement['time_saved'] / original_results['total_time']) * 100
        
        if original_results['actual_rounds'] and improved_results['actual_rounds']:
            improvement['rounds_saved'] = original_results['actual_rounds'] - improved_results['actual_rounds']
            improvement['rounds_saved_pct'] = (improvement['rounds_saved'] / original_results['actual_rounds']) * 100
        
        # 添加到结果数据
        results_data.append({
            'dataset': dataset,
            'model': model,
            'original_test_acc': original_results['test_accuracy'],
            'improved_test_acc': improved_results['test_accuracy'],
            'original_train_acc': original_results['train_accuracy'],
            'improved_train_acc': improved_results['train_accuracy'],
            'original_time': original_results['total_time'],
            'improved_time': improved_results['total_time'],
            'original_rounds': original_results['actual_rounds'],
            'improved_rounds': improved_results['actual_rounds'],
            'early_stopped': improved_results['early_stopped'],
            'best_val_acc': improved_results['best_val_acc'],
            **improvement
        })
    
    return results_data

def generate_report(results_data):
    """生成实验报告"""
    
    print("=" * 80)
    print("联邦学习模型架构改进效果验证实验结果报告")
    print("=" * 80)
    print()
    
    # 创建DataFrame
    df = pd.DataFrame(results_data)
    
    # 基本结果表格
    print("1. 基本实验结果对比")
    print("-" * 50)
    
    for _, row in df.iterrows():
        print(f"\n【{row['dataset'].upper()} + {row['model'].upper()}】")
        print(f"测试准确率:")
        print(f"  原始版本: {row['original_test_acc']:.2f}%")
        print(f"  改进版本: {row['improved_test_acc']:.2f}%")
        if 'test_acc_gain' in row and row['test_acc_gain'] is not None:
            print(f"  提升: {row['test_acc_gain']:+.2f}% ({row['test_acc_gain_pct']:+.1f}%)")
        
        print(f"训练时间:")
        print(f"  原始版本: {row['original_time']:.1f}秒")
        print(f"  改进版本: {row['improved_time']:.1f}秒")
        if 'time_saved' in row and row['time_saved'] is not None:
            print(f"  节省: {row['time_saved']:.1f}秒 ({row['time_saved_pct']:.1f}%)")
        
        print(f"训练轮次:")
        print(f"  原始版本: {row['original_rounds']}轮")
        print(f"  改进版本: {row['improved_rounds']}轮 {'(早停)' if row['early_stopped'] else ''}")
        if 'rounds_saved' in row and row['rounds_saved'] is not None:
            print(f"  节省: {row['rounds_saved']}轮 ({row['rounds_saved_pct']:.1f}%)")
    
    # 总体统计
    print("\n\n2. 总体改进效果统计")
    print("-" * 50)
    
    valid_acc_improvements = [row['test_acc_gain'] for row in results_data if row.get('test_acc_gain') is not None]
    valid_time_savings = [row['time_saved_pct'] for row in results_data if row.get('time_saved_pct') is not None]
    valid_rounds_savings = [row['rounds_saved_pct'] for row in results_data if row.get('rounds_saved_pct') is not None]
    
    if valid_acc_improvements:
        print(f"准确率提升:")
        print(f"  平均提升: {np.mean(valid_acc_improvements):.2f}%")
        print(f"  最大提升: {max(valid_acc_improvements):.2f}%")
        print(f"  最小提升: {min(valid_acc_improvements):.2f}%")
    
    if valid_time_savings:
        print(f"时间节省:")
        print(f"  平均节省: {np.mean(valid_time_savings):.1f}%")
        print(f"  最大节省: {max(valid_time_savings):.1f}%")
        print(f"  最小节省: {min(valid_time_savings):.1f}%")
    
    if valid_rounds_savings:
        print(f"轮次节省:")
        print(f"  平均节省: {np.mean(valid_rounds_savings):.1f}%")
        print(f"  最大节省: {max(valid_rounds_savings):.1f}%")
        print(f"  最小节省: {min(valid_rounds_savings):.1f}%")
    
    early_stop_count = sum([1 for row in results_data if row['early_stopped']])
    print(f"早停触发率: {early_stop_count}/{len(results_data)} ({early_stop_count/len(results_data)*100:.1f}%)")
    
    # 保存详细结果到CSV
    df.to_csv('results/detailed_results.csv', index=False)
    print(f"\n详细结果已保存到: results/detailed_results.csv")
    
    return df

def create_visualizations(df):
    """创建可视化图表"""
    
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 测试准确率对比
    ax1 = axes[0, 0]
    datasets = [f"{row['dataset']}-{row['model']}" for _, row in df.iterrows()]
    original_acc = df['original_test_acc'].values
    improved_acc = df['improved_test_acc'].values
    
    x = np.arange(len(datasets))
    width = 0.35
    
    ax1.bar(x - width/2, original_acc, width, label='原始版本', alpha=0.8)
    ax1.bar(x + width/2, improved_acc, width, label='改进版本', alpha=0.8)
    ax1.set_xlabel('数据集-模型')
    ax1.set_ylabel('测试准确率 (%)')
    ax1.set_title('测试准确率对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 训练时间对比
    ax2 = axes[0, 1]
    original_time = df['original_time'].values
    improved_time = df['improved_time'].values
    
    ax2.bar(x - width/2, original_time, width, label='原始版本', alpha=0.8)
    ax2.bar(x + width/2, improved_time, width, label='改进版本', alpha=0.8)
    ax2.set_xlabel('数据集-模型')
    ax2.set_ylabel('训练时间 (秒)')
    ax2.set_title('训练时间对比')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 准确率提升
    ax3 = axes[1, 0]
    acc_gains = [row.get('test_acc_gain', 0) for _, row in df.iterrows()]
    colors = ['green' if gain > 0 else 'red' for gain in acc_gains]
    
    bars = ax3.bar(x, acc_gains, color=colors, alpha=0.7)
    ax3.set_xlabel('数据集-模型')
    ax3.set_ylabel('准确率提升 (%)')
    ax3.set_title('准确率提升情况')
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets, rotation=45)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3)
    
    # 在柱子上添加数值标签
    for bar, gain in zip(bars, acc_gains):
        if gain != 0:
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{gain:.2f}%', ha='center', va='bottom')
    
    # 4. 时间节省百分比
    ax4 = axes[1, 1]
    time_savings = [row.get('time_saved_pct', 0) for _, row in df.iterrows()]
    
    bars = ax4.bar(x, time_savings, color='blue', alpha=0.7)
    ax4.set_xlabel('数据集-模型')
    ax4.set_ylabel('时间节省 (%)')
    ax4.set_title('训练时间节省情况')
    ax4.set_xticks(x)
    ax4.set_xticklabels(datasets, rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 在柱子上添加数值标签
    for bar, saving in zip(bars, time_savings):
        if saving != 0:
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{saving:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/experiment_results_visualization.png', dpi=300, bbox_inches='tight')
    print("可视化图表已保存到: results/experiment_results_visualization.png")
    
    plt.show()

def main():
    """主函数"""
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    print("分析实验结果...")
    results_data = analyze_experiment_results()
    
    print("生成实验报告...")
    df = generate_report(results_data)
    
    print("创建可视化图表...")
    create_visualizations(df)
    
    print("\n实验结果分析完成！")

if __name__ == '__main__':
    main()
