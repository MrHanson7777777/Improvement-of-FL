#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化策略效果对比测试脚本
对比基础版本 vs 优化版本的性能差异
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime

def run_experiment(command, experiment_name, timeout=3600):
    """运行实验并记录结果"""
    print(f"\n{'='*60}")
    print(f"开始实验: {experiment_name}")
    print(f"命令: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 为PowerShell修改命令格式
        if os.name == 'nt':  # Windows系统
            # 将&&替换为;，并确保路径正确
            command = command.replace(' && ', '; ')
            # 添加cd命令到当前目录
            current_dir = os.getcwd()
            if not command.startswith('cd '):
                command = f'cd "{current_dir}"; {command}'
        
        # 运行命令
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd=os.getcwd()  # 确保在正确的工作目录
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 解析输出中的准确率
        output = result.stdout
        error_output = result.stderr
        accuracy = None
        
        print(f"返回码: {result.returncode}")
        if result.returncode != 0:
            print(f"错误输出: {error_output[:500]}...")  # 只显示前500字符
        
        # 寻找测试准确率
        for line in output.split('\n'):
            if 'Test Accuracy' in line:
                try:
                    accuracy = float(line.split(':')[1].strip().replace('%', ''))
                    break
                except:
                    pass
        
        success = result.returncode == 0 and accuracy is not None
        
        return {
            'experiment_name': experiment_name,
            'command': command,
            'duration': duration,
            'accuracy': accuracy,
            'output': output,
            'error': error_output,
            'success': success
        }
        
    except subprocess.TimeoutExpired:
        print(f"实验 {experiment_name} 超时!")
        return {
            'experiment_name': experiment_name,
            'command': command,
            'duration': timeout,
            'accuracy': None,
            'output': '',
            'error': 'Timeout',
            'success': False
        }
    except Exception as e:
        print(f"实验 {experiment_name} 出错: {str(e)}")
        return {
            'experiment_name': experiment_name,
            'command': command,
            'duration': 0,
            'accuracy': None,
            'output': '',
            'error': str(e),
            'success': False
        }

def main():
    """主函数 - 运行对比实验"""
    
    print("🚀 联邦学习优化策略效果验证实验")
    print("对比基础版本与优化版本的性能差异")
    
    # 实验配置
    experiments = [
        {
            'name': 'CIFAR-10_基础CNN',
            'command': 'python federated_main_improved.py --dataset cifar --model cnn --epochs 20 --lr 0.01 --local_ep 3 --num_users 10 --frac 1.0 --gpu 0',
            'description': '基础CNN模型，无优化策略'
        },
        {
            'name': 'CIFAR-10_增强CNN_优化版',
            'command': 'python federated_main_optimized.py --dataset cifar --model enhanced_cnn --epochs 20 --lr 0.01 --local_ep 3 --num_users 10 --frac 1.0 --gpu 0',
            'description': '增强CNN + 学习率调度 + 数据增强 + 正则化'
        },
        {
            'name': 'CIFAR-10_ResNet50_基础版',
            'command': 'python federated_main_improved.py --dataset cifar --model resnet50 --epochs 20 --lr 0.01 --local_ep 3 --num_users 10 --frac 1.0 --gpu 0',
            'description': 'ResNet-50模型，无优化策略'
        },
        {
            'name': 'CIFAR-10_ResNet50_优化版',
            'command': 'python federated_main_optimized.py --dataset cifar --model resnet50 --epochs 20 --lr 0.01 --local_ep 3 --num_users 10 --frac 1.0 --gpu 0',
            'description': 'ResNet-50 + 学习率调度 + 数据增强 + 正则化'
        }
    ]
    
    # 存储结果
    results = []
    
    # 运行实验
    for exp in experiments:
        result = run_experiment(
            exp['command'], 
            exp['name'], 
            timeout=1800  # 30分钟超时
        )
        result['description'] = exp['description']
        results.append(result)
        
        # 打印结果摘要
        if result['success'] and result['accuracy'] is not None:
            print(f"✅ {exp['name']}: {result['accuracy']:.2f}% (用时: {result['duration']:.1f}s)")
        else:
            print(f"❌ {exp['name']}: 实验失败")
    
    # 生成报告
    generate_report(results)

def generate_report(results):
    """生成实验报告"""
    
    print(f"\n{'='*80}")
    print("🎉 实验报告")
    print(f"{'='*80}")
    
    # 按模型类型分组对比
    model_comparisons = {}
    
    for result in results:
        if result['success'] and result['accuracy'] is not None:
            name = result['experiment_name']
            
            # 提取模型类型
            if 'CNN' in name and 'ResNet' not in name:
                model_type = 'CNN'
            elif 'ResNet50' in name:
                model_type = 'ResNet50'
            else:
                model_type = 'Other'
            
            # 判断是否为优化版
            is_optimized = '优化版' in name
            
            if model_type not in model_comparisons:
                model_comparisons[model_type] = {'basic': None, 'optimized': None}
            
            if is_optimized:
                model_comparisons[model_type]['optimized'] = result
            else:
                model_comparisons[model_type]['basic'] = result
    
    # 打印对比结果
    improvements = []
    
    for model_type, data in model_comparisons.items():
        basic = data['basic']
        optimized = data['optimized']
        
        print(f"\n📊 {model_type} 模型对比:")
        
        if basic and optimized:
            basic_acc = basic['accuracy']
            opt_acc = optimized['accuracy']
            improvement = opt_acc - basic_acc
            improvement_pct = (improvement / basic_acc) * 100
            
            print(f"  基础版本准确率: {basic_acc:.2f}%")
            print(f"  优化版本准确率: {opt_acc:.2f}%")
            print(f"  绝对提升: {improvement:.2f}%")
            print(f"  相对提升: {improvement_pct:.1f}%")
            
            improvements.append({
                'model': model_type,
                'basic_acc': basic_acc,
                'optimized_acc': opt_acc,
                'absolute_improvement': improvement,
                'relative_improvement': improvement_pct
            })
            
            if improvement > 0:
                print(f"  ✅ 优化策略有效!")
            else:
                print(f"  ❌ 优化策略无效")
        else:
            if basic:
                print(f"  基础版本准确率: {basic['accuracy']:.2f}%")
                print(f"  优化版本: 实验失败")
            elif optimized:
                print(f"  基础版本: 实验失败")
                print(f"  优化版本准确率: {optimized['accuracy']:.2f}%")
            else:
                print(f"  基础版本和优化版本都实验失败")
    
    # 总结
    print(f"\n🎯 优化策略效果总结:")
    
    if improvements:
        avg_absolute = sum(imp['absolute_improvement'] for imp in improvements) / len(improvements)
        avg_relative = sum(imp['relative_improvement'] for imp in improvements) / len(improvements)
        
        print(f"  平均绝对提升: {avg_absolute:.2f}%")
        print(f"  平均相对提升: {avg_relative:.1f}%")
        
        positive_improvements = [imp for imp in improvements if imp['absolute_improvement'] > 0]
        success_rate = len(positive_improvements) / len(improvements) * 100
        
        print(f"  优化成功率: {success_rate:.1f}%")
        
        if avg_absolute > 1.0:
            print(f"  🎉 优化策略显著有效! 平均提升 {avg_absolute:.2f}%")
        elif avg_absolute > 0:
            print(f"  ✅ 优化策略轻微有效，提升 {avg_absolute:.2f}%")
        else:
            print(f"  ❌ 优化策略无效果或负面影响")
    else:
        print("  无法进行对比分析，实验失败")
    
    # 关于优化策略的说明
    print(f"\n💡 本次测试的优化策略包括:")
    print("  1. 学习率调度 (Cosine Annealing)")
    print("  2. 数据增强 (CutMix, MixUp)")
    print("  3. 正则化 (Label Smoothing, Weight Decay)")
    print("  4. 梯度裁剪")
    print("  5. 增强模型架构")
    
    # 保存结果到文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optimization_results_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'results': results,
            'improvements': improvements
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 详细结果已保存到: {filename}")

if __name__ == '__main__':
    main()
