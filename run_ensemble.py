#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行集成学习分类器
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ensemble_classifier import EnsembleNewsClassifier

def main():
    """主函数"""
    print("=" * 60)
    print("集成学习新闻分类器")
    print("=" * 60)
    
    # 创建集成分类器
    ensemble_classifier = EnsembleNewsClassifier()
    
    # 选择集成方法
    print("\n请选择集成方法:")
    print("1. 投票集成 (Voting)")
    print("2. 堆叠集成 (Stacking)")
    
    choice = input("请输入选择 (1-2): ").strip()
    
    if choice == '1':
        # 投票集成
        print("\n正在使用投票集成方法...")
        results = ensemble_classifier.run_complete_pipeline(ensemble_type='voting')
    elif choice == '2':
        # 堆叠集成
        print("\n正在使用堆叠集成方法...")
        results = ensemble_classifier.run_complete_pipeline(ensemble_type='stacking')
    else:
        print("无效选择，使用默认的投票集成方法")
        results = ensemble_classifier.run_complete_pipeline(ensemble_type='voting')
    
    if results:
        print(f"\n最终结果:")
        print(f"准确率: {results['accuracy']:.4f}")
        print(f"F1-Macro: {results['f1_macro']:.4f}")
        print(f"F1-Weighted: {results['f1_weighted']:.4f}")
        print(f"交叉验证: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")
        
        # 保存结果到文件
        with open('results/ensemble_results.txt', 'w', encoding='utf-8') as f:
            f.write("集成学习分类器结果\n")
            f.write("=" * 30 + "\n")
            f.write(f"准确率: {results['accuracy']:.4f}\n")
            f.write(f"F1-Macro: {results['f1_macro']:.4f}\n")
            f.write(f"F1-Weighted: {results['f1_weighted']:.4f}\n")
            f.write(f"交叉验证: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})\n")
        
        print("\n结果已保存到 results/ensemble_results.txt")
    else:
        print("训练失败，请检查数据文件是否存在")

if __name__ == "__main__":
    main() 