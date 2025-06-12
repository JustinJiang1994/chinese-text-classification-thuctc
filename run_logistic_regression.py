#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行逻辑回归新闻分类器
"""

import os
import sys
import time
from datetime import datetime

# 添加src目录到路径
sys.path.append('src')

from logistic_regression_classifier import LogisticRegressionNewsClassifier

def main():
    print("=" * 60)
    print("中文新闻逻辑回归分类器")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 创建分类器实例
        classifier = LogisticRegressionNewsClassifier()
        
        # 运行完整流程（不使用网格搜索以节省时间）
        results = classifier.run_complete_pipeline(use_grid_search=False)
        
        # 计算运行时间
        end_time = time.time()
        runtime = end_time - start_time
        
        print("\n" + "=" * 60)
        print("任务完成总结")
        print("=" * 60)
        print(f"运行时间: {runtime:.2f} 秒")
        print(f"准确率: {results['accuracy']:.4f}")
        print(f"F1-Macro: {results['f1_macro']:.4f}")
        print(f"F1-Weighted: {results['f1_weighted']:.4f}")
        print(f"交叉验证: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 保存结果到文件
        with open('results/logistic_regression_results.txt', 'w', encoding='utf-8') as f:
            f.write("逻辑回归分类器结果\n")
            f.write("=" * 40 + "\n")
            f.write(f"运行时间: {runtime:.2f} 秒\n")
            f.write(f"准确率: {results['accuracy']:.4f}\n")
            f.write(f"F1-Macro: {results['f1_macro']:.4f}\n")
            f.write(f"F1-Weighted: {results['f1_weighted']:.4f}\n")
            f.write(f"交叉验证: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})\n")
            f.write(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print("\n结果已保存到 results/logistic_regression_results.txt")
        print("混淆矩阵已保存到 results/logistic_regression_confusion_matrix.png")
        print("模型已保存到 models/logistic_regression_model.pkl")
        
    except Exception as e:
        print(f"运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 