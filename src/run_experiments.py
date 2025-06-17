#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一实验运行脚本
整合所有实验功能，提供简洁的接口
"""

import argparse
import json
from pathlib import Path
from experiments import NewsClassifierExperiment

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='新闻分类实验运行器')
    parser.add_argument('--experiment', type=str, default='quick',
                       choices=['quick', 'imbalance', 'full'],
                       help='实验类型: quick(快速), imbalance(不平衡), full(完整)')
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='样本数量 (仅用于quick和imbalance实验)')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存结果')
    parser.add_argument('--no-plot', action='store_true',
                       help='不生成图表')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 更新配置
    if args.no_save:
        config['save_results'] = False
    if args.no_plot:
        config['plot_results'] = False
    
    # 创建实验管理器
    experiment = NewsClassifierExperiment(config)
    
    # 运行实验
    if args.experiment == 'quick':
        print("=== 运行快速实验 ===")
        results = experiment.run_quick_experiment(args.sample_size)
    elif args.experiment == 'imbalance':
        print("=== 运行类别不平衡实验 ===")
        results = experiment.run_imbalance_experiment(args.sample_size)
    elif args.experiment == 'full':
        print("=== 运行完整实验 ===")
        results = experiment.run_full_experiment()
    
    # 打印总结
    experiment.print_summary(results)

def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # 返回默认配置
    return {
        'data_dir': 'results',
        'results_dir': 'results',
        'sample_size': 10000,
        'max_features': 8000,
        'test_size': 0.2,
        'random_state': 42,
        'models': ['naive_bayes', 'svm', 'random_forest', 'logistic_regression'],
        'sampling_methods': ['original', 'smote', 'tomek_links', 'smoteenn'],
        'save_results': True,
        'plot_results': True
    }

if __name__ == "__main__":
    main() 