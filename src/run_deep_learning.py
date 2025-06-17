#!/usr/bin/env python3
"""
深度学习实验运行脚本
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.deep_learning_experiments import DeepLearningExperiment

def main():
    parser = argparse.ArgumentParser(description='运行深度学习实验')
    parser.add_argument('--data_path', type=str, default='results/train.csv',
                       help='训练数据路径')
    parser.add_argument('--sample_size', type=int, default=5000,
                       help='样本数量 (None表示使用全部数据)')
    parser.add_argument('--output_dir', type=str, default='results/deep_learning_experiments',
                       help='输出目录')
    parser.add_argument('--config', type=str, default='config/deep_learning_config.json',
                       help='配置文件路径')
    parser.add_argument('--models', nargs='+', 
                       choices=['textcnn', 'lstm', 'gru', 'rnn', 'fasttext'],
                       help='要训练的模型列表')
    parser.add_argument('--epochs', type=int, default=20,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--embed_dim', type=int, default=128,
                       help='词嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='隐藏层维度')
    parser.add_argument('--max_len', type=int, default=100,
                       help='最大序列长度')
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出')
    
    args = parser.parse_args()
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_path):
        print(f"错误: 数据文件不存在: {args.data_path}")
        print("请先运行数据预处理: python src/run_preprocessing.py")
        sys.exit(1)
    
    # 加载配置
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    # 命令行参数覆盖配置文件
    if args.models:
        config['models'] = args.models
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.embed_dim:
        config['embed_dim'] = args.embed_dim
    if args.hidden_dim:
        config['hidden_dim'] = args.hidden_dim
    if args.max_len:
        config['max_len'] = args.max_len
    
    # 设置默认值
    defaults = {
        'embed_dim': 128,
        'hidden_dim': 128,
        'num_layers': 1,
        'bidirectional': False,
        'kernel_sizes': [3, 4, 5],
        'num_filters': 100,
        'ngram_range': 2,
        'max_len': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 20,
        'patience': 5,
        'dropout': 0.5,
        'min_freq': 2
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    if args.verbose:
        print("配置:")
        print(json.dumps(config, indent=2, ensure_ascii=False))
    
    # 创建实验
    experiment = DeepLearningExperiment(config)
    
    # 运行实验
    print(f"开始深度学习实验...")
    print(f"数据路径: {args.data_path}")
    print(f"样本数量: {args.sample_size if args.sample_size else '全部'}")
    print(f"模型: {config['models']}")
    print(f"输出目录: {args.output_dir}")
    
    results = experiment.run_experiment(args.data_path, args.sample_size)
    
    # 保存结果
    experiment.save_results(args.output_dir)
    
    # 绘制图表
    experiment.plot_results(args.output_dir)
    
    # 打印结果摘要
    print("\n" + "="*50)
    print("实验结果摘要")
    print("="*50)
    
    # 按F1-macro排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1_macro'], reverse=True)
    
    print(f"{'模型':<12} {'F1-macro':<10} {'F1-weighted':<12} {'训练时间':<10} {'参数数量':<10}")
    print("-" * 60)
    
    for model_name, result in sorted_results:
        print(f"{model_name.upper():<12} {result['f1_macro']:<10.4f} {result['f1_weighted']:<12.4f} "
              f"{result['training_time']:<10.1f}s {result['model_params']:<10,}")
    
    print("\n" + "="*50)
    best_model, best_result = sorted_results[0]
    print(f"最佳模型: {best_model.upper()}")
    print(f"F1-macro: {best_result['f1_macro']:.4f}")
    print(f"F1-weighted: {best_result['f1_weighted']:.4f}")
    print(f"训练时间: {best_result['training_time']:.1f}秒")
    print(f"模型参数: {best_result['model_params']:,}")
    print("="*50)
    
    print(f"\n结果已保存到: {args.output_dir}")

if __name__ == '__main__':
    main() 