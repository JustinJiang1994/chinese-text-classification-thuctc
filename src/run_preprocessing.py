#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理运行脚本
使用配置文件进行数据预处理
"""

import json
import argparse
from pathlib import Path
from data_preprocessor import DataPreprocessor
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_file: str = "config/preprocessing_config.json") -> dict:
    """
    加载配置文件
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        配置字典
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"成功加载配置文件: {config_file}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise

def run_preprocessing(config: dict, input_file: str = None):
    """
    运行数据预处理
    
    Args:
        config: 配置字典
        input_file: 输入文件路径（可选，会覆盖配置文件中的设置）
    """
    # 获取配置
    preprocess_config = config['data_preprocessing']
    experiment_config = config['experiment_settings']
    
    # 确定输入文件
    if input_file is None:
        if experiment_config['use_sample_data']:
            input_file = experiment_config['sample_data_file']
        else:
            input_file = experiment_config['full_data_file']
    
    # 检查输入文件是否存在
    if not Path(input_file).exists():
        logger.error(f"输入文件不存在: {input_file}")
        return
    
    # 创建预处理器
    preprocessor = DataPreprocessor(
        stopwords_file=preprocess_config['stopwords_file'],
        random_seed=preprocess_config['random_seed']
    )
    
    # 更新配置
    preprocessor.config.update({
        'min_text_length': preprocess_config['text_filtering']['min_text_length'],
        'max_text_length': preprocess_config['text_filtering']['max_text_length'],
        'train_ratio': preprocess_config['data_splitting']['train_ratio'],
        'val_ratio': preprocess_config['data_splitting']['val_ratio'],
        'test_ratio': preprocess_config['data_splitting']['test_ratio'],
        'stratify': preprocess_config['data_splitting']['stratify'],
        'remove_empty': preprocess_config['text_filtering']['remove_empty']
    })
    
    # 运行预处理流程
    output_dir = preprocess_config['output']['output_dir']
    stats = preprocessor.preprocess_pipeline(input_file, output_dir)
    
    logger.info(f"数据预处理完成！")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"随机种子: {preprocess_config['random_seed']}")
    
    return stats

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据预处理脚本')
    parser.add_argument('--config', type=str, default='config/preprocessing_config.json',
                       help='配置文件路径')
    parser.add_argument('--input', type=str, default=None,
                       help='输入文件路径（可选，会覆盖配置文件中的设置）')
    parser.add_argument('--use-sample', action='store_true',
                       help='使用样本数据（覆盖配置文件设置）')
    parser.add_argument('--use-full', action='store_true',
                       help='使用完整数据（覆盖配置文件设置）')
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 处理命令行参数
        if args.use_sample:
            config['experiment_settings']['use_sample_data'] = True
        elif args.use_full:
            config['experiment_settings']['use_sample_data'] = False
        
        # 运行预处理
        stats = run_preprocessing(config, args.input)
        
        if stats:
            print(f"\n✅ 预处理成功完成！")
            print(f"📊 处理统计:")
            print(f"   - 原始数据: {stats['original_count']} 条")
            print(f"   - 清洗后数据: {stats['cleaned_count']} 条")
            print(f"   - 训练集: {stats['train_count']} 条")
            print(f"   - 验证集: {stats['val_count']} 条")
            print(f"   - 测试集: {stats['test_count']} 条")
            print(f"   - 处理时间: {stats['processing_time']:.2f} 秒")
        
    except Exception as e:
        logger.error(f"预处理失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 