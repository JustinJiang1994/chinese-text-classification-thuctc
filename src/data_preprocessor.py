#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理流程脚本
包含数据清洗、数据切分等预处理步骤
"""

import pandas as pd
import numpy as np
import re
import jieba
from pathlib import Path
import logging
from typing import List, Set, Tuple, Dict
import time
from sklearn.model_selection import train_test_split
import json

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, stopwords_file: str = "data/stopwords.txt", random_seed: int = 42):
        """
        初始化数据预处理器
        
        Args:
            stopwords_file: 停用词文件路径
            random_seed: 随机种子，确保结果可复现
        """
        self.stopwords_file = Path(stopwords_file)
        self.stopwords = set()
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # 加载停用词
        self.load_stopwords()
        
        # 预处理配置
        self.config = {
            'min_text_length': 10,  # 最小文本长度
            'max_text_length': 2000,  # 最大文本长度
            'train_ratio': 0.7,  # 训练集比例
            'val_ratio': 0.15,   # 验证集比例
            'test_ratio': 0.15,  # 测试集比例
            'stratify': True,    # 是否分层采样
            'remove_empty': True # 是否移除空文本
        }
        
    def load_stopwords(self):
        """加载停用词"""
        try:
            with open(self.stopwords_file, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        self.stopwords.add(word)
            logger.info(f"成功加载 {len(self.stopwords)} 个停用词")
        except Exception as e:
            logger.error(f"加载停用词失败: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        清洗单个文本
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 1. 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 2. 去除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 3. 去除邮箱
        text = re.sub(r'\S+@\S+', '', text)
        
        # 4. 去除特殊字符和数字，保留中文和英文
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', text)
        
        # 5. 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 6. 分词并去除停用词
        if text:
            words = jieba.lcut(text)
            cleaned_words = [word for word in words if word not in self.stopwords and len(word.strip()) > 0]
            text = ' '.join(cleaned_words)
        
        return text
    
    def filter_texts(self, df: pd.DataFrame, text_column: str = 'content') -> pd.DataFrame:
        """
        过滤文本数据
        
        Args:
            df: 包含文本的DataFrame
            text_column: 文本列名
            
        Returns:
            过滤后的DataFrame
        """
        logger.info(f"开始过滤文本数据...")
        
        # 计算文本长度
        df['text_length'] = df[text_column].str.len()
        
        # 记录过滤前的数量
        original_count = len(df)
        
        # 按长度过滤
        df_filtered = df[
            (df['text_length'] >= self.config['min_text_length']) &
            (df['text_length'] <= self.config['max_text_length'])
        ]
        
        # 移除空文本
        if self.config['remove_empty']:
            df_filtered = df_filtered[df_filtered[text_column].str.strip() != '']
        
        # 统计过滤效果
        filtered_count = len(df_filtered)
        removed_count = original_count - filtered_count
        
        logger.info(f"过滤完成！")
        logger.info(f"原始数据量: {original_count}")
        logger.info(f"过滤后数据量: {filtered_count}")
        logger.info(f"移除数据量: {removed_count} ({removed_count/original_count*100:.1f}%)")
        
        return df_filtered
    
    def split_data(self, df: pd.DataFrame, text_column: str = 'content') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        切分数据集为训练集、验证集、测试集（不分层采样，始终随机划分）
        
        Args:
            df: 包含文本的DataFrame
            text_column: 文本列名
            
        Returns:
            训练集、验证集、测试集
        """
        logger.info(f"开始切分数据集...（不分层采样，始终随机划分）")
        
        # 确保比例总和为1
        total_ratio = self.config['train_ratio'] + self.config['val_ratio'] + self.config['test_ratio']
        if abs(total_ratio - 1.0) > 1e-6:
            logger.warning(f"数据集比例总和不为1，已自动调整")
            # 重新计算比例
            self.config['train_ratio'] = self.config['train_ratio'] / total_ratio
            self.config['val_ratio'] = self.config['val_ratio'] / total_ratio
            self.config['test_ratio'] = self.config['test_ratio'] / total_ratio
        
        # 随机采样切分
        train_df, temp_df = train_test_split(
            df, 
            test_size=(1 - self.config['train_ratio']),
            random_state=self.random_seed,
            shuffle=True
        )
        val_ratio_adjusted = self.config['val_ratio'] / (self.config['val_ratio'] + self.config['test_ratio'])
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio_adjusted),
            random_state=self.random_seed,
            shuffle=True
        )
        
        logger.info(f"数据集切分完成！")
        logger.info(f"训练集: {len(train_df)} 样本 ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"验证集: {len(val_df)} 样本 ({len(val_df)/len(df)*100:.1f}%)")
        logger.info(f"测试集: {len(test_df)} 样本 ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def get_class_distribution(self, df: pd.DataFrame) -> Dict:
        """
        获取类别分布
        
        Args:
            df: DataFrame
            
        Returns:
            类别分布字典
        """
        if 'category' not in df.columns:
            return {}
        
        distribution = df['category'].value_counts().to_dict()
        return distribution
    
    def preprocess_pipeline(self, input_file: str, output_dir: str = "results") -> Dict:
        """
        完整的数据预处理流程
        
        Args:
            input_file: 输入文件路径
            output_dir: 输出目录
            
        Returns:
            预处理统计信息
        """
        logger.info(f"开始数据预处理流程...")
        start_time = time.time()
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 读取数据
        logger.info(f"读取数据文件: {input_file}")
        df = pd.read_csv(input_file, encoding='utf-8')
        logger.info(f"原始数据量: {len(df)} 条")
        
        # 2. 数据清洗
        logger.info("开始数据清洗...")
        df['content_cleaned'] = df['content'].apply(self.clean_text)
        
        # 3. 数据过滤
        df_filtered = self.filter_texts(df, 'content_cleaned')
        
        # 4. 数据集切分
        train_df, val_df, test_df = self.split_data(df_filtered, 'content_cleaned')
        
        # 5. 保存处理结果
        # 保存完整清洗后的数据
        cleaned_file = output_path / "data_cleaned.csv"
        df_filtered.to_csv(cleaned_file, index=False, encoding='utf-8')
        
        # 保存训练集
        train_file = output_path / "train.csv"
        train_df.to_csv(train_file, index=False, encoding='utf-8')
        
        # 保存验证集
        val_file = output_path / "val.csv"
        val_df.to_csv(val_file, index=False, encoding='utf-8')
        
        # 保存测试集
        test_file = output_path / "test.csv"
        test_df.to_csv(test_file, index=False, encoding='utf-8')
        
        # 6. 生成统计信息
        stats = {
            'original_count': len(df),
            'cleaned_count': len(df_filtered),
            'train_count': len(train_df),
            'val_count': len(val_df),
            'test_count': len(test_df),
            'removed_count': len(df) - len(df_filtered),
            'processing_time': time.time() - start_time,
            'random_seed': self.random_seed,
            'config': self.config,
            'class_distribution': {
                'original': self.get_class_distribution(df),
                'cleaned': self.get_class_distribution(df_filtered),
                'train': self.get_class_distribution(train_df),
                'val': self.get_class_distribution(val_df),
                'test': self.get_class_distribution(test_df)
            }
        }
        
        # 保存统计信息
        stats_file = output_path / "preprocessing_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # 7. 输出统计信息
        self.print_statistics(stats)
        
        logger.info(f"数据预处理完成！总耗时: {stats['processing_time']:.2f} 秒")
        logger.info(f"结果保存在: {output_path}")
        
        return stats
    
    def print_statistics(self, stats: Dict):
        """打印统计信息"""
        print("\n" + "="*60)
        print("数据预处理统计信息")
        print("="*60)
        print(f"原始数据量: {stats['original_count']}")
        print(f"清洗后数据量: {stats['cleaned_count']}")
        print(f"移除数据量: {stats['removed_count']} ({stats['removed_count']/stats['original_count']*100:.1f}%)")
        print(f"训练集: {stats['train_count']} 样本")
        print(f"验证集: {stats['val_count']} 样本")
        print(f"测试集: {stats['test_count']} 样本")
        print(f"处理时间: {stats['processing_time']:.2f} 秒")
        print(f"随机种子: {stats['random_seed']}")
        
        print("\n类别分布:")
        for split_name, distribution in stats['class_distribution'].items():
            print(f"\n{split_name.upper()}:")
            for category, count in sorted(distribution.items()):
                print(f"  {category}: {count}")
        
        print("="*60)


def main():
    """主函数"""
    # 创建数据预处理器
    preprocessor = DataPreprocessor(random_seed=42)
    
    # 处理样本数据
    input_file = "sample.csv"
    
    if Path(input_file).exists():
        stats = preprocessor.preprocess_pipeline(input_file)
        print(f"\n预处理完成！结果保存在 results/ 目录下")
    else:
        logger.warning(f"输入文件 {input_file} 不存在，请先运行数据抽取脚本")


if __name__ == "__main__":
    main() 