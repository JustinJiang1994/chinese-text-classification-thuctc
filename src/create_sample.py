#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建样本数据文件
从完整数据集中抽取少量样本，用于GitHub示范
"""

import pandas as pd
import random
from pathlib import Path

def create_sample_data(input_file: str = "results/extracted_news_data.csv", 
                      output_file: str = "sample.csv",
                      samples_per_category: int = 5):
    """
    从完整数据集中创建样本文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        samples_per_category: 每个类别抽取的样本数量
    """
    
    print(f"正在从 {input_file} 创建样本文件...")
    
    # 读取完整数据
    df = pd.read_csv(input_file, encoding='utf-8')
    
    # 按类别分组并随机抽取样本
    sample_data = []
    categories = df['category'].unique()
    
    for category in categories:
        category_data = df[df['category'] == category]
        
        # 如果该类别的数据少于指定数量，则全部抽取
        if len(category_data) <= samples_per_category:
            samples = category_data
        else:
            # 随机抽取指定数量的样本
            samples = category_data.sample(n=samples_per_category, random_state=42)
        
        sample_data.append(samples)
    
    # 合并所有样本
    sample_df = pd.concat(sample_data, ignore_index=True)
    
    # 保存样本文件
    sample_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"样本文件已保存到: {output_file}")
    print(f"总样本数: {len(sample_df)}")
    print(f"类别数: {len(categories)}")
    print(f"每个类别样本数: {samples_per_category}")
    
    # 显示类别分布
    print("\n类别分布:")
    category_counts = sample_df['category'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count} 个样本")
    
    # 显示文件大小
    file_size = Path(output_file).stat().st_size
    print(f"\n文件大小: {file_size / 1024:.1f} KB")

if __name__ == "__main__":
    create_sample_data() 