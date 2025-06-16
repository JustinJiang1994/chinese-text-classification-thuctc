#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据合并脚本
将各个类别的新闻数据合并成一个CSV文件
"""

import os
import pandas as pd
import glob

def merge_news_data():
    """合并新闻数据"""
    print("=" * 60)
    print("合并新闻数据")
    print("=" * 60)
    
    # 数据目录
    data_dir = 'data'
    
    # 获取所有新闻CSV文件
    news_files = glob.glob(os.path.join(data_dir, '*_news.csv'))
    
    if not news_files:
        print("未找到新闻数据文件")
        return False
    
    print(f"找到 {len(news_files)} 个新闻数据文件:")
    for file in news_files:
        print(f"  - {os.path.basename(file)}")
    
    # 合并数据
    all_data = []
    
    for file in news_files:
        print(f"\n正在处理: {os.path.basename(file)}")
        
        try:
            # 读取数据
            df = pd.read_csv(file)
            print(f"  读取 {len(df)} 条记录")
            
            # 从文件名提取类别
            category = os.path.basename(file).replace('_news.csv', '')
            
            # 确保有content列
            if 'content' not in df.columns:
                print(f"  警告: {file} 没有content列，跳过")
                continue
            
            # 添加类别列
            df['category'] = category
            
            # 选择需要的列
            if 'title' in df.columns:
                df = df[['title', 'content', 'category']]
            else:
                df = df[['content', 'category']]
            
            all_data.append(df)
            
        except Exception as e:
            print(f"  错误: 处理 {file} 时出错 - {e}")
            continue
    
    if not all_data:
        print("没有成功读取任何数据文件")
        return False
    
    # 合并所有数据
    print("\n正在合并数据...")
    merged_df = pd.concat(all_data, ignore_index=True)
    
    print(f"合并完成，总共 {len(merged_df)} 条记录")
    print(f"类别分布:")
    print(merged_df['category'].value_counts())
    
    # 保存合并后的数据
    output_file = os.path.join(data_dir, 'news_data.csv')
    merged_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\n合并后的数据已保存到: {output_file}")
    
    return True

if __name__ == "__main__":
    merge_news_data() 