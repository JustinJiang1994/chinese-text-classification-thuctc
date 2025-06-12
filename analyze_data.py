#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文新闻数据集详细统计分析（高频词已去除停用词）
"""

import pandas as pd
import numpy as np
import jieba
import os
from collections import Counter
import re

def load_stopwords(filepath='data/stopwords.txt'):
    stopwords = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                stopwords.add(line.strip())
    except Exception as e:
        print(f"加载停用词失败: {e}")
    return stopwords

def analyze_category(category, file_path, stopwords):
    """分析单个类别的数据"""
    try:
        df = pd.read_csv(file_path)
        df = df.dropna()
        
        # 基本统计
        total_news = len(df)
        
        # 文本长度统计
        char_lengths = df['content'].str.len()
        word_lengths = df['content'].apply(lambda x: len(list(jieba.cut(str(x)))))
        
        # 计算统计量
        stats = {
            'category': category,
            'total_news': total_news,
            'avg_chars': char_lengths.mean(),
            'median_chars': char_lengths.median(),
            'std_chars': char_lengths.std(),
            'min_chars': char_lengths.min(),
            'max_chars': char_lengths.max(),
            'avg_words': word_lengths.mean(),
            'median_words': word_lengths.median(),
            'std_words': word_lengths.std(),
            'min_words': word_lengths.min(),
            'max_words': word_lengths.max(),
            'q25_chars': char_lengths.quantile(0.25),
            'q75_chars': char_lengths.quantile(0.75),
            'q25_words': word_lengths.quantile(0.25),
            'q75_words': word_lengths.quantile(0.75)
        }
        
        # 词频统计（去除停用词，取前10个高频词）
        all_text = ' '.join(df['content'].astype(str))
        words = [w for w in jieba.cut(all_text) if w.strip() and w not in stopwords]
        word_freq = Counter(words)
        top_words = word_freq.most_common(10)
        
        return stats, top_words
        
    except Exception as e:
        print(f"分析 {category} 时出错: {e}")
        return None, None

def main():
    # 定义类别和文件路径
    categories = {
        '汽车': 'data/car_news.csv',
        '娱乐': 'data/entertainment_news.csv',
        '财经': 'data/finance_news.csv',
        '家居': 'data/home_news.csv',
        '房产': 'data/house_news.csv',
        '国际': 'data/international_news.csv',
        '军事': 'data/military_news.csv',
        '社会': 'data/society_news.csv',
        '体育': 'data/sports_news.csv',
        '科技': 'data/technology_news.csv'
    }
    
    print("中文新闻数据集详细统计分析（高频词已去除停用词）")
    print("=" * 80)
    
    stopwords = load_stopwords()
    all_stats = []
    all_top_words = {}
    
    # 分析每个类别
    for category, file_path in categories.items():
        print(f"\n正在分析 {category} 类别...")
        stats, top_words = analyze_category(category, file_path, stopwords)
        
        if stats:
            all_stats.append(stats)
            all_top_words[category] = top_words
            
            print(f"  ✓ 完成分析")
    
    # 生成详细报告
    print("\n" + "=" * 80)
    print("详细统计报告")
    print("=" * 80)
    
    # 1. 基本统计表
    print("\n1. 基本统计信息")
    print("-" * 80)
    print(f"{'类别':<6} {'新闻数':<8} {'平均字符':<10} {'中位字符':<10} {'平均词数':<10} {'中位词数':<10}")
    print("-" * 80)
    
    for stats in all_stats:
        print(f"{stats['category']:<6} {stats['total_news']:<8,} {stats['avg_chars']:<10.1f} {stats['median_chars']:<10.1f} {stats['avg_words']:<10.1f} {stats['median_words']:<10.1f}")
    
    # 2. 字符长度分布
    print("\n2. 字符长度分布")
    print("-" * 80)
    print(f"{'类别':<6} {'最小值':<8} {'Q25':<8} {'中位数':<8} {'Q75':<8} {'最大值':<8} {'标准差':<8}")
    print("-" * 80)
    
    for stats in all_stats:
        print(f"{stats['category']:<6} {stats['min_chars']:<8} {stats['q25_chars']:<8.0f} {stats['median_chars']:<8.0f} {stats['q75_chars']:<8.0f} {stats['max_chars']:<8} {stats['std_chars']:<8.1f}")
    
    # 3. 词数分布
    print("\n3. 词数分布")
    print("-" * 80)
    print(f"{'类别':<6} {'最小值':<8} {'Q25':<8} {'中位数':<8} {'Q75':<8} {'最大值':<8} {'标准差':<8}")
    print("-" * 80)
    
    for stats in all_stats:
        print(f"{stats['category']:<6} {stats['min_words']:<8} {stats['q25_words']:<8.0f} {stats['median_words']:<8.0f} {stats['q75_words']:<8.0f} {stats['max_words']:<8} {stats['std_words']:<8.1f}")
    
    # 4. 高频词统计（已去除停用词）
    print("\n4. 各类别高频词（前5个，已去除停用词）")
    print("-" * 80)
    
    for category, top_words in all_top_words.items():
        print(f"\n{category}:")
        for i, (word, count) in enumerate(top_words[:5], 1):
            print(f"  {i}. {word} ({count:,}次)")
    
    # 5. 总体统计
    print("\n5. 总体统计")
    print("-" * 80)
    
    total_news = sum(stats['total_news'] for stats in all_stats)
    avg_chars_all = np.mean([stats['avg_chars'] for stats in all_stats])
    avg_words_all = np.mean([stats['avg_words'] for stats in all_stats])
    
    print(f"总新闻数: {total_news:,}")
    print(f"平均字符数: {avg_chars_all:.1f}")
    print(f"平均词数: {avg_words_all:.1f}")
    print(f"类别数: {len(all_stats)}")
    
    # 6. 数据不平衡分析
    print("\n6. 数据不平衡分析")
    print("-" * 80)
    
    news_counts = [stats['total_news'] for stats in all_stats]
    max_count = max(news_counts)
    min_count = min(news_counts)
    imbalance_ratio = max_count / min_count
    
    print(f"最大类别数量: {max_count:,}")
    print(f"最小类别数量: {min_count:,}")
    print(f"不平衡比例: {imbalance_ratio:.2f}:1")
    
    # 保存详细报告到文件
    with open('detailed_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("中文新闻数据集详细统计分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 写入基本统计
        f.write("1. 基本统计信息\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'类别':<6} {'新闻数':<8} {'平均字符':<10} {'中位字符':<10} {'平均词数':<10} {'中位词数':<10}\n")
        f.write("-" * 80 + "\n")
        
        for stats in all_stats:
            f.write(f"{stats['category']:<6} {stats['total_news']:<8,} {stats['avg_chars']:<10.1f} {stats['median_chars']:<10.1f} {stats['avg_words']:<10.1f} {stats['median_words']:<10.1f}\n")
        
        # 写入高频词
        f.write("\n2. 各类别高频词（前10个）\n")
        f.write("-" * 80 + "\n")
        
        for category, top_words in all_top_words.items():
            f.write(f"\n{category}:\n")
            for i, (word, count) in enumerate(top_words[:10], 1):
                f.write(f"  {i}. {word} ({count:,}次)\n")
        
        # 写入总体统计
        f.write(f"\n3. 总体统计\n")
        f.write("-" * 80 + "\n")
        f.write(f"总新闻数: {total_news:,}\n")
        f.write(f"平均字符数: {avg_chars_all:.1f}\n")
        f.write(f"平均词数: {avg_words_all:.1f}\n")
        f.write(f"类别数: {len(all_stats)}\n")
        f.write(f"不平衡比例: {imbalance_ratio:.2f}:1\n")
    
    print(f"\n详细报告已保存到: detailed_analysis_report.txt")

if __name__ == "__main__":
    main() 