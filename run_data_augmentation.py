#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行数据增强
"""

import sys
import os
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_augmentation import TextAugmenter

def main():
    """主函数"""
    print("=" * 60)
    print("数据增强工具")
    print("=" * 60)
    
    # 检查数据文件是否存在
    data_file = 'data/news_data.csv'
    if not os.path.exists(data_file):
        print(f"错误: 数据文件 {data_file} 不存在")
        print("请先运行 python merge_data.py 合并数据文件")
        return
    
    # 加载数据
    print("正在加载数据...")
    df = pd.read_csv(data_file)
    print(f"原始数据量: {len(df)} 条")
    
    # 显示原始数据分布
    print("\n原始数据分布:")
    class_counts = df['category'].value_counts()
    for category, count in class_counts.items():
        print(f"  {category}: {count}")
    
    # 创建数据增强器
    augmenter = TextAugmenter()
    
    # 选择增强策略
    print("\n请选择数据增强策略:")
    print("1. 平衡数据集 (每个类别5000条样本)")
    print("2. 只对少数类别进行增强")
    print("3. 测试数据增强功能")
    
    choice = input("请输入选择 (1-3): ").strip()
    
    if choice == '1':
        # 平衡数据集
        print("\n正在平衡数据集...")
        balanced_df = augmenter.balance_dataset(df, target_samples_per_class=5000)
        
        # 保存平衡后的数据
        output_file = 'data/news_data_balanced.csv'
        balanced_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n平衡后的数据已保存到: {output_file}")
        
    elif choice == '2':
        # 只对少数类别进行增强
        print("\n正在对少数类别进行数据增强...")
        augmented_df = augmenter.augment_minority_classes(df, augmentation_ratio=0.5)
        
        # 保存增强后的数据
        output_file = 'data/news_data_augmented.csv'
        augmented_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n增强后的数据已保存到: {output_file}")
        
    elif choice == '3':
        # 测试数据增强功能
        print("\n正在测试数据增强功能...")
        test_data_augmentation()
        
    else:
        print("无效选择")
        return
    
    print("\n数据增强完成!")


def test_data_augmentation():
    """测试数据增强功能"""
    from src.data_augmentation import test_data_augmentation
    test_data_augmentation()


if __name__ == "__main__":
    main() 