#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻数据抽取器
从data目录下的各个类别文件夹中抽取新闻文本，并为每个文件打上对应的类别标签
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsDataExtractor:
    """新闻数据抽取器"""
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化数据抽取器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.categories = []
        self.extracted_data = []
        
    def get_categories(self) -> List[str]:
        """
        获取所有新闻类别
        
        Returns:
            类别列表
        """
        if not self.categories:
            # 获取data目录下的所有文件夹（排除stopwords.txt等文件）
            self.categories = [
                folder.name for folder in self.data_dir.iterdir() 
                if folder.is_dir() and not folder.name.startswith('.')
            ]
            self.categories.sort()  # 排序确保一致性
            logger.info(f"发现 {len(self.categories)} 个新闻类别: {self.categories}")
        
        return self.categories
    
    def extract_text_from_file(self, file_path: Path) -> str:
        """
        从单个文件中提取文本内容
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件内容
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            return content
        except Exception as e:
            logger.warning(f"读取文件 {file_path} 失败: {e}")
            return ""
    
    def extract_category_data(self, category: str) -> List[Dict]:
        """
        抽取单个类别的所有数据
        
        Args:
            category: 类别名称
            
        Returns:
            该类别的数据列表
        """
        category_dir = self.data_dir / category
        if not category_dir.exists():
            logger.warning(f"类别目录不存在: {category_dir}")
            return []
        
        category_data = []
        txt_files = list(category_dir.glob("*.txt"))
        
        logger.info(f"开始抽取类别 '{category}' 的数据，共 {len(txt_files)} 个文件")
        
        for file_path in txt_files:
            content = self.extract_text_from_file(file_path)
            if content:  # 只添加非空内容
                category_data.append({
                    'file_id': file_path.stem,  # 文件名（不含扩展名）
                    'category': category,
                    'content': content,
                    'file_path': str(file_path)
                })
        
        logger.info(f"类别 '{category}' 抽取完成，有效文件 {len(category_data)} 个")
        return category_data
    
    def extract_all_data(self) -> List[Dict]:
        """
        抽取所有类别的数据
        
        Returns:
            所有数据的列表
        """
        categories = self.get_categories()
        all_data = []
        
        for category in categories:
            category_data = self.extract_category_data(category)
            all_data.extend(category_data)
        
        self.extracted_data = all_data
        logger.info(f"数据抽取完成，总共 {len(all_data)} 个文件")
        return all_data
    
    def save_to_json(self, output_file: str = "results/extracted_news_data.json"):
        """
        将抽取的数据保存为JSON格式
        
        Args:
            output_file: 输出文件路径
        """
        if not self.extracted_data:
            logger.warning("没有数据可保存，请先调用 extract_all_data()")
            return
        
        # 创建输出目录
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存数据
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.extracted_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据已保存到: {output_path}")
    
    def save_to_csv(self, output_file: str = "results/extracted_news_data.csv"):
        """
        将抽取的数据保存为CSV格式（仅包含类别标签和新闻内容）
        
        Args:
            output_file: 输出文件路径
        """
        if not self.extracted_data:
            logger.warning("没有数据可保存，请先调用 extract_all_data()")
            return
        
        # 创建输出目录
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 只保留category和content字段
        simplified_data = []
        for item in self.extracted_data:
            simplified_data.append({
                'category': item['category'],
                'content': item['content']
            })
        
        # 转换为DataFrame并保存
        df = pd.DataFrame(simplified_data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"数据已保存到: {output_path} (仅包含类别和内容)")
    
    def get_statistics(self) -> Dict:
        """
        获取数据统计信息
        
        Returns:
            统计信息字典
        """
        if not self.extracted_data:
            logger.warning("没有数据可统计，请先调用 extract_all_data()")
            return {}
        
        # 按类别统计
        category_counts = {}
        total_chars = 0
        
        for item in self.extracted_data:
            category = item['category']
            content = item['content']
            
            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += 1
            
            total_chars += len(content)
        
        stats = {
            'total_files': len(self.extracted_data),
            'total_categories': len(category_counts),
            'total_characters': total_chars,
            'avg_chars_per_file': total_chars / len(self.extracted_data) if self.extracted_data else 0,
            'category_distribution': category_counts
        }
        
        return stats
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        if not stats:
            return
        
        print("\n" + "="*50)
        print("数据统计信息")
        print("="*50)
        print(f"总文件数: {stats['total_files']}")
        print(f"总类别数: {stats['total_categories']}")
        print(f"总字符数: {stats['total_characters']:,}")
        print(f"平均每文件字符数: {stats['avg_chars_per_file']:.1f}")
        print("\n类别分布:")
        for category, count in sorted(stats['category_distribution'].items()):
            percentage = (count / stats['total_files']) * 100
            print(f"  {category}: {count} 个文件 ({percentage:.1f}%)")
        print("="*50)


def main():
    """主函数"""
    # 创建数据抽取器
    extractor = NewsDataExtractor()
    
    # 抽取所有数据
    logger.info("开始抽取新闻数据...")
    extractor.extract_all_data()
    
    # 打印统计信息
    extractor.print_statistics()
    
    # 保存数据
    extractor.save_to_json()
    extractor.save_to_csv()
    
    logger.info("数据抽取和保存完成！")


if __name__ == "__main__":
    main() 