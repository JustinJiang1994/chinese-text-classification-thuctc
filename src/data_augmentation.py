#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强模块
实现多种文本数据增强技术
"""

import random
import re
import jieba
import pandas as pd
import numpy as np
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import nlpaug.augmenter.word as naw
    import nlpaug.augmenter.sentence as nas
    from nlpaug.util import Action
except ImportError:
    print("警告: nlpaug未安装，部分数据增强功能不可用")
    naw = None
    nas = None

try:
    import synonyms
except ImportError:
    print("警告: synonyms未安装，同义词替换功能不可用")
    synonyms = None


class TextAugmenter:
    """文本数据增强器"""
    
    def __init__(self, stopwords_file='data/stopwords.txt'):
        """
        初始化文本增强器
        
        Args:
            stopwords_file: 停用词文件路径
        """
        self.stopwords = self.load_stopwords(stopwords_file)
        
    def load_stopwords(self, stopwords_file):
        """加载停用词"""
        try:
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                stopwords = set([line.strip() for line in f])
            return stopwords
        except FileNotFoundError:
            print(f"警告: 停用词文件 {stopwords_file} 不存在，使用空停用词表")
            return set()
    
    def random_deletion(self, text: str, deletion_ratio: float = 0.1) -> str:
        """
        随机删除增强
        
        Args:
            text: 输入文本
            deletion_ratio: 删除比例
            
        Returns:
            增强后的文本
        """
        words = jieba.lcut(text)
        if len(words) <= 3:  # 如果词数太少，不进行删除
            return text
        
        n_deletion = max(1, int(len(words) * deletion_ratio))
        words_to_delete = random.sample(range(len(words)), n_deletion)
        
        # 删除选中的词
        augmented_words = [word for i, word in enumerate(words) if i not in words_to_delete]
        
        return ''.join(augmented_words)
    
    def random_insertion(self, text: str, insertion_ratio: float = 0.1) -> str:
        """
        随机插入增强
        
        Args:
            text: 输入文本
            insertion_ratio: 插入比例
            
        Returns:
            增强后的文本
        """
        words = jieba.lcut(text)
        if len(words) == 0:
            return text
        
        n_insertion = max(1, int(len(words) * insertion_ratio))
        
        # 随机选择要插入的词
        words_to_insert = random.choices(words, k=n_insertion)
        
        # 随机选择插入位置
        for word_to_insert in words_to_insert:
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, word_to_insert)
        
        return ''.join(words)
    
    def random_swap(self, text: str, swap_ratio: float = 0.1) -> str:
        """
        随机交换增强
        
        Args:
            text: 输入文本
            swap_ratio: 交换比例
            
        Returns:
            增强后的文本
        """
        words = jieba.lcut(text)
        if len(words) <= 1:
            return text
        
        n_swap = max(1, int(len(words) * swap_ratio))
        
        for _ in range(n_swap):
            if len(words) >= 2:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ''.join(words)
    
    def synonym_replacement(self, text: str, replacement_ratio: float = 0.1) -> str:
        """
        同义词替换增强
        
        Args:
            text: 输入文本
            replacement_ratio: 替换比例
            
        Returns:
            增强后的文本
        """
        if synonyms is None:
            return text
        
        words = jieba.lcut(text)
        if len(words) == 0:
            return text
        
        n_replacement = max(1, int(len(words) * replacement_ratio))
        
        for _ in range(n_replacement):
            if len(words) > 0:
                idx = random.randint(0, len(words) - 1)
                word = words[idx]
                
                try:
                    # 获取同义词
                    synonyms_list = synonyms.nearby(word)[0]
                    if len(synonyms_list) > 1:
                        # 随机选择一个同义词
                        synonym = random.choice(synonyms_list[1:])
                        words[idx] = synonym
                except:
                    continue
        
        return ''.join(words)
    
    def back_translation_augmentation(self, text: str) -> str:
        """
        回译增强（简化版）
        
        Args:
            text: 输入文本
            
        Returns:
            增强后的文本
        """
        # 这里使用简化的回译增强，实际项目中可以使用翻译API
        # 对于中文文本，我们可以使用一些常见的替换规则
        
        replacements = {
            '非常': ['很', '特别', '十分'],
            '重要': ['关键', '核心', '主要'],
            '发展': ['进展', '推进', '促进'],
            '问题': ['难题', '挑战', '困难'],
            '解决': ['处理', '应对', '克服'],
            '提高': ['提升', '增强', '改善'],
            '增加': ['增长', '上升', '提升'],
            '减少': ['降低', '下降', '削减'],
            '支持': ['帮助', '协助', '援助'],
            '反对': ['抵制', '拒绝', '排斥']
        }
        
        augmented_text = text
        for original, alternatives in replacements.items():
            if original in augmented_text:
                replacement = random.choice(alternatives)
                augmented_text = augmented_text.replace(original, replacement, 1)
        
        return augmented_text
    
    def nlpaug_synonym_replacement(self, text: str, replacement_ratio: float = 0.1) -> str:
        """
        使用nlpaug进行同义词替换
        
        Args:
            text: 输入文本
            replacement_ratio: 替换比例
            
        Returns:
            增强后的文本
        """
        if naw is None:
            return text
        
        try:
            aug = naw.SynonymAug(aug_src='wordnet', aug_p=replacement_ratio)
            augmented_text = aug.augment(text)[0]
            return augmented_text
        except:
            return text
    
    def nlpaug_contextual_replacement(self, text: str, replacement_ratio: float = 0.1) -> str:
        """
        使用nlpaug进行上下文替换
        
        Args:
            text: 输入文本
            replacement_ratio: 替换比例
            
        Returns:
            增强后的文本
        """
        if naw is None:
            return text
        
        try:
            aug = naw.ContextualWordEmbsAug(
                model_path='bert-base-chinese',
                action="substitute",
                aug_p=replacement_ratio
            )
            augmented_text = aug.augment(text)[0]
            return augmented_text
        except:
            return text
    
    def augment_text(self, text: str, methods: List[str] = None, 
                    augmentation_ratio: float = 0.1) -> List[str]:
        """
        对单个文本进行多种增强
        
        Args:
            text: 输入文本
            methods: 增强方法列表
            augmentation_ratio: 增强比例
            
        Returns:
            增强后的文本列表
        """
        if methods is None:
            methods = ['random_deletion', 'random_insertion', 'random_swap', 'synonym_replacement']
        
        augmented_texts = []
        
        for method in methods:
            try:
                if method == 'random_deletion':
                    augmented_text = self.random_deletion(text, augmentation_ratio)
                elif method == 'random_insertion':
                    augmented_text = self.random_insertion(text, augmentation_ratio)
                elif method == 'random_swap':
                    augmented_text = self.random_swap(text, augmentation_ratio)
                elif method == 'synonym_replacement':
                    augmented_text = self.synonym_replacement(text, augmentation_ratio)
                elif method == 'back_translation':
                    augmented_text = self.back_translation_augmentation(text)
                elif method == 'nlpaug_synonym':
                    augmented_text = self.nlpaug_synonym_replacement(text, augmentation_ratio)
                elif method == 'nlpaug_contextual':
                    augmented_text = self.nlpaug_contextual_replacement(text, augmentation_ratio)
                else:
                    continue
                
                # 确保增强后的文本不为空且与原文不同
                if augmented_text and augmented_text != text and len(augmented_text) > 10:
                    augmented_texts.append(augmented_text)
                    
            except Exception as e:
                print(f"增强方法 {method} 失败: {e}")
                continue
        
        return augmented_texts
    
    def balance_dataset(self, df: pd.DataFrame, target_samples_per_class: int = 5000) -> pd.DataFrame:
        """
        平衡数据集，对少数类别进行数据增强
        
        Args:
            df: 输入数据框
            target_samples_per_class: 每个类别的目标样本数
            
        Returns:
            平衡后的数据框
        """
        print("正在平衡数据集...")
        
        balanced_data = []
        
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            current_samples = len(category_df)
            
            print(f"类别 '{category}': {current_samples} 个样本")
            
            if current_samples >= target_samples_per_class:
                # 如果样本数足够，随机采样
                balanced_df = category_df.sample(n=target_samples_per_class, random_state=42)
                balanced_data.append(balanced_df)
            else:
                # 如果样本数不足，进行数据增强
                needed_samples = target_samples_per_class - current_samples
                print(f"  需要增强 {needed_samples} 个样本")
                
                # 先添加原始样本
                balanced_data.append(category_df)
                
                # 对原始样本进行增强
                augmented_samples = []
                texts = category_df['content'].tolist()
                
                for text in texts:
                    if len(augmented_samples) >= needed_samples:
                        break
                    
                    # 对每个文本进行多种增强
                    augmented_texts = self.augment_text(
                        text, 
                        methods=['random_deletion', 'random_insertion', 'synonym_replacement'],
                        augmentation_ratio=0.1
                    )
                    
                    for aug_text in augmented_texts:
                        if len(augmented_samples) >= needed_samples:
                            break
                        augmented_samples.append({
                            'content': aug_text,
                            'category': category
                        })
                
                # 如果增强的样本还不够，重复一些样本
                if len(augmented_samples) < needed_samples:
                    remaining = needed_samples - len(augmented_samples)
                    additional_samples = category_df.sample(n=min(remaining, len(category_df)), random_state=42)
                    
                    for _, row in additional_samples.iterrows():
                        if len(augmented_samples) >= needed_samples:
                            break
                        augmented_samples.append({
                            'content': row['content'],
                            'category': row['category']
                        })
                
                # 创建增强样本的数据框
                if augmented_samples:
                    augmented_df = pd.DataFrame(augmented_samples)
                    balanced_data.append(augmented_df)
                    print(f"  生成了 {len(augmented_samples)} 个增强样本")
        
        # 合并所有数据
        balanced_df = pd.concat(balanced_data, ignore_index=True)
        
        print(f"平衡后总样本数: {len(balanced_df)}")
        print("各类别样本数:")
        for category in balanced_df['category'].unique():
            count = len(balanced_df[balanced_df['category'] == category])
            print(f"  {category}: {count}")
        
        return balanced_df
    
    def augment_minority_classes(self, df: pd.DataFrame, 
                               augmentation_ratio: float = 0.5) -> pd.DataFrame:
        """
        只对少数类别进行数据增强
        
        Args:
            df: 输入数据框
            augmentation_ratio: 增强比例
            
        Returns:
            增强后的数据框
        """
        print("正在对少数类别进行数据增强...")
        
        # 计算每个类别的样本数
        class_counts = df['category'].value_counts()
        median_count = class_counts.median()
        
        print(f"中位数样本数: {median_count}")
        
        augmented_data = []
        
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            current_count = len(category_df)
            
            print(f"类别 '{category}': {current_count} 个样本")
            
            # 添加原始样本
            augmented_data.append(category_df)
            
            # 如果样本数少于中位数，进行增强
            if current_count < median_count:
                needed_samples = int((median_count - current_count) * augmentation_ratio)
                print(f"  需要增强 {needed_samples} 个样本")
                
                texts = category_df['content'].tolist()
                augmented_samples = []
                
                for text in texts:
                    if len(augmented_samples) >= needed_samples:
                        break
                    
                    augmented_texts = self.augment_text(
                        text,
                        methods=['random_deletion', 'random_insertion', 'synonym_replacement'],
                        augmentation_ratio=0.1
                    )
                    
                    for aug_text in augmented_texts:
                        if len(augmented_samples) >= needed_samples:
                            break
                        augmented_samples.append({
                            'content': aug_text,
                            'category': category
                        })
                
                if augmented_samples:
                    augmented_df = pd.DataFrame(augmented_samples)
                    augmented_data.append(augmented_df)
                    print(f"  生成了 {len(augmented_samples)} 个增强样本")
        
        # 合并数据
        augmented_df = pd.concat(augmented_data, ignore_index=True)
        
        print(f"增强后总样本数: {len(augmented_df)}")
        return augmented_df


def test_data_augmentation():
    """测试数据增强功能"""
    print("测试数据增强功能...")
    
    # 创建测试文本
    test_text = "这是一个非常重要的新闻，我们需要认真处理这个问题。"
    
    # 创建增强器
    augmenter = TextAugmenter()
    
    # 测试各种增强方法
    print(f"原始文本: {test_text}")
    print()
    
    methods = [
        ('random_deletion', '随机删除'),
        ('random_insertion', '随机插入'),
        ('random_swap', '随机交换'),
        ('synonym_replacement', '同义词替换'),
        ('back_translation', '回译增强')
    ]
    
    for method, description in methods:
        try:
            if method == 'random_deletion':
                result = augmenter.random_deletion(test_text)
            elif method == 'random_insertion':
                result = augmenter.random_insertion(test_text)
            elif method == 'random_swap':
                result = augmenter.random_swap(test_text)
            elif method == 'synonym_replacement':
                result = augmenter.synonym_replacement(test_text)
            elif method == 'back_translation':
                result = augmenter.back_translation_augmentation(test_text)
            
            print(f"{description}: {result}")
        except Exception as e:
            print(f"{description}: 失败 - {e}")
    
    print()
    
    # 测试多种增强
    print("多种增强方法组合:")
    augmented_texts = augmenter.augment_text(test_text)
    for i, aug_text in enumerate(augmented_texts, 1):
        print(f"  增强{i}: {aug_text}")


if __name__ == "__main__":
    test_data_augmentation() 