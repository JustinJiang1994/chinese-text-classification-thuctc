#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速传统机器学习实验
用于在小样本数据上测试算法效果
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import jieba
import re

def load_sample_data():
    """加载小样本数据进行快速测试"""
    print("加载小样本数据...")
    
    # 使用sample.csv进行快速测试
    sample_file = Path("sample.csv")
    if sample_file.exists():
        data = pd.read_csv(sample_file)
        print(f"加载了 {len(data)} 个样本")
        return data
    else:
        print("未找到sample.csv，使用完整数据集的前1000个样本")
        data = pd.read_csv("results/train.csv", nrows=1000)
        return data

def preprocess_text(text):
    """文本预处理"""
    if pd.isna(text):
        return ""
    
    # 移除特殊字符和数字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', str(text))
    # 分词
    words = jieba.cut(text)
    return ' '.join(words)

def quick_experiment():
    """快速实验"""
    print("=== 快速传统机器学习实验 ===")
    
    # 1. 加载数据
    data = load_sample_data()
    
    # 2. 显示类别分布
    print("\n类别分布:")
    category_counts = data['category'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    
    # 3. 文本预处理
    print("\n正在预处理文本...")
    data['processed_content'] = data['content'].apply(preprocess_text)
    
    # 4. 特征提取
    print("正在提取TF-IDF特征...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9
    )
    
    X = vectorizer.fit_transform(data['processed_content'])
    print(f"特征维度: {X.shape[1]}")
    
    # 5. 标签编码
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['category'])
    print(f"类别数量: {len(label_encoder.classes_)}")
    
    # 6. 简单训练测试分割
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # 7. 创建模型
    models = {
        '朴素贝叶斯': MultinomialNB(alpha=1.0),
        'SVM': LinearSVC(C=1.0, class_weight='balanced', random_state=42, max_iter=1000),
        '随机森林': RandomForestClassifier(n_estimators=50, max_depth=8, class_weight='balanced', random_state=42),
        '逻辑回归': LogisticRegression(C=1.0, class_weight='balanced', random_state=42, max_iter=1000)
    }
    
    # 8. 训练和评估
    results = {}
    
    for name, model in models.items():
        print(f"\n训练 {name}...")
        start_time = time.time()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        training_time = time.time() - start_time
        
        # 计算指标
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'training_time': training_time
        }
        
        print(f"  F1-macro: {f1_macro:.4f}")
        print(f"  F1-weighted: {f1_weighted:.4f}")
        print(f"  训练时间: {training_time:.2f}秒")
        
        # 显示详细分类报告
        print(f"\n{name} 详细分类报告:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # 9. 结果总结
    print("\n=== 实验结果总结 ===")
    print("模型性能排名 (按F1-macro):")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1_macro'], reverse=True)
    
    for i, (name, result) in enumerate(sorted_results, 1):
        print(f"{i}. {name}: F1-macro={result['f1_macro']:.4f}, F1-weighted={result['f1_weighted']:.4f}")
    
    return results

if __name__ == "__main__":
    quick_experiment() 