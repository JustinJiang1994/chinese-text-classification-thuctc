#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM分类器运行脚本
用于训练和评估基于LSTM的深度学习分类器
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 添加src目录到路径
sys.path.append('src')

from lstm_classifier import LSTMClassifier

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def main():
    """主函数"""
    print("=" * 60)
    print("LSTM分类器训练和评估")
    print("=" * 60)
    
    # 记录开始时间
    start_time = time.time()
    
    # 创建LSTM分类器实例
    print("初始化LSTM分类器...")
    classifier = LSTMClassifier(
        max_words=10000,      # 词汇表大小
        max_len=200,          # 序列最大长度
        embedding_dim=128,    # 词嵌入维度
        lstm_units=128        # LSTM单元数
    )
    
    # 运行完整流程
    print("\n开始训练和评估流程...")
    accuracy = classifier.run_full_pipeline(
        data_file='data/news_data.csv',
        stopwords_file='data/stopwords.txt',
        test_size=0.2,
        random_state=42
    )
    
    # 记录结束时间
    end_time = time.time()
    training_time = end_time - start_time
    
    if accuracy is not None:
        print("\n" + "=" * 60)
        print("LSTM分类器结果总结")
        print("=" * 60)
        print(f"最终准确率: {accuracy:.4f}")
        print(f"训练时间: {training_time:.2f} 秒")
        print(f"模型已保存到: models/lstm_model.pkl")
        print(f"结果已保存到: results/lstm_results.pkl")
        print(f"训练历史图: results/lstm_training_history.png")
        print(f"混淆矩阵图: results/lstm_confusion_matrix.png")
        
        # 保存详细结果
        save_detailed_results(accuracy, training_time, classifier)
        
    else:
        print("训练流程执行失败")


def save_detailed_results(accuracy, training_time, classifier):
    """保存详细结果"""
    results = {
        'model_name': 'LSTM',
        'accuracy': accuracy,
        'training_time': training_time,
        'model_params': {
            'max_words': classifier.max_words,
            'max_len': classifier.max_len,
            'embedding_dim': classifier.embedding_dim,
            'lstm_units': classifier.lstm_units
        },
        'class_names': classifier.class_names
    }
    
    # 保存到文本文件
    results_file = 'results/lstm_detailed_results.txt'
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("LSTM分类器详细结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"模型名称: LSTM\n")
        f.write(f"准确率: {accuracy:.4f}\n")
        f.write(f"训练时间: {training_time:.2f} 秒\n\n")
        f.write("模型参数:\n")
        f.write(f"  词汇表大小: {classifier.max_words}\n")
        f.write(f"  序列最大长度: {classifier.max_len}\n")
        f.write(f"  词嵌入维度: {classifier.embedding_dim}\n")
        f.write(f"  LSTM单元数: {classifier.lstm_units}\n\n")
        f.write("类别名称:\n")
        for i, class_name in enumerate(classifier.class_names):
            f.write(f"  {i}: {class_name}\n")
    
    print(f"详细结果已保存到: {results_file}")


def test_model():
    """测试已训练的模型"""
    print("=" * 60)
    print("测试LSTM模型")
    print("=" * 60)
    
    # 创建分类器实例
    classifier = LSTMClassifier()
    
    # 加载模型
    if not classifier.load_model():
        print("无法加载模型，请先训练模型")
        return
    
    # 加载停用词
    stopwords = classifier.load_stopwords()
    
    # 测试文本
    test_texts = [
        "今天股市大涨，投资者信心增强",
        "新款手机发布，性能提升明显",
        "电影票房创新高，观众反响热烈",
        "科技公司发布新产品，市场反应积极"
    ]
    
    print("\n测试预测:")
    print("-" * 40)
    
    for i, text in enumerate(test_texts, 1):
        predicted_label, confidence, probabilities = classifier.predict(text, stopwords)
        if predicted_label is not None:
            print(f"文本 {i}: {text}")
            print(f"预测类别: {predicted_label}")
            print(f"置信度: {confidence:.4f}")
            print(f"所有类别概率: {dict(zip(classifier.class_names, probabilities))}")
            print("-" * 40)


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_model()
    else:
        main() 