#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM分类器测试脚本
用于测试已训练的LSTM模型
"""

import os
import sys
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


def load_test_data(data_file='data/news_data.csv'):
    """加载测试数据"""
    try:
        df = pd.read_csv(data_file)
        print(f"测试数据加载成功，共 {len(df)} 条记录")
        return df
    except FileNotFoundError:
        print(f"错误: 数据文件 {data_file} 不存在")
        return None


def test_model_performance():
    """测试模型性能"""
    print("=" * 60)
    print("LSTM模型性能测试")
    print("=" * 60)
    
    # 创建分类器实例
    classifier = LSTMClassifier()
    
    # 加载模型
    if not classifier.load_model():
        print("无法加载模型，请先训练模型")
        return
    
    # 加载测试数据
    df = load_test_data()
    if df is None:
        return
    
    # 加载停用词
    stopwords = classifier.load_stopwords()
    
    # 预处理测试数据
    print("正在预处理测试数据...")
    df['processed_text'] = df['content'].apply(lambda x: classifier.preprocess_text(x, stopwords))
    df = df[df['processed_text'].str.len() > 0].reset_index(drop=True)
    
    # 随机选择一部分数据进行测试
    test_size = min(100, len(df))
    test_df = df.sample(n=test_size, random_state=42).reset_index(drop=True)
    
    print(f"测试样本数量: {len(test_df)}")
    
    # 进行预测
    print("正在进行预测...")
    predictions = []
    confidences = []
    true_labels = []
    
    for idx, row in test_df.iterrows():
        predicted_label, confidence, probabilities = classifier.predict(row['content'], stopwords)
        if predicted_label is not None:
            predictions.append(predicted_label)
            confidences.append(confidence)
            true_labels.append(row['category'])
    
    # 计算准确率
    accuracy = accuracy_score(true_labels, predictions)
    print(f"\n测试准确率: {accuracy:.4f}")
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(true_labels, predictions, 
                              target_names=classifier.class_names))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(true_labels, predictions, classifier.class_names)
    
    # 保存测试结果
    save_test_results(true_labels, predictions, confidences, accuracy, test_df)


def plot_confusion_matrix(true_labels, predictions, class_names, save_path='results/lstm_test_confusion_matrix.png'):
    """绘制测试混淆矩阵"""
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names,
               yticklabels=class_names)
    plt.title('LSTM分类器测试混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"测试混淆矩阵已保存到: {save_path}")


def save_test_results(true_labels, predictions, confidences, accuracy, test_df):
    """保存测试结果"""
    results = {
        'accuracy': accuracy,
        'predictions': predictions,
        'true_labels': true_labels,
        'confidences': confidences,
        'test_data': test_df
    }
    
    # 保存到pickle文件
    results_path = 'results/lstm_test_results.pkl'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    # 保存到文本文件
    text_path = 'results/lstm_test_results.txt'
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write("LSTM分类器测试结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"测试准确率: {accuracy:.4f}\n")
        f.write(f"测试样本数量: {len(test_df)}\n\n")
        
        f.write("详细预测结果:\n")
        f.write("-" * 40 + "\n")
        for i, (true_label, pred_label, confidence) in enumerate(zip(true_labels, predictions, confidences)):
            f.write(f"样本 {i+1}:\n")
            f.write(f"  真实类别: {true_label}\n")
            f.write(f"  预测类别: {pred_label}\n")
            f.write(f"  置信度: {confidence:.4f}\n")
            f.write(f"  预测正确: {'✓' if true_label == pred_label else '✗'}\n")
            f.write("-" * 40 + "\n")
    
    print(f"测试结果已保存到: {results_path}")
    print(f"详细结果已保存到: {text_path}")


def interactive_test():
    """交互式测试"""
    print("=" * 60)
    print("LSTM模型交互式测试")
    print("=" * 60)
    
    # 创建分类器实例
    classifier = LSTMClassifier()
    
    # 加载模型
    if not classifier.load_model():
        print("无法加载模型，请先训练模型")
        return
    
    # 加载停用词
    stopwords = classifier.load_stopwords()
    
    print("请输入要分类的新闻文本（输入'quit'退出）:")
    print("-" * 40)
    
    while True:
        text = input("请输入文本: ").strip()
        
        if text.lower() in ['quit', 'exit', '退出']:
            break
        
        if not text:
            continue
        
        # 进行预测
        predicted_label, confidence, probabilities = classifier.predict(text, stopwords)
        
        if predicted_label is not None:
            print(f"\n预测结果:")
            print(f"  预测类别: {predicted_label}")
            print(f"  置信度: {confidence:.4f}")
            print(f"  所有类别概率:")
            
            # 按概率排序显示
            prob_dict = dict(zip(classifier.class_names, probabilities))
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            
            for class_name, prob in sorted_probs[:5]:  # 显示前5个
                print(f"    {class_name}: {prob:.4f}")
        
        print("-" * 40)


def main():
    """主函数"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "performance":
            test_model_performance()
        elif sys.argv[1] == "interactive":
            interactive_test()
        else:
            print("用法:")
            print("  python test_lstm.py performance  # 测试模型性能")
            print("  python test_lstm.py interactive  # 交互式测试")
    else:
        print("请选择测试模式:")
        print("1. 性能测试")
        print("2. 交互式测试")
        
        choice = input("请输入选择 (1/2): ").strip()
        
        if choice == "1":
            test_model_performance()
        elif choice == "2":
            interactive_test()
        else:
            print("无效选择")


if __name__ == "__main__":
    main() 