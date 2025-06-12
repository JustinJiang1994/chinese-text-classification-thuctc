#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试朴素贝叶斯分类器
"""

import sys
import os

# 添加src目录到路径
sys.path.append('src')

from naive_bayes_classifier import NaiveBayesNewsClassifier

def test_model():
    """测试已训练的模型"""
    print("=" * 50)
    print("测试朴素贝叶斯分类器")
    print("=" * 50)
    
    # 创建分类器实例
    classifier = NaiveBayesNewsClassifier()
    
    # 加载已训练的模型
    classifier.load_model()
    
    # 测试文本
    test_texts = [
        "特斯拉发布新款电动汽车，续航里程大幅提升",
        "NBA季后赛激战正酣，勇士队击败湖人队",
        "央行宣布降息政策，股市应声上涨",
        "新款iPhone发布，搭载最新A17芯片",
        "房地产市场调控政策出台，房价趋于稳定",
        "国际足联世界杯预选赛即将开始",
        "好莱坞明星获得奥斯卡最佳男主角奖",
        "军事演习在南海举行，多国参与",
        "家居装修新趋势：简约风格受欢迎",
        "社会新闻：志愿者帮助贫困学生"
    ]
    
    print("\n测试文本:")
    print("-" * 30)
    for i, text in enumerate(test_texts, 1):
        print(f"{i}. {text}")
    
    print("\n预测结果:")
    print("-" * 30)
    
    # 进行预测
    predicted_classes, probabilities = classifier.predict(test_texts)
    
    for i, (text, pred_class, prob) in enumerate(zip(test_texts, predicted_classes, probabilities), 1):
        max_prob = max(prob)
        print(f"{i}. 文本: {text[:30]}...")
        print(f"   预测类别: {pred_class}")
        print(f"   置信度: {max_prob:.4f}")
        print()

if __name__ == "__main__":
    test_model() 