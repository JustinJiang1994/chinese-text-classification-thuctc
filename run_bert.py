#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行BERT新闻分类器
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.bert_classifier import BERTNewsClassifier

def main():
    """主函数"""
    print("=" * 60)
    print("BERT新闻分类器")
    print("=" * 60)
    
    # 创建BERT分类器
    bert_classifier = BERTNewsClassifier()
    
    # 运行完整流程
    results = bert_classifier.run_complete_pipeline()
    
    if results:
        print(f"\n最终结果:")
        print(f"准确率: {results['accuracy']:.4f}")
        print(f"F1-Macro: {results['f1_macro']:.4f}")
        print(f"F1-Weighted: {results['f1_weighted']:.4f}")
        
        # 保存结果到文件
        with open('results/bert_results.txt', 'w', encoding='utf-8') as f:
            f.write("BERT分类器结果\n")
            f.write("=" * 30 + "\n")
            f.write(f"准确率: {results['accuracy']:.4f}\n")
            f.write(f"F1-Macro: {results['f1_macro']:.4f}\n")
            f.write(f"F1-Weighted: {results['f1_weighted']:.4f}\n")
        
        print("\n结果已保存到 results/bert_results.txt")
    else:
        print("训练失败，请检查数据文件是否存在")

if __name__ == "__main__":
    main() 