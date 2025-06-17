#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传统机器学习分类实验
包含多种算法和类别不平衡处理策略
"""

import pandas as pd
import numpy as np
import json
import time
import warnings
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import xgboost as xgb
import jieba
import re

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

class TraditionalMLExperiment:
    def __init__(self, data_dir="results"):
        self.data_dir = Path(data_dir)
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.vectorizer = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """加载训练、验证和测试数据"""
        print("正在加载数据...")
        
        # 加载数据
        self.train_data = pd.read_csv(self.data_dir / "train.csv")
        self.val_data = pd.read_csv(self.data_dir / "val.csv")
        self.test_data = pd.read_csv(self.data_dir / "test.csv")
        
        print(f"训练集: {len(self.train_data)} 样本")
        print(f"验证集: {len(self.val_data)} 样本")
        print(f"测试集: {len(self.test_data)} 样本")
        
        # 显示类别分布
        self._show_class_distribution()
        
    def _show_class_distribution(self):
        """显示类别分布"""
        print("\n=== 类别分布 ===")
        train_dist = self.train_data['category'].value_counts()
        print("训练集类别分布:")
        for category, count in train_dist.items():
            print(f"  {category}: {count}")
            
        # 计算不平衡比例
        max_count = train_dist.max()
        min_count = train_dist.min()
        imbalance_ratio = max_count / min_count
        print(f"\n类别不平衡比例: {imbalance_ratio:.2f}:1")
        
    def preprocess_text(self, text):
        """文本预处理"""
        if pd.isna(text):
            return ""
        
        # 移除特殊字符和数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', str(text))
        # 分词
        words = jieba.cut(text)
        return ' '.join(words)
    
    def prepare_features(self, max_features=10000, ngram_range=(1, 2)):
        """准备TF-IDF特征"""
        print("正在准备TF-IDF特征...")
        
        # 预处理文本
        train_texts = self.train_data['content'].apply(self.preprocess_text)
        val_texts = self.val_data['content'].apply(self.preprocess_text)
        test_texts = self.test_data['content'].apply(self.preprocess_text)
        
        # TF-IDF向量化
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95
        )
        
        # 训练TF-IDF
        X_train = self.vectorizer.fit_transform(train_texts)
        X_val = self.vectorizer.transform(val_texts)
        X_test = self.vectorizer.transform(test_texts)
        
        # 标签编码
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(self.train_data['category'])
        y_val = self.label_encoder.transform(self.val_data['category'])
        y_test = self.label_encoder.transform(self.test_data['category'])
        
        print(f"特征维度: {X_train.shape[1]}")
        print(f"类别数量: {len(self.label_encoder.classes_)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_models(self):
        """创建各种机器学习模型"""
        print("正在创建模型...")
        
        # 计算类别权重
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        class_weight_dict = dict(zip(range(len(class_weights)), class_weights))
        
        # 1. 朴素贝叶斯 (基线模型)
        self.models['naive_bayes'] = MultinomialNB(alpha=1.0)
        
        # 2. 支持向量机
        self.models['svm'] = LinearSVC(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        # 3. 随机森林
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # 4. 逻辑回归
        self.models['logistic_regression'] = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        # 5. XGBoost
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=1,
            random_state=42,
            n_jobs=-1
        )
        
        print(f"创建了 {len(self.models)} 个模型")
        
    def train_models(self, X_train, y_train):
        """训练所有模型"""
        print("正在训练模型...")
        
        for name, model in self.models.items():
            print(f"训练 {name}...")
            start_time = time.time()
            
            model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            print(f"  {name} 训练完成，耗时: {training_time:.2f}秒")
    
    def evaluate_models(self, X_val, y_val, X_test, y_test):
        """评估所有模型"""
        print("正在评估模型...")
        
        for name, model in self.models.items():
            print(f"\n评估 {name}...")
            
            # 验证集预测
            y_val_pred = model.predict(X_val)
            
            # 测试集预测
            y_test_pred = model.predict(X_test)
            
            # 计算指标
            val_f1_macro = f1_score(y_val, y_val_pred, average='macro')
            val_f1_weighted = f1_score(y_val, y_val_pred, average='weighted')
            test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
            test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
            
            # 保存结果
            self.results[name] = {
                'val_f1_macro': val_f1_macro,
                'val_f1_weighted': val_f1_weighted,
                'test_f1_macro': test_f1_macro,
                'test_f1_weighted': test_f1_weighted,
                'val_predictions': y_val_pred,
                'test_predictions': y_test_pred
            }
            
            print(f"  验证集 F1-macro: {val_f1_macro:.4f}")
            print(f"  验证集 F1-weighted: {val_f1_weighted:.4f}")
            print(f"  测试集 F1-macro: {test_f1_macro:.4f}")
            print(f"  测试集 F1-weighted: {test_f1_weighted:.4f}")
    
    def handle_imbalance(self, X_train, y_train):
        """处理类别不平衡问题"""
        print("正在处理类别不平衡问题...")
        
        # 1. SMOTE过采样
        print("应用SMOTE过采样...")
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # 2. 随机欠采样
        print("应用随机欠采样...")
        undersampler = RandomUnderSampler(random_state=42)
        X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)
        
        # 3. SMOTEENN (SMOTE + Edited Nearest Neighbors)
        print("应用SMOTEENN...")
        smoteenn = SMOTEENN(random_state=42)
        X_train_smoteenn, y_train_smoteenn = smoteenn.fit_resample(X_train, y_train)
        
        return {
            'original': (X_train, y_train),
            'smote': (X_train_smote, y_train_smote),
            'undersample': (X_train_under, y_train_under),
            'smoteenn': (X_train_smoteenn, y_train_smoteenn)
        }
    
    def run_imbalance_experiments(self, X_val, y_val, X_test, y_test):
        """运行类别不平衡处理实验"""
        print("\n=== 类别不平衡处理实验 ===")
        
        imbalance_results = {}
        
        for method_name, (X_train, y_train) in self.balanced_datasets.items():
            print(f"\n使用 {method_name} 方法...")
            
            # 重新训练模型
            for name, model in self.models.items():
                print(f"  训练 {name}...")
                model.fit(X_train, y_train)
                
                # 预测和评估
                y_val_pred = model.predict(X_val)
                y_test_pred = model.predict(X_test)
                
                val_f1_macro = f1_score(y_val, y_val_pred, average='macro')
                test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
                
                if method_name not in imbalance_results:
                    imbalance_results[method_name] = {}
                
                imbalance_results[method_name][name] = {
                    'val_f1_macro': val_f1_macro,
                    'test_f1_macro': test_f1_macro
                }
                
                print(f"    {name} - 验证集F1-macro: {val_f1_macro:.4f}, 测试集F1-macro: {test_f1_macro:.4f}")
        
        return imbalance_results
    
    def save_results(self):
        """保存实验结果"""
        print("正在保存实验结果...")
        
        # 保存详细结果
        results_file = self.data_dir / "traditional_ml_results.json"
        
        # 转换numpy数组为列表以便JSON序列化
        serializable_results = {}
        for model_name, result in self.results.items():
            serializable_results[model_name] = {
                'val_f1_macro': float(result['val_f1_macro']),
                'val_f1_weighted': float(result['val_f1_weighted']),
                'test_f1_macro': float(result['test_f1_macro']),
                'test_f1_weighted': float(result['test_f1_weighted'])
            }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {results_file}")
    
    def plot_results(self):
        """绘制实验结果"""
        print("正在绘制结果图表...")
        
        # 创建图表目录
        plots_dir = self.data_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. 模型性能对比
        models = list(self.results.keys())
        val_f1_macro = [self.results[m]['val_f1_macro'] for m in models]
        test_f1_macro = [self.results[m]['test_f1_macro'] for m in models]
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, val_f1_macro, width, label='验证集 F1-macro', alpha=0.8)
        plt.bar(x + width/2, test_f1_macro, width, label='测试集 F1-macro', alpha=0.8)
        
        plt.xlabel('模型')
        plt.ylabel('F1-macro 分数')
        plt.title('传统机器学习模型性能对比')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "model_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 类别不平衡处理效果对比
        if hasattr(self, 'imbalance_results'):
            self._plot_imbalance_results(plots_dir)
    
    def _plot_imbalance_results(self, plots_dir):
        """绘制类别不平衡处理结果"""
        methods = list(self.imbalance_results.keys())
        models = list(self.imbalance_results[methods[0]].keys())
        
        # 为每个模型创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, model in enumerate(models):
            if i < len(axes):
                val_scores = [self.imbalance_results[method][model]['val_f1_macro'] 
                            for method in methods]
                test_scores = [self.imbalance_results[method][model]['test_f1_macro'] 
                             for method in methods]
                
                x = np.arange(len(methods))
                width = 0.35
                
                axes[i].bar(x - width/2, val_scores, width, label='验证集', alpha=0.8)
                axes[i].bar(x + width/2, test_scores, width, label='测试集', alpha=0.8)
                
                axes[i].set_title(f'{model} 性能对比')
                axes[i].set_ylabel('F1-macro 分数')
                axes[i].set_xticks(x)
                axes[i].set_xticklabels(methods, rotation=45)
                axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / "imbalance_handling_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_experiment(self):
        """运行完整实验"""
        print("=== 开始传统机器学习实验 ===")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 准备特征
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_features()
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        
        # 3. 创建模型
        self.create_models()
        
        # 4. 训练和评估原始模型
        print("\n=== 原始数据训练 ===")
        self.train_models(X_train, y_train)
        self.evaluate_models(X_val, y_val, X_test, y_test)
        
        # 5. 处理类别不平衡
        self.balanced_datasets = self.handle_imbalance(X_train, y_train)
        
        # 6. 运行不平衡处理实验
        self.imbalance_results = self.run_imbalance_experiments(X_val, y_val, X_test, y_test)
        
        # 7. 保存结果
        self.save_results()
        
        # 8. 绘制图表
        self.plot_results()
        
        print("\n=== 实验完成 ===")

def main():
    """主函数"""
    experiment = TraditionalMLExperiment()
    experiment.run_experiment()

if __name__ == "__main__":
    main() 