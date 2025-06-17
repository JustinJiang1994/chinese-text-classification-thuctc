#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
类别不平衡处理实验
专门测试各种重采样技术对不平衡数据的效果
"""

import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
import jieba
import re
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ImbalanceExperiment:
    def __init__(self, data_dir="results"):
        self.data_dir = Path(data_dir)
        self.results = {}
        
    def load_data(self, sample_size=5000):
        """加载数据"""
        print("正在加载数据...")
        
        # 使用训练集的前N个样本进行实验
        data = pd.read_csv(self.data_dir / "train.csv", nrows=sample_size)
        print(f"加载了 {len(data)} 个样本")
        
        return data
    
    def preprocess_text(self, text):
        """文本预处理"""
        if pd.isna(text):
            return ""
        
        # 移除特殊字符和数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', str(text))
        # 分词
        words = jieba.cut(text)
        return ' '.join(words)
    
    def prepare_features(self, data, max_features=8000):
        """准备特征"""
        print("正在预处理文本...")
        data['processed_content'] = data['content'].apply(self.preprocess_text)
        
        print("正在提取TF-IDF特征...")
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        )
        
        X = vectorizer.fit_transform(data['processed_content'])
        print(f"特征维度: {X.shape[1]}")
        
        # 标签编码
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(data['category'])
        print(f"类别数量: {len(label_encoder.classes_)}")
        
        return X, y, vectorizer, label_encoder
    
    def analyze_imbalance(self, y, label_encoder):
        """分析类别不平衡情况"""
        print("\n=== 类别不平衡分析 ===")
        
        # 计算类别分布
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(label_encoder.classes_, counts))
        
        print("类别分布:")
        for category, count in sorted(class_distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"  {category}: {count}")
        
        # 计算不平衡指标
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count
        
        print(f"\n不平衡比例: {imbalance_ratio:.2f}:1")
        print(f"最大类别样本数: {max_count}")
        print(f"最小类别样本数: {min_count}")
        
        return class_distribution, imbalance_ratio
    
    def create_sampling_methods(self):
        """创建各种重采样方法"""
        sampling_methods = {
            '原始数据': None,
            'SMOTE': SMOTE(random_state=42, k_neighbors=3),
            'ADASYN': ADASYN(random_state=42),
            'BorderlineSMOTE': BorderlineSMOTE(random_state=42, k_neighbors=3),
            '随机欠采样': RandomUnderSampler(random_state=42),
            'TomekLinks': TomekLinks(),
            'EditedNearestNeighbours': EditedNearestNeighbours(),
            'SMOTEENN': SMOTEENN(random_state=42),
            'SMOTETomek': SMOTETomek(random_state=42)
        }
        
        return sampling_methods
    
    def create_models(self):
        """创建机器学习模型"""
        models = {
            '朴素贝叶斯': MultinomialNB(alpha=1.0),
            'SVM': LinearSVC(C=1.0, class_weight='balanced', random_state=42, max_iter=1000),
            '随机森林': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42),
            '逻辑回归': LogisticRegression(C=1.0, class_weight='balanced', random_state=42, max_iter=1000)
        }
        
        return models
    
    def apply_sampling(self, X, y, sampling_method, method_name):
        """应用重采样方法"""
        if sampling_method is None:
            return X, y
        
        print(f"  应用 {method_name}...")
        try:
            X_resampled, y_resampled = sampling_method.fit_resample(X, y)
            print(f"    原始样本数: {len(y)} -> 重采样后: {len(y_resampled)}")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"    {method_name} 失败: {e}")
            return X, y
    
    def evaluate_model(self, model, X_train, y_train, X_test, y_test, model_name, sampling_name):
        """评估模型性能"""
        # 训练模型
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算指标
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        
        # 计算每个类别的F1分数
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        return {
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_micro': f1_micro,
            'f1_per_class': f1_per_class,
            'training_time': training_time,
            'predictions': y_pred
        }
    
    def run_experiment(self, sample_size=5000):
        """运行完整实验"""
        print("=== 类别不平衡处理实验 ===")
        
        # 1. 加载数据
        data = self.load_data(sample_size)
        
        # 2. 准备特征
        X, y, vectorizer, label_encoder = self.prepare_features(data)
        
        # 3. 分析不平衡情况
        class_distribution, imbalance_ratio = self.analyze_imbalance(y, label_encoder)
        
        # 4. 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n训练集: {len(y_train)} 样本")
        print(f"测试集: {len(y_test)} 样本")
        
        # 5. 创建重采样方法和模型
        sampling_methods = self.create_sampling_methods()
        models = self.create_models()
        
        # 6. 运行实验
        print("\n=== 开始实验 ===")
        
        for sampling_name, sampling_method in sampling_methods.items():
            print(f"\n--- {sampling_name} ---")
            
            # 应用重采样
            X_train_resampled, y_train_resampled = self.apply_sampling(
                X_train, y_train, sampling_method, sampling_name
            )
            
            # 训练和评估所有模型
            for model_name, model in models.items():
                print(f"  训练 {model_name}...")
                
                # 评估模型
                result = self.evaluate_model(
                    model, X_train_resampled, y_train_resampled, 
                    X_test, y_test, model_name, sampling_name
                )
                
                # 保存结果
                key = f"{sampling_name}_{model_name}"
                self.results[key] = result
                
                print(f"    F1-macro: {result['f1_macro']:.4f}")
                print(f"    F1-weighted: {result['f1_weighted']:.4f}")
                print(f"    训练时间: {result['training_time']:.2f}秒")
        
        # 7. 保存结果
        self.save_results(class_distribution, imbalance_ratio)
        
        # 8. 绘制结果
        self.plot_results(label_encoder.classes_)
        
        print("\n=== 实验完成 ===")
    
    def save_results(self, class_distribution, imbalance_ratio):
        """保存实验结果"""
        print("正在保存实验结果...")
        
        # 准备可序列化的结果
        serializable_results = {}
        for key, result in self.results.items():
            serializable_results[key] = {
                'f1_macro': float(result['f1_macro']),
                'f1_weighted': float(result['f1_weighted']),
                'f1_micro': float(result['f1_micro']),
                'f1_per_class': result['f1_per_class'].tolist(),
                'training_time': float(result['training_time'])
            }
        
        # 添加实验元数据
        experiment_summary = {
            'class_distribution': {k: int(v) for k, v in class_distribution.items()},
            'imbalance_ratio': float(imbalance_ratio),
            'results': serializable_results
        }
        
        # 保存到文件
        results_file = self.data_dir / "imbalance_experiment_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_summary, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {results_file}")
    
    def plot_results(self, class_names):
        """绘制实验结果"""
        print("正在绘制结果图表...")
        
        # 创建图表目录
        plots_dir = self.data_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. 重采样方法对比
        self._plot_sampling_comparison(plots_dir)
        
        # 2. 模型性能对比
        self._plot_model_comparison(plots_dir)
        
        # 3. 详细性能分析
        self._plot_detailed_analysis(plots_dir, class_names)
    
    def _plot_sampling_comparison(self, plots_dir):
        """绘制重采样方法对比图"""
        # 提取重采样方法名称
        sampling_methods = set()
        models = set()
        
        for key in self.results.keys():
            parts = key.split('_', 1)
            if len(parts) == 2:
                sampling_methods.add(parts[0])
                models.add(parts[1])
        
        sampling_methods = sorted(list(sampling_methods))
        models = sorted(list(models))
        
        # 创建对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, model in enumerate(models):
            if i < len(axes):
                f1_scores = []
                method_names = []
                
                for method in sampling_methods:
                    key = f"{method}_{model}"
                    if key in self.results:
                        f1_scores.append(self.results[key]['f1_macro'])
                        method_names.append(method)
                
                axes[i].bar(range(len(f1_scores)), f1_scores, alpha=0.8)
                axes[i].set_title(f'{model} - 不同重采样方法对比')
                axes[i].set_ylabel('F1-macro 分数')
                axes[i].set_xticks(range(len(method_names)))
                axes[i].set_xticklabels(method_names, rotation=45, ha='right')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "sampling_methods_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_comparison(self, plots_dir):
        """绘制模型性能对比图"""
        # 为每个重采样方法创建模型对比图
        sampling_methods = set()
        models = set()
        
        for key in self.results.keys():
            parts = key.split('_', 1)
            if len(parts) == 2:
                sampling_methods.add(parts[0])
                models.add(parts[1])
        
        sampling_methods = sorted(list(sampling_methods))
        models = sorted(list(models))
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()
        
        for i, method in enumerate(sampling_methods):
            if i < len(axes):
                f1_scores = []
                model_names = []
                
                for model in models:
                    key = f"{method}_{model}"
                    if key in self.results:
                        f1_scores.append(self.results[key]['f1_macro'])
                        model_names.append(model)
                
                axes[i].bar(range(len(f1_scores)), f1_scores, alpha=0.8)
                axes[i].set_title(f'{method} - 模型性能对比')
                axes[i].set_ylabel('F1-macro 分数')
                axes[i].set_xticks(range(len(model_names)))
                axes[i].set_xticklabels(model_names, rotation=45, ha='right')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "model_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_detailed_analysis(self, plots_dir, class_names):
        """绘制详细性能分析"""
        # 找到最佳组合
        best_key = max(self.results.keys(), key=lambda k: self.results[k]['f1_macro'])
        best_result = self.results[best_key]
        
        # 绘制最佳模型的类别性能
        plt.figure(figsize=(12, 6))
        f1_per_class = best_result['f1_per_class']
        
        plt.bar(range(len(class_names)), f1_per_class, alpha=0.8)
        plt.title(f'最佳模型 ({best_key}) - 各类别F1分数')
        plt.xlabel('类别')
        plt.ylabel('F1分数')
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "best_model_class_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_summary(self):
        """打印实验总结"""
        print("\n=== 实验总结 ===")
        
        # 找到最佳结果
        best_key = max(self.results.keys(), key=lambda k: self.results[k]['f1_macro'])
        best_result = self.results[best_key]
        
        print(f"最佳组合: {best_key}")
        print(f"最佳F1-macro: {best_result['f1_macro']:.4f}")
        print(f"最佳F1-weighted: {best_result['f1_weighted']:.4f}")
        
        # 按重采样方法分组显示结果
        sampling_methods = set()
        for key in self.results.keys():
            parts = key.split('_', 1)
            if len(parts) == 2:
                sampling_methods.add(parts[0])
        
        print(f"\n各重采样方法的最佳F1-macro:")
        for method in sorted(sampling_methods):
            method_results = [self.results[k]['f1_macro'] for k in self.results.keys() if k.startswith(method)]
            if method_results:
                best_method_score = max(method_results)
                print(f"  {method}: {best_method_score:.4f}")

def main():
    """主函数"""
    experiment = ImbalanceExperiment()
    experiment.run_experiment(sample_size=10000)  # 使用1万个样本进行实验
    experiment.print_summary()

if __name__ == "__main__":
    main() 