#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本特征提取方法对比实验
支持多种特征提取方法的性能对比
"""

import pandas as pd
import numpy as np
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import jieba
import re

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

class FeatureExtractionExperiment:
    """特征提取方法对比实验类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化特征提取实验
        
        Args:
            config: 实验配置字典
        """
        self.config = config or self._get_default_config()
        self.results_dir = Path(self.config['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
        # 机器学习模型
        self.models = self._create_models()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'data_dir': 'results',
            'results_dir': 'results/feature_experiments',
            'sample_size': 3000,
            'test_size': 0.2,
            'random_state': 42,
            'feature_methods': [
                'tfidf_basic',
                'tfidf_ngram',
                'count_vectorizer',
                'lda_topics',
                'tfidf_svd'
            ],
            'models': ['naive_bayes', 'svm', 'random_forest', 'logistic_regression'],
            'save_results': True,
            'plot_results': True,
            'feature_params': {
                'tfidf_basic': {
                    'max_features': 3000,
                    'ngram_range': (1, 1),
                    'min_df': 2,
                    'max_df': 0.9
                },
                'tfidf_ngram': {
                    'max_features': 5000,
                    'ngram_range': (1, 2),
                    'min_df': 2,
                    'max_df': 0.9
                },
                'count_vectorizer': {
                    'max_features': 3000,
                    'ngram_range': (1, 1),
                    'min_df': 2,
                    'max_df': 0.9
                },
                'lda_topics': {
                    'n_components': 50,
                    'max_iter': 10,
                    'random_state': 42
                },
                'tfidf_svd': {
                    'max_features': 5000,
                    'n_components': 100,
                    'random_state': 42
                }
            }
        }
    
    def _create_models(self) -> Dict[str, Any]:
        """创建机器学习模型"""
        return {
            'naive_bayes': MultinomialNB(alpha=1.0),
            'svm': LinearSVC(C=1.0, class_weight='balanced', random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, 
                                                   class_weight='balanced', random_state=42),
            'logistic_regression': LogisticRegression(C=1.0, class_weight='balanced', 
                                                     random_state=42, max_iter=1000)
        }
    
    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if pd.isna(text):
            return ""
        
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 去除URL和邮箱
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # 只保留中文和英文
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z\s]', '', text)
        
        # 分词
        words = jieba.cut(text)
        return ' '.join(words)
    
    def extract_features(self, texts: List[str], method: str, fit_vectorizer: bool = True, vectorizer=None) -> Tuple[np.ndarray, Any]:
        """提取特征"""
        params = self.config['feature_params'].get(method, {})
        
        if method == 'tfidf_basic':
            if fit_vectorizer:
                vectorizer = TfidfVectorizer(**params)
                features = vectorizer.fit_transform(texts)
            else:
                features = vectorizer.transform(texts)
            return features, vectorizer
        
        elif method == 'tfidf_ngram':
            if fit_vectorizer:
                vectorizer = TfidfVectorizer(**params)
                features = vectorizer.fit_transform(texts)
            else:
                features = vectorizer.transform(texts)
            return features, vectorizer
        
        elif method == 'count_vectorizer':
            if fit_vectorizer:
                vectorizer = CountVectorizer(**params)
                features = vectorizer.fit_transform(texts)
            else:
                features = vectorizer.transform(texts)
            return features, vectorizer
        
        elif method == 'lda_topics':
            # 先用TF-IDF，再用LDA
            if fit_vectorizer:
                tfidf = TfidfVectorizer(max_features=3000, min_df=2, max_df=0.9)
                lda = LatentDirichletAllocation(n_components=params['n_components'], 
                                              max_iter=params['max_iter'], 
                                              random_state=params['random_state'])
                pipeline = Pipeline([('tfidf', tfidf), ('lda', lda)])
                features = pipeline.fit_transform(texts)
                vectorizer = pipeline
            else:
                features = vectorizer.transform(texts)
            return features, vectorizer
        
        elif method == 'tfidf_svd':
            # TF-IDF + SVD降维
            if fit_vectorizer:
                tfidf = TfidfVectorizer(max_features=params['max_features'], 
                                       ngram_range=(1, 2), min_df=2, max_df=0.9)
                svd = TruncatedSVD(n_components=params['n_components'], 
                                  random_state=params['random_state'])
                pipeline = Pipeline([('tfidf', tfidf), ('svd', svd)])
                features = pipeline.fit_transform(texts)
                vectorizer = pipeline
            else:
                features = vectorizer.transform(texts)
            return features, vectorizer
        
        else:
            raise ValueError(f"不支持的特征提取方法: {method}")
    
    def run_feature_experiment(self, sample_size: int = None) -> Dict[str, Any]:
        """运行特征提取对比实验"""
        sample_size = sample_size or self.config['sample_size']
        
        print("=== 特征提取方法对比实验 ===")
        print(f"样本数量: {sample_size}")
        
        # 加载数据
        data = self._load_data(sample_size)
        print(f"加载数据: {len(data)} 样本")
        
        # 预处理文本
        print("预处理文本...")
        data['processed_content'] = data['content'].apply(self.preprocess_text)
        
        # 标签编码
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(data['category'])
        
        # 分割数据
        X_texts = data['processed_content'].tolist()
        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            X_texts, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state'], stratify=y
        )
        
        results = {}
        feature_methods = self.config['feature_methods']
        
        for method in feature_methods:
            print(f"\n--- 测试 {method} ---")
            
            try:
                # 提取特征
                start_time = time.time()
                X_train, vectorizer = self.extract_features(X_train_texts, method, fit_vectorizer=True)
                X_test, _ = self.extract_features(X_test_texts, method, fit_vectorizer=False, vectorizer=vectorizer)
                feature_time = time.time() - start_time
                
                print(f"特征维度: {X_train.shape[1]}")
                print(f"特征提取时间: {feature_time:.2f}秒")
                
                # 测试所有模型
                method_results = {}
                for model_name, model in self.models.items():
                    # 对于有负值的特征，跳过朴素贝叶斯
                    if method in ['tfidf_svd', 'lda_topics'] and model_name == 'naive_bayes':
                        print(f"  跳过 {model_name} (不支持负值特征)")
                        continue
                        
                    print(f"  训练 {model_name}...")
                    
                    start_time = time.time()
                    model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    y_pred = model.predict(X_test)
                    f1_macro = f1_score(y_test, y_pred, average='macro')
                    f1_weighted = f1_score(y_test, y_pred, average='weighted')
                    
                    method_results[model_name] = {
                        'f1_macro': f1_macro,
                        'f1_weighted': f1_weighted,
                        'training_time': training_time,
                        'feature_time': feature_time,
                        'feature_dim': X_train.shape[1]
                    }
                    
                    print(f"    F1-macro: {f1_macro:.4f}")
                    print(f"    训练时间: {training_time:.2f}秒")
                
                results[method] = method_results
                
            except Exception as e:
                print(f"  {method} 失败: {str(e)}")
                results[method] = {'error': str(e)}
        
        # 保存结果
        if self.config['save_results']:
            self._save_results(results, 'feature_extraction_experiment')
        
        # 绘制结果
        if self.config['plot_results']:
            self._plot_results(results, label_encoder.classes_)
        
        return results
    
    def _load_data(self, sample_size: int) -> pd.DataFrame:
        """加载数据"""
        # 直接从训练集抽取样本
        data = pd.read_csv(Path(self.config['data_dir']) / "train.csv", nrows=sample_size)
        print(f"从训练集抽取: {len(data)} 样本")
        
        return data
    
    def _save_results(self, results: Dict[str, Any], experiment_name: str):
        """保存实验结果"""
        results_file = self.results_dir / f"{experiment_name}_results.json"
        
        # 确保结果可序列化
        serializable_results = self._make_serializable(results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {results_file}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """确保对象可JSON序列化"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def _plot_results(self, results: Dict[str, Any], class_names: np.ndarray):
        """绘制实验结果"""
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. 特征方法性能对比
        self._plot_feature_methods_comparison(results, plots_dir)
        
        # 2. 模型在不同特征下的性能
        self._plot_model_performance_across_features(results, plots_dir)
        
        # 3. 特征维度对比
        self._plot_feature_dimensions(results, plots_dir)
        
        # 4. 计算时间对比
        self._plot_computation_time(results, plots_dir)
    
    def _plot_feature_methods_comparison(self, results: Dict[str, Any], plots_dir: Path):
        """绘制特征方法性能对比"""
        plt.figure(figsize=(12, 8))
        
        methods = []
        f1_scores = []
        
        for method, method_results in results.items():
            if 'error' not in method_results:
                # 计算平均F1-macro分数
                scores = [result['f1_macro'] for result in method_results.values() 
                         if isinstance(result, dict) and 'f1_macro' in result]
                if scores:
                    avg_score = np.mean(scores)
                    methods.append(method)
                    f1_scores.append(avg_score)
        
        # 排序
        sorted_indices = np.argsort(f1_scores)[::-1]
        methods = [methods[i] for i in sorted_indices]
        f1_scores = [f1_scores[i] for i in sorted_indices]
        
        bars = plt.bar(range(len(methods)), f1_scores, color='skyblue', alpha=0.7)
        plt.xlabel('特征提取方法')
        plt.ylabel('平均F1-macro分数')
        plt.title('不同特征提取方法性能对比')
        plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # 添加数值标签
        for bar, score in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'feature_methods_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_performance_across_features(self, results: Dict[str, Any], plots_dir: Path):
        """绘制模型在不同特征下的性能"""
        plt.figure(figsize=(14, 8))
        
        models = list(self.models.keys())
        methods = [method for method in results.keys() if 'error' not in results[method]]
        
        # 准备数据
        data = []
        for method in methods:
            for model in models:
                if model in results[method] and 'f1_macro' in results[method][model]:
                    data.append({
                        'method': method,
                        'model': model,
                        'f1_macro': results[method][model]['f1_macro']
                    })
        
        if data:
            df = pd.DataFrame(data)
            
            # 创建热力图
            pivot_table = df.pivot(index='model', columns='method', values='f1_macro')
            
            sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', 
                       cbar_kws={'label': 'F1-macro分数'})
            plt.title('不同模型在不同特征提取方法下的性能')
            plt.xlabel('特征提取方法')
            plt.ylabel('机器学习模型')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(plots_dir / 'model_performance_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_feature_dimensions(self, results: Dict[str, Any], plots_dir: Path):
        """绘制特征维度对比"""
        plt.figure(figsize=(10, 6))
        
        methods = []
        dimensions = []
        
        for method, method_results in results.items():
            if 'error' not in method_results:
                # 获取特征维度
                for model_result in method_results.values():
                    if isinstance(model_result, dict) and 'feature_dim' in model_result:
                        methods.append(method)
                        dimensions.append(model_result['feature_dim'])
                        break
        
        if methods:
            bars = plt.bar(range(len(methods)), dimensions, color='lightgreen', alpha=0.7)
            plt.xlabel('特征提取方法')
            plt.ylabel('特征维度')
            plt.title('不同特征提取方法的特征维度对比')
            plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
            
            # 添加数值标签
            for bar, dim in zip(bars, dimensions):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(dimensions)*0.01,
                        f'{dim}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'feature_dimensions.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_computation_time(self, results: Dict[str, Any], plots_dir: Path):
        """绘制计算时间对比"""
        plt.figure(figsize=(12, 8))
        
        methods = []
        feature_times = []
        training_times = []
        
        for method, method_results in results.items():
            if 'error' not in method_results:
                # 计算平均时间
                feature_time_list = []
                training_time_list = []
                
                for model_result in method_results.values():
                    if isinstance(model_result, dict):
                        if 'feature_time' in model_result:
                            feature_time_list.append(model_result['feature_time'])
                        if 'training_time' in model_result:
                            training_time_list.append(model_result['training_time'])
                
                if feature_time_list and training_time_list:
                    methods.append(method)
                    feature_times.append(np.mean(feature_time_list))
                    training_times.append(np.mean(training_time_list))
        
        if methods:
            x = np.arange(len(methods))
            width = 0.35
            
            plt.bar(x - width/2, feature_times, width, label='特征提取时间', alpha=0.7)
            plt.bar(x + width/2, training_times, width, label='模型训练时间', alpha=0.7)
            
            plt.xlabel('特征提取方法')
            plt.ylabel('时间 (秒)')
            plt.title('不同特征提取方法的计算时间对比')
            plt.xticks(x, methods, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / 'computation_time.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def print_summary(self, results: Dict[str, Any]):
        """打印实验总结"""
        print("\n" + "="*50)
        print("特征提取实验总结")
        print("="*50)
        
        # 找出最佳特征方法
        best_method = None
        best_score = 0
        
        for method, method_results in results.items():
            if 'error' not in method_results:
                scores = [result['f1_macro'] for result in method_results.values() 
                         if isinstance(result, dict) and 'f1_macro' in result]
                if scores:
                    avg_score = np.mean(scores)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_method = method
        
        if best_method:
            print(f"最佳特征提取方法: {best_method}")
            print(f"平均F1-macro分数: {best_score:.4f}")
        
        # 打印各方法性能排名
        print("\n特征方法性能排名:")
        method_scores = []
        for method, method_results in results.items():
            if 'error' not in method_results:
                scores = [result['f1_macro'] for result in method_results.values() 
                         if isinstance(result, dict) and 'f1_macro' in result]
                if scores:
                    avg_score = np.mean(scores)
                    method_scores.append((method, avg_score))
        
        method_scores.sort(key=lambda x: x[1], reverse=True)
        for i, (method, score) in enumerate(method_scores, 1):
            print(f"{i}. {method}: {score:.4f}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='特征提取方法对比实验')
    parser.add_argument('--sample-size', type=int, default=3000,
                       help='样本数量')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存结果')
    parser.add_argument('--no-plot', action='store_true',
                       help='不生成图表')
    
    args = parser.parse_args()
    
    # 加载配置
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    # 更新配置
    if args.no_save:
        config['save_results'] = False
    if args.no_plot:
        config['plot_results'] = False
    
    # 创建实验
    experiment = FeatureExtractionExperiment(config)
    
    # 运行实验
    results = experiment.run_feature_experiment(args.sample_size)
    
    # 打印总结
    experiment.print_summary(results)


if __name__ == "__main__":
    main()
