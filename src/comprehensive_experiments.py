#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合交叉实验模块
同时考虑重采样方法、特征提取方法和机器学习模型的三维交叉实验
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
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
import jieba
import re

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

class ComprehensiveExperiment:
    """综合交叉实验类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化综合实验
        
        Args:
            config: 实验配置字典
        """
        self.config = config or self._get_default_config()
        self.results_dir = Path(self.config['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
        # 初始化组件
        self.models = self._create_models()
        self.sampling_methods = self._create_sampling_methods()
        self.feature_extractors = self._create_feature_extractors()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'data_dir': 'results',
            'results_dir': 'results/comprehensive_experiments',
            'sample_size': 3000,
            'test_size': 0.2,
            'random_state': 42,
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
    
    def _create_sampling_methods(self) -> Dict[str, Any]:
        """创建重采样方法"""
        return {
            'original': None,
            'smote': SMOTE(random_state=42, k_neighbors=3),
            'tomek_links': TomekLinks(),
            'smoteenn': SMOTEENN(random_state=42)
        }
    
    def _create_feature_extractors(self) -> Dict[str, Any]:
        """创建特征提取器"""
        return {
            'tfidf_basic': self._create_tfidf_basic,
            'count_vectorizer': self._create_count_vectorizer,
            'tfidf_svd': self._create_tfidf_svd
        }
    
    def _create_tfidf_basic(self, params: Dict[str, Any]) -> TfidfVectorizer:
        """创建基础TF-IDF向量化器"""
        return TfidfVectorizer(**params)
    
    def _create_count_vectorizer(self, params: Dict[str, Any]) -> CountVectorizer:
        """创建词频向量化器"""
        return CountVectorizer(**params)
    
    def _create_tfidf_svd(self, params: Dict[str, Any]) -> Pipeline:
        """创建TF-IDF + SVD降维"""
        tfidf = TfidfVectorizer(max_features=params['max_features'], 
                               ngram_range=(1, 2), min_df=2, max_df=0.9)
        svd = TruncatedSVD(n_components=params['n_components'], 
                          random_state=params['random_state'])
        return Pipeline([('tfidf', tfidf), ('svd', svd)])
    
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
        
        if method in ['tfidf_basic', 'count_vectorizer']:
            if fit_vectorizer:
                vectorizer = self.feature_extractors[method](params)
                features = vectorizer.fit_transform(texts)
            else:
                features = vectorizer.transform(texts)
            return features, vectorizer
        
        elif method == 'tfidf_svd':
            if fit_vectorizer:
                pipeline = self.feature_extractors[method](params)
                features = pipeline.fit_transform(texts)
                vectorizer = pipeline
            else:
                features = vectorizer.transform(texts)
            return features, vectorizer
        
        else:
            raise ValueError(f"不支持的特征提取方法: {method}")
    
    def apply_sampling(self, X: np.ndarray, y: np.ndarray, sampling_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """应用重采样方法"""
        sampling_method = self.sampling_methods.get(sampling_name)
        
        if sampling_method is None:
            return X, y
        
        try:
            X_resampled, y_resampled = sampling_method.fit_resample(X, y)
            return X_resampled, y_resampled
        except Exception as e:
            print(f"    {sampling_name} 失败: {e}")
            return X, y
    
    def run_comprehensive_experiment(self, sample_size: int = None) -> Dict[str, Any]:
        """运行综合交叉实验"""
        sample_size = sample_size or self.config['sample_size']
        
        print("=== 综合交叉实验 ===")
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
        
        # 实验配置
        sampling_methods = list(self.sampling_methods.keys())
        feature_methods = list(self.feature_extractors.keys())
        models = list(self.models.keys())
        
        total_experiments = len(sampling_methods) * len(feature_methods) * len(models)
        print(f"总实验数: {total_experiments}")
        print(f"重采样方法: {len(sampling_methods)} 种")
        print(f"特征提取方法: {len(feature_methods)} 种")
        print(f"机器学习模型: {len(models)} 种")
        
        results = {}
        best_score = 0
        best_combination = None
        experiment_count = 0
        
        # 三维交叉实验
        for sampling_name in sampling_methods:
            print(f"\n--- 重采样方法: {sampling_name} ---")
            
            for feature_name in feature_methods:
                print(f"  --- 特征方法: {feature_name} ---")
                
                try:
                    # 提取特征
                    X_train, vectorizer = self.extract_features(X_train_texts, feature_name, fit_vectorizer=True)
                    X_test, _ = self.extract_features(X_test_texts, feature_name, fit_vectorizer=False, vectorizer=vectorizer)
                    
                    # 应用重采样
                    X_train_resampled, y_train_resampled = self.apply_sampling(X_train, y_train, sampling_name)
                    
                    print(f"    特征维度: {X_train.shape[1]}")
                    print(f"    重采样后样本数: {len(y_train_resampled)}")
                    
                    # 测试所有模型
                    for model_name, model in self.models.items():
                        # 对于有负值的特征，跳过朴素贝叶斯
                        if feature_name == 'tfidf_svd' and model_name == 'naive_bayes':
                            continue
                        
                        experiment_count += 1
                        print(f"      [{experiment_count}/{total_experiments}] 训练 {model_name}...")
                        
                        try:
                            start_time = time.time()
                            model.fit(X_train_resampled, y_train_resampled)
                            training_time = time.time() - start_time
                            
                            y_pred = model.predict(X_test)
                            f1_macro = f1_score(y_test, y_pred, average='macro')
                            f1_weighted = f1_score(y_test, y_pred, average='weighted')
                            
                            # 记录结果
                            key = f"{sampling_name}_{feature_name}_{model_name}"
                            results[key] = {
                                'f1_macro': f1_macro,
                                'f1_weighted': f1_weighted,
                                'training_time': training_time,
                                'feature_dim': X_train.shape[1],
                                'resampled_samples': len(y_train_resampled),
                                'sampling_method': sampling_name,
                                'feature_method': feature_name,
                                'model_name': model_name
                            }
                            
                            # 更新最佳组合
                            if f1_macro > best_score:
                                best_score = f1_macro
                                best_combination = key
                            
                            print(f"        F1-macro: {f1_macro:.4f}")
                            
                        except Exception as e:
                            print(f"        {model_name} 失败: {e}")
                            key = f"{sampling_name}_{feature_name}_{model_name}"
                            results[key] = {'error': str(e)}
                
                except Exception as e:
                    print(f"    {feature_name} 失败: {e}")
                    for model_name in models:
                        key = f"{sampling_name}_{feature_name}_{model_name}"
                        results[key] = {'error': str(e)}
        
        # 添加总结信息
        results['best_combination'] = {
            'name': best_combination,
            'f1_macro': best_score
        }
        results['experiment_summary'] = {
            'total_experiments': total_experiments,
            'successful_experiments': len([r for r in results.values() if isinstance(r, dict) and 'f1_macro' in r]),
            'best_score': best_score,
            'best_combination': best_combination
        }
        
        # 保存结果
        if self.config['save_results']:
            self._save_results(results, 'comprehensive_experiment')
        
        # 绘制结果
        if self.config['plot_results']:
            self._plot_results(results, label_encoder.classes_)
        
        return results
    
    def _load_data(self, sample_size: int) -> pd.DataFrame:
        """加载数据"""
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
        
        # 1. 最佳组合分析
        self._plot_best_combinations(results, plots_dir)
        
        # 2. 三维热力图
        self._plot_3d_heatmap(results, plots_dir)
        
        # 3. 各维度性能分析
        self._plot_dimension_analysis(results, plots_dir)
        
        # 4. 性能分布
        self._plot_performance_distribution(results, plots_dir)
    
    def _plot_best_combinations(self, results: Dict[str, Any], plots_dir: Path):
        """绘制最佳组合分析"""
        # 提取成功的实验结果
        successful_results = {}
        for key, result in results.items():
            if isinstance(result, dict) and 'f1_macro' in result:
                successful_results[key] = result
        
        if not successful_results:
            return
        
        # 按F1-macro排序
        sorted_results = sorted(successful_results.items(), key=lambda x: x[1]['f1_macro'], reverse=True)
        top_10 = sorted_results[:10]
        
        plt.figure(figsize=(12, 8))
        
        combinations = [item[0] for item in top_10]
        scores = [item[1]['f1_macro'] for item in top_10]
        
        bars = plt.bar(range(len(combinations)), scores, color='lightcoral', alpha=0.7)
        plt.xlabel('实验组合')
        plt.ylabel('F1-macro分数')
        plt.title('Top 10 最佳实验组合')
        plt.xticks(range(len(combinations)), combinations, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # 添加数值标签
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'best_combinations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_3d_heatmap(self, results: Dict[str, Any], plots_dir: Path):
        """绘制三维热力图"""
        # 提取成功的实验结果
        successful_results = {}
        for key, result in results.items():
            if isinstance(result, dict) and 'f1_macro' in result:
                successful_results[key] = result
        
        if not successful_results:
            return
        
        # 准备数据
        data = []
        for key, result in successful_results.items():
            parts = key.split('_')
            if len(parts) >= 3:
                sampling = parts[0]
                feature = parts[1]
                model = '_'.join(parts[2:])  # 处理模型名可能包含下划线的情况
                
                data.append({
                    'sampling': sampling,
                    'feature': feature,
                    'model': model,
                    'f1_macro': result['f1_macro']
                })
        
        if data:
            df = pd.DataFrame(data)
            
            # 为每个重采样方法创建热力图
            sampling_methods = df['sampling'].unique()
            n_methods = len(sampling_methods)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, sampling in enumerate(sampling_methods):
                if i < len(axes):
                    subset = df[df['sampling'] == sampling]
                    if not subset.empty:
                        pivot_table = subset.pivot(index='model', columns='feature', values='f1_macro')
                        
                        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', 
                                   cbar_kws={'label': 'F1-macro分数'}, ax=axes[i])
                        axes[i].set_title(f'{sampling} - 特征×模型性能')
                        axes[i].set_xlabel('特征提取方法')
                        axes[i].set_ylabel('机器学习模型')
            
            plt.tight_layout()
            plt.savefig(plots_dir / '3d_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_dimension_analysis(self, results: Dict[str, Any], plots_dir: Path):
        """绘制各维度性能分析"""
        # 提取成功的实验结果
        successful_results = {}
        for key, result in results.items():
            if isinstance(result, dict) and 'f1_macro' in result:
                successful_results[key] = result
        
        if not successful_results:
            return
        
        # 分析各维度的平均性能
        sampling_performance = {}
        feature_performance = {}
        model_performance = {}
        
        for key, result in successful_results.items():
            parts = key.split('_')
            if len(parts) >= 3:
                sampling = parts[0]
                feature = parts[1]
                model = '_'.join(parts[2:])
                
                # 统计各维度性能
                if sampling not in sampling_performance:
                    sampling_performance[sampling] = []
                sampling_performance[sampling].append(result['f1_macro'])
                
                if feature not in feature_performance:
                    feature_performance[feature] = []
                feature_performance[feature].append(result['f1_macro'])
                
                if model not in model_performance:
                    model_performance[model] = []
                model_performance[model].append(result['f1_macro'])
        
        # 绘制各维度性能对比
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 重采样方法性能
        sampling_means = {k: np.mean(v) for k, v in sampling_performance.items()}
        axes[0].bar(sampling_means.keys(), sampling_means.values(), alpha=0.7)
        axes[0].set_title('重采样方法平均性能')
        axes[0].set_ylabel('平均F1-macro分数')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 特征方法性能
        feature_means = {k: np.mean(v) for k, v in feature_performance.items()}
        axes[1].bar(feature_means.keys(), feature_means.values(), alpha=0.7)
        axes[1].set_title('特征提取方法平均性能')
        axes[1].set_ylabel('平均F1-macro分数')
        axes[1].tick_params(axis='x', rotation=45)
        
        # 模型性能
        model_means = {k: np.mean(v) for k, v in model_performance.items()}
        axes[2].bar(model_means.keys(), model_means.values(), alpha=0.7)
        axes[2].set_title('机器学习模型平均性能')
        axes[2].set_ylabel('平均F1-macro分数')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'dimension_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_distribution(self, results: Dict[str, Any], plots_dir: Path):
        """绘制性能分布"""
        # 提取成功的实验结果
        successful_results = {}
        for key, result in results.items():
            if isinstance(result, dict) and 'f1_macro' in result:
                successful_results[key] = result
        
        if not successful_results:
            return
        
        f1_scores = [result['f1_macro'] for result in successful_results.values()]
        
        plt.figure(figsize=(10, 6))
        plt.hist(f1_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('F1-macro分数')
        plt.ylabel('实验数量')
        plt.title('F1-macro分数分布')
        plt.axvline(np.mean(f1_scores), color='red', linestyle='--', label=f'平均值: {np.mean(f1_scores):.3f}')
        plt.axvline(np.median(f1_scores), color='green', linestyle='--', label=f'中位数: {np.median(f1_scores):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'performance_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_summary(self, results: Dict[str, Any]):
        """打印实验总结"""
        print("\n" + "="*60)
        print("综合交叉实验总结")
        print("="*60)
        
        # 实验统计
        if 'experiment_summary' in results:
            summary = results['experiment_summary']
            print(f"总实验数: {summary['total_experiments']}")
            print(f"成功实验数: {summary['successful_experiments']}")
            print(f"成功率: {summary['successful_experiments']/summary['total_experiments']*100:.1f}%")
        
        # 最佳组合
        if 'best_combination' in results:
            best = results['best_combination']
            print(f"\n最佳组合: {best['name']}")
            print(f"最佳F1-macro分数: {best['f1_macro']:.4f}")
        
        # 提取成功的实验结果
        successful_results = {}
        for key, result in results.items():
            if isinstance(result, dict) and 'f1_macro' in result:
                successful_results[key] = result
        
        if successful_results:
            # 各维度最佳
            sampling_best = {}
            feature_best = {}
            model_best = {}
            
            for key, result in successful_results.items():
                parts = key.split('_')
                if len(parts) >= 3:
                    sampling = parts[0]
                    feature = parts[1]
                    model = '_'.join(parts[2:])
                    
                    # 更新各维度最佳
                    if sampling not in sampling_best or result['f1_macro'] > sampling_best[sampling]['f1_macro']:
                        sampling_best[sampling] = {'key': key, 'f1_macro': result['f1_macro']}
                    
                    if feature not in feature_best or result['f1_macro'] > feature_best[feature]['f1_macro']:
                        feature_best[feature] = {'key': key, 'f1_macro': result['f1_macro']}
                    
                    if model not in model_best or result['f1_macro'] > model_best[model]['f1_macro']:
                        model_best[model] = {'key': key, 'f1_macro': result['f1_macro']}
            
            print(f"\n各维度最佳组合:")
            print(f"最佳重采样方法: {max(sampling_best.items(), key=lambda x: x[1]['f1_macro'])[0]} ({max(sampling_best.items(), key=lambda x: x[1]['f1_macro'])[1]['f1_macro']:.4f})")
            print(f"最佳特征方法: {max(feature_best.items(), key=lambda x: x[1]['f1_macro'])[0]} ({max(feature_best.items(), key=lambda x: x[1]['f1_macro'])[1]['f1_macro']:.4f})")
            print(f"最佳模型: {max(model_best.items(), key=lambda x: x[1]['f1_macro'])[0]} ({max(model_best.items(), key=lambda x: x[1]['f1_macro'])[1]['f1_macro']:.4f})")
            
            # Top 5 组合
            sorted_results = sorted(successful_results.items(), key=lambda x: x[1]['f1_macro'], reverse=True)
            print(f"\nTop 5 最佳组合:")
            for i, (key, result) in enumerate(sorted_results[:5], 1):
                print(f"{i}. {key}: {result['f1_macro']:.4f}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='综合交叉实验')
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
    experiment = ComprehensiveExperiment(config)
    
    # 运行实验
    results = experiment.run_comprehensive_experiment(args.sample_size)
    
    # 打印总结
    experiment.print_summary(results)


if __name__ == "__main__":
    main() 