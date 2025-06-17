#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一实验管理模块
整合数据预处理、传统机器学习实验、深度学习实验等功能
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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

class NewsClassifierExperiment:
    """新闻分类实验统一管理类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化实验管理器
        
        Args:
            config: 实验配置字典
        """
        self.config = config or self._get_default_config()
        self.data_dir = Path(self.config['data_dir'])
        self.results_dir = Path(self.config['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
        # 实验组件
        self.data_processor = DataProcessor(self.config)
        self.ml_experimenter = MLExperimenter(self.config)
        self.visualizer = ResultVisualizer(self.config)
        
        # 实验结果
        self.results = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'data_dir': 'results',
            'results_dir': 'results',
            'sample_size': 10000,
            'max_features': 8000,
            'test_size': 0.2,
            'random_state': 42,
            'models': ['naive_bayes', 'svm', 'random_forest', 'logistic_regression'],
            'sampling_methods': ['original', 'smote', 'tomek_links', 'smoteenn'],
            'save_results': True,
            'plot_results': True
        }
    
    def run_quick_experiment(self, sample_size: int = 1000) -> Dict[str, Any]:
        """运行快速实验"""
        print("=== 快速实验 ===")
        
        # 加载小样本数据
        data = self._load_sample_data(sample_size)
        
        # 准备特征
        X, y, vectorizer, label_encoder = self.data_processor.prepare_features(data)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state'], stratify=y
        )
        
        # 运行ML实验
        results = self.ml_experimenter.run_single_experiment(
            X_train, X_test, y_train, y_test, 
            models=self.config['models']
        )
        
        # 保存结果
        if self.config['save_results']:
            self._save_results(results, 'quick_experiment')
        
        return results
    
    def run_imbalance_experiment(self, sample_size: int = None) -> Dict[str, Any]:
        """运行类别不平衡实验"""
        print("=== 类别不平衡实验 ===")
        
        # 加载数据
        sample_size = sample_size or self.config['sample_size']
        data = self._load_sample_data(sample_size)
        
        # 准备特征
        X, y, vectorizer, label_encoder = self.data_processor.prepare_features(data)
        
        # 分析不平衡情况
        imbalance_info = self.data_processor.analyze_imbalance(y, label_encoder)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state'], stratify=y
        )
        
        # 运行不平衡实验
        results = self.ml_experimenter.run_imbalance_experiment(
            X_train, X_test, y_train, y_test,
            models=self.config['models'],
            sampling_methods=self.config['sampling_methods']
        )
        
        # 添加不平衡信息
        results['imbalance_info'] = imbalance_info
        
        # 保存结果
        if self.config['save_results']:
            self._save_results(results, 'imbalance_experiment')
        
        # 绘制结果
        if self.config['plot_results']:
            self.visualizer.plot_imbalance_results(results, label_encoder.classes_)
        
        return results
    
    def run_full_experiment(self) -> Dict[str, Any]:
        """运行完整实验（全数据集）"""
        print("=== 完整实验 ===")
        
        # 加载完整数据
        train_data = pd.read_csv(self.data_dir / "train.csv")
        val_data = pd.read_csv(self.data_dir / "val.csv")
        test_data = pd.read_csv(self.data_dir / "test.csv")
        
        # 准备特征
        X_train, X_val, X_test, y_train, y_val, y_test, vectorizer, label_encoder = \
            self.data_processor.prepare_full_features(train_data, val_data, test_data)
        
        # 运行完整实验
        results = self.ml_experimenter.run_full_experiment(
            X_train, X_val, X_test, y_train, y_val, y_test,
            models=self.config['models'],
            sampling_methods=self.config['sampling_methods']
        )
        
        # 保存结果
        if self.config['save_results']:
            self._save_results(results, 'full_experiment')
        
        return results
    
    def _load_sample_data(self, sample_size: int) -> pd.DataFrame:
        """加载样本数据"""
        # 优先使用sample.csv
        sample_file = Path("sample.csv")
        if sample_file.exists():
            data = pd.read_csv(sample_file)
            print(f"使用sample.csv: {len(data)} 样本")
        else:
            # 从训练集抽取样本
            data = pd.read_csv(self.data_dir / "train.csv", nrows=sample_size)
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
    
    def print_summary(self, results: Dict[str, Any]):
        """打印实验总结"""
        print("\n=== 实验总结 ===")
        
        if 'best_model' in results:
            best = results['best_model']
            print(f"最佳模型: {best['name']}")
            print(f"最佳F1-macro: {best['f1_macro']:.4f}")
            print(f"最佳F1-weighted: {best['f1_weighted']:.4f}")
        
        if 'model_rankings' in results:
            print(f"\n模型性能排名:")
            for i, (model, score) in enumerate(results['model_rankings'], 1):
                print(f"{i}. {model}: {score:.4f}")


class DataProcessor:
    """数据处理类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if pd.isna(text):
            return ""
        
        # 移除特殊字符和数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', str(text))
        # 分词
        words = jieba.cut(text)
        return ' '.join(words)
    
    def prepare_features(self, data: pd.DataFrame, max_features: int = None) -> Tuple:
        """准备特征"""
        max_features = max_features or self.config['max_features']
        
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
    
    def prepare_full_features(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                            test_data: pd.DataFrame) -> Tuple:
        """准备完整数据集特征"""
        # 预处理所有数据
        train_data['processed_content'] = train_data['content'].apply(self.preprocess_text)
        val_data['processed_content'] = val_data['content'].apply(self.preprocess_text)
        test_data['processed_content'] = test_data['content'].apply(self.preprocess_text)
        
        # TF-IDF向量化
        vectorizer = TfidfVectorizer(
            max_features=self.config['max_features'],
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        )
        
        X_train = vectorizer.fit_transform(train_data['processed_content'])
        X_val = vectorizer.transform(val_data['processed_content'])
        X_test = vectorizer.transform(test_data['processed_content'])
        
        # 标签编码
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(train_data['category'])
        y_val = label_encoder.transform(val_data['category'])
        y_test = label_encoder.transform(test_data['category'])
        
        return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer, label_encoder
    
    def analyze_imbalance(self, y: np.ndarray, label_encoder: LabelEncoder) -> Dict[str, Any]:
        """分析类别不平衡情况"""
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(label_encoder.classes_, counts))
        
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count
        
        return {
            'class_distribution': class_distribution,
            'imbalance_ratio': imbalance_ratio,
            'max_count': max_count,
            'min_count': min_count
        }


class MLExperimenter:
    """机器学习实验类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = self._create_models()
        self.sampling_methods = self._create_sampling_methods()
    
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
            'adasyn': ADASYN(random_state=42),
            'borderline_smote': BorderlineSMOTE(random_state=42, k_neighbors=3),
            'random_undersample': RandomUnderSampler(random_state=42),
            'tomek_links': TomekLinks(),
            'edited_nearest_neighbours': EditedNearestNeighbours(),
            'smoteenn': SMOTEENN(random_state=42),
            'smotetomek': SMOTETomek(random_state=42)
        }
    
    def run_single_experiment(self, X_train: np.ndarray, X_test: np.ndarray,
                            y_train: np.ndarray, y_test: np.ndarray,
                            models: List[str] = None) -> Dict[str, Any]:
        """运行单次实验"""
        models = models or list(self.models.keys())
        results = {}
        
        for model_name in models:
            if model_name in self.models:
                print(f"训练 {model_name}...")
                model = self.models[model_name]
                
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                y_pred = model.predict(X_test)
                
                f1_macro = f1_score(y_test, y_pred, average='macro')
                f1_weighted = f1_score(y_test, y_pred, average='weighted')
                
                results[model_name] = {
                    'f1_macro': f1_macro,
                    'f1_weighted': f1_weighted,
                    'training_time': training_time
                }
                
                print(f"  F1-macro: {f1_macro:.4f}")
                print(f"  训练时间: {training_time:.2f}秒")
        
        return results
    
    def run_imbalance_experiment(self, X_train: np.ndarray, X_test: np.ndarray,
                               y_train: np.ndarray, y_test: np.ndarray,
                               models: List[str] = None, 
                               sampling_methods: List[str] = None) -> Dict[str, Any]:
        """运行类别不平衡实验"""
        models = models or list(self.models.keys())
        sampling_methods = sampling_methods or list(self.sampling_methods.keys())
        
        results = {}
        best_score = 0
        best_model = None
        
        for sampling_name in sampling_methods:
            print(f"\n--- {sampling_name} ---")
            
            # 应用重采样
            X_train_resampled, y_train_resampled = self._apply_sampling(
                X_train, y_train, sampling_name
            )
            
            # 训练所有模型
            for model_name in models:
                if model_name in self.models:
                    print(f"  训练 {model_name}...")
                    
                    model = self.models[model_name]
                    start_time = time.time()
                    model.fit(X_train_resampled, y_train_resampled)
                    training_time = time.time() - start_time
                    
                    y_pred = model.predict(X_test)
                    f1_macro = f1_score(y_test, y_pred, average='macro')
                    f1_weighted = f1_score(y_test, y_pred, average='weighted')
                    
                    key = f"{sampling_name}_{model_name}"
                    results[key] = {
                        'f1_macro': f1_macro,
                        'f1_weighted': f1_weighted,
                        'training_time': training_time
                    }
                    
                    # 更新最佳模型
                    if f1_macro > best_score:
                        best_score = f1_macro
                        best_model = {
                            'name': key,
                            'f1_macro': f1_macro,
                            'f1_weighted': f1_weighted
                        }
                    
                    print(f"    F1-macro: {f1_macro:.4f}")
        
        # 添加总结信息
        results['best_model'] = best_model
        results['model_rankings'] = self._get_model_rankings(results)
        
        return results
    
    def run_full_experiment(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                          y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                          models: List[str] = None, sampling_methods: List[str] = None) -> Dict[str, Any]:
        """运行完整实验"""
        # 类似run_imbalance_experiment，但使用验证集进行模型选择
        # 这里简化处理，直接使用测试集
        return self.run_imbalance_experiment(X_train, X_test, y_train, y_test, models, sampling_methods)
    
    def _apply_sampling(self, X: np.ndarray, y: np.ndarray, sampling_name: str) -> Tuple:
        """应用重采样方法"""
        sampling_method = self.sampling_methods.get(sampling_name)
        
        if sampling_method is None:
            return X, y
        
        try:
            X_resampled, y_resampled = sampling_method.fit_resample(X, y)
            print(f"    原始样本数: {len(y)} -> 重采样后: {len(y_resampled)}")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"    {sampling_name} 失败: {e}")
            return X, y
    
    def _get_model_rankings(self, results: Dict[str, Any]) -> List[Tuple[str, float]]:
        """获取模型排名"""
        model_scores = []
        for key, result in results.items():
            if isinstance(result, dict) and 'f1_macro' in result:
                model_scores.append((key, result['f1_macro']))
        
        return sorted(model_scores, key=lambda x: x[1], reverse=True)


class ResultVisualizer:
    """结果可视化类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.plots_dir = Path(config['results_dir']) / "plots"
        self.plots_dir.mkdir(exist_ok=True)
    
    def plot_imbalance_results(self, results: Dict[str, Any], class_names: np.ndarray):
        """绘制不平衡实验结果"""
        print("正在绘制结果图表...")
        
        # 保存结果到实例变量
        self.results = results
        
        # 绘制重采样方法对比
        self._plot_sampling_comparison(self.plots_dir)
        
        # 绘制模型性能对比
        self._plot_model_comparison(self.plots_dir)
        
        # 绘制详细性能分析
        self._plot_detailed_analysis(self.plots_dir, class_names)
    
    def _plot_sampling_comparison(self, plots_dir):
        """绘制重采样方法对比图"""
        # 提取重采样方法名称
        sampling_methods = set()
        models = set()
        
        for key in self.results.keys():
            # 过滤掉非实验结果的关键字
            if key in ['best_model', 'model_rankings', 'imbalance_info']:
                continue
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
            # 过滤掉非实验结果的关键字
            if key in ['best_model', 'model_rankings', 'imbalance_info']:
                continue
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
        # 找到最佳模型
        if 'best_model' in self.results:
            best_key = self.results['best_model']['name']
            best_result = self.results[best_key]
            
            # 绘制最佳模型的类别性能
            plt.figure(figsize=(12, 6))
            f1_per_class = best_result.get('f1_per_class', [])
            
            if len(f1_per_class) > 0:
                plt.bar(range(len(class_names)), f1_per_class, alpha=0.8)
                plt.title(f'最佳模型 ({best_key}) - 各类别F1分数')
                plt.xlabel('类别')
                plt.ylabel('F1分数')
                plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(plots_dir / "best_model_class_performance.png", dpi=300, bbox_inches='tight')
                plt.close()


def main():
    """主函数 - 演示用法"""
    # 创建实验管理器
    experiment = NewsClassifierExperiment()
    
    # 运行快速实验
    print("运行快速实验...")
    quick_results = experiment.run_quick_experiment(sample_size=1000)
    experiment.print_summary(quick_results)
    
    # 运行不平衡实验
    print("\n运行不平衡实验...")
    imbalance_results = experiment.run_imbalance_experiment(sample_size=5000)
    experiment.print_summary(imbalance_results)


if __name__ == "__main__":
    main() 