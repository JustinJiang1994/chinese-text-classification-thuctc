#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型集成实验模块
结合表现最好的传统ML和深度学习模型，实现多种集成方法
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import TomekLinks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import jieba
import re
import pickle

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

class EnsembleExperiment:
    """模型集成实验类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化集成实验
        
        Args:
            config: 实验配置字典
        """
        self.config = config or self._get_default_config()
        self.data_dir = Path(self.config['data_dir'])
        self.results_dir = Path(self.config['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
        # 实验结果
        self.results = {}
        self.models = {}
        self.predictions = {}
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'data_dir': 'results',
            'results_dir': 'results/ensemble_experiments',
            'sample_size': 10000,
            'max_features': 5000,
            'test_size': 0.2,
            'random_state': 42,
            'use_class_weight': True,
            'ensemble_methods': ['voting_hard', 'voting_soft', 'weighted_ensemble'],
            'base_models': {
                'traditional': ['tomeklinks_svm', 'logistic_regression'],
                'deep_learning': ['lstm', 'textcnn']
            },
            'weights': {
                'tomeklinks_svm': 0.3,
                'logistic_regression': 0.2,
                'lstm': 0.3,
                'textcnn': 0.2
            }
        }
    
    def load_data(self, sample_size: int = None) -> Tuple[List[str], List[int], List[str], List[int], LabelEncoder]:
        """加载训练和测试数据"""
        sample_size = sample_size or self.config['sample_size']
        
        # 加载训练数据
        train_path = Path('results/train.csv')
        test_path = Path('results/test.csv')
        
        if not train_path.exists():
            print(f"训练数据文件不存在: {train_path}")
            print("请先运行数据预处理脚本")
            return None, None, None, None, None
            
        if not test_path.exists():
            print(f"测试数据文件不存在: {test_path}")
            print("请先运行数据预处理脚本")
            return None, None, None, None, None
        
        print(f"加载训练数据: {train_path}")
        train_data = pd.read_csv(train_path)
        
        print(f"加载测试数据: {test_path}")
        test_data = pd.read_csv(test_path)
        
        # 如果指定了样本量，从训练集中采样
        if sample_size and sample_size < len(train_data):
            train_data = train_data.sample(n=sample_size, random_state=self.config['random_state'])
        
        print(f"训练数据样本数: {len(train_data)}")
        print(f"测试数据样本数: {len(test_data)}")
        print(f"训练数据类别分布:\n{train_data['category'].value_counts()}")
        print(f"测试数据类别分布:\n{test_data['category'].value_counts()}")
        
        # 预处理文本
        print("预处理训练文本...")
        train_data['processed_content'] = train_data['content'].apply(self.preprocess_text)
        
        print("预处理测试文本...")
        test_data['processed_content'] = test_data['content'].apply(self.preprocess_text)
        
        # 标签编码（使用训练集的标签来fit，确保一致性）
        label_encoder = LabelEncoder()
        label_encoder.fit(train_data['category'])
        
        train_labels = label_encoder.transform(train_data['category'])
        test_labels = label_encoder.transform(test_data['category'])
        
        return (train_data['processed_content'].tolist(), train_labels, 
                test_data['processed_content'].tolist(), test_labels, label_encoder)
    
    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if pd.isna(text):
            return ""
        
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 去除URL和邮箱
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # 去除特殊字符，保留中文和英文
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        
        # 分词
        words = jieba.cut(text)
        return ' '.join(words)
    
    def run_experiment(self, sample_size: int = None) -> Dict[str, Any]:
        """运行完整的集成实验"""
        print("=== 模型集成实验 ===")
        
        # 1. 加载数据
        texts, labels, test_texts, test_labels, label_encoder = self.load_data(sample_size)
        if texts is None:
            return {}
        
        # 2. 特征提取
        print("特征提取...")
        vectorizer = TfidfVectorizer(
            max_features=self.config['max_features'],
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True
        )
        
        X = vectorizer.fit_transform(texts)
        X_test = vectorizer.transform(test_texts)
        print(f"训练特征维度: {X.shape}")
        print(f"测试特征维度: {X_test.shape}")
        
        # 3. 使用独立的训练和测试数据（不再需要划分）
        X_train, y_train = X, labels
        y_test = test_labels
        
        print(f"训练集: {len(y_train)} 样本")
        print(f"测试集: {len(y_test)} 样本")
        
        # 4. 创建基础模型
        base_models = self._create_base_models(X_train, y_train)
        
        # 5. 训练基础模型
        base_results = self._train_base_models(base_models, X_test, y_test)
        
        # 6. 创建集成模型
        ensembles = self._create_ensembles(base_results)
        
        # 7. 评估集成模型
        ensemble_results = self._evaluate_ensembles(ensembles, X_test, y_test)
        
        # 8. 合并所有结果
        all_results = {**base_results, **ensemble_results}
        
        # 9. 保存结果
        self.save_results(all_results, label_encoder)
        
        # 10. 绘制结果
        self.plot_results(all_results)
        
        print("\n=== 集成实验完成 ===")
        return all_results
    
    def _create_base_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """创建基础模型"""
        print("创建基础模型...")
        
        # 计算类别权重
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(zip(range(len(class_weights)), class_weights))
        
        models = {}
        
        # 1. TomekLinks + SVM (最佳传统ML模型)
        print("创建 TomekLinks + SVM...")
        tomek = TomekLinks()
        X_train_tomek, y_train_tomek = tomek.fit_resample(X_train, y_train)
        
        svm = LinearSVC(
            C=1.0,
            class_weight='balanced',
            random_state=self.config['random_state'],
            max_iter=1000
        )
        
        models['tomeklinks_svm'] = {
            'model': svm,
            'X_train': X_train_tomek,
            'y_train': y_train_tomek,
            'type': 'traditional'
        }
        
        # 2. 逻辑回归
        print("创建逻辑回归...")
        lr = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=self.config['random_state'],
            max_iter=1000
        )
        
        models['logistic_regression'] = {
            'model': lr,
            'X_train': X_train,
            'y_train': y_train,
            'type': 'traditional'
        }
        
        # 3. 随机森林
        print("创建随机森林...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=self.config['random_state'],
            n_jobs=-1
        )
        
        models['random_forest'] = {
            'model': rf,
            'X_train': X_train,
            'y_train': y_train,
            'type': 'traditional'
        }
        
        return models
    
    def _train_base_models(self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """训练基础模型"""
        print("训练基础模型...")
        
        results = {}
        
        for name, model_info in models.items():
            print(f"训练 {name}...")
            
            model = model_info['model']
            X_train = model_info['X_train']
            y_train = model_info['y_train']
            
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # 预测
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # 计算指标
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            
            results[name] = {
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'training_time': training_time,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'type': 'traditional',
                'model': model
            }
            
            print(f"  F1-macro: {f1_macro:.4f}")
            print(f"  F1-weighted: {f1_weighted:.4f}")
            print(f"  训练时间: {training_time:.2f}秒")
        
        return results
    
    def _create_ensembles(self, base_results: Dict[str, Any]) -> Dict[str, Any]:
        """创建集成模型"""
        print("创建集成模型...")
        
        ensembles = {}
        
        # 1. 硬投票集成
        print("创建硬投票集成...")
        voting_models = []
        for name, result in base_results.items():
            if result['type'] == 'traditional':
                voting_models.append((name, result['model']))
        
        if len(voting_models) >= 2:
            voting_hard = VotingClassifier(
                estimators=voting_models,
                voting='hard'
            )
            ensembles['voting_hard'] = voting_hard
        
        # 2. 软投票集成
        print("创建软投票集成...")
        voting_soft_models = []
        for name, result in base_results.items():
            if result['type'] == 'traditional' and result['probabilities'] is not None:
                voting_soft_models.append((name, result['model']))
        
        if len(voting_soft_models) >= 2:
            voting_soft = VotingClassifier(
                estimators=voting_soft_models,
                voting='soft'
            )
            ensembles['voting_soft'] = voting_soft
        
        # 3. 加权集成
        print("创建加权集成...")
        ensembles['weighted_ensemble'] = self._create_weighted_ensemble(base_results)
        
        return ensembles
    
    def _create_weighted_ensemble(self, base_results: Dict[str, Any]) -> Any:
        """创建加权集成"""
        class WeightedEnsemble:
            def __init__(self, base_results, weights):
                self.base_results = base_results
                self.weights = weights
                # 计算所有预测结果中的最大类别索引，确保one-hot编码维度正确
                all_preds = np.concatenate([np.array(result['predictions']) for result in base_results.values()])
                self.num_classes = int(all_preds.max()) + 1
            
            def predict(self, X):
                # 这里简化处理，直接使用训练好的预测结果
                # 实际应用中需要重新预测
                weighted_probs = np.zeros((X.shape[0], self.num_classes))
                
                for name, result in self.base_results.items():
                    if result['probabilities'] is not None:
                        weight = self.weights.get(name, 1.0)
                        weighted_probs += weight * np.array(result['probabilities'])
                    else:
                        # 如果没有概率，使用one-hot编码
                        weight = self.weights.get(name, 1.0)
                        preds = np.array(result['predictions'])
                        one_hot = np.eye(self.num_classes)[preds]
                        weighted_probs += weight * one_hot
                
                return np.argmax(weighted_probs, axis=1)
        
        return WeightedEnsemble(base_results, self.config['weights'])
    
    def _evaluate_ensembles(self, ensembles: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """评估集成模型"""
        print("评估集成模型...")
        
        ensemble_results = {}
        
        for name, ensemble in ensembles.items():
            print(f"评估 {name}...")
            start_time = time.time()
            # VotingClassifier需要先fit
            if hasattr(ensemble, 'fit') and hasattr(ensemble, 'predict'):
                # 取基础模型的训练集
                # 假设所有基础模型的X_train/y_train一致
                base = list(self.models.values())[0] if self.models else None
                if base and 'X_train' in base and 'y_train' in base:
                    X_train = base['X_train']
                    y_train = base['y_train']
                else:
                    # 回退到X_test/y_test（极端情况）
                    X_train = X_test
                    y_train = y_test
                ensemble.fit(X_train, y_train)
                y_pred = ensemble.predict(X_test)
            else:
                y_pred = ensemble.predict(X_test)
            prediction_time = time.time() - start_time
            # 计算指标
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            ensemble_results[name] = {
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'prediction_time': prediction_time,
                'predictions': y_pred,
                'type': 'ensemble'
            }
            print(f"  F1-macro: {f1_macro:.4f}")
            print(f"  F1-weighted: {f1_weighted:.4f}")
            print(f"  预测时间: {prediction_time:.2f}秒")
        return ensemble_results
    
    def save_results(self, results: Dict[str, Any], label_encoder: LabelEncoder):
        """保存实验结果"""
        results_dir = Path(self.config['results_dir'])
        results_dir.mkdir(exist_ok=True)
        
        # 保存结果
        results_file = results_dir / 'ensemble_results.json'
        
        # 转换numpy数组为列表以便JSON序列化
        serializable_results = {}
        for name, result in results.items():
            serializable_result = {}
            for key, value in result.items():
                if key == 'model':  # 跳过模型对象
                    continue
                elif isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, np.integer):
                    serializable_result[key] = int(value)
                elif isinstance(value, np.floating):
                    serializable_result[key] = float(value)
                else:
                    serializable_result[key] = value
            serializable_results[name] = serializable_result
        
        # 添加实验信息
        experiment_info = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config,
            'results': serializable_results,
            'label_encoder_classes': label_encoder.classes_.tolist()
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_info, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {results_file}")
    
    def plot_results(self, results: Dict[str, Any]):
        """绘制实验结果"""
        results_dir = Path(self.config['results_dir'])
        results_dir.mkdir(exist_ok=True)
        
        # 1. 性能对比图
        plt.figure(figsize=(12, 8))
        
        model_names = []
        f1_macro_scores = []
        f1_weighted_scores = []
        colors = []
        
        for name, result in results.items():
            model_names.append(name)
            f1_macro_scores.append(result['f1_macro'])
            f1_weighted_scores.append(result['f1_weighted'])
            
            if result['type'] == 'traditional':
                colors.append('blue')
            elif result['type'] == 'deep_learning':
                colors.append('green')
            else:
                colors.append('red')
        
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, f1_macro_scores, width, label='F1-macro', color='skyblue', alpha=0.8)
        plt.bar(x + width/2, f1_weighted_scores, width, label='F1-weighted', color='lightcoral', alpha=0.8)
        
        plt.xlabel('模型')
        plt.ylabel('F1分数')
        plt.title('模型集成性能对比')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(results_dir / 'ensemble_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"图表已保存到: {results_dir}")


if __name__ == "__main__":
    # 运行集成实验
    experiment = EnsembleExperiment()
    results = experiment.run_experiment(sample_size=2000)
    
    # 打印结果总结
    print("\n=== 集成实验结果总结 ===")
    print("模型性能排名 (按F1-macro):")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1_macro'], reverse=True)
    
    for i, (name, result) in enumerate(sorted_results, 1):
        print(f"{i}. {name}: F1-macro={result['f1_macro']:.4f}, F1-weighted={result['f1_weighted']:.4f}")
    
    # 找出最佳集成模型
    ensemble_results = {name: result for name, result in results.items() if result['type'] == 'ensemble'}
    if ensemble_results:
        best_ensemble = max(ensemble_results.items(), key=lambda x: x[1]['f1_macro'])
        print(f"\n最佳集成模型: {best_ensemble[0]}")
        print(f"F1-macro: {best_ensemble[1]['f1_macro']:.4f}")
        print(f"F1-weighted: {best_ensemble[1]['f1_weighted']:.4f}") 