#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成学习分类器
实现多种集成学习方法
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class EnsembleNewsClassifier:
    """集成学习新闻分类器"""
    
    def __init__(self):
        self.vectorizer = None
        self.ensemble_classifier = None
        self.label_encoder = LabelEncoder()
        self.stopwords = set()
        self.base_classifiers = {}
        self.load_stopwords()
        
    def load_stopwords(self):
        """加载停用词"""
        try:
            with open('data/stopwords.txt', 'r', encoding='utf-8') as f:
                self.stopwords = set(line.strip() for line in f)
            print(f"已加载 {len(self.stopwords)} 个停用词")
        except Exception as e:
            print(f"加载停用词失败: {e}")
    
    def preprocess_text(self, text):
        """文本预处理：分词并去除停用词"""
        if pd.isna(text):
            return ""
        
        # 使用jieba分词
        words = jieba.cut(str(text))
        # 过滤停用词和空字符
        filtered_words = [word for word in words if word.strip() and word not in self.stopwords]
        return ' '.join(filtered_words)
    
    def load_data(self, data_file='data/news_data.csv', max_samples_per_class=5000):
        """加载数据"""
        print("正在加载数据...")
        
        try:
            df = pd.read_csv(data_file)
            print(f"数据加载成功，共 {len(df)} 条记录")
            
            # 限制每个类别的样本数量
            data_list = []
            for category in df['category'].unique():
                category_df = df[df['category'] == category]
                if len(category_df) > max_samples_per_class:
                    category_df = category_df.sample(n=max_samples_per_class, random_state=42)
                data_list.append(category_df)
            
            df = pd.concat(data_list, ignore_index=True)
            print(f"平衡后数据量: {len(df)} 条")
            
            return df
        except FileNotFoundError:
            print(f"错误: 数据文件 {data_file} 不存在")
            return None
    
    def prepare_data(self, df, test_size=0.2):
        """准备训练和测试数据"""
        print("正在准备数据...")
        
        # 文本预处理
        df['processed_content'] = df['content'].apply(self.preprocess_text)
        
        # 编码标签
        y = self.label_encoder.fit_transform(df['category'])
        X = df['processed_content']
        
        # 分割数据
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"训练集大小: {len(self.X_train)}")
        print(f"测试集大小: {len(self.X_test)}")
        print(f"类别数量: {len(self.label_encoder.classes_)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def create_vectorizer(self, max_features=10000):
        """创建TF-IDF向量化器"""
        print("正在创建TF-IDF向量化器...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # 使用1-gram和2-gram
            min_df=2,            # 最小文档频率
            max_df=0.95,         # 最大文档频率
            stop_words=None      # 我们已经在预处理中处理了停用词
        )
        
        return self.vectorizer
    
    def create_base_classifiers(self):
        """创建基础分类器"""
        print("正在创建基础分类器...")
        
        # 逻辑回归
        self.base_classifiers['lr'] = LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=42,
            class_weight='balanced',
            solver='liblinear',
            multi_class='ovr'
        )
        
        # 支持向量机
        self.base_classifiers['svc'] = SVC(
            C=1.0,
            kernel='linear',
            random_state=42,
            class_weight='balanced',
            probability=True
        )
        
        # 朴素贝叶斯
        self.base_classifiers['nb'] = MultinomialNB(alpha=0.5)
        
        # 随机森林
        self.base_classifiers['rf'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        print(f"已创建 {len(self.base_classifiers)} 个基础分类器")
        return self.base_classifiers
    
    def create_voting_classifier(self, voting='soft', weights=None):
        """创建投票集成分类器"""
        print("正在创建投票集成分类器...")
        
        if weights is None:
            weights = [0.3, 0.3, 0.2, 0.2]  # 默认权重
        
        estimators = [
            ('lr', self.base_classifiers['lr']),
            ('svc', self.base_classifiers['svc']),
            ('nb', self.base_classifiers['nb']),
            ('rf', self.base_classifiers['rf'])
        ]
        
        self.ensemble_classifier = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights
        )
        
        print(f"投票集成分类器创建完成 (voting={voting})")
        return self.ensemble_classifier
    
    def create_stacking_classifier(self, cv=5):
        """创建Stacking集成分类器"""
        print("正在创建Stacking集成分类器...")
        
        estimators = [
            ('lr', self.base_classifiers['lr']),
            ('svc', self.base_classifiers['svc']),
            ('nb', self.base_classifiers['nb']),
            ('rf', self.base_classifiers['rf'])
        ]
        
        # 元分类器
        meta_classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        
        self.ensemble_classifier = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_classifier,
            cv=cv,
            n_jobs=-1
        )
        
        print("Stacking集成分类器创建完成")
        return self.ensemble_classifier
    
    def train_individual_classifiers(self):
        """训练各个基础分类器"""
        print("正在训练基础分类器...")
        
        # 向量化数据
        X_train_vec = self.vectorizer.fit_transform(self.X_train)
        X_test_vec = self.vectorizer.transform(self.X_test)
        
        individual_results = {}
        
        for name, classifier in self.base_classifiers.items():
            print(f"训练 {name} 分类器...")
            
            # 训练分类器
            classifier.fit(X_train_vec, self.y_train)
            
            # 预测
            y_pred = classifier.predict(X_test_vec)
            
            # 计算评估指标
            accuracy = accuracy_score(self.y_test, y_pred)
            f1_macro = f1_score(self.y_test, y_pred, average='macro')
            f1_weighted = f1_score(self.y_test, y_pred, average='weighted')
            
            individual_results[name] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'predictions': y_pred
            }
            
            print(f"  {name} - 准确率: {accuracy:.4f}, F1-Macro: {f1_macro:.4f}")
        
        return individual_results
    
    def train_ensemble(self, ensemble_type='voting', **kwargs):
        """训练集成分类器"""
        print(f"正在训练{ensemble_type}集成分类器...")
        
        # 向量化数据
        X_train_vec = self.vectorizer.fit_transform(self.X_train)
        X_test_vec = self.vectorizer.transform(self.X_test)
        
        # 创建集成分类器
        if ensemble_type == 'voting':
            voting = kwargs.get('voting', 'soft')
            weights = kwargs.get('weights', None)
            self.create_voting_classifier(voting=voting, weights=weights)
        elif ensemble_type == 'stacking':
            cv = kwargs.get('cv', 5)
            self.create_stacking_classifier(cv=cv)
        else:
            raise ValueError(f"不支持的集成类型: {ensemble_type}")
        
        # 训练集成分类器
        self.ensemble_classifier.fit(X_train_vec, self.y_train)
        
        # 预测
        self.y_pred = self.ensemble_classifier.predict(X_test_vec)
        self.y_pred_proba = self.ensemble_classifier.predict_proba(X_test_vec)
        
        # 计算评估指标
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.f1_macro = f1_score(self.y_test, self.y_pred, average='macro')
        self.f1_weighted = f1_score(self.y_test, self.y_pred, average='weighted')
        
        # 交叉验证
        cv_scores = cross_val_score(self.ensemble_classifier, X_train_vec, self.y_train, cv=5, scoring='accuracy')
        self.cv_mean = cv_scores.mean()
        self.cv_std = cv_scores.std()
        
        print(f"集成分类器训练完成!")
        print(f"准确率: {self.accuracy:.4f}")
        print(f"F1-Macro: {self.f1_macro:.4f}")
        print(f"F1-Weighted: {self.f1_weighted:.4f}")
        print(f"交叉验证: {self.cv_mean:.4f} (+/- {self.cv_std * 2:.4f})")
        
        return self.accuracy, self.f1_macro, self.f1_weighted
    
    def evaluate(self):
        """评估模型"""
        print("正在评估集成分类器...")
        
        # 分类报告
        print("\n分类报告:")
        print(classification_report(
            self.y_test, self.y_pred,
            target_names=self.label_encoder.classes_,
            digits=4
        ))
        
        # 各类别详细表现
        print("\n各类别详细表现:")
        print("-" * 60)
        print(f"{'类别':<8} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'支持数':<8}")
        print("-" * 60)
        
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_test, self.y_pred, average=None
        )
        
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"{class_name:<8} {precision[i]:<8.4f} {recall[i]:<8.4f} {f1[i]:<8.4f} {support[i]:<8}")
        
        return {
            'accuracy': self.accuracy,
            'f1_macro': self.f1_macro,
            'f1_weighted': self.f1_weighted,
            'cv_mean': self.cv_mean,
            'cv_std': self.cv_std,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    
    def plot_confusion_matrix(self):
        """绘制混淆矩阵"""
        print("正在生成混淆矩阵...")
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title('集成分类器混淆矩阵', fontsize=16, fontweight='bold')
        plt.xlabel('预测类别', fontsize=12)
        plt.ylabel('真实类别', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('results/ensemble_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_classifiers(self, individual_results):
        """比较各个分类器的性能"""
        print("正在比较分类器性能...")
        
        # 准备数据
        classifiers = list(individual_results.keys()) + ['ensemble']
        accuracies = [individual_results[name]['accuracy'] for name in individual_results.keys()]
        accuracies.append(self.accuracy)
        
        f1_scores = [individual_results[name]['f1_macro'] for name in individual_results.keys()]
        f1_scores.append(self.f1_macro)
        
        # 绘制比较图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 准确率比较
        bars1 = ax1.bar(classifiers, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'purple'])
        ax1.set_title('分类器准确率比较', fontsize=14, fontweight='bold')
        ax1.set_ylabel('准确率')
        ax1.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # F1分数比较
        bars2 = ax2.bar(classifiers, f1_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'purple'])
        ax2.set_title('分类器F1-Macro分数比较', fontsize=14, fontweight='bold')
        ax2.set_ylabel('F1-Macro分数')
        ax2.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, f1 in zip(bars2, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{f1:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results/classifier_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印详细比较结果
        print("\n分类器性能详细比较:")
        print("-" * 60)
        print(f"{'分类器':<12} {'准确率':<8} {'F1-Macro':<8} {'F1-Weighted':<8}")
        print("-" * 60)
        
        for name in individual_results.keys():
            acc = individual_results[name]['accuracy']
            f1_macro = individual_results[name]['f1_macro']
            f1_weighted = individual_results[name]['f1_weighted']
            print(f"{name:<12} {acc:<8.4f} {f1_macro:<8.4f} {f1_weighted:<8.4f}")
        
        print(f"{'ensemble':<12} {self.accuracy:<8.4f} {self.f1_macro:<8.4f} {self.f1_weighted:<8.4f}")
    
    def save_model(self, model_path='models/ensemble_model.pkl'):
        """保存模型"""
        print(f"正在保存模型到 {model_path}...")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'vectorizer': self.vectorizer,
            'ensemble_classifier': self.ensemble_classifier,
            'base_classifiers': self.base_classifiers,
            'label_encoder': self.label_encoder,
            'stopwords': self.stopwords
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print("模型保存完成!")
    
    def load_model(self, model_path='models/ensemble_model.pkl'):
        """加载模型"""
        print(f"正在加载模型从 {model_path}...")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.ensemble_classifier = model_data['ensemble_classifier']
        self.base_classifiers = model_data['base_classifiers']
        self.label_encoder = model_data['label_encoder']
        self.stopwords = model_data['stopwords']
        
        print("模型加载完成!")
    
    def predict(self, texts):
        """预测新文本的类别"""
        if self.ensemble_classifier is None or self.vectorizer is None:
            raise ValueError("模型未训练，请先调用train_ensemble()方法")
        
        # 预处理文本
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # 向量化
        X_vec = self.vectorizer.transform(processed_texts)
        
        # 预测
        predictions = self.ensemble_classifier.predict(X_vec)
        probabilities = self.ensemble_classifier.predict_proba(X_vec)
        
        # 转换回类别名称
        predicted_classes = self.label_encoder.inverse_transform(predictions)
        
        return predicted_classes, probabilities
    
    def run_complete_pipeline(self, ensemble_type='voting', **kwargs):
        """运行完整的训练和评估流程"""
        print("开始集成学习新闻分类任务...")
        print("=" * 60)
        
        # 1. 加载数据
        df = self.load_data()
        if df is None:
            return None
        
        # 2. 准备数据
        self.prepare_data(df)
        
        # 3. 创建向量化器
        self.create_vectorizer()
        
        # 4. 创建基础分类器
        self.create_base_classifiers()
        
        # 5. 训练基础分类器
        individual_results = self.train_individual_classifiers()
        
        # 6. 训练集成分类器
        self.train_ensemble(ensemble_type=ensemble_type, **kwargs)
        
        # 7. 评估模型
        results = self.evaluate()
        
        # 8. 绘制混淆矩阵
        self.plot_confusion_matrix()
        
        # 9. 比较分类器性能
        self.compare_classifiers(individual_results)
        
        # 10. 保存模型
        self.save_model()
        
        print("\n" + "=" * 60)
        print("集成学习分类任务完成！")
        
        return results


if __name__ == "__main__":
    # 创建集成分类器
    ensemble_classifier = EnsembleNewsClassifier()
    
    # 运行完整流程
    results = ensemble_classifier.run_complete_pipeline(ensemble_type='voting')
    
    if results:
        print(f"\n最终结果:")
        print(f"准确率: {results['accuracy']:.4f}")
        print(f"F1-Macro: {results['f1_macro']:.4f}")
        print(f"F1-Weighted: {results['f1_weighted']:.4f}") 