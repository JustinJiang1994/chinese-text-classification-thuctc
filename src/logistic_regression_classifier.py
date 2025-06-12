#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于逻辑回归的中文新闻分类器
"""

import pandas as pd
import numpy as np
import jieba
import pickle
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LogisticRegressionNewsClassifier:
    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = LabelEncoder()
        self.stopwords = set()
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
    
    def load_data(self, max_samples_per_class=5000):
        """加载数据并平衡各类别样本数量"""
        print("正在加载数据...")
        
        categories = {
            '汽车': 'data/car_news.csv',
            '娱乐': 'data/entertainment_news.csv',
            '财经': 'data/finance_news.csv',
            '家居': 'data/home_news.csv',
            '房产': 'data/house_news.csv',
            '国际': 'data/international_news.csv',
            '军事': 'data/military_news.csv',
            '社会': 'data/society_news.csv',
            '体育': 'data/sports_news.csv',
            '科技': 'data/technology_news.csv'
        }
        
        data_list = []
        for category, file_path in categories.items():
            try:
                df = pd.read_csv(file_path)
                df['category'] = category
                
                # 限制每个类别的样本数量
                if len(df) > max_samples_per_class:
                    df = df.sample(n=max_samples_per_class, random_state=42)
                
                data_list.append(df)
                print(f"已加载 {category} 类别: {len(df)} 条数据")
            except Exception as e:
                print(f"加载 {file_path} 失败: {e}")
        
        self.data = pd.concat(data_list, ignore_index=True)
        print(f"总数据量: {len(self.data)} 条")
        return self.data
    
    def prepare_data(self, test_size=0.2):
        """准备训练和测试数据"""
        print("正在准备数据...")
        
        # 文本预处理
        self.data['processed_content'] = self.data['content'].apply(self.preprocess_text)
        
        # 编码标签
        y = self.label_encoder.fit_transform(self.data['category'])
        X = self.data['processed_content']
        
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
    
    def create_classifier(self, C=1.0, max_iter=1000):
        """创建逻辑回归分类器"""
        print("正在创建逻辑回归分类器...")
        
        self.classifier = LogisticRegression(
            C=C,                    # 正则化参数
            max_iter=max_iter,      # 最大迭代次数
            random_state=42,        # 随机种子
            class_weight='balanced', # 处理类别不平衡
            solver='liblinear',     # 使用liblinear求解器
            multi_class='ovr'       # 一对多策略
        )
        
        return self.classifier
    
    def hyperparameter_tuning(self, X_train_vec, y_train):
        """超参数调优"""
        print("正在进行超参数调优...")
        
        # 定义参数网格
        param_grid = {
            'C': [0.1, 1.0, 10.0, 100.0],
            'max_iter': [1000, 2000]
        }
        
        # 使用网格搜索
        grid_search = GridSearchCV(
            LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear', multi_class='ovr'),
            param_grid,
            cv=3,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        # 在训练集上进行网格搜索
        grid_search.fit(X_train_vec, y_train)
        
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
        
        # 使用最佳参数创建分类器
        self.classifier = grid_search.best_estimator_
        
        return self.classifier
    
    def train(self, use_grid_search=False):
        """训练模型"""
        print("正在训练模型...")
        
        # 向量化训练数据
        X_train_vec = self.vectorizer.fit_transform(self.X_train)
        X_test_vec = self.vectorizer.transform(self.X_test)
        
        if use_grid_search:
            # 使用网格搜索进行超参数调优
            self.hyperparameter_tuning(X_train_vec, self.y_train)
        else:
            # 使用默认参数训练
            self.classifier.fit(X_train_vec, self.y_train)
        
        # 预测
        self.y_pred = self.classifier.predict(X_test_vec)
        self.y_pred_proba = self.classifier.predict_proba(X_test_vec)
        
        # 计算评估指标
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.f1_macro = f1_score(self.y_test, self.y_pred, average='macro')
        self.f1_weighted = f1_score(self.y_test, self.y_pred, average='weighted')
        
        # 交叉验证
        cv_scores = cross_val_score(self.classifier, X_train_vec, self.y_train, cv=5, scoring='accuracy')
        self.cv_mean = cv_scores.mean()
        self.cv_std = cv_scores.std()
        
        print(f"训练完成！")
        print(f"准确率: {self.accuracy:.4f}")
        print(f"F1-Macro: {self.f1_macro:.4f}")
        print(f"F1-Weighted: {self.f1_weighted:.4f}")
        print(f"交叉验证: {self.cv_mean:.4f} (+/- {self.cv_std * 2:.4f})")
        
        return self.accuracy, self.f1_macro, self.f1_weighted
    
    def evaluate(self):
        """详细评估模型"""
        print("\n详细评估结果:")
        print("=" * 60)
        
        # 分类报告
        report = classification_report(
            self.y_test, 
            self.y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        print("\n各类别详细指标:")
        print("-" * 40)
        for class_name in self.label_encoder.classes_:
            if class_name in report:
                print(f"{class_name}:")
                print(f"  精确率: {report[class_name]['precision']:.4f}")
                print(f"  召回率: {report[class_name]['recall']:.4f}")
                print(f"  F1分数: {report[class_name]['f1-score']:.4f}")
                print(f"  支持数: {report[class_name]['support']}")
        
        return report
    
    def plot_confusion_matrix(self):
        """绘制混淆矩阵"""
        print("正在生成混淆矩阵...")
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('逻辑回归分类器混淆矩阵', fontsize=16, fontweight='bold')
        plt.xlabel('预测类别', fontsize=12)
        plt.ylabel('真实类别', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('results/logistic_regression_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, model_path='models/logistic_regression_model.pkl'):
        """保存模型"""
        print(f"正在保存模型到 {model_path}...")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'stopwords': self.stopwords
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print("模型保存完成！")
    
    def load_model(self, model_path='models/logistic_regression_model.pkl'):
        """加载模型"""
        print(f"正在加载模型从 {model_path}...")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        self.stopwords = model_data['stopwords']
        
        print("模型加载完成！")
    
    def predict(self, texts):
        """预测新文本的类别"""
        if self.classifier is None or self.vectorizer is None:
            raise ValueError("模型未训练，请先调用train()方法")
        
        # 预处理文本
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # 向量化
        X_vec = self.vectorizer.transform(processed_texts)
        
        # 预测
        predictions = self.classifier.predict(X_vec)
        probabilities = self.classifier.predict_proba(X_vec)
        
        # 转换回类别名称
        predicted_classes = self.label_encoder.inverse_transform(predictions)
        
        return predicted_classes, probabilities
    
    def run_complete_pipeline(self, use_grid_search=False):
        """运行完整的训练和评估流程"""
        print("开始逻辑回归新闻分类任务...")
        print("=" * 60)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 准备数据
        self.prepare_data()
        
        # 3. 创建向量化器
        self.create_vectorizer()
        
        # 4. 创建分类器
        self.create_classifier()
        
        # 5. 训练模型
        self.train(use_grid_search=use_grid_search)
        
        # 6. 评估模型
        self.evaluate()
        
        # 7. 绘制混淆矩阵
        self.plot_confusion_matrix()
        
        # 8. 保存模型
        self.save_model()
        
        print("\n" + "=" * 60)
        print("逻辑回归分类任务完成！")
        
        return {
            'accuracy': self.accuracy,
            'f1_macro': self.f1_macro,
            'f1_weighted': self.f1_weighted,
            'cv_mean': self.cv_mean,
            'cv_std': self.cv_std
        }

if __name__ == "__main__":
    # 创建分类器实例
    classifier = LogisticRegressionNewsClassifier()
    
    # 运行完整流程（不使用网格搜索以节省时间）
    results = classifier.run_complete_pipeline(use_grid_search=False)
    
    print(f"\n最终结果:")
    print(f"准确率: {results['accuracy']:.4f}")
    print(f"F1-Macro: {results['f1_macro']:.4f}")
    print(f"F1-Weighted: {results['f1_weighted']:.4f}")
    print(f"交叉验证: {results['cv_mean']:.4f} (+/- {results['cv_std'] * 2:.4f})") 