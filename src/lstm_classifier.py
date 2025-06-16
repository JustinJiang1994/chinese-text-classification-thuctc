#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于LSTM的中文新闻分类器
使用深度学习LSTM网络进行文本分类
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import jieba
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
except ImportError:
    print("请安装tensorflow: pip install tensorflow")
    exit(1)


class LSTMClassifier:
    """基于LSTM的深度学习分类器"""
    
    def __init__(self, max_words=10000, max_len=200, embedding_dim=128, lstm_units=128):
        """
        初始化LSTM分类器
        
        Args:
            max_words: 词汇表大小
            max_len: 序列最大长度
            embedding_dim: 词嵌入维度
            lstm_units: LSTM单元数
        """
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        self.model = None
        self.class_names = None
        self.history = None
        self.class_weights = None
        
    def load_stopwords(self, stopwords_file='data/stopwords.txt'):
        """加载停用词"""
        try:
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                stopwords = set([line.strip() for line in f])
            return stopwords
        except FileNotFoundError:
            print(f"警告: 停用词文件 {stopwords_file} 不存在，使用空停用词表")
            return set()
    
    def preprocess_text(self, text, stopwords):
        """文本预处理"""
        if pd.isna(text):
            return ""
        
        # 去除特殊字符和数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', str(text))
        
        # 分词
        words = jieba.lcut(text)
        
        # 去除停用词和短词
        words = [word for word in words if word not in stopwords and len(word) > 1]
        
        return ' '.join(words)
    
    def load_data(self, data_file='data/news_data.csv'):
        """加载数据"""
        print("正在加载数据...")
        try:
            df = pd.read_csv(data_file)
            print(f"数据加载成功，共 {len(df)} 条记录")
            return df
        except FileNotFoundError:
            print(f"错误: 数据文件 {data_file} 不存在")
            return None
    
    def prepare_data(self, df, stopwords):
        """准备训练数据"""
        print("正在预处理数据...")
        
        # 文本预处理
        df['processed_text'] = df['content'].apply(lambda x: self.preprocess_text(x, stopwords))
        
        # 移除空文本
        df = df[df['processed_text'].str.len() > 0].reset_index(drop=True)
        print(f"预处理后剩余 {len(df)} 条记录")
        
        # 标签编码
        self.class_names = df['category'].unique()
        y = self.label_encoder.fit_transform(df['category'])
        
        # 文本序列化
        texts = df['processed_text'].tolist()
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        
        # 转换为序列
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        # 转换为one-hot编码
        y_categorical = to_categorical(y, num_classes=len(self.class_names))
        
        print(f"词汇表大小: {len(self.tokenizer.word_index) + 1}")
        print(f"类别数量: {len(self.class_names)}")
        print(f"输入形状: {X.shape}")
        print(f"标签形状: {y_categorical.shape}")
        
        # 计算类别权重
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(len(self.class_names)),
            y=y
        )
        self.class_weights = dict(enumerate(class_weights))
        print("类别权重:")
        for idx, weight in self.class_weights.items():
            print(f"  {self.class_names[idx]}: {weight:.4f}")
        return X, y_categorical, y
    
    def build_model(self, num_classes):
        """构建LSTM模型"""
        print("正在构建LSTM模型...")
        
        model = Sequential([
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_len),
            Bidirectional(LSTM(self.lstm_units, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(self.lstm_units // 2)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("模型结构:")
        model.summary()
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """训练模型"""
        print("正在训练LSTM模型...")
        
        # 回调函数
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
        ]
        
        # 训练模型
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            class_weight=self.class_weights
        )
        
        print("模型训练完成!")
        
    def evaluate(self, X_test, y_test, y_test_original):
        """评估模型"""
        print("正在评估模型...")
        
        # 预测
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 计算准确率
        accuracy = accuracy_score(y_test_original, y_pred)
        print(f"测试集准确率: {accuracy:.4f}")
        
        # 分类报告
        print("\n分类报告:")
        print(classification_report(y_test_original, y_pred, 
                                  target_names=self.class_names))
        
        return y_pred, y_pred_proba, accuracy
    
    def plot_training_history(self, save_path='results/lstm_training_history.png'):
        """绘制训练历史"""
        if self.history is None:
            print("没有训练历史可绘制")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 准确率
        ax1.plot(self.history.history['accuracy'], label='训练准确率')
        ax1.plot(self.history.history['val_accuracy'], label='验证准确率')
        ax1.set_title('模型准确率')
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('准确率')
        ax1.legend()
        ax1.grid(True)
        
        # 损失
        ax2.plot(self.history.history['loss'], label='训练损失')
        ax2.plot(self.history.history['val_loss'], label='验证损失')
        ax2.set_title('模型损失')
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('损失')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"训练历史图已保存到: {save_path}")
    
    def plot_confusion_matrix(self, y_test, y_pred, save_path='results/lstm_confusion_matrix.png'):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('LSTM分类器混淆矩阵')
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"混淆矩阵已保存到: {save_path}")
    
    def save_model(self, model_path='models/lstm_model.pkl'):
        """保存模型"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 保存模型组件
        model_data = {
            'model': self.model,
            'tokenizer': self.tokenizer,
            'label_encoder': self.label_encoder,
            'class_names': self.class_names,
            'max_words': self.max_words,
            'max_len': self.max_len,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path='models/lstm_model.pkl'):
        """加载模型"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.tokenizer = model_data['tokenizer']
            self.label_encoder = model_data['label_encoder']
            self.class_names = model_data['class_names']
            self.max_words = model_data['max_words']
            self.max_len = model_data['max_len']
            self.embedding_dim = model_data['embedding_dim']
            self.lstm_units = model_data['lstm_units']
            
            print(f"模型已从 {model_path} 加载")
            return True
        except FileNotFoundError:
            print(f"模型文件 {model_path} 不存在")
            return False
    
    def predict(self, text, stopwords):
        """预测单个文本"""
        if self.model is None:
            print("模型未加载，请先训练或加载模型")
            return None
        
        # 预处理文本
        processed_text = self.preprocess_text(text, stopwords)
        
        # 转换为序列
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_len, padding='post', truncating='post')
        
        # 预测
        prediction = self.model.predict(padded_sequence)
        predicted_class = np.argmax(prediction[0])
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
        confidence = prediction[0][predicted_class]
        
        return predicted_label, confidence, prediction[0]
    
    def run_full_pipeline(self, data_file='data/news_data.csv', 
                         stopwords_file='data/stopwords.txt',
                         test_size=0.2, random_state=42):
        """运行完整的训练和评估流程"""
        print("=" * 60)
        print("LSTM分类器完整流程")
        print("=" * 60)
        
        # 加载数据
        df = self.load_data(data_file)
        if df is None:
            return None
        
        # 加载停用词
        stopwords = self.load_stopwords(stopwords_file)
        
        # 准备数据
        X, y_categorical, y_original = self.prepare_data(df, stopwords)
        
        # 划分数据集
        X_train, X_test, y_train, y_test, y_train_orig, y_test_orig = train_test_split(
            X, y_categorical, y_original, test_size=test_size, random_state=random_state, stratify=y_original
        )
        
        # 进一步划分验证集
        X_train, X_val, y_train, y_val, y_train_orig, y_val_orig = train_test_split(
            X_train, y_train, y_train_orig, test_size=0.2, random_state=random_state, stratify=y_train_orig
        )
        
        print(f"训练集大小: {X_train.shape[0]}")
        print(f"验证集大小: {X_val.shape[0]}")
        print(f"测试集大小: {X_test.shape[0]}")
        
        # 构建模型
        self.model = self.build_model(len(self.class_names))
        
        # 训练模型
        self.train(X_train, y_train, X_val, y_val)
        
        # 评估模型
        y_pred, y_pred_proba, accuracy = self.evaluate(X_test, y_test, y_test_orig)
        
        # 绘制图表
        self.plot_training_history()
        self.plot_confusion_matrix(y_test_orig, y_pred)
        
        # 保存模型
        self.save_model()
        
        # 保存结果
        results = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_labels': y_test_orig,
            'class_names': self.class_names
        }
        
        results_path = 'results/lstm_results.pkl'
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"结果已保存到: {results_path}")
        
        return accuracy


if __name__ == "__main__":
    # 创建分类器实例
    classifier = LSTMClassifier(
        max_words=10000,
        max_len=200,
        embedding_dim=128,
        lstm_units=128
    )
    
    # 运行完整流程
    accuracy = classifier.run_full_pipeline()
    
    if accuracy is not None:
        print(f"\nLSTM分类器最终准确率: {accuracy:.4f}")
    else:
        print("流程执行失败") 