#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于BERT的中文新闻分类器
使用预训练BERT模型进行特征提取和分类
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import jieba
import re
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

try:
    import torch
    from transformers import BertTokenizer, BertModel, BertForSequenceClassification
    from transformers import Trainer, TrainingArguments
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("请安装transformers和torch: pip install transformers torch")
    exit(1)


class NewsDataset(Dataset):
    """新闻数据集类"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTNewsClassifier:
    """基于BERT的新闻分类器"""
    
    def __init__(self, model_name='hfl/chinese-roberta-wwm-ext', max_length=512):
        """
        初始化BERT分类器
        
        Args:
            model_name: 预训练模型名称
            max_length: 最大序列长度
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.class_names = None
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
        
        # 去除特殊字符和数字，保留中文和标点
        text = re.sub(r'[^\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef]', ' ', str(text))
        
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
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
    
    def prepare_data(self, df, stopwords, max_samples_per_class=5000):
        """准备训练数据"""
        print("正在预处理数据...")
        
        # 文本预处理
        df['processed_text'] = df['content'].apply(lambda x: self.preprocess_text(x, stopwords))
        
        # 移除空文本
        df = df[df['processed_text'].str.len() > 0].reset_index(drop=True)
        print(f"预处理后剩余 {len(df)} 条记录")
        
        # 限制每个类别的样本数量
        data_list = []
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            if len(category_df) > max_samples_per_class:
                category_df = category_df.sample(n=max_samples_per_class, random_state=42)
            data_list.append(category_df)
        
        df = pd.concat(data_list, ignore_index=True)
        print(f"平衡后数据量: {len(df)} 条")
        
        # 标签编码
        self.class_names = df['category'].unique()
        y = self.label_encoder.fit_transform(df['category'])
        texts = df['processed_text'].tolist()
        
        print(f"类别数量: {len(self.class_names)}")
        print("各类别样本数:")
        for i, class_name in enumerate(self.class_names):
            count = np.sum(y == i)
            print(f"  {class_name}: {count}")
        
        return texts, y
    
    def create_model(self, num_classes):
        """创建BERT模型"""
        print(f"正在加载预训练模型: {self.model_name}")
        
        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        
        # 加载预训练模型
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_classes,
            problem_type="single_label_classification"
        )
        
        print("模型加载完成!")
        return self.model
    
    def extract_bert_features(self, texts):
        """提取BERT特征"""
        print("正在提取BERT特征...")
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未初始化，请先调用create_model()")
        
        # 设置为评估模式
        self.model.eval()
        
        features = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # 编码文本
                encoding = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # 获取BERT输出
                outputs = self.model(**encoding, output_hidden_states=True)
                
                # 使用[CLS]标记的输出作为句子表示
                batch_features = outputs.hidden_states[-1][:, 0, :].numpy()
                features.extend(batch_features)
        
        return np.array(features)
    
    def train(self, texts, labels, epochs=3, batch_size=16, learning_rate=2e-5):
        """训练BERT模型"""
        print("正在训练BERT模型...")
        
        # 创建数据集
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        train_dataset = NewsDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = NewsDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        # 计算类别权重
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        
        print("类别权重:")
        for i, weight in enumerate(class_weights):
            print(f"  {self.class_names[i]}: {weight:.4f}")
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir='./bert_news_classifier',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # 训练模型
        trainer.train()
        
        # 保存模型
        trainer.save_model('./models/bert_news_classifier')
        self.tokenizer.save_pretrained('./models/bert_news_classifier')
        
        print("BERT模型训练完成!")
        
        return trainer
    
    def evaluate(self, texts, labels):
        """评估模型"""
        print("正在评估模型...")
        
        # 创建测试数据集
        test_dataset = NewsDataset(texts, labels, self.tokenizer, self.max_length)
        
        # 预测
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in DataLoader(test_dataset, batch_size=32):
                outputs = self.model(**batch)
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(batch['labels'].cpu().numpy())
        
        # 计算评估指标
        accuracy = accuracy_score(true_labels, predictions)
        f1_macro = f1_score(true_labels, predictions, average='macro')
        f1_weighted = f1_score(true_labels, predictions, average='weighted')
        
        print(f"准确率: {accuracy:.4f}")
        print(f"F1-Macro: {f1_macro:.4f}")
        print(f"F1-Weighted: {f1_weighted:.4f}")
        
        # 分类报告
        print("\n分类报告:")
        print(classification_report(
            true_labels, predictions, 
            target_names=self.class_names,
            digits=4
        ))
        
        return accuracy, f1_macro, f1_weighted, predictions, true_labels
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """绘制混淆矩阵"""
        print("正在生成混淆矩阵...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('BERT分类器混淆矩阵', fontsize=16, fontweight='bold')
        plt.xlabel('预测类别', fontsize=12)
        plt.ylabel('真实类别', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('results/bert_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, model_path='models/bert_model.pkl'):
        """保存模型"""
        print(f"正在保存模型到 {model_path}...")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'tokenizer': self.tokenizer,
            'label_encoder': self.label_encoder,
            'class_names': self.class_names,
            'class_weights': self.class_weights,
            'model_name': self.model_name,
            'max_length': self.max_length
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print("模型保存完成!")
    
    def load_model(self, model_path='models/bert_model.pkl'):
        """加载模型"""
        print(f"正在加载模型从 {model_path}...")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.tokenizer = model_data['tokenizer']
        self.label_encoder = model_data['label_encoder']
        self.class_names = model_data['class_names']
        self.class_weights = model_data['class_weights']
        self.model_name = model_data['model_name']
        self.max_length = model_data['max_length']
        
        print("模型加载完成!")
    
    def predict(self, texts):
        """预测新文本的类别"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未加载，请先调用load_model()方法")
        
        # 预处理文本
        stopwords = self.load_stopwords()
        processed_texts = [self.preprocess_text(text, stopwords) for text in texts]
        
        # 创建数据集
        dataset = NewsDataset(processed_texts, [0] * len(processed_texts), self.tokenizer, self.max_length)
        
        # 预测
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=32):
                outputs = self.model(**batch)
                probs = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(outputs.logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        # 转换回类别名称
        predicted_classes = self.label_encoder.inverse_transform(predictions)
        
        return predicted_classes, np.array(probabilities)
    
    def run_complete_pipeline(self, data_file='data/news_data.csv'):
        """运行完整的训练和评估流程"""
        print("开始BERT新闻分类任务...")
        print("=" * 60)
        
        # 1. 加载数据
        df = self.load_data(data_file)
        if df is None:
            return None
        
        # 2. 加载停用词
        stopwords = self.load_stopwords()
        
        # 3. 准备数据
        texts, labels = self.prepare_data(df, stopwords)
        
        # 4. 创建模型
        self.create_model(len(self.class_names))
        
        # 5. 训练模型
        trainer = self.train(texts, labels)
        
        # 6. 评估模型
        accuracy, f1_macro, f1_weighted, predictions, true_labels = self.evaluate(texts, labels)
        
        # 7. 绘制混淆矩阵
        self.plot_confusion_matrix(true_labels, predictions)
        
        # 8. 保存模型
        self.save_model()
        
        print("\n" + "=" * 60)
        print("BERT分类任务完成！")
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }


if __name__ == "__main__":
    # 创建BERT分类器
    bert_classifier = BERTNewsClassifier()
    
    # 运行完整流程
    results = bert_classifier.run_complete_pipeline()
    
    if results:
        print(f"\n最终结果:")
        print(f"准确率: {results['accuracy']:.4f}")
        print(f"F1-Macro: {results['f1_macro']:.4f}")
        print(f"F1-Weighted: {results['f1_weighted']:.4f}") 