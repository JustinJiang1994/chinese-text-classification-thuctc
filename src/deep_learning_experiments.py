"""
深度学习实验模块
基于PyTorch实现LSTM、GRU、RNN、TextCNN、FastText等模型
"""

import os
import json
import time
import pickle
import argparse
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import jieba
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Vocabulary:
    """词汇表类"""
    
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = Counter()
        
    def build_vocab(self, texts: List[str]):
        """构建词汇表"""
        # 统计词频
        for text in texts:
            words = jieba.lcut(text)
            self.word_freq.update(words)
        
        # 添加满足最小频率的词
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def encode(self, text: str, max_len: int = 100) -> List[int]:
        """将文本编码为索引序列"""
        words = jieba.lcut(text)
        indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words[:max_len]]
        
        # 填充或截断
        if len(indices) < max_len:
            indices += [self.word2idx['<PAD>']] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
            
        return indices
    
    def __len__(self):
        return len(self.word2idx)

class NewsDataset(Dataset):
    """新闻数据集类"""
    
    def __init__(self, texts: List[str], labels: List[int], vocab: Vocabulary, max_len: int = 100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 编码文本
        indices = self.vocab.encode(text, self.max_len)
        
        return {
            'text': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

class TextCNN(nn.Module):
    """TextCNN模型"""
    
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int, 
                 kernel_sizes: List[int] = [3, 4, 5], num_filters: int = 100, dropout: float = 0.5):
        super(TextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = embedded.unsqueeze(1)  # (batch_size, 1, seq_len, embed_dim)
        
        # 卷积
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))  # (batch_size, num_filters, seq_len-k+1, 1)
            conv_out = conv_out.squeeze(3)  # (batch_size, num_filters, seq_len-k+1)
            pooled = torch.max_pool1d(conv_out, conv_out.size(2))  # (batch_size, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))  # (batch_size, num_filters)
        
        # 拼接
        concatenated = torch.cat(conv_outputs, dim=1)  # (batch_size, len(kernel_sizes) * num_filters)
        
        # 全连接
        output = self.dropout(concatenated)
        output = self.fc(output)
        
        return output

class RNNModel(nn.Module):
    """RNN模型基类"""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, 
                 num_layers: int = 1, dropout: float = 0.5, bidirectional: bool = False):
        super(RNNModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # RNN层
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0,
                         bidirectional=bidirectional)
        
        # 全连接层
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # RNN前向传播
        output, hidden = self.rnn(embedded)
        
        # 取最后一个时间步的输出
        if self.bidirectional:
            # 双向RNN，拼接最后两个隐藏状态
            last_output = torch.cat((output[:, -1, :self.hidden_dim], 
                                   output[:, 0, self.hidden_dim:]), dim=1)
        else:
            last_output = output[:, -1, :]
        
        # 全连接层
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output

class LSTMModel(RNNModel):
    """LSTM模型"""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, 
                 num_layers: int = 1, dropout: float = 0.5, bidirectional: bool = False):
        super(LSTMModel, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes, 
                                       num_layers, dropout, bidirectional)
        
        # 替换RNN为LSTM
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0,
                          bidirectional=bidirectional)

class GRUModel(RNNModel):
    """GRU模型"""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, 
                 num_layers: int = 1, dropout: float = 0.5, bidirectional: bool = False):
        super(GRUModel, self).__init__(vocab_size, embed_dim, hidden_dim, num_classes, 
                                      num_layers, dropout, bidirectional)
        
        # 替换RNN为GRU
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0,
                         bidirectional=bidirectional)

class FastText(nn.Module):
    """FastText模型"""
    
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int, 
                 ngram_range: int = 2, dropout: float = 0.5):
        super(FastText, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.ngram_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim, padding_idx=0) 
            for _ in range(ngram_range - 1)
        ])
        self.fc = nn.Linear(embed_dim * ngram_range, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.ngram_range = ngram_range
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        
        # 1-gram embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = embedded.mean(dim=1)  # (batch_size, embed_dim)
        
        # n-gram embeddings (简化版本)
        ngram_embeddings = [embedded]
        
        for i in range(self.ngram_range - 1):
            n = i + 2
            if seq_len >= n:
                # 创建n-gram索引
                ngram_indices = []
                for j in range(seq_len - n + 1):
                    # 简单的n-gram哈希方法
                    ngram_hash = torch.sum(x[:, j:j+n] * (100 ** torch.arange(n)), dim=1)
                    ngram_indices.append(ngram_hash)
                
                if ngram_indices:
                    ngram_tensor = torch.stack(ngram_indices, dim=1)  # (batch_size, num_ngrams)
                    # 使用模运算确保索引在词汇表范围内
                    ngram_tensor = ngram_tensor % (x.size(1) - 1) + 1  # 避免0（padding）
                    ngram_embedded = self.ngram_embeddings[i](ngram_tensor)  # (batch_size, num_ngrams, embed_dim)
                    ngram_embedded = ngram_embedded.mean(dim=1)  # (batch_size, embed_dim)
                    ngram_embeddings.append(ngram_embedded)
                else:
                    ngram_embeddings.append(torch.zeros_like(embedded))
            else:
                ngram_embeddings.append(torch.zeros_like(embedded))
        
        # 拼接所有n-gram embeddings
        concatenated = torch.cat(ngram_embeddings, dim=1)
        
        # 全连接层
        output = self.dropout(concatenated)
        output = self.fc(output)
        
        return output

class DeepLearningExperiment:
    """深度学习实验类"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab = None
        self.label2idx = None
        self.idx2label = None
        self.results = {}
        
    def load_data(self, data_path: str, sample_size: Optional[int] = None) -> Tuple[List[str], List[int]]:
        """加载数据"""
        print(f"加载数据: {data_path}")
        df = pd.read_csv(data_path)
        
        if sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        texts = df['content'].tolist()
        labels = df['category'].tolist()
        
        # 创建标签映射
        unique_labels = sorted(set(labels))
        self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        
        # 转换标签为索引
        label_indices = [self.label2idx[label] for label in labels]
        
        print(f"数据加载完成: {len(texts)} 个样本, {len(unique_labels)} 个类别")
        return texts, label_indices
    
    def build_vocabulary(self, texts: List[str]):
        """构建词汇表"""
        print("构建词汇表...")
        self.vocab = Vocabulary(min_freq=self.config.get('min_freq', 2))
        self.vocab.build_vocab(texts)
        print(f"词汇表大小: {len(self.vocab)}")
    
    def create_datasets(self, texts: List[str], labels: List[int]) -> Tuple[Dataset, Dataset, Dataset]:
        """创建数据集"""
        # 划分数据集
        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, labels, test_size=0.3, random_state=42, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # 创建数据集
        train_dataset = NewsDataset(X_train, y_train, self.vocab, self.config['max_len'])
        val_dataset = NewsDataset(X_val, y_val, self.vocab, self.config['max_len'])
        test_dataset = NewsDataset(X_test, y_test, self.vocab, self.config['max_len'])
        
        return train_dataset, val_dataset, test_dataset
    
    def get_model(self, model_name: str, num_classes: int) -> nn.Module:
        """获取模型"""
        vocab_size = len(self.vocab)
        embed_dim = self.config['embed_dim']
        
        if model_name == 'textcnn':
            return TextCNN(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_classes=num_classes,
                kernel_sizes=self.config.get('kernel_sizes', [3, 4, 5]),
                num_filters=self.config.get('num_filters', 100),
                dropout=self.config.get('dropout', 0.5)
            )
        elif model_name == 'lstm':
            return LSTMModel(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=self.config.get('hidden_dim', 128),
                num_classes=num_classes,
                num_layers=self.config.get('num_layers', 1),
                dropout=self.config.get('dropout', 0.5),
                bidirectional=self.config.get('bidirectional', False)
            )
        elif model_name == 'gru':
            return GRUModel(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=self.config.get('hidden_dim', 128),
                num_classes=num_classes,
                num_layers=self.config.get('num_layers', 1),
                dropout=self.config.get('dropout', 0.5),
                bidirectional=self.config.get('bidirectional', False)
            )
        elif model_name == 'rnn':
            return RNNModel(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=self.config.get('hidden_dim', 128),
                num_classes=num_classes,
                num_layers=self.config.get('num_layers', 1),
                dropout=self.config.get('dropout', 0.5),
                bidirectional=self.config.get('bidirectional', False)
            )
        elif model_name == 'fasttext':
            return FastText(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_classes=num_classes,
                ngram_range=self.config.get('ngram_range', 2),
                dropout=self.config.get('dropout', 0.5)
            )
        else:
            raise ValueError(f"不支持的模型: {model_name}")
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """训练模型"""
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        
        best_val_loss = float('inf')
        best_model_state = None
        patience = self.config.get('patience', 5)
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        print(f"开始训练 {model.__class__.__name__}...")
        
        for epoch in range(self.config['epochs']):
            # 训练阶段
            model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}'):
                text = batch['text'].to(self.device)
                label = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                output = model(text)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    text = batch['text'].to(self.device)
                    label = batch['label'].to(self.device)
                    
                    output = model(text)
                    loss = criterion(output, label)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停在第 {epoch+1} 轮")
                    break
        
        # 加载最佳模型
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Dict:
        """评估模型"""
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                text = batch['text'].to(self.device)
                label = batch['label'].to(self.device)
                
                output = model(text)
                predictions = torch.argmax(output, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
        
        # 计算指标
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        
        # 分类报告
        report = classification_report(all_labels, all_predictions, 
                                     target_names=[self.idx2label[i] for i in range(len(self.idx2label))],
                                     output_dict=True)
        
        return {
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': report,
            'predictions': all_predictions,
            'true_labels': all_labels
        }
    
    def run_experiment(self, data_path: str, sample_size: Optional[int] = None) -> Dict:
        """运行实验"""
        print("=" * 50)
        print("开始深度学习实验")
        print("=" * 50)
        
        # 加载数据
        texts, labels = self.load_data(data_path, sample_size)
        
        # 构建词汇表
        self.build_vocabulary(texts)
        
        # 创建数据集
        train_dataset, val_dataset, test_dataset = self.create_datasets(texts, labels)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], 
                                shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], 
                              shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], 
                               shuffle=False, num_workers=0)
        
        num_classes = len(self.label2idx)
        models = self.config['models']
        
        for model_name in models:
            print(f"\n{'='*20} 训练 {model_name.upper()} {'='*20}")
            
            start_time = time.time()
            
            # 获取模型
            model = self.get_model(model_name, num_classes)
            
            # 训练模型
            train_result = self.train_model(model, train_loader, val_loader)
            
            # 评估模型
            eval_result = self.evaluate_model(train_result['model'], test_loader)
            
            training_time = time.time() - start_time
            
            # 保存结果
            self.results[model_name] = {
                'f1_macro': eval_result['f1_macro'],
                'f1_weighted': eval_result['f1_weighted'],
                'training_time': training_time,
                'best_val_loss': train_result['best_val_loss'],
                'train_losses': train_result['train_losses'],
                'val_losses': train_result['val_losses'],
                'classification_report': eval_result['classification_report'],
                'model_params': sum(p.numel() for p in model.parameters())
            }
            
            print(f"{model_name.upper()} 结果:")
            print(f"  F1-macro: {eval_result['f1_macro']:.4f}")
            print(f"  F1-weighted: {eval_result['f1_weighted']:.4f}")
            print(f"  训练时间: {training_time:.2f}秒")
            print(f"  模型参数: {self.results[model_name]['model_params']:,}")
        
        return self.results
    
    def save_results(self, output_dir: str):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存实验结果
        results_file = os.path.join(output_dir, 'deep_learning_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            # 转换numpy类型为Python类型
            json_results = {}
            for model_name, result in self.results.items():
                json_results[model_name] = {
                    'f1_macro': float(result['f1_macro']),
                    'f1_weighted': float(result['f1_weighted']),
                    'training_time': float(result['training_time']),
                    'best_val_loss': float(result['best_val_loss']),
                    'model_params': int(result['model_params']),
                    'train_losses': [float(x) for x in result['train_losses']],
                    'val_losses': [float(x) for x in result['val_losses']]
                }
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        # 保存词汇表和标签映射
        vocab_file = os.path.join(output_dir, 'vocab.pkl')
        with open(vocab_file, 'wb') as f:
            pickle.dump(self.vocab, f)
        
        label_file = os.path.join(output_dir, 'label_mapping.pkl')
        with open(label_file, 'wb') as f:
            pickle.dump({'label2idx': self.label2idx, 'idx2label': self.idx2label}, f)
        
        print(f"结果已保存到: {output_dir}")
    
    def plot_results(self, output_dir: str):
        """绘制结果图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 模型性能对比
        plt.figure(figsize=(12, 8))
        
        models = list(self.results.keys())
        f1_macro_scores = [self.results[model]['f1_macro'] for model in models]
        f1_weighted_scores = [self.results[model]['f1_weighted'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, f1_macro_scores, width, label='F1-macro', alpha=0.8)
        plt.bar(x + width/2, f1_weighted_scores, width, label='F1-weighted', alpha=0.8)
        
        plt.xlabel('模型')
        plt.ylabel('F1分数')
        plt.title('深度学习模型性能对比')
        plt.xticks(x, [model.upper() for model in models])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (macro, weighted) in enumerate(zip(f1_macro_scores, f1_weighted_scores)):
            plt.text(i - width/2, macro + 0.01, f'{macro:.3f}', ha='center', va='bottom')
            plt.text(i + width/2, weighted + 0.01, f'{weighted:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 训练损失曲线
        plt.figure(figsize=(15, 10))
        
        for i, model_name in enumerate(models, 1):
            plt.subplot(2, 3, i)
            
            train_losses = self.results[model_name]['train_losses']
            val_losses = self.results[model_name]['val_losses']
            
            epochs = range(1, len(train_losses) + 1)
            plt.plot(epochs, train_losses, 'b-', label='训练损失', alpha=0.8)
            plt.plot(epochs, val_losses, 'r-', label='验证损失', alpha=0.8)
            
            plt.title(f'{model_name.upper()} 训练曲线')
            plt.xlabel('轮次')
            plt.ylabel('损失')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 模型参数和训练时间对比
        plt.figure(figsize=(12, 5))
        
        params = [self.results[model]['model_params'] for model in models]
        times = [self.results[model]['training_time'] for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 参数数量
        ax1.bar(models, params, alpha=0.8, color='skyblue')
        ax1.set_xlabel('模型')
        ax1.set_ylabel('参数数量')
        ax1.set_title('模型参数数量对比')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, param in enumerate(params):
            ax1.text(i, param + max(params)*0.01, f'{param:,}', ha='center', va='bottom')
        
        # 训练时间
        ax2.bar(models, times, alpha=0.8, color='lightcoral')
        ax2.set_xlabel('模型')
        ax2.set_ylabel('训练时间 (秒)')
        ax2.set_title('模型训练时间对比')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, time_val in enumerate(times):
            ax2.text(i, time_val + max(times)*0.01, f'{time_val:.1f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_efficiency_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"图表已保存到: {output_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='深度学习实验')
    parser.add_argument('--data_path', type=str, default='results/train.csv',
                       help='数据文件路径')
    parser.add_argument('--sample_size', type=int, default=5000,
                       help='样本数量')
    parser.add_argument('--output_dir', type=str, default='results/deep_learning_experiments',
                       help='输出目录')
    parser.add_argument('--config', type=str, default='config/deep_learning_config.json',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        # 默认配置
        config = {
            'models': ['textcnn', 'lstm', 'gru', 'rnn', 'fasttext'],
            'embed_dim': 128,
            'hidden_dim': 128,
            'num_layers': 1,
            'bidirectional': False,
            'kernel_sizes': [3, 4, 5],
            'num_filters': 100,
            'ngram_range': 2,
            'max_len': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 20,
            'patience': 5,
            'dropout': 0.5,
            'min_freq': 2
        }
    
    # 创建实验
    experiment = DeepLearningExperiment(config)
    
    # 运行实验
    results = experiment.run_experiment(args.data_path, args.sample_size)
    
    # 保存结果
    experiment.save_results(args.output_dir)
    
    # 绘制图表
    experiment.plot_results(args.output_dir)
    
    # 打印最佳模型
    best_model = max(results.items(), key=lambda x: x[1]['f1_macro'])
    print(f"\n最佳模型: {best_model[0].upper()}")
    print(f"F1-macro: {best_model[1]['f1_macro']:.4f}")
    print(f"F1-weighted: {best_model[1]['f1_weighted']:.4f}")

if __name__ == '__main__':
    main() 