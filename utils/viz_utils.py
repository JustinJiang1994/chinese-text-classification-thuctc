"""
可视化工具函数
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_model_comparison(results: Dict[str, Any], 
                         sampling_methods: List[str], 
                         models: List[str],
                         save_path: str = "results/plots/model_comparison.png"):
    """绘制模型性能对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, model in enumerate(models):
        if i < len(axes):
            f1_scores = []
            method_names = []
            
            for method in sampling_methods:
                key = f"{method}_{model}"
                if key in results:
                    f1_scores.append(results[key]['f1_macro'])
                    method_names.append(method)
            
            axes[i].bar(range(len(f1_scores)), f1_scores, alpha=0.8)
            axes[i].set_title(f'{model} - 不同重采样方法对比')
            axes[i].set_ylabel('F1-macro 分数')
            axes[i].set_xticks(range(len(method_names)))
            axes[i].set_xticklabels(method_names, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_sampling_comparison(results: Dict[str, Any], 
                           sampling_methods: List[str], 
                           models: List[str],
                           save_path: str = "results/plots/sampling_comparison.png"):
    """绘制重采样方法对比图"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for i, method in enumerate(sampling_methods):
        if i < len(axes):
            f1_scores = []
            model_names = []
            
            for model in models:
                key = f"{method}_{model}"
                if key in results:
                    f1_scores.append(results[key]['f1_macro'])
                    model_names.append(model)
            
            axes[i].bar(range(len(f1_scores)), f1_scores, alpha=0.8)
            axes[i].set_title(f'{method} - 模型性能对比')
            axes[i].set_ylabel('F1-macro 分数')
            axes[i].set_xticks(range(len(model_names)))
            axes[i].set_xticklabels(model_names, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_performance(f1_per_class: np.ndarray, 
                          class_names: List[str],
                          save_path: str = "results/plots/class_performance.png"):
    """绘制各类别性能图"""
    plt.figure(figsize=(12, 6))
    
    plt.bar(range(len(class_names)), f1_per_class, alpha=0.8)
    plt.title('各类别F1分数')
    plt.xlabel('类别')
    plt.ylabel('F1分数')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_results(results: Dict[str, Any], class_names: List[str] = None):
    """绘制所有结果图表"""
    # 提取重采样方法和模型名称
    sampling_methods = set()
    models = set()
    
    for key in results.keys():
        if '_' in key and key not in ['best_model', 'model_rankings']:
            parts = key.split('_', 1)
            if len(parts) == 2:
                sampling_methods.add(parts[0])
                models.add(parts[1])
    
    sampling_methods = list(sampling_methods)
    models = list(models)
    
    # 绘制图表
    plot_model_comparison(results, sampling_methods, models)
    plot_sampling_comparison(results, sampling_methods, models)
    
    # 如果有最佳模型，绘制类别性能
    if 'best_model' in results and class_names:
        best_key = results['best_model']['name']
        if best_key in results and 'f1_per_class' in results[best_key]:
            plot_class_performance(
                results[best_key]['f1_per_class'], 
                class_names
            ) 