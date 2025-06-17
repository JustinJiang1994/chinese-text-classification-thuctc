"""
数据处理工具函数
"""

import pandas as pd
import numpy as np
import re
import jieba
from pathlib import Path
from typing import Dict, Any, Tuple
import json

def preprocess_text(text: str) -> str:
    """文本预处理"""
    if pd.isna(text):
        return ""
    
    # 移除特殊字符和数字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', str(text))
    # 分词
    words = jieba.cut(text)
    return ' '.join(words)

def load_data(data_path: str, sample_size: int = None) -> pd.DataFrame:
    """加载数据"""
    data_path = Path(data_path)
    
    if sample_size:
        data = pd.read_csv(data_path, nrows=sample_size)
    else:
        data = pd.read_csv(data_path)
    
    return data

def save_results(results: Dict[str, Any], file_path: str):
    """保存实验结果"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 确保结果可序列化
    serializable_results = make_serializable(results)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)

def make_serializable(obj: Any) -> Any:
    """确保对象可JSON序列化"""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f) 