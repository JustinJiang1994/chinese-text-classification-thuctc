"""
评估工具函数
"""

import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from typing import Dict, Any, List, Tuple

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                  class_names: List[str] = None) -> Dict[str, Any]:
    """评估模型性能"""
    results = {
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'f1_micro': f1_score(y_true, y_pred, average='micro'),
        'f1_per_class': f1_score(y_true, y_pred, average=None)
    }
    
    if class_names:
        results['classification_report'] = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )
    
    return results

def get_model_rankings(results: Dict[str, Any]) -> List[Tuple[str, float]]:
    """获取模型排名"""
    model_scores = []
    for key, result in results.items():
        if isinstance(result, dict) and 'f1_macro' in result:
            model_scores.append((key, result['f1_macro']))
    
    return sorted(model_scores, key=lambda x: x[1], reverse=True)

def find_best_model(results: Dict[str, Any]) -> Dict[str, Any]:
    """找到最佳模型"""
    best_score = 0
    best_model = None
    
    for key, result in results.items():
        if isinstance(result, dict) and 'f1_macro' in result:
            if result['f1_macro'] > best_score:
                best_score = result['f1_macro']
                best_model = {
                    'name': key,
                    'f1_macro': result['f1_macro'],
                    'f1_weighted': result['f1_weighted']
                }
    
    return best_model 