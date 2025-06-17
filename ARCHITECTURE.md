# 项目架构说明

## 简化后的项目结构

```
news_classifier/
├── src/
│   ├── experiments.py          # 统一实验管理模块 (核心)
│   ├── run_experiments.py      # 实验运行脚本 (入口)
│   ├── data_preprocessor.py    # 数据预处理 (保留)
│   ├── data_extractor.py       # 数据抽取 (保留)
│   └── create_sample.py        # 样本创建 (保留)
├── utils/                      # 工具函数包
│   ├── __init__.py
│   ├── data_utils.py           # 数据处理工具
│   ├── eval_utils.py           # 评估工具
│   └── viz_utils.py            # 可视化工具
├── config/
│   ├── preprocessing_config.json  # 预处理配置
│   └── experiment_config.json     # 实验配置
├── results/                    # 实验结果
├── data/                       # 原始数据
├── sample.csv                  # 样本数据
├── requirements.txt            # 依赖包
└── README.md                   # 项目说明
```

## 架构优化说明

### 1. 核心模块整合

**之前的问题**:
- `traditional_ml_experiments.py` (411行)
- `imbalance_experiment.py` (415行)
- `quick_experiment.py` (141行)
- 大量重复代码

**优化后**:
- `experiments.py` - 统一实验管理模块
- `run_experiments.py` - 简洁的命令行接口
- 消除重复代码，提高可维护性

### 2. 模块化设计

#### NewsClassifierExperiment (主控制器)
```python
# 统一接口
experiment = NewsClassifierExperiment(config)
results = experiment.run_quick_experiment(1000)
results = experiment.run_imbalance_experiment(10000)
results = experiment.run_full_experiment()
```

#### DataProcessor (数据处理)
```python
# 特征准备
X, y, vectorizer, label_encoder = data_processor.prepare_features(data)
imbalance_info = data_processor.analyze_imbalance(y, label_encoder)
```

#### MLExperimenter (机器学习实验)
```python
# 实验执行
results = ml_experimenter.run_single_experiment(X_train, X_test, y_train, y_test)
results = ml_experimenter.run_imbalance_experiment(X_train, X_test, y_train, y_test)
```

#### ResultVisualizer (结果可视化)
```python
# 图表生成
visualizer.plot_imbalance_results(results, class_names)
```

### 3. 配置管理

**统一配置文件**: `config/experiment_config.json`
```json
{
  "models": ["naive_bayes", "svm", "random_forest", "logistic_regression"],
  "sampling_methods": ["original", "smote", "tomek_links", "smoteenn"],
  "max_features": 8000,
  "random_state": 42
}
```

### 4. 工具函数包

**utils/ 目录**:
- `data_utils.py` - 数据处理工具
- `eval_utils.py` - 评估工具
- `viz_utils.py` - 可视化工具

### 5. 使用方式

#### 命令行使用
```bash
# 快速实验
python src/run_experiments.py --experiment quick --sample-size 1000

# 不平衡实验
python src/run_experiments.py --experiment imbalance --sample-size 10000

# 完整实验
python src/run_experiments.py --experiment full

# 使用自定义配置
python src/run_experiments.py --experiment imbalance --config config/experiment_config.json
```

#### 编程使用
```python
from src.experiments import NewsClassifierExperiment

# 创建实验管理器
experiment = NewsClassifierExperiment()

# 运行实验
results = experiment.run_imbalance_experiment(10000)

# 打印结果
experiment.print_summary(results)
```

## 优势

### 1. 代码复用
- 消除重复代码
- 统一数据处理流程
- 共享评估和可视化功能

### 2. 易于扩展
- 模块化设计
- 清晰的接口
- 配置驱动

### 3. 易于维护
- 单一职责原则
- 清晰的依赖关系
- 统一的错误处理

### 4. 易于使用
- 简洁的命令行接口
- 灵活的配置选项
- 详细的结果输出

## 迁移指南

### 从旧架构迁移

1. **删除重复文件**:
   - `traditional_ml_experiments.py`
   - `imbalance_experiment.py`
   - `quick_experiment.py`

2. **使用新接口**:
   ```python
   # 旧方式
   python src/quick_experiment.py
   python src/imbalance_experiment.py
   
   # 新方式
   python src/run_experiments.py --experiment quick
   python src/run_experiments.py --experiment imbalance
   ```

3. **配置迁移**:
   - 将实验参数移到 `config/experiment_config.json`
   - 使用统一的配置管理

## 后续扩展

### 1. 深度学习模块
```python
class DLExperimenter:
    def run_bert_experiment(self):
        pass
    
    def run_cnn_experiment(self):
        pass
```

### 2. 模型保存和加载
```python
class ModelManager:
    def save_model(self, model, path):
        pass
    
    def load_model(self, path):
        pass
```

### 3. 实验日志
```python
class ExperimentLogger:
    def log_experiment(self, config, results):
        pass
```

这个新架构大大简化了项目结构，提高了代码的可维护性和可扩展性。 