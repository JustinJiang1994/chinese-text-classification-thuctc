# 新闻分类数据集

这是一个中文新闻分类数据集，基于[THUCNews](http://thuctc.thunlp.org/)数据集处理而成。

## 数据集来源

本数据集基于清华大学自然语言处理实验室发布的[THUCNews中文文本分类数据集](http://thuctc.thunlp.org/)。THUCNews是根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档（2.19 GB），均为UTF-8纯文本格式。

### 完整数据集下载

| 文件 | 描述 | 大小 | 下载链接 |
|------|------|------|----------|
| THUCNews.zip | THUCNews中文文本数据集 | 1.56GB | [下载](http://thuctc.thunlp.org/) |

## 数据集信息

- **总文件数**: 836,074 个新闻文件
- **总类别数**: 14 个新闻类别
- **总字符数**: 785,258,026 个字符
- **平均每文件字符数**: 939.2 个字符

## 类别分布

| 类别 | 文件数量 | 占比 |
|------|----------|------|
| 科技 | 162,929 | 19.5% |
| 股票 | 154,398 | 18.5% |
| 体育 | 131,603 | 15.7% |
| 娱乐 | 92,632 | 11.1% |
| 时政 | 63,086 | 7.5% |
| 社会 | 50,849 | 6.1% |
| 教育 | 41,936 | 5.0% |
| 财经 | 37,098 | 4.4% |
| 家居 | 32,586 | 3.9% |
| 游戏 | 24,373 | 2.9% |
| 房产 | 20,050 | 2.4% |
| 时尚 | 13,368 | 1.6% |
| 彩票 | 7,588 | 0.9% |
| 星座 | 3,578 | 0.4% |

## 数据预处理结果

### 样本数据集预处理结果
- **原始数据量**: 70 条
- **清洗后数据量**: 69 条
- **移除数据量**: 1 条 (1.4%)
- **训练集**: 48 条 (69.6%)
- **验证集**: 10 条 (14.5%)
- **测试集**: 11 条 (15.9%)
- **处理时间**: 0.39 秒

### 完整数据集预处理结果
- **原始数据量**: 836,074 条
- **清洗后数据量**: 787,500 条
- **移除数据量**: 48,574 条 (5.8%)
- **训练集**: 551,249 条 (70.0%)
- **验证集**: 118,125 条 (15.0%)
- **测试集**: 118,126 条 (15.0%)
- **处理时间**: 1,357.29 秒 (约22.6分钟)

### 完整数据集类别分布

| 类别 | 原始数量 | 清洗后数量 | 训练集 | 验证集 | 测试集 |
|------|----------|------------|--------|--------|--------|
| 体育 | 131,603 | 125,673 | 88,217 | 18,826 | 18,630 |
| 娱乐 | 92,632 | 88,937 | 62,201 | 13,381 | 13,355 |
| 家居 | 32,586 | 30,903 | 21,559 | 4,770 | 4,574 |
| 彩票 | 7,588 | 6,524 | 4,483 | 1,003 | 1,038 |
| 房产 | 20,050 | 18,759 | 13,147 | 2,751 | 2,861 |
| 教育 | 41,936 | 36,385 | 25,433 | 5,549 | 5,403 |
| 时尚 | 13,368 | 13,185 | 9,316 | 1,939 | 1,930 |
| 时政 | 63,086 | 60,683 | 42,579 | 8,999 | 9,105 |
| 星座 | 3,578 | 3,454 | 2,339 | 519 | 596 |
| 游戏 | 24,373 | 23,429 | 16,414 | 3,474 | 3,541 |
| 社会 | 50,849 | 46,330 | 32,475 | 6,897 | 6,958 |
| 科技 | 162,929 | 153,043 | 106,972 | 22,982 | 23,089 |
| 股票 | 154,398 | 146,601 | 102,658 | 21,981 | 21,962 |
| 财经 | 37,098 | 33,594 | 23,456 | 5,054 | 5,084 |

### 数据清洗效果
- **数据质量**: 清洗后保留了94.2%的数据
- **主要清洗步骤**:
  - HTML标签去除
  - URL和邮箱去除
  - 特殊字符处理（只保留中文和英文）
  - 停用词去除（使用2,610个停用词）
  - 长度过滤（10-2000字符）
  - 空文本移除

### 数据集特点
- **随机种子**: 42（确保结果可复现）
- **切分方式**: 随机采样（不分层）
- **数据平衡性**: 各类别在训练/验证/测试集中保持相对一致的分布
- **数据规模**: 适合大规模机器学习实验

## 数据格式

### 完整数据集
- **文件**: `results/extracted_news_data.csv` (2.0GB)
- **格式**: CSV
- **列**: 
  - `category`: 新闻类别标签
  - `content`: 新闻内容

### 样本数据集
- **文件**: `sample.csv` (132KB)
- **格式**: CSV
- **内容**: 每个类别抽取5个样本，共70个样本
- **用途**: 用于GitHub展示和快速测试

## 数据示例

```csv
category,content
体育,"春兰杯决赛有奖竞猜启动 选择冠军赢取空调大奖..."
科技,"超广角蔡司镜头 索尼W530仅售1168元..."
娱乐,"某明星最新动态..."
```

## 使用说明

1. **完整数据集**: 适用于训练机器学习模型
2. **样本数据集**: 适用于快速测试和演示

## 文件结构

```
news_classifier/
├── data/                    # 原始数据目录
│   ├── 体育/               # 体育类新闻
│   ├── 科技/               # 科技类新闻
│   ├── stopwords.txt       # 停用词文件
│   └── ...                 # 其他类别
├── results/                 # 处理结果
│   ├── extracted_news_data.csv  # 完整数据集
│   ├── extracted_news_data.json # JSON格式数据
│   ├── data_cleaned.csv    # 清洗后数据
│   ├── train.csv           # 训练集
│   ├── val.csv             # 验证集
│   └── test.csv            # 测试集
├── sample.csv              # 样本数据集
├── src/                    # 源代码
│   ├── data_extractor.py   # 数据抽取脚本
│   ├── create_sample.py    # 样本创建脚本
│   ├── data_preprocessor.py # 数据预处理脚本
│   └── run_preprocessing.py # 预处理运行脚本
├── config/                 # 配置文件
│   └── preprocessing_config.json # 预处理配置
└── README.md               # 项目说明
```

## 快速开始

### 1. 数据抽取

使用 `src/data_extractor.py` 可以从原始数据目录重新生成处理后的数据集：

```bash
python src/data_extractor.py
```

### 2. 数据预处理

使用 `src/run_preprocessing.py` 进行数据清洗和切分：

```bash
# 处理样本数据
python src/run_preprocessing.py

# 处理完整数据
python src/run_preprocessing.py --use-full
```

### 3. 样本数据生成

使用 `src/create_sample.py` 可以生成样本数据：

```bash
python src/create_sample.py
```

## 数据预处理流程

### 预处理步骤

1. **数据清洗 (Text Cleaning)**
   - HTML标签去除
   - URL和邮箱去除
   - 特殊字符处理（只保留中文和英文）
   - 停用词去除
   - 空格规范化

2. **数据过滤 (Text Filtering)**
   - 长度过滤（10-2000字符）
   - 空文本移除

3. **数据集切分 (Data Splitting)**
   - 分层采样确保各类别比例一致
   - 切分比例：训练集70% / 验证集15% / 测试集15%

### 可复现性保证

- **随机种子**: 固定为42，确保所有随机操作结果可复现
- **分层采样**: 确保各类别在训练/验证/测试集中的比例一致
- **配置管理**: 所有参数都记录在配置文件和统计信息中

### 输出文件

预处理完成后，会在 `results/` 目录下生成：

- `data_cleaned.csv`: 清洗后的完整数据
- `train.csv`: 训练集
- `val.csv`: 验证集
- `test.csv`: 测试集
- `preprocessing_stats.json`: 预处理统计信息

## 版本控制说明

### .gitignore 配置

本项目配置了 `.gitignore` 文件，确保大文件和数据不会被上传到GitHub：

#### 被忽略的文件
- **原始数据目录**: `data/` (包含836,074个文件)
- **处理结果**: `results/extracted_news_data.csv` (2.0GB)
- **JSON数据**: `results/extracted_news_data.json` (2.1GB)
- **模型文件**: `models/` 目录
- **日志文件**: `logs/` 目录

#### 会被上传的文件
- **样本数据**: `sample.csv` (132KB) - 用于GitHub展示
- **源代码**: `src/` 目录
- **配置文件**: `.gitignore`, `requirements.txt`
- **停用词**: `data/stopwords.txt`

### 使用建议

1. **开发时**: 使用 `sample.csv` 进行快速测试
2. **训练时**: 使用完整的 `results/extracted_news_data.csv`
3. **部署时**: 从THUCNews官网下载完整数据集

## 相关工具

- **[THUCTC工具包](http://thuctc.thunlp.org/)**: 清华大学自然语言处理实验室发布的中文文本分类工具包
- **测试结果**: 使用THUCTC工具包在此数据集上进行评测，准确率可以达到88.6%

## 注意事项

- 数据为中文新闻文本
- 包含标点符号和换行符
- 适合用于文本分类任务
- 建议使用UTF-8编码处理
- 原始数据来源于新浪新闻RSS订阅频道
- 确保在提交前检查 `git status` 确认大文件被正确忽略

## 引用

如果您在研究中使用了本数据集，请引用原始THUCNews数据集：

**中文：** 孙茂松，李景阳，郭志芃，赵宇，郑亚斌，司宪策，刘知远. THUCTC：一个高效的中文文本分类工具包. 2016.

**英文：** Maosong Sun, Jingyang Li, Zhipeng Guo, Yu Zhao, Yabin Zheng, Xiance Si, Zhiyuan Liu. THUCTC: An Efficient Chinese Text Classifier. 2016.

---

## 最佳实践指南

### 实验结果总结

#### 1. 模型性能层次结构
基于实验结果，模型性能呈现明显的层次结构：

**🏆 顶级性能 (F1-macro > 0.95)**
- **预训练模型**: RoBERTa (0.9716) > BERT (0.9696)
- **适用场景**: 生产环境、高精度需求、大规模数据
- **代价**: 训练时间长 (~2小时)、计算资源需求高

**🥈 高性能 (F1-macro > 0.90)**
- **深度学习**: LSTM (0.9249) > GRU (0.9222) > TextCNN (0.9155)
- **适用场景**: 大规模数据、序列建模、局部特征捕获
- **代价**: 中等训练时间 (5-6小时)、中等资源需求

**🥉 实用性能 (F1-macro > 0.80)**
- **传统ML**: TomekLinks+SVM (0.8631) > 原始数据+SVM (0.8585)
- **适用场景**: 快速部署、中等精度需求、资源受限
- **优势**: 训练极快 (秒级)、资源需求低

#### 2. 关键发现
- **类别不平衡处理**: TomekLinks比SMOTE更有效，轻微清理边界噪声点即可显著提升性能
- **特征提取**: TF-IDF基础特征在大多数情况下表现最佳，避免过度降维
- **模型稳定性**: SVM和逻辑回归表现最稳定，随机森林对参数敏感
- **训练效率**: 传统ML训练速度是深度学习的1000倍以上

### 不同场景下的模型选择建议

#### 🚀 生产环境部署

**高精度需求 (>95%)**
```python
# 推荐模型
model = "RoBERTa"  # F1-macro: 0.9716
# 备选方案
model = "BERT"     # F1-macro: 0.9696

# 配置建议
- 使用完整数据集 (551,249条)
- 启用类别加权
- 使用GPU训练
- 早停机制防止过拟合
```

**平衡性能需求 (90-95%)**
```python
# 推荐模型
model = "LSTM"     # F1-macro: 0.9249
# 备选方案
model = "GRU"      # F1-macro: 0.9222
model = "TextCNN"  # F1-macro: 0.9155

# 配置建议
- 使用大规模数据集 (>100,000条)
- 启用类别加权
- 使用早停机制
- 考虑模型集成
```

**快速部署需求 (80-90%)**
```python
# 推荐模型
model = "TomekLinks+SVM"  # F1-macro: 0.8631
# 备选方案
model = "原始数据+SVM"    # F1-macro: 0.8585

# 配置建议
- 使用中等规模数据集 (10,000-50,000条)
- 启用类别权重
- 使用TF-IDF特征
- 快速迭代优化
```

#### ⚡ 快速原型开发

**极速验证 (<1秒)**
```python
# 推荐模型
model = "朴素贝叶斯"  # 训练时间: 0.003秒
# 配置建议
- 使用小样本数据 (<1,000条)
- 简单特征提取
- 快速基线建立
```

**快速深度学习 (<1分钟)**
```python
# 推荐模型
model = "FastText"  # 训练时间: 32秒
# 配置建议
- 使用中等样本 (2,000-5,000条)
- 启用类别加权
- 快速超参数调优
```

**传统ML快速迭代 (<10秒)**
```python
# 推荐模型
model = "SVM系列"  # 训练时间: 0.09秒
# 配置建议
- 使用重采样方法
- 特征工程优化
- 多模型对比
```

#### 💰 资源受限环境

**最小内存 (<10M参数)**
```python
# 推荐模型
model = "FastText"  # 8.5M参数
# 配置建议
- 减少词嵌入维度
- 使用小词汇表
- 简化模型结构
```

**CPU友好**
```python
# 推荐模型
model = "传统ML系列"  # SVM, 逻辑回归, 朴素贝叶斯
# 配置建议
- 使用TF-IDF特征
- 避免复杂重采样
- 优化特征维度
```

**低延迟推理**
```python
# 推荐模型
model = "朴素贝叶斯"  # 推理最快
model = "SVM"        # 推理较快
# 配置建议
- 预计算特征
- 模型压缩
- 缓存机制
```

#### 📊 不同数据规模策略

**小样本数据 (<1,000条)**
```python
# 推荐策略
1. 使用传统ML + 重采样
2. 特征: TF-IDF (1-2gram)
3. 模型: SVM + TomekLinks
4. 评估: 交叉验证
```

**中等样本数据 (1,000-10,000条)**
```python
# 推荐策略
1. 传统ML + 重采样
2. 尝试FastText
3. 特征: TF-IDF (1-2gram)
4. 模型: SVM, 逻辑回归, FastText
```

**大规模数据 (10,000-100,000条)**
```python
# 推荐策略
1. 深度学习模型
2. 预训练模型微调
3. 特征: 词嵌入 + 序列建模
4. 模型: LSTM, GRU, TextCNN
```

**超大规模数据 (>100,000条)**
```python
# 推荐策略
1. 预训练模型
2. 大规模深度学习
3. 模型集成
4. 模型: RoBERTa, BERT, LSTM
```

### 性能优化技巧

#### 1. 数据预处理优化

**文本清洗策略**
```python
# 推荐清洗步骤
1. HTML标签去除
2. URL和邮箱去除
3. 特殊字符处理（保留中文和英文）
4. 停用词去除
5. 长度过滤（10-2000字符）

# 优化技巧
- 使用正则表达式批量处理
- 并行化文本清洗
- 缓存清洗结果
```

**特征工程优化**
```python
# TF-IDF优化
vectorizer = TfidfVectorizer(
    max_features=5000,      # 根据数据规模调整
    ngram_range=(1, 2),     # 1-2gram通常最佳
    min_df=2,              # 过滤低频词
    max_df=0.9,            # 过滤高频词
    sublinear_tf=True      # 使用对数缩放
)

# 特征选择
- 使用卡方检验选择特征
- 基于信息增益选择
- 递归特征消除
```

#### 2. 类别不平衡处理优化

**重采样策略选择**
```python
# 推荐策略（按效果排序）
1. TomekLinks          # 轻微清理，效果最佳
2. 原始数据 + 类别权重 # 简单有效
3. SMOTE              # 对朴素贝叶斯效果显著
4. ADASYN             # 自适应采样

# 避免使用
- 随机欠采样（损失信息）
- 过度重采样（过拟合风险）
```

**类别权重优化**
```python
# 计算类别权重
from sklearn.utils.class_weight import compute_class_weight

# 方法1: 平衡权重
class_weight = 'balanced'

# 方法2: 自定义权重
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
)

# 方法3: 基于频率的权重
class_weights = 1.0 / (class_counts + 1e-6)
```

#### 3. 模型训练优化

**深度学习优化**
```python
# 学习率调度
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=3
)

# 早停机制
early_stopping = EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True
)

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

**传统ML优化**
```python
# SVM优化
svm = LinearSVC(
    C=1.0,                    # 正则化参数
    class_weight='balanced',  # 类别权重
    max_iter=1000,           # 最大迭代次数
    random_state=42          # 随机种子
)

# 逻辑回归优化
lr = LogisticRegression(
    C=1.0,                    # 正则化参数
    class_weight='balanced',  # 类别权重
    max_iter=1000,           # 最大迭代次数
    solver='liblinear'       # 优化器选择
)
```

#### 4. 评估和调优优化

**交叉验证策略**
```python
# 分层K折交叉验证
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,           # 5折交叉验证
    shuffle=True,         # 随机打乱
    random_state=42       # 随机种子
)

# 时间序列交叉验证（如果数据有时间顺序）
from sklearn.model_selection import TimeSeriesSplit
```

**超参数调优**
```python
# 网格搜索
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1.0, 10.0],
    'class_weight': ['balanced', None]
}

grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

# 随机搜索（更快）
from sklearn.model_selection import RandomizedSearchCV
```

#### 5. 部署优化

**模型压缩**
```python
# 特征选择
from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(chi2, k=1000)
X_selected = selector.fit_transform(X_train, y_train)

# 模型量化
import joblib
joblib.dump(model, 'compressed_model.pkl', compress=3)

# 模型剪枝（深度学习）
pruned_model = prune_model(model, amount=0.3)
```

**推理优化**
```python
# 批量推理
def batch_predict(model, texts, batch_size=32):
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        pred = model.predict(batch)
        predictions.extend(pred)
    return predictions

# 缓存机制
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_predict(text):
    return model.predict([text])[0]
```

#### 6. 监控和维护

**性能监控**
```python
# 模型性能跟踪
import mlflow

mlflow.log_metric("f1_macro", f1_macro)
mlflow.log_metric("training_time", training_time)
mlflow.log_param("model_type", model_type)

# 数据漂移检测
def detect_drift(reference_data, current_data):
    # 统计分布比较
    # 特征重要性变化
    # 性能下降检测
    pass
```

**模型更新策略**
```python
# A/B测试
def ab_test(model_a, model_b, test_data):
    pred_a = model_a.predict(test_data)
    pred_b = model_b.predict(test_data)
    # 统计显著性测试
    return statistical_test(pred_a, pred_b)

# 渐进式更新
def gradual_update(old_model, new_model, update_ratio=0.1):
    # 逐步替换模型
    pass
```

---

## 项目状态

✅ **已完成功能**
- 数据提取和预处理
- 传统机器学习实验
- 类别不平衡处理
- 模块化架构重构
- 配置驱动实验
- 可视化工具
- 深度学习实验准备
- 模型优化和调参
- 深度学习模型（BERT, RoBERTa等）

🔄 **进行中**


📋 **计划中**
- 模型部署和API
- 实时预测服务 

## 集成模型实验结果（独立测试集）

本实验采用2000条训练数据、118,126条独立测试数据，评估了多种集成方法与单模型的性能：

| 排名 | 模型              | F1-macro | F1-weighted | 类型     |
|------|-------------------|----------|-------------|----------|
| 1    | voting_hard       | 0.9502   | 0.9540      | 集成     |
| 2    | voting_soft       | 0.9388   | 0.9462      | 集成     |
| 3    | weighted_ensemble | 0.8588   | 0.9027      | 集成     |
| 4    | tomeklinks_svm    | 0.8585   | 0.9025      | 传统ML   |
| 5    | logistic_regression | 0.8496 | 0.8889      | 传统ML   |
| 6    | random_forest     | 0.7345   | 0.7989      | 传统ML   |

- **硬投票集成**：F1-macro = 0.9502，显著优于所有单模型
- **软投票集成**：F1-macro = 0.9388，提升明显
- **加权集成**：与最佳单模型相当
- **最佳单模型**：TomekLinks+SVM，F1-macro = 0.8585

### 关键结论
- 集成方法（尤其投票集成）在大规模独立测试集上依然能显著提升性能
- 训练/测试数据完全独立，结果更真实可信
- 集成方法适合生产环境和高精度需求场景

--- 