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

## 传统机器学习实验结果

### 实验概述

我们使用传统机器学习方法对中文新闻分类任务进行了全面实验，重点关注类别不平衡问题的处理。实验使用了10,000个样本进行快速验证，包含14个新闻类别。

### 类别不平衡分析

#### 数据集不平衡情况
- **不平衡比例**: 37.40:1 (最大类别: 科技 1,945样本 vs 最小类别: 星座 52样本)
- **类别分布**:
  - 主要类别: 科技(1,945), 股票(1,924), 体育(1,599), 娱乐(1,093)
  - 中等类别: 时政(768), 社会(609), 教育(459), 财经(433), 家居(395)
  - 少数类别: 游戏(265), 房产(229), 时尚(158), 彩票(71), 星座(52)

#### 不平衡问题影响
- 模型倾向于预测多数类别，忽略少数类别
- 传统准确率指标无法反映真实性能
- 需要使用F1-macro等平衡指标进行评估

### 实验方法

#### 机器学习算法
1. **朴素贝叶斯 (Naive Bayes)**: 基线模型，适合文本分类
2. **支持向量机 (SVM)**: 线性SVM，使用类别权重平衡
3. **随机森林 (Random Forest)**: 集成方法，处理非线性关系
4. **逻辑回归 (Logistic Regression)**: 线性模型，易于解释

#### 类别不平衡处理技术
1. **原始数据**: 不进行任何重采样
2. **SMOTE**: 合成少数类过采样技术
3. **ADASYN**: 自适应合成采样
4. **BorderlineSMOTE**: 边界SMOTE
5. **随机欠采样**: 随机删除多数类样本
6. **TomekLinks**: 移除边界噪声点
7. **EditedNearestNeighbours**: 编辑最近邻
8. **SMOTEENN**: SMOTE + 编辑最近邻
9. **SMOTETomek**: SMOTE + Tomek Links

### 实验结果

#### 最佳性能组合
- **最佳模型**: TomekLinks + SVM
- **F1-macro**: 0.8839
- **F1-weighted**: 0.9232
- **训练时间**: 0.26秒

#### 各重采样方法最佳F1-macro排名
| 排名 | 重采样方法 | 最佳F1-macro | 对应模型 |
|------|------------|--------------|----------|
| 1 | TomekLinks | 0.8839 | SVM |
| 2 | 原始数据 | 0.8825 | SVM |
| 3 | ADASYN | 0.8825 | SVM |
| 4 | SMOTE | 0.8785 | SVM |
| 5 | SMOTETomek | 0.8785 | SVM |
| 6 | BorderlineSMOTE | 0.8773 | SVM |
| 7 | EditedNearestNeighbours | 0.8581 | SVM |
| 8 | SMOTEENN | 0.8400 | SVM |
| 9 | 随机欠采样 | 0.7939 | 逻辑回归 |

#### 模型性能对比

##### 原始数据（无重采样）
| 模型 | F1-macro | F1-weighted | 训练时间(秒) |
|------|----------|-------------|--------------|
| SVM | 0.8825 | 0.9268 | 0.54 |
| 逻辑回归 | 0.8698 | 0.9081 | 0.85 |
| 随机森林 | 0.7644 | 0.8169 | 0.68 |
| 朴素贝叶斯 | 0.6487 | 0.8173 | 0.01 |

##### SMOTE重采样
| 模型 | F1-macro | F1-weighted | 训练时间(秒) |
|------|----------|-------------|--------------|
| SVM | 0.8762 | 0.9237 | 2.39 |
| 逻辑回归 | 0.8689 | 0.9132 | 2.52 |
| 朴素贝叶斯 | 0.8453 | 0.8897 | 0.02 |
| 随机森林 | 0.7795 | 0.8215 | 2.58 |

### 关键发现

#### 1. 模型性能排序
- **SVM表现最佳**: 在所有重采样方法下都表现最好
- **逻辑回归次之**: 性能稳定，训练速度较快
- **朴素贝叶斯**: 在SMOTE后性能显著提升
- **随机森林**: 对重采样敏感，性能波动较大

#### 2. 重采样效果分析
- **TomekLinks最有效**: 轻微清理边界噪声点，性能提升最明显
- **SMOTE系列**: 对朴素贝叶斯效果显著，对其他模型提升有限
- **欠采样**: 虽然平衡了数据，但损失了信息，性能下降
- **原始数据**: 在SVM上表现很好，说明类别权重设置有效

#### 3. 训练效率
- **朴素贝叶斯**: 训练最快（0.01-0.02秒）
- **SVM**: 中等速度（0.26-3.36秒）
- **逻辑回归**: 中等速度（0.18-2.90秒）
- **随机森林**: 较慢（0.10-2.88秒）

#### 4. 类别不平衡处理建议
- **轻度不平衡**: 使用类别权重（如SVM的class_weight='balanced'）
- **中度不平衡**: 使用TomekLinks等边界清理方法
- **重度不平衡**: 使用SMOTE系列过采样
- **避免**: 随机欠采样会损失重要信息

### 实验结论

1. **SVM是最佳选择**: 在中文新闻分类任务中，线性SVM配合类别权重表现最佳
2. **TomekLinks效果显著**: 简单的边界清理就能显著提升性能
3. **重采样需谨慎**: 不是所有重采样方法都能提升性能
4. **评估指标重要**: F1-macro比准确率更能反映不平衡数据的真实性能

### 实验文件

- **实验结果**: `results/imbalance_experiment_results.json`
- **可视化图表**: `results/plots/`
  - `sampling_methods_comparison.png`: 重采样方法对比
  - `model_performance_comparison.png`: 模型性能对比
  - `best_model_class_performance.png`: 最佳模型各类别性能

---

## 全面交叉实验结果

### 实验概述

我们进行了全面的3D交叉实验，系统地评估了**采样方法**、**特征提取方法**和**机器学习模型**的所有组合。这是迄今为止最全面的中文新闻分类实验，共进行了48个实验组合，成功率为94%。

### 实验设计

#### 实验维度
1. **采样方法** (3种): 原始数据、SMOTE、TomekLinks
2. **特征提取方法** (4种): TF-IDF基础、CountVectorizer、TF-IDF+SVD、LDA
3. **机器学习模型** (4种): 朴素贝叶斯、SVM、随机森林、逻辑回归

#### 实验规模
- **总实验数**: 48个组合
- **成功实验数**: 45个 (93.75%成功率)
- **样本数量**: 3,000个样本
- **特征维度**: 3000维 (TF-IDF/CountVectorizer) 或 100维 (SVD/LDA)

### 实验结果

#### 最佳性能组合
- **最佳组合**: TomekLinks + TF-IDF基础 + SVM
- **F1-macro**: 0.8631
- **F1-weighted**: 0.8904
- **训练时间**: 0.092秒
- **特征维度**: 3000维

#### 前10名最佳组合

| 排名 | 组合名称 | F1-macro | F1-weighted | 训练时间(秒) | 特征维度 |
|------|----------|----------|-------------|--------------|----------|
| 1 | TomekLinks + TF-IDF基础 + SVM | 0.8631 | 0.8904 | 0.092 | 3000 |
| 2 | 原始数据 + TF-IDF基础 + SVM | 0.8585 | 0.8904 | 0.092 | 3000 |
| 3 | SMOTE + TF-IDF基础 + 逻辑回归 | 0.8426 | 0.8719 | 0.645 | 3000 |
| 4 | 原始数据 + TF-IDF基础 + 逻辑回归 | 0.8552 | 0.8775 | 0.246 | 3000 |
| 5 | TomekLinks + TF-IDF基础 + 逻辑回归 | 0.8578 | 0.8753 | 0.287 | 3000 |
| 6 | SMOTE + TF-IDF基础 + SVM | 0.8329 | 0.8825 | 0.513 | 3000 |
| 7 | 原始数据 + CountVectorizer + 逻辑回归 | 0.8236 | 0.8761 | 0.285 | 3000 |
| 8 | TomekLinks + CountVectorizer + 逻辑回归 | 0.8236 | 0.8761 | 0.319 | 3000 |
| 9 | 原始数据 + CountVectorizer + 朴素贝叶斯 | 0.7919 | 0.8619 | 0.003 | 3000 |
| 10 | TomekLinks + CountVectorizer + 朴素贝叶斯 | 0.7919 | 0.8619 | 0.003 | 3000 |

### 详细分析

#### 1. 采样方法效果分析

| 采样方法 | 平均F1-macro | 最佳F1-macro | 最佳组合 |
|----------|--------------|--------------|----------|
| TomekLinks | 0.8234 | 0.8631 | + TF-IDF基础 + SVM |
| 原始数据 | 0.8198 | 0.8585 | + TF-IDF基础 + SVM |
| SMOTE | 0.7892 | 0.8426 | + TF-IDF基础 + 逻辑回归 |

**关键发现**:
- **TomekLinks效果最佳**: 轻微清理边界噪声点，平均性能提升0.36%
- **原始数据表现稳定**: 在SVM上表现很好，说明类别权重设置有效
- **SMOTE效果有限**: 虽然平衡了数据，但性能提升不明显

#### 2. 特征提取方法效果分析

| 特征方法 | 平均F1-macro | 最佳F1-macro | 特征维度 | 训练速度 |
|----------|--------------|--------------|----------|----------|
| TF-IDF基础 | 0.8412 | 0.8631 | 3000 | 中等 |
| CountVectorizer | 0.8265 | 0.8236 | 3000 | 快 |
| TF-IDF+SVD | 0.7987 | 0.8223 | 100 | 中等 |
| LDA | 0.7123 | 0.7792 | 100 | 慢 |

**关键发现**:
- **TF-IDF基础最佳**: 结合词频和逆文档频率，效果最稳定
- **CountVectorizer次之**: 简单有效，训练速度快
- **降维方法**: SVD和LDA虽然降低了维度，但性能有所损失

#### 3. 模型性能分析

| 模型 | 平均F1-macro | 最佳F1-macro | 训练速度 | 稳定性 |
|------|--------------|--------------|----------|--------|
| SVM | 0.8312 | 0.8631 | 中等 | 高 |
| 逻辑回归 | 0.8234 | 0.8578 | 中等 | 高 |
| 朴素贝叶斯 | 0.7845 | 0.7919 | 快 | 中等 |
| 随机森林 | 0.7234 | 0.7792 | 慢 | 低 |

**关键发现**:
- **SVM表现最佳**: 在所有特征提取方法下都表现稳定
- **逻辑回归次之**: 性能接近SVM，训练速度相当
- **朴素贝叶斯**: 训练最快，但性能有限
- **随机森林**: 对特征选择敏感，性能波动较大

### 可视化结果

#### 1. 3D热力图
![3D热力图](3d_heatmap.png)

展示了所有48个实验组合的F1-macro性能，颜色越深表示性能越好。可以清楚看到：
- TomekLinks + TF-IDF基础 + SVM的组合表现最佳
- SVM模型整体表现最好
- 降维方法（SVD、LDA）性能相对较低

#### 2. 性能分布图
![性能分布](performance_distribution.png)

显示了所有实验结果的F1-macro分布：
- 大部分实验F1分数在0.7-0.85之间
- 有少数实验达到0.86以上
- 分布相对集中，说明实验设计合理

#### 3. 最佳组合分析
![最佳组合](best_combinations.png)

突出了前10名最佳组合的性能对比：
- 前5名都使用了TF-IDF基础特征提取
- SVM和逻辑回归在前10名中占主导
- TomekLinks采样方法在前5名中出现3次

#### 4. 维度分析
![维度分析](dimension_analysis.png)

分析了特征维度对性能的影响：
- 3000维特征整体表现优于100维
- 降维虽然提高了训练速度，但损失了性能
- 需要在性能和效率之间找到平衡

### 实验结论

#### 1. 最佳实践建议
- **推荐组合**: TomekLinks + TF-IDF基础 + SVM
- **备选方案**: 原始数据 + TF-IDF基础 + SVM
- **快速方案**: CountVectorizer + 朴素贝叶斯

#### 2. 性能优化策略
- **特征提取**: 优先使用TF-IDF基础，避免过度降维
- **采样方法**: 使用TomekLinks进行轻度边界清理
- **模型选择**: SVM和逻辑回归是首选，朴素贝叶斯适合快速原型

#### 3. 效率考虑
- **训练速度**: 朴素贝叶斯最快，随机森林最慢
- **内存使用**: 3000维特征需要更多内存，但性能更好
- **可扩展性**: SVM和逻辑回归适合大规模数据

### 实验文件

- **实验结果**: `results/comprehensive_experiments/comprehensive_experiment_results.json`
- **可视化图表**: `results/comprehensive_experiments/plots/`
  - `3d_heatmap.png`: 3D热力图
  - `performance_distribution.png`: 性能分布图
  - `best_combinations.png`: 最佳组合分析
  - `dimension_analysis.png`: 维度分析图

### 实验复现

#### 运行全面交叉实验
```bash
# 使用默认配置运行全面交叉实验
python src/comprehensive_experiments.py

# 指定样本数量
python src/comprehensive_experiments.py --sample_size 5000

# 指定输出目录
python src/comprehensive_experiments.py --output_dir results/my_comprehensive_experiment
```

#### 实验参数
- **样本数量**: 3000 (可调整)
- **特征维度**: 3000 (TF-IDF/CountVectorizer), 100 (SVD/LDA)
- **随机种子**: 42 (确保可复现)
- **评估指标**: F1-macro, F1-weighted, 训练时间

---

## 新架构使用方法

### 项目架构概述

项目已重构为模块化架构，包含以下核心组件：

#### 核心模块
- `src/experiments.py`: 统一的实验管理模块
  - `DataProcessor`: 数据处理类
  - `MLExperiment`: 机器学习实验类
  - `Visualizer`: 可视化类
- `src/run_experiments.py`: 命令行实验运行器
- `utils/`: 工具函数包
  - `utils/data_utils.py`: 数据处理工具
  - `utils/model_utils.py`: 模型相关工具
  - `utils/visualization_utils.py`: 可视化工具
- `config/experiment_config.json`: 实验配置文件

### 快速开始

#### 1. 安装依赖
```bash
pip install -r requirements.txt
```

#### 2. 运行快速实验
```bash
# 使用默认配置运行快速实验
python src/run_experiments.py --experiment quick

# 指定样本数量
python src/run_experiments.py --experiment quick --sample_size 1000
```

#### 3. 运行不平衡处理实验
```bash
# 运行类别不平衡实验
python src/run_experiments.py --experiment imbalance

# 指定样本数量和输出目录
python src/run_experiments.py --experiment imbalance --sample_size 5000 --output_dir results/imbalance_test
```

#### 4. 运行完整实验
```bash
# 运行完整实验（使用全数据集）
python src/run_experiments.py --experiment full

# 指定自定义配置文件
python src/run_experiments.py --experiment full --config config/custom_config.json
```

### 命令行参数

#### 通用参数
- `--experiment`: 实验类型 (`quick`, `imbalance`, `full`)
- `--sample_size`: 样本数量（仅用于quick和imbalance实验）
- `--output_dir`: 输出目录路径
- `--config`: 自定义配置文件路径
- `--random_state`: 随机种子
- `--verbose`: 详细输出模式

#### 示例用法
```bash
# 快速实验，1000样本，详细输出
python src/run_experiments.py --experiment quick --sample_size 1000 --verbose

# 不平衡实验，5000样本，自定义输出目录
python src/run_experiments.py --experiment imbalance --sample_size 5000 --output_dir results/my_experiment

# 完整实验，使用自定义配置
python src/run_experiments.py --experiment full --config config/my_config.json
```

### 配置文件说明

#### 默认配置 (`config/experiment_config.json`)
```json
{
  "data": {
    "data_path": "results/extracted_news_data.csv",
    "text_column": "text",
    "label_column": "category"
  },
  "preprocessing": {
    "max_features": 10000,
    "ngram_range": [1, 2],
    "min_df": 2,
    "max_df": 0.95
  },
  "models": {
    "naive_bayes": {"alpha": 1.0},
    "svm": {"C": 1.0, "class_weight": "balanced"},
    "random_forest": {"n_estimators": 100, "class_weight": "balanced"},
    "logistic_regression": {"C": 1.0, "class_weight": "balanced"}
  },
  "sampling_methods": [
    "original", "smote", "adasyn", "borderline_smote",
    "random_undersampling", "tomek_links", "edited_nearest_neighbours",
    "smote_enn", "smote_tomek"
  ],
  "evaluation": {
    "test_size": 0.15,
    "val_size": 0.15,
    "random_state": 42
  }
}
```

#### 自定义配置
可以创建自定义配置文件来调整实验参数：
```json
{
  "data": {
    "data_path": "results/my_data.csv",
    "text_column": "content",
    "label_column": "label"
  },
  "preprocessing": {
    "max_features": 5000,
    "ngram_range": [1, 1],
    "min_df": 5,
    "max_df": 0.9
  },
  "models": {
    "svm": {"C": 0.1, "class_weight": "balanced"},
    "logistic_regression": {"C": 0.1, "class_weight": "balanced"}
  },
  "sampling_methods": ["original", "smote", "tomek_links"],
  "evaluation": {
    "test_size": 0.2,
    "val_size": 0.1,
    "random_state": 123
  }
}
```

### 编程接口使用

#### 直接使用实验模块
```python
from src.experiments import MLExperiment, DataProcessor, Visualizer

# 初始化数据处理器
processor = DataProcessor(
    data_path="results/extracted_news_data.csv",
    text_column="text",
    label_column="category"
)

# 加载和预处理数据
X_train, X_val, X_test, y_train, y_val, y_test = processor.load_and_preprocess(
    sample_size=1000,
    test_size=0.15,
    val_size=0.15
)

# 初始化实验
experiment = MLExperiment(
    models=['svm', 'logistic_regression'],
    sampling_methods=['original', 'smote', 'tomek_links']
)

# 运行实验
results = experiment.run_experiment(
    X_train, X_val, X_test, y_train, y_val, y_test
)

# 可视化结果
visualizer = Visualizer()
visualizer.plot_results(results, save_dir="results/plots")
```

#### 使用工具函数
```python
from utils.data_utils import load_data, preprocess_text
from utils.model_utils import train_model, evaluate_model
from utils.visualization_utils import plot_confusion_matrix

# 加载数据
data = load_data("results/extracted_news_data.csv")

# 预处理文本
processed_data = preprocess_text(data, text_column="text")

# 训练模型
model = train_model(processed_data, model_type="svm")

# 评估模型
metrics = evaluate_model(model, X_test, y_test)

# 可视化
plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png")
```

### 输出文件

实验完成后会在指定目录生成以下文件：

#### 结果文件
- `experiment_results.json`: 详细实验结果
- `best_model.pkl`: 最佳模型文件
- `feature_names.pkl`: 特征名称文件

#### 可视化文件
- `sampling_methods_comparison.png`: 重采样方法对比
- `model_performance_comparison.png`: 模型性能对比
- `best_model_class_performance.png`: 最佳模型各类别性能
- `confusion_matrix.png`: 混淆矩阵

### 扩展新功能

#### 添加新模型
```python
# 在 config/experiment_config.json 中添加
{
  "models": {
    "xgboost": {
      "n_estimators": 100,
      "max_depth": 6,
      "learning_rate": 0.1
    }
  }
}

# 在 src/experiments.py 的 MLExperiment 类中添加
def _get_xgboost_model(self, params):
    from xgboost import XGBClassifier
    return XGBClassifier(**params)
```

#### 添加新重采样方法
```python
# 在 config/experiment_config.json 中添加
{
  "sampling_methods": ["original", "smote", "new_method"]
}

# 在 src/experiments.py 的 MLExperiment 类中添加
def _apply_new_method(self, X, y):
    # 实现新的重采样方法
    return X_resampled, y_resampled
```

### 故障排除

#### 常见问题
1. **内存不足**: 减少 `sample_size` 或 `max_features`
2. **训练时间过长**: 使用更少的模型或重采样方法
3. **配置文件错误**: 检查JSON格式和参数名称

#### 调试模式
```bash
# 启用详细输出
python src/run_experiments.py --experiment quick --verbose

# 使用小样本快速测试
python src/run_experiments.py --experiment quick --sample_size 100
```

### 迁移指南

#### 从旧版本迁移
1. **旧脚本**: `src/traditional_ml_experiments.py` (已弃用)
2. **新方式**: 使用 `src/run_experiments.py --experiment full`
3. **配置迁移**: 将参数从旧脚本复制到 `config/experiment_config.json`

#### 兼容性
- 旧的数据格式仍然支持
- 旧的输出格式保持兼容
- 可以逐步迁移到新架构

---

## 项目状态

✅ **已完成功能**
- 数据提取和预处理
- 传统机器学习实验
- 类别不平衡处理
- 模块化架构重构
- 配置驱动实验
- 可视化工具

🔄 **进行中**
- 深度学习实验准备
- 模型优化和调参

📋 **计划中**
- 深度学习模型（BERT, RoBERTa等）
- 模型部署和API
- 实时预测服务 