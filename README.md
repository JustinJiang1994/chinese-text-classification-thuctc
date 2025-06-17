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