# .gitignore 配置说明

## 概述

本项目的 `.gitignore` 文件配置确保大文件和数据不会被上传到GitHub，保持仓库的轻量级。

## 主要规则

### 1. 数据文件排除
- **`data/`**: 排除整个原始数据目录
- **`results/`**: 排除处理结果目录（包含2GB+的大文件）
- **`*.csv`, `*.json`, `*.txt`**: 排除所有数据文件格式

### 2. 保留的文件
- **`sample.csv`**: 保留样本文件用于GitHub展示
- **`data/stopwords.txt`**: 保留停用词文件
- **`.gitkeep`**: 保留空目录结构

### 3. 模型和日志
- **`models/`**: 排除模型文件目录
- **`logs/`**: 排除日志文件目录

### 4. 大文件格式
排除以下大文件格式：
- 机器学习模型: `*.pkl`, `*.h5`, `*.bin`, `*.safetensors`
- 压缩文件: `*.zip`, `*.tar.gz`, `*.rar`
- 媒体文件: `*.mp4`, `*.mp3`, `*.jpg`
- 文档文件: `*.pdf`, `*.doc`, `*.xlsx`

## 当前被忽略的文件

根据当前配置，以下文件不会被上传到GitHub：

```
results/extracted_news_data.csv     (2.0GB)
results/extracted_news_data.json    (2.1GB)
data/体育/                          (包含131,603个文件)
data/科技/                          (包含162,929个文件)
... (所有原始数据目录)
```

## 会被上传的文件

```
sample.csv                          (132KB - 样本数据)
README.md                           (项目说明)
src/                                (源代码)
.gitignore                          (忽略规则)
requirements.txt                    (依赖文件)
```

## 使用建议

1. **开发时**: 使用 `sample.csv` 进行快速测试
2. **训练时**: 使用完整的 `results/extracted_news_data.csv`
3. **部署时**: 从THUCNews官网下载完整数据集

## 注意事项

- 确保在提交前检查 `git status` 确认大文件被正确忽略
- 如果需要分享完整数据，请使用外部链接或数据托管服务
- 定期清理临时文件和缓存文件 