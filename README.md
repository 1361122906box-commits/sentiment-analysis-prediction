# 中文舆情情感分析与趋势预测

数据挖掘与商务智能课程项目：对中文评论进行正面、中性、负面分类，使用 BERT 进行情感分析、SVM 进行对比评估，并通过 LSTM 和 ECharts 展示情感变化趋势。

## 仓库内容

| 文件 | 用途 |
| --- | --- |
| `数据清洗脚本.py` | 评论数据清洗 |
| `data_loader.py` | 训练集、测试集与人工标注数据加载 |
| `model_trainer.py` | BERT 微调与模型加载 |
| `main_analysis.py` | 单事件情感分析与结果导出 |
| `model_evaluator.py` | BERT、SVM 指标与混淆矩阵对比 |
| `lstm_predictor.py` | 情感时间序列预测 |
| `visualization_data_exporter.py` | 可视化数据整理与批量导出 |
| `batch_processor.py` | 批量处理入口 |
| `font_utils.py` | 中文图表字体配置 |
| `可交互的可视化大屏/` | HTML、JavaScript 可视化页面 |
| `other/` | 早期实验脚本，部分使用原作者本机绝对路径 |

## 环境准备

原项目使用 Python 3.8、PyTorch 2.0.1、Transformers 4.36.0。`environment.yml` 保留了原 Windows Conda 环境导出信息；`requirements.txt` 是原环境快照，包含 `file:///...` 本机构建路径，不适合直接在其他电脑运行 `pip install -r requirements.txt`。

可从以下核心依赖开始配置独立环境，再按平台补齐依赖；本次上传未重新安装依赖或验证完整训练流程：

```bash
conda create -n sentiment python=3.8
conda activate sentiment
python -m pip install "numpy<2" "pandas<2.1" "matplotlib<3.8" "scikit-learn<1.4" seaborn jieba chardet "torch==2.0.1" "transformers==4.36.0" "tokenizers==0.15.2"
```

GPU 环境需另行选择与设备及 CUDA 相匹配的 PyTorch 安装方式。绘图需安装适合当前操作系统的中文字体。

## 数据与模型

公开仓库仅收录源码和环境说明。原始评论、清洗后数据、训练集、人工标注文件、分析结果、模型权重、虚拟环境和备份均由 `.gitignore` 排除，本地文件保留。克隆仓库后，需要自行准备获准使用的数据和模型，才能运行分析或在大屏中查看结果。

### 数据目录

在项目根目录准备：

```text
数据源/                         # 原始评论 JSON，供清洗脚本使用
清洗后的数据源/                 # 清洗后 JSON，批处理入口会检查文件存在
训练集/
  train_dataset.json
  test_dataset.json
  manual_annotation_template.csv
output/                         # 图表与分析结果输出
```

训练集、测试集使用 JSON 数组，每条记录包含 `content`、`topic`、`time`。例如，以下为格式示例，并非真实评论：

```json
[
  {
    "content": "这是一条用于说明格式的示例评论。",
    "topic": "泡泡玛特直播事故",
    "time": "2025-01-01 12:00:00"
  }
]
```

人工标注 CSV 至少需要 `content`、`sentiment` 列；情感标签为 `positive`、`neutral`、`negative`。训练脚本使用规则生成初始训练标签，以人工标注数据进行验证。

当前分析函数实际从训练集和测试集中按 `topic` 筛选评论；请让话题名称与入口脚本中的事件名称一致。仅提供清洗后的 JSON 不足以运行完整分析。

### 模型目录

在以下位置准备兼容 Hugging Face Transformers 的模型文件（配置、分词器与权重）：

```text
finetuned_models/
  bert-base-chinese/             # 基础中文 BERT 模型
  bert-sentiment-weibo/          # 本项目微调后的三分类模型
```

`python model_trainer.py` 使用基础模型和本地训练数据进行微调，并将结果写入 `bert-sentiment-weibo`。分析时优先加载微调模型；若加载失败，代码会尝试基础模型。基础模型的分类头未经过任务训练，不能将其输出视为已经验证的情感分析结果。

## 运行

所有命令均在项目根目录执行。先准备好上述数据和模型，并创建输出目录：

```bash
python -c "from pathlib import Path; Path('output').mkdir(exist_ok=True)"
python 数据清洗脚本.py
python model_trainer.py
python main_analysis.py
```

批量分析与模型评估入口分别为：

```bash
python batch_processor.py
python model_evaluator.py
```

事件名称与路径在入口脚本中配置。预测和评估结果受训练数据、规则标签与时间序列长度影响，适用于课程实验与方法比较。

## 查看可视化大屏

将分析生成的 `*_可视化数据.json`、`*_情感分析结果.json` 复制到 `可交互的可视化大屏/`，文件名需与 `dashboard.js` 中的配置一致。在项目根目录启动本地 HTTP 服务：

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory 可交互的可视化大屏
```

访问 `http://127.0.0.1:8000`。页面通过 `fetch` 加载 JSON，需使用 HTTP 服务访问；ECharts 和词云扩展从外部 CDN 加载，需要网络连接。仓库不包含评论结果 JSON，未准备这些文件时大屏无法展示分析内容。
