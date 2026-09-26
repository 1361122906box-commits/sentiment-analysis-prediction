# 中文舆情情感分析与趋势预测

数据挖掘与商务智能课程项目：对中文评论进行正面、中性、负面分类，使用 BERT 进行情感分析、SVM 进行对比评估，并通过 LSTM 和 ECharts 展示情感变化趋势。

仓库提供一套可直接运行的、完全虚构的演示数据。克隆仓库后，不需要原作者的私有数据或模型，也可以先生成并打开演示大屏；如需运行 BERT，则可按下面的固定版本命令下载公开的中文基础模型并用演示数据微调。

## 五分钟体验

安装 Python 3.10–3.12 和 Git 后，在终端执行：

```bash
git clone https://github.com/1361122906box-commits/sentiment-analysis-prediction.git
cd sentiment-analysis-prediction
python reproduce.py demo
python reproduce.py serve --directory artifacts/demo
```

打开 `http://127.0.0.1:8000`，用 `Ctrl+C` 停止服务。此路径仅使用 Python 标准库，不需要安装机器学习依赖、下载权重或访问 CDN。页面明确标注“虚构示例数据”和“关键词规则分类”。重复生成时请指定新目录，例如 `python reproduce.py demo --output artifacts/demo-2`。

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
| `reproduce.py` | 可移植的演示、模型下载、训练、评估和大屏导出入口 |
| `examples/comments.csv` | 36 条虚构评论，含 train、validation、test 三个划分 |
| `tests/test_reproduction.py` | 数据校验、离线大屏、模型流程回归测试 |
| `THIRD_PARTY.md` | 随仓库分发的前端脚本及授权信息 |
| `other/` | 早期实验脚本，部分使用原作者本机绝对路径 |

## 环境准备

推荐 Python 3.10–3.12。仓库中的 `requirements.txt` 已改为跨机器可安装的固定版本；`environment.yml` 提供对应的 Conda 环境。CPU 可以运行演示和小规模训练，GPU 需要按自己的 CUDA 版本安装匹配的 PyTorch。

```bash
conda env create -f environment.yml
conda activate sentiment-repro
python -m unittest discover -s tests -v
```

也可以使用 Python 虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

上面使用 Windows 命令直接调用虚拟环境里的 Python，无需修改 PowerShell 执行策略。后续 BERT 命令中的 `python` 也要替换成 `.venv\Scripts\python.exe`。macOS/Linux 对应路径为 `.venv/bin/python`。

## 数据与模型

原作者的真实评论、清洗后数据、训练集、人工标注文件、分析结果、模型权重、虚拟环境和备份均未上传。仓库内的 `examples/comments.csv` 是新编写的虚构数据，仅用于验证代码路径，不能代表真实业务效果。

### 数据目录

以下目录仅用于原始课程脚本；使用 `reproduce.py` 的默认示例流程无需准备它们。在项目根目录准备：

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

如果只想查看可视化效果，不需要准备模型。运行 `demo` 会使用仓库内的虚构数据和规则分类，生成自包含的大屏。

原始课程脚本在以下位置加载兼容 Hugging Face Transformers 的模型文件（配置、分词器与权重）。新的 `reproduce.py` 默认将下载和训练产物放到 `artifacts/`：

```text
finetuned_models/
  bert-base-chinese/             # 基础中文 BERT 模型
  bert-sentiment-weibo/          # 本项目微调后的三分类模型
```

推荐使用可复现入口下载固定提交的公开基础模型：

```bash
python reproduce.py download --model artifacts/base-model
```

下载命令使用 `google-bert/bert-base-chinese` 的固定 revision，并写入 `download-source.json`。它不会覆盖已有的项目模型目录。`python reproduce.py train` 会用演示数据训练三分类模型，写入 `artifacts/trained-model`；这只是验证流程的示例训练，不是对真实舆情数据的质量承诺。

基础模型来源：[官方模型页面](https://huggingface.co/google-bert/bert-base-chinese/tree/84b432f646e4047ce1b5db001d43a348cd3f6bd0)。只下载分词器、配置和一份 safetensors 权重，约 410 MB。需要能连接 Hugging Face；中断后可重新运行同一下载命令续传。不会自动回退到未经训练的分类模型。

## 运行

所有命令均在项目根目录执行。最快的离线演示不需要下载模型：

```bash
python reproduce.py demo
python reproduce.py serve --directory artifacts/demo
```

浏览器打开命令输出的 `http://127.0.0.1:8000`，即可看到评论、情感分布、趋势、词云和规则基线预测。演示大屏随输出目录复制 ECharts 和词云脚本，不依赖 CDN。

完整的 BERT、SVM、LSTM 流程：

```bash
python reproduce.py download --model artifacts/base-model
python reproduce.py train --base-model artifacts/base-model --model artifacts/trained-model --epochs 3
python reproduce.py analyze --model artifacts/trained-model --output artifacts/bert-dashboard
python reproduce.py serve --directory artifacts/bert-dashboard
```

`analyze` 会在 test 划分上输出 BERT 和 SVM 指标，训练 LSTM 预测未来 6 个时间点，并导出同样的大屏。若机器资源有限，可先用 `--head-only --epochs 1` 做流程冒烟测试。首次下载基础模型需要网络；之后训练和分析可以使用本地模型文件。

训练仅使用 `train` 划分，按 `validation` 的宏平均 F1 保存最佳模型，评估仅使用 `test`。默认示例仅有 18 条训练、9 条验证和9 条测试评论，指标用于验证程序执行，不代表真实泛化性能。`--head-only` 会冻结 BERT 编码器，仅训练分类相关参数；LSTM 预测未做未来时间区间回测。

模型和大屏输出目录已存在时，命令会拒绝覆盖；重复实验请指定新的 `--model`、`--output`。替换为自有 CSV 时，可在 `train` 和 `analyze` 两条命令中传入 `--data 路径.csv`。CSV 字段为 `content,sentiment,split,topic,time`，与示例一致；不同划分中出现相同评论会被拒绝。

原始课程脚本仍可按原方式使用，但它要求用户自行准备 `数据源/`、`清洗后的数据源/`、`训练集/` 和 `finetuned_models/`：

```bash
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

`reproduce.py demo` 或 `reproduce.py analyze` 已经生成了包含 `events.json`、数据 JSON 和本地前端脚本的完整目录，直接运行：

```bash
python reproduce.py serve --directory artifacts/demo
```

页面通过 `fetch` 加载 JSON，因此要使用 HTTP 服务访问，而不是直接双击 HTML。原始课程结果也可以复制到 `可交互的可视化大屏/`，但需要同时提供与 `dashboard.js` 配置相匹配的文件名。

## 复现边界

演示数据和模型流程可以从仓库、公开模型地址和命令完整重建；原作者的真实数据、微调权重与历史输出不在仓库中，因此不能从 GitHub 恢复这些私有实验材料。真实数据的效果、训练时间和 GPU 占用会随数据规模和硬件变化。

已在 Windows + Python 3.12 的独立环境中验证标准库演示，以及从公开模型下载到三轮 BERT 训练、SVM 评估、LSTM 和大屏导出的完整路径。其他操作系统和 GPU 尚未实机验证。具体检查见 [复现验证记录](VALIDATION.md)。
