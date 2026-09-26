# 复现验证记录

验证日期：2026-09-26。环境：Windows、Python 3.12.14、CPU，使用 `requirements.txt` 安装的独立虚拟环境。其他操作系统、Python 3.10/3.11 和 GPU 尚未实机验证。

验证目录由 Git 暂存的仓库文件导出，不包含原项目真实数据、微调模型、历史输出或虚拟环境。基础 BERT 单独通过公开下载命令取得，未使用原项目 `finetuned_models/` 中的权重。

## 已完成验证

- `python reproduce.py demo`：仅使用标准库生成自包含大屏和 36 条虚构评论的结果。
- `python -m unittest discover -s tests -v`：6 个测试覆盖样例划分、跨划分重复评论拒绝、输出目录保护、资源校验、规则否定词、小型 BERT 训练与重载、SVM 留出集评估和 LSTM。
- Git 导出副本的资源 SHA-256 校验通过；`.gitattributes` 防止系统换行转换改变下载脚本的字节内容。
- `python reproduce.py download --model <测试目录>/base-model`：从固定 Hugging Face revision 成功下载 5 个必需文件，包括真实预训练 safetensors 权重。
- `python reproduce.py train --base-model <测试目录>/base-model --model artifacts/trained-model --epochs 3`：使用真实基础 BERT 完成三轮训练，按验证集宏平均 F1 保存模型；未使用 `--head-only`。
- `python reproduce.py analyze --model artifacts/trained-model --output artifacts/bert-dashboard`：成功重载模型，在 9 条 test 评论上评估 BERT 与 SVM，并完成 30 轮 LSTM 训练、6 个时间点的预测和大屏导出。
- 本地 HTTP 服务和浏览器页面加载正常，展示 36 条样例评论的分类汇总、时间趋势、词云与预测，并明确标注虚构数据与分类方法。

测试中的小型 BERT 为随机初始化的离线模型，只验证程序连接是否正常；公开基础 BERT 的三轮训练流程另外单独运行。样例数据规模很小，以上检查证明流程可以执行，不证明真实场景的预测质量。

## 重复实验

`download` 可继续下载同一来源的目录；`demo`、`train`、`analyze` 拒绝覆盖既有输出。重新运行时请指定新的 `--output` 或 `--model`，避免与上次实验冲突。
