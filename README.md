# chinese-sentiment-bert

《人工智能导论》课程设计作业 —— 基于预训练语言模型的中文文本情感分析系统

人工智能导论课程设计作业-文本情感分析系统，由**曾理、谢嘉辉、王文挺、邱润峰**四人共同完成。

## 项目简介

本项目选题为《人工智能导论》课程项目指南中的**题目7：自然语言处理（情感分析方向）**，核心任务为中文文本情感分析：

1. **情感极性分类（核心）**：判断输入中文文本的情感极性（积极 / 中性 / 消极）；
2. **方面级情感分析（扩展）**：抽取评论文本中的方面词，并判断每个方面对应的情感极性。

项目以 **BERT 系列预训练模型微调**为主线，以情感词典规则法、TF-IDF+SVM、TextCNN、BiLSTM 等经典方法为基线，构建统一的中文文本情感分析系统，详见开题报告（`docs/`）。

## 项目进度

| 周次 | 阶段 | 状态 |
| --- | --- | --- |
| 第 1–3 周 | 项目分组 | ✅ 已完成 |
| 第 4–6 周 | 前期准备（文献阅读、数据收集） | ✅ 已完成 |
| 第 7 周 | 开题报告 | ✅ 见 `docs/` |
| 第 8–11 周 | 技术方案设计与基线实现 | ✅ 已完成（详见 `docs/项目进展报告.md`） |
| 第 12 周 | 进度检查（技术方案与实验方案） | ⬜ 待开始 |
| 第 13–15 周 | 实验分析与系统封装 | ⬜ 待开始 |
| 第 16 周 | 项目汇报 | ⬜ 待开始 |

## 仓库结构

```
├── README.md                    # 项目说明
├── docs/                        # 项目文档（开题报告、项目进展报告）
│   ├── 开题报告-情感分析系统.md
│   ├── 开题报告-情感分析系统.docx
│   └── 项目进展报告.md
├── scripts/                     # 辅助脚本
│   ├── setup_env.ps1            # 一键环境配置（venv + GPU 检测 + 依赖安装）
│   ├── download_data.py         # 数据集自动下载（ModelScope，SHA256 校验）
│   ├── prepare_data.py          # 一键：下载 + 清洗 + 划分（团队数据保持一致）
│   ├── run_baselines.py         # 词典法 + SVM 基线实验
│   ├── crawl_douban.py          # 豆瓣短评爬虫
│   ├── convert_douban.py        # 豆瓣数据转换（星级 -> 情感标签）
│   ├── make_docx.py             # 报告生成脚本
│   └── 1.py                     # 测试脚本
├── src/                         # 项目核心代码
│   ├── config.py
│   ├── data/                    # 数据处理（preprocess / dataset）
│   ├── models/                  # 模型（textcnn / bilstm / bert / absa_joint / lexicon）
│   ├── utils/                   # 评测指标
│   ├── train.py  train_absa.py  # 训练入口
│   ├── evaluate.py              # 评测入口
│   ├── predict.py  predict_absa.py  # 推理入口
│   └── app.py                   # Flask 演示界面
├── requirements.txt             # 依赖清单
├── 参考资料/                    # 课程资料与参考文献（本地，不入库）
├── 工具/                        # 本机辅助工具（本地，不入库）
└── data/                        # 原始数据集（本地，不入库）
```

## 快速开始

```powershell
# 1. 配置环境（自动创建 .venv、按 GPU 装 CUDA 版 PyTorch、安装依赖）
powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1

# 2. 下载并整理数据集（SHA256 校验，团队数据一致）
.venv\Scripts\python.exe scripts\prepare_data.py

# 3. 训练与评测
.venv\Scripts\python.exe src\train.py --model bert --dataset chnsenticorp
.venv\Scripts\python.exe src\evaluate.py --model bert --dataset chnsenticorp

# 4. 启动 Flask 演示界面（http://127.0.0.1:5000）
.venv\Scripts\python.exe src\app.py
```

## 数据集

### 情感极性分类（已收集）

| 数据集 | 规模 | 来源与说明 |
| --- | --- | --- |
| 中文酒店评论（ChnSentiCorp_htl_all） | 7,766 条 | ModelScope `AiNiklaus/ChnSentiCorp`（Apache-2.0），label 0=负面 / 1=正面 |
| 微博情感（weibo_senti_100k） | train 100k + test 20k | ModelScope `swift/weibo_senti_100k`（Apache-2.0），官方 train/test 划分 |

### 方面级情感分析（已收集）

| 数据集 | 规模 | 来源与说明 |
| --- | --- | --- |
| SemEval-2014 Task 4 | Restaurant 3041+800 / Laptop 3045+800 | ModelScope `anzhou0798/semeval2014`，标准 ABSA XML |

### 数据整理

- 原始数据保存在 `data/raw/`（本地，不入库），统一整理结果在 `data/processed/`（JSONL 格式，不入库）；
- **团队成员获取数据（推荐）**：clone 仓库后运行 `python scripts/prepare_data.py`，脚本会自动从 ModelScope 下载全部数据集（SHA256 校验，与仓库作者数据完全一致）并完成清洗、标签映射与 train/val/test 划分，无需手动查找下载；
- 单独下载原始数据：`python scripts/download_data.py`；
- 处理模块：`src/data/preprocess.py`（清洗/加载/解析）、`src/data/dataset.py`（划分/统计/读写）。

## 运行环境（计划）

- Python 3.x、PyTorch、HuggingFace Transformers、scikit-learn、jieba

