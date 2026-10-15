# 代码目录（src）

本目录存放中文文本情感分析系统的核心代码，按功能划分为：

```
src/
├── config.py               # 全局配置（数据路径、模型超参数等）
├── data/                   # 数据处理
│   ├── preprocess.py       # 文本清洗、分词、标签映射
│   └── dataset.py          # 数据集加载与训练/验证/测试划分
├── models/                 # 模型定义
│   ├── textcnn.py          # TextCNN 基线模型
│   ├── bilstm.py           # BiLSTM 基线模型
│   └── bert_sentiment.py   # BERT 系列情感分类主模型
├── utils/                  # 工具函数
│   └── metrics.py          # 评价指标（Accuracy、Macro-F1 等）
├── train.py                # 训练入口
├── evaluate.py             # 评测入口
└── predict.py              # 推理入口（可对接演示界面）
```

> 说明：原始数据集放在仓库外的 `data/`（不入库）；模型权重、日志等训练产物也不入库。

