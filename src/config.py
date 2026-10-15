# -*- coding: utf-8 -*-
"""全局配置：数据路径、模型超参数等。按实际项目情况调整。"""

# 数据与输出路径
DATA_DIR = "../data"          # 原始数据目录（不入库）
OUTPUT_DIR = "../outputs"     # 输出目录：模型权重、日志等（不入库）

# 训练超参数
SEED = 42
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
MAX_SEQ_LEN = 128
DROPOUT = 0.1

# 预训练模型（按实际选型调整）
PRETRAINED_MODEL_NAME = "bert-base-chinese"
# 可选：bert-base-chinese / hfl/chinese-bert-wwm / hfl/chinese-roberta-wwm-ext

# 情感分类标签
LABELS = ["积极", "中性", "消极"]

