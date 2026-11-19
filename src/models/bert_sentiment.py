# -*- coding: utf-8 -*-
"""BERT 系列情感分类主模型（预训练模型微调）。

结构：预训练模型（[CLS] 表示）-> dropout -> 全连接 -> Softmax。
基于 HuggingFace Transformers 实现。
"""

import os

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class BertSentiment(nn.Module):
    def __init__(self, pretrained_name="bert-base-chinese", num_labels=2,
                 dropout=0.1):
        super().__init__()
        self.pretrained_name = pretrained_name
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(pretrained_name)
        self.config.num_labels = num_labels
        self.backbone = AutoModel.from_pretrained(pretrained_name,
                                                  config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.last_hidden_state[:, 0]    # [CLS]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)              # (B, num_labels)


def get_bert_tokenizer(pretrained_name="bert-base-chinese"):
    """返回对应的分词器（本地缓存优先，避免重复下载）。"""
    return AutoTokenizer.from_pretrained(pretrained_name)


if __name__ == "__main__":
    # 自检：需要先安装 transformers 并联网下载模型
    try:
        tokenizer = get_bert_tokenizer()
        m = BertSentiment(num_labels=2)
        enc = tokenizer("这家店很好吃。", return_tensors="pt")
        y = m(**enc)
        print("BertSentiment output:", tuple(y.shape))
    except Exception as e:
        print("自检跳过（需安装 transformers / 联网下载模型）:", type(e).__name__, e)
