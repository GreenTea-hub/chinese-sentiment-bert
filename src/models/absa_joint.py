# -*- coding: utf-8 -*-
"""ABSA 联合训练模型：BERT 共享编码 + 方面抽取（BIO 序列标注）+ 方面情感分类。

设计（对应开题报告 6.4 的"BERT-based 联合模型"）：
    - 共享编码器：BERT（英文用 bert-base-uncased，中文可换中文权重）
    - 头 1 方面抽取：token 级 BIO 序列标注（B-方面/I-方面/O），CE 损失
    - 头 2 方面情感：对每个标注出的方面 span，取 span 内 hidden 平均，
      接全连接输出 4 类极性（positive/negative/neutral/conflict），CE 损失
    - 联合训练：两路损失相加，同时优化共享 BERT 与两个头
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class BertAbsaJoint(nn.Module):
    """联合 ABSA 模型：方面抽取 + 方面情感分类（多任务联合训练）。"""

    def __init__(self, pretrained_name="bert-base-uncased",
                 num_polarities=4, dropout=0.1):
        super().__init__()
        self.pretrained_name = pretrained_name
        self.num_polarities = num_polarities
        self.config = AutoConfig.from_pretrained(pretrained_name)
        self.backbone = AutoModel.from_pretrained(pretrained_name,
                                                  config=self.config)
        self.dropout = nn.Dropout(dropout)
        # 头1：BIO 抽取（0=B, 1=I, 2=O）
        self.extractor = nn.Linear(self.config.hidden_size, 3)
        # 头2：span 情感（pos/neg/neu/conf）
        self.polarity_cls = nn.Linear(self.config.hidden_size, num_polarities)

    def forward(self, input_ids, attention_mask, aspect_spans=None,
                return_spans=False):
        """训练/推理统一入口。

        aspect_spans: (batch, max_spans, 2) 的 (start_tok, end_tok)（含 end）。
                      训练时提供 gold span 计算情感头损失；
                      return_spans=True 时用抽取头结果解码 span。
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state          # (B, L, H)
        hidden = self.dropout(hidden)

        # 抽取头：逐 token 预测 BIO
        extract_logits = self.extractor(hidden)  # (B, L, 3)

        polarity_logits = None
        if aspect_spans is not None:
            polarity_logits = self._span_polarity(hidden, aspect_spans)
        elif return_spans:
            pred_spans = self._decode_spans(extract_logits, attention_mask)
            if len(pred_spans) > 0:
                polarity_logits = self._span_polarity(
                    hidden, torch.tensor(pred_spans, dtype=torch.long,
                                         device=hidden.device).unsqueeze(0))
        return extract_logits, polarity_logits

    def _span_polarity(self, hidden, aspect_spans):
        """对每个 aspect span 求平均 hidden 并预测极性。"""
        batch_size, seq_len, hidden_size = hidden.shape
        device = hidden.device
        spans = aspect_spans.to(device)         # (B, M, 2)
        valid = (spans[:, :, 1] > spans[:, :, 0])  # (B, M) 有效 span
        mask = valid.float()                    # (B, M)
        span_embs = []
        for b in range(batch_size):
            for m in range(spans.size(1)):
                s, e = spans[b, m, 0].item(), spans[b, m, 1].item()
                if e > s:
                    span_embs.append(hidden[b, s:e].mean(dim=0))
        if not span_embs:
            return None
        emb = torch.stack(span_embs)            # (N, H)
        return self.polarity_cls(self.dropout(emb))

    def _decode_spans(self, extract_logits, attention_mask):
        """从 BIO 预测解码方面 span（推理时用）。返回 [(s, e), ...]。"""
        pred = extract_logits.argmax(dim=-1)    # (B, L)
        batch0 = pred[0]
        max_len = int(attention_mask[0].sum().item())
        spans = []
        cur = None
        for i in range(max_len):
            tag = int(batch0[i].item())
            if tag == 0:                        # B
                cur = i
            elif tag == 1 and cur is not None:  # I
                pass
            elif tag == 2:                      # O
                if cur is not None:
                    spans.append((cur, i))
                    cur = None
        if cur is not None:
            spans.append((cur, max_len))
        return spans


if __name__ == "__main__":
    m = BertAbsaJoint()
    ids = torch.randint(0, 100, (1, 20))
    mask = torch.ones_like(ids)
    spans = torch.tensor([[[2, 5], [0, 0]]])     # 一个有效 span
    ex, pol = m(ids, mask, aspect_spans=spans)
    print("extract:", tuple(ex.shape), "polarity:", None if pol is None else tuple(pol.shape))
