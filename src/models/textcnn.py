# -*- coding: utf-8 -*-
"""TextCNN 情感分类基线模型（Kim, 2014）。

结构：embedding -> 多尺寸卷积(2,3,4) -> 全局最大池化 -> 拼接 -> dropout -> 全连接。
输入：分词后的 token id 序列（由 train.py 的 build_vocab + tokenize 提供）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, filter_sizes=(2, 3, 4),
                 num_filters=128, num_labels=2, dropout=0.5,
                 padding_idx=0, max_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      padding_idx=padding_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=ks)
            for ks in filter_sizes
        ])
        total_filters = num_filters * len(filter_sizes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(total_filters, num_labels)
        self.max_len = max_len

    def forward(self, x, mask=None):
        # x: (batch, seq_len) 的长整型 token ids；mask 兼容参数（不使用）
        emb = self.embedding(x)                 # (B, L, E)
        emb = emb.transpose(1, 2)               # (B, E, L)
        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(emb))               # (B, F, L-ks+1)
            c = F.max_pool1d(c, c.size(2))      # (B, F, 1)
            conv_outs.append(c.squeeze(2))      # (B, F)
        out = torch.cat(conv_outs, dim=1)       # (B, F*len)
        out = self.dropout(out)
        return self.fc(out)                     # (B, num_labels)


if __name__ == "__main__":
    # 简单自检
    m = TextCNN(vocab_size=100, num_labels=2)
    x = torch.randint(0, 100, (4, 32))
    y = m(x)
    print("TextCNN output:", tuple(y.shape))
