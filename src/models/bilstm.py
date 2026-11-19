# -*- coding: utf-8 -*-
"""BiLSTM 情感分类基线模型。

结构：embedding -> 双向 LSTM（2 层）-> 取最后一个时间步 -> dropout -> 全连接。
输入：分词后的 token id 序列。
"""

import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256,
                 num_layers=2, num_labels=2, dropout=0.5,
                 padding_idx=0, max_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        self.max_len = max_len

    def forward(self, x, mask=None, lengths=None):
        # x: (batch, seq_len); mask: (batch, seq_len) 兼容参数，
        # 若不传 lengths 则按 mask 计算真实长度（padding 处为 0）
        emb = self.embedding(x)                     # (B, L, E)
        if lengths is None and mask is not None:
            lengths = mask.sum(dim=1).cpu().clamp(min=1)
        if lengths is not None:
            emb = nn.utils.rnn.pack_padded_sequence(
                emb, lengths, batch_first=True, enforce_sorted=False)
            out, (h_n, _) = self.lstm(emb)
            # h_n: (num_layers*2, B, H) -> 取最后一层两个方向拼接
            last_fwd = h_n[-2]                      # (B, H)
            last_bwd = h_n[-1]                      # (B, H)
            last = torch.cat([last_fwd, last_bwd], dim=1)  # (B, 2H)
        else:
            out, (h_n, _) = self.lstm(emb)
            last_fwd = h_n[-2]
            last_bwd = h_n[-1]
            last = torch.cat([last_fwd, last_bwd], dim=1)
        last = self.dropout(last)
        return self.fc(last)                        # (B, num_labels)


if __name__ == "__main__":
    m = BiLSTM(vocab_size=100, num_labels=2)
    x = torch.randint(0, 100, (4, 32))
    lens = torch.tensor([32, 20, 15, 8])
    y = m(x, lens)
    print("BiLSTM output:", tuple(y.shape))
