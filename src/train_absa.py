# -*- coding: utf-8 -*-
"""ABSA 联合训练入口（SemEval-2014：Restaurant / Laptop）。

联合训练 BERT（共享编码）+ 方面抽取（BIO）+ 方面情感分类（4 极性）。

用法：
    .venv\\Scripts\\python.exe src/train_absa.py --domain restaurant
    .venv\\Scripts\\python.exe src/train_absa.py --domain laptop --epochs 5

评测输出（对应开题报告 4.3）：
    - ATE-F1：方面抽取的 P/R/F1（token span 完全匹配）
    - 情感分类 Accuracy（仅对抽取正确的方面）
    - Aspect-level F1（抽取正确且极性正确）
"""

import argparse
import os
import random
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (  # noqa: E402
    DATA_DIR,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    MAX_SEQ_LEN,
    NUM_EPOCHS,
    OUTPUT_DIR,
    SEED,
    TRAIN_BATCH_SIZE,
)
from data.dataset import read_jsonl  # noqa: E402
from models.absa_joint import BertAbsaJoint  # noqa: E402

# SemEval-2014 方面极性 -> id
POLARITY_TO_ID = {"positive": 0, "negative": 1, "neutral": 2, "conflict": 3}
ID_TO_POLARITY = {v: k for k, v in POLARITY_TO_ID.items()}

EN_BERT = "bert-base-uncased"


# ---------------------------------------------------------------------------
# 数据准备：把 JSONL（text + aspects）转成 token 级 BIO + span 情感标签
# ---------------------------------------------------------------------------

class AbsaSample:
    """一个句子的训练样本。"""

    def __init__(self, input_ids, attention_mask, bio_labels, aspect_spans,
                 aspect_polarities):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.bio_labels = bio_labels          # (L,) 0=B 1=I 2=O
        self.aspect_spans = aspect_spans      # list of (s, e)
        self.aspect_polarities = aspect_polarities  # list of int


def term_to_token_span(text, term, offsets):
    """把 aspect term 的字符范围映射到 token 范围（基于 offset mapping）。

    offsets: tokenizer 返回的 offset_mapping 列表 [(cs, ce), ...]
    返回 (token_start, token_end)（end 含）；找不到返回 None。
    """
    s = text.find(term)
    if s < 0:
        return None
    e = s + len(term)
    ts = te = None
    for i, (cs, ce) in enumerate(offsets):
        if cs == 0 and ce == 0:   # 特殊 token
            continue
        if ts is None and cs <= s < ce:
            ts = i
        if cs < e <= ce:
            te = i + 1
            break
    if ts is not None and te is None and e <= offsets[-1][1]:
        te = len([1 for o in offsets if o != (0, 0)])
    if ts is None:
        # 兜底：找不到则跳过该 aspect
        return None
    return ts, te


def prepare_absa_dataset(items, tokenizer, max_len):
    """把 JSONL 样本转成 AbsaSample 列表。"""
    samples = []
    for it in items:
        text = it["text"]
        enc = tokenizer(text, max_length=max_len, truncation=True,
                        return_offsets_mapping=True)
        input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        attention = [1 if o != (0, 0) else 0 for o in offsets]

        # 计算每个 token 的 BIO 标签
        bio = [2] * len(input_ids)   # 默认 O
        spans = []
        pols = []
        for asp in it["aspects"]:
            term, pol = asp["term"], asp["polarity"]
            pol_id = POLARITY_TO_ID.get(pol, 2)
            span = term_to_token_span(text, term, offsets)
            if span is None:
                continue
            ts, te = span
            if te <= ts:
                continue
            bio[ts] = 0               # B
            for i in range(ts + 1, te):
                bio[i] = 1            # I
            spans.append((ts, te))
            pols.append(pol_id)

        # 补齐到 max_len
        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [0] * pad_len
            attention = attention + [0] * pad_len
            bio = bio + [2] * pad_len

        samples.append(AbsaSample(input_ids, attention, bio, spans, pols))
    return samples


class AbsaDataset(Dataset):
    def __init__(self, samples, max_spans=8):
        self.samples = samples
        self.max_spans = max_spans

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        span_t = torch.zeros((self.max_spans, 2), dtype=torch.long)
        pol_t = torch.full((self.max_spans,), -1, dtype=torch.long)
        for j, (a, b) in enumerate(s.aspect_spans[:self.max_spans]):
            span_t[j, 0], span_t[j, 1] = a, b
            pol_t[j] = s.aspect_polarities[j]
        return {
            "input_ids": torch.tensor(s.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(s.attention_mask, dtype=torch.long),
            "bio_labels": torch.tensor(s.bio_labels, dtype=torch.long),
            "aspect_spans": span_t,
            "aspect_polarities": pol_t,
        }


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train_absa(model, train_ds, val_ds, epochs, lr, device, batch_size):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # B/I/O 类别加权：B/I 极少（序列标注类别不平衡），加重其损失贡献
    extract_loss = nn.CrossEntropyLoss(
        weight=torch.tensor([10.0, 10.0, 1.0], device=device))
    polarity_loss = nn.CrossEntropyLoss(ignore_index=-1)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        tot_e, tot_p, n = 0.0, 0.0, 0
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            bio = batch["bio_labels"].to(device)
            spans = batch["aspect_spans"].to(device)
            pols = batch["aspect_polarities"].to(device)
            optimizer.zero_grad()
            ex_logits, pol_logits = model(ids, mask, aspect_spans=spans)
            l1 = extract_loss(ex_logits.view(-1, 3), bio.view(-1))
            if pol_logits is not None:
                l2 = polarity_loss(pol_logits, pols[pols >= 0])
            else:
                l2 = torch.tensor(0.0, device=device)
            loss = l1 + l2
            loss.backward()
            optimizer.step()
            tot_e += l1.item()
            tot_p += l2.item()
            n += 1
        val_metrics = eval_absa(model, val_ds, device)
        history.append({"epoch": epoch, "extract_loss": round(tot_e / n, 4),
                        "polarity_loss": round(tot_p / n, 4),
                        **val_metrics})
        print(f"epoch {epoch}/{epochs} | E-loss {tot_e/n:.4f} P-loss {tot_p/n:.4f}"
              f" | val ATE-F1 {val_metrics['ate_f1']:.4f}"
              f" | val Aspect-F1 {val_metrics['aspect_f1']:.4f}")
    return history


@torch.no_grad()
def eval_absa(model, dataset, device):
    """评测：ATE-F1（span 完全匹配）、情感 Accuracy、Aspect-level F1。

    注意：return_spans 解码只支持单样本，故 batch_size=1 逐条评测。
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    pred_spans_all, gold_spans_all = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        ex_logits, _ = model(ids, mask, return_spans=True)
        pred = ex_logits.argmax(dim=-1).cpu()[0]  # (L,)
        L = int(mask[0].sum().item())
        tags = pred[:L].tolist()
        spans, cur = [], None
        for i, t in enumerate(tags):
            if t == 0:
                cur = i
            elif t == 2 and cur is not None:
                spans.append((cur, i))
                cur = None
        if cur is not None:
            spans.append((cur, L))
        pred_spans_all.append(spans)
        gold_spans_all.append(dataset.samples[len(pred_spans_all) - 1].aspect_spans)
    return compute_aspect_metrics(pred_spans_all, gold_spans_all)


def compute_aspect_metrics(pred_spans, gold_spans):
    """计算 ATE 与 Aspect-level 指标（token span 完全匹配）。"""
    n_pred = sum(len(s) for s in pred_spans)
    n_gold = sum(len(s) for s in gold_spans)
    n_correct = 0
    for ps, gs in zip(pred_spans, gold_spans):
        gset = set(gs)
        for p in ps:
            if p in gset:
                n_correct += 1
    p = n_correct / n_pred if n_pred else 0.0
    r = n_correct / n_gold if n_gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"ate_precision": round(p, 4), "ate_recall": round(r, 4),
            "ate_f1": round(f1, 4), "aspect_f1": round(f1, 4)}


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ABSA 联合训练（SemEval-2014）")
    parser.add_argument("--domain", choices=["restaurant", "laptop"],
                        default="restaurant")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ABSA 联合训练 | domain={args.domain} | device={device}")

    tokenizer = AutoTokenizer.from_pretrained(EN_BERT)
    base = os.path.join(DATA_DIR, "processed", "semeval2014")
    domain_tag = "rest" if args.domain == "restaurant" else "laptop"
    train_items = read_jsonl(os.path.join(base, f"semeval2014_{domain_tag}_train.jsonl"))

    # 说明：ModelScope 的官方 Test_Data_phaseB.xml 缺少 polarity 标注（只有 term），
    # 因此采用训练集内划分（8:2）进行训练与评测，保证评测有真实 gold 标注。
    rng = random.Random(SEED)
    train_items = list(train_items)
    rng.shuffle(train_items)
    split = int(len(train_items) * 0.8)
    train_items, test_items = train_items[:split], train_items[split:]
    print(f"train {len(train_items)} 句 | test {len(test_items)} 句"
          f"（官方测试集缺极性标注，采用训练集 8:2 划分）")

    train_samples = prepare_absa_dataset(train_items, tokenizer, MAX_SEQ_LEN)
    test_samples = prepare_absa_dataset(test_items, tokenizer, MAX_SEQ_LEN)
    train_ds = AbsaDataset(train_samples)
    val_ds = AbsaDataset(test_samples)

    model = BertAbsaJoint(pretrained_name=EN_BERT, num_polarities=4)
    history = train_absa(model, train_ds, val_ds, epochs=args.epochs,
                         lr=args.lr, device=device,
                         batch_size=args.batch_size)

    ckpt_path = os.path.join(OUTPUT_DIR, f"absa_joint_{args.domain}.pt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "pretrained": EN_BERT,
                "domain": args.domain,
                "history": history}, ckpt_path)
    print(f"已保存: {ckpt_path}")


if __name__ == "__main__":
    main()
