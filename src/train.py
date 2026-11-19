# -*- coding: utf-8 -*-
"""训练入口：加载数据 -> 构建模型 -> 训练 -> 保存最优模型。

用法（在项目根目录）：
    .venv\\Scripts\\python.exe src/train.py --model textcnn --dataset chnsenticorp
    .venv\\Scripts\\python.exe src/train.py --model bilstm  --dataset weibo
    .venv\\Scripts\\python.exe src/train.py --model bert    --dataset chnsenticorp --epochs 3

支持模型：textcnn / bilstm / bert
支持数据集：chnsenticorp / weibo（均为 data/processed 下的 JSONL）
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (  # noqa: E402
    DATA_DIR,
    DROPOUT,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    MAX_SEQ_LEN,
    NUM_EPOCHS,
    OUTPUT_DIR,
    PRETRAINED_MODEL_NAME,
    SEED,
    TRAIN_BATCH_SIZE,
)
from data.dataset import read_jsonl  # noqa: E402
from models.bilstm import BiLSTM  # noqa: E402
from models.textcnn import TextCNN  # noqa: E402


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """通用数据集：token ids / attention_mask / label_id。"""

    def __init__(self, texts, labels, token_ids, label_ids):
        self.texts = texts
        self.labels = labels
        self.token_ids = token_ids
        self.label_ids = label_ids

    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, i):
        return {
            "token_ids": torch.tensor(self.token_ids[i], dtype=torch.long),
            "attention_mask": torch.tensor(
                [1 if t != 0 else 0 for t in self.token_ids[i]], dtype=torch.long
            ),
            "label_id": torch.tensor(self.label_ids[i], dtype=torch.long),
        }


def build_vocab(all_texts, max_vocab=50000):
    """基于 jieba 分词构建词表（TextCNN / BiLSTM 使用）。"""
    import jieba
    from collections import Counter
    counter = Counter()
    for t in all_texts:
        counter.update(jieba.lcut(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, _ in counter.most_common(max_vocab - 2):
        vocab[w] = len(vocab)
    return vocab


def tokenize_texts(texts, vocab, max_len):
    """jieba 分词 -> token id 序列（padding/截断到 max_len）。"""
    import jieba
    rows = []
    for t in texts:
        ids = [vocab.get(w, 1) for w in jieba.lcut(t)]
        ids = ids[:max_len]
        ids = ids + [0] * (max_len - len(ids))
        rows.append(ids)
    return rows


def load_split(dataset):
    """加载指定数据集的 train/val 划分并返回 label 映射。"""
    base = os.path.join(DATA_DIR, "processed", dataset)
    train = read_jsonl(os.path.join(base, "train.jsonl"))
    val = read_jsonl(os.path.join(base, "val.jsonl"))
    label_ids = sorted({it["label_id"] for it in train + val})
    label_to_idx = {v: i for i, v in enumerate(label_ids)}
    num_labels = len(label_ids)
    return train, val, label_to_idx, num_labels


def make_ds(items, token_ids, label_to_idx):
    return TextDataset([it["text"] for it in items],
                       [it["label"] for it in items],
                       token_ids,
                       [label_to_idx[it["label_id"]] for it in items])


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train_model(model, train_ds, val_ds, epochs, lr, device, batch_size,
                eval_batch_size):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False)

    best_acc = 0.0
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total = 0.0, 0
        for batch in loader:
            x = batch["token_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["label_id"].to(device)
            optimizer.zero_grad()
            logits = model(x, mask)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            total += len(y)
        train_loss = total_loss / total

        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in eval_loader:
                x = batch["token_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                y = batch["label_id"].to(device)
                pred = model(x, mask).argmax(dim=1)
                correct += (pred == y).sum().item()
        val_acc = correct / len(val_ds)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "val_acc": round(val_acc, 4)})
        print(f"epoch {epoch}/{epochs} | loss {train_loss:.4f} | val_acc {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    return best_state, best_acc, history


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="训练情感分类模型")
    parser.add_argument("--model", choices=["textcnn", "bilstm", "bert"],
                        default="textcnn")
    parser.add_argument("--dataset", choices=["chnsenticorp", "weibo"],
                        default="chnsenticorp")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", default=None,
                        help="cpu / cuda，默认自动选择")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"模型: {args.model} | 数据集: {args.dataset} | 设备: {device}")

    train, val, label_to_idx, num_labels = load_split(args.dataset)
    vocab = None

    if args.model == "bert":
        from models.bert_sentiment import BertSentiment, get_bert_tokenizer
        tokenizer = get_bert_tokenizer(PRETRAINED_MODEL_NAME)

        def encode(items):
            enc = tokenizer([it["text"] for it in items],
                            padding="max_length", truncation=True,
                            max_length=MAX_SEQ_LEN, return_tensors="pt")
            return enc["input_ids"].numpy().tolist()

        train_ds = make_ds(train, encode(train), label_to_idx)
        val_ds = make_ds(val, encode(val), label_to_idx)
        model = BertSentiment(pretrained_name=PRETRAINED_MODEL_NAME,
                              num_labels=num_labels, dropout=DROPOUT)
    else:
        all_texts = [it["text"] for it in train + val]
        vocab = build_vocab(all_texts)
        train_ds = make_ds(train, tokenize_texts(
            [it["text"] for it in train], vocab, MAX_SEQ_LEN), label_to_idx)
        val_ds = make_ds(val, tokenize_texts(
            [it["text"] for it in val], vocab, MAX_SEQ_LEN), label_to_idx)
        if args.model == "textcnn":
            model = TextCNN(vocab_size=len(vocab), num_labels=num_labels,
                            dropout=DROPOUT, max_len=MAX_SEQ_LEN)
        else:
            model = BiLSTM(vocab_size=len(vocab), num_labels=num_labels,
                           dropout=DROPOUT, max_len=MAX_SEQ_LEN)

    best_state, best_acc, history = train_model(
        model, train_ds, val_ds, epochs=args.epochs, lr=args.lr,
        device=device, batch_size=args.batch_size,
        eval_batch_size=EVAL_BATCH_SIZE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUTPUT_DIR, f"{args.model}_{args.dataset}.pt")
    torch.save({"state_dict": best_state,
                "model": args.model,
                "dataset": args.dataset,
                "num_labels": num_labels,
                "vocab": vocab,
                "label_to_idx": label_to_idx,
                "max_len": MAX_SEQ_LEN,
                "pretrained": PRETRAINED_MODEL_NAME if args.model == "bert" else None,
                "history": history,
                "best_acc": best_acc}, ckpt_path)
    print(f"最优 val_acc: {best_acc:.4f} | 已保存: {ckpt_path}")


if __name__ == "__main__":
    main()
