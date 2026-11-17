# -*- coding: utf-8 -*-
"""数据集加载与划分：统一数据格式、训练/验证/测试划分、统计信息。"""

import json
import os
import random

from .preprocess import load_all, clean_text

# 统一的内部标签
LABEL_NAMES = ["negative", "positive", "neutral"]
LABEL_ID_TO_NAME = {i: n for i, n in enumerate(LABEL_NAMES)}


def train_val_test_split(items, ratios=(0.8, 0.1, 0.1), seed=42):
    """按比例划分训练/验证/测试集（先打乱，保持同分布）。"""
    assert abs(sum(ratios) - 1.0) < 1e-6, "划分比例之和必须为 1"
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def dataset_stats(items, name="dataset"):
    """统计类别分布等基础信息。"""
    from collections import Counter
    stats = {"name": name, "total": len(items)}
    labels = Counter()
    for it in items:
        if "label_id" in it:
            labels[it["label_id"]] += 1
    if labels:
        stats["label_distribution"] = {
            LABEL_ID_TO_NAME.get(k, str(k)): v for k, v in sorted(labels.items())
        }
    if items:
        lens = [len(it["text"]) for it in items if "text" in it]
        if lens:
            stats["text_len"] = {"min": min(lens), "max": max(lens),
                                 "avg": round(sum(lens) / len(lens), 2)}
    return stats


def write_jsonl(items, path):
    """把样本列表写成 JSONL（UTF-8）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def read_jsonl(path):
    """读取 JSONL 文件。"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_classification_datasets(raw_dir, out_dir, ratios=(0.8, 0.1, 0.1),
                                  seed=42):
    """把情感极性分类数据集（ChnSentiCorp、weibo）整理为统一 JSONL。

    - ChnSentiCorp：单文件，按 ratios 划分 train/val/test。
    - weibo_senti_100k：官方已提供 train/test 划分——
      train.csv 再按 ratios 划出 val（8:1:1 中 val 取 10%），
      test.csv 整体作为官方测试集。

    输出：
      out_dir/chnsenticorp/{train,val,test}.jsonl
      out_dir/weibo/{train,val,test}.jsonl
    """
    ds = load_all(raw_dir)
    result = {}
    for key, items in ds.items():
        if not items or "aspects" in items[0]:
            continue  # 方面级数据由 build_aspect_datasets 处理
        name = key.replace("_train", "").replace("_test", "")
        if key.endswith("_test"):
            # 官方 test 划分直接作为测试集写入
            sub = os.path.join(out_dir, name, "test.jsonl")
            write_jsonl(items, sub)
            result[f"{name}_test"] = items
            print(f"[{name}] test: {len(items)} 条 -> {sub}")
            continue
        if key.endswith("_train"):
            # 该数据集有官方 test 文件：train 文件只划 train/val
            train, val, _ = train_val_test_split(items, ratios=ratios, seed=seed)
            for split, part in zip(("train", "val"), (train, val)):
                sub = os.path.join(out_dir, name, f"{split}.jsonl")
                write_jsonl(part, sub)
                result[f"{name}_{split}"] = part
                print(f"[{name}] {split}: {len(part)} 条 -> {sub}")
            continue
        # 单文件数据集（如 ChnSentiCorp）：按 ratios 划分 train/val/test
        train, val, test = train_val_test_split(items, ratios=ratios, seed=seed)
        for split, part in zip(("train", "val", "test"), (train, val, test)):
            sub = os.path.join(out_dir, name, f"{split}.jsonl")
            write_jsonl(part, sub)
            result[f"{name}_{split}"] = part
            print(f"[{name}] {split}: {len(part)} 条 -> {sub}")
    return result


def build_aspect_datasets(raw_dir, out_dir):
    """把 SemEval-2014 方面级数据整理为统一 JSONL。

    输出：out_dir/semeval2014/{rest_train,rest_test,laptop_train,laptop_test}.jsonl
    样本格式：{"text": ..., "aspects": [{"term": ..., "polarity": ...}]}
    """
    ds = load_all(raw_dir)
    result = {}
    for key, items in ds.items():
        if not items or "aspects" not in items[0]:
            continue
        name = key  # 例如 semeval2014_rest_train
        sub = os.path.join(out_dir, "semeval2014", f"{name}.jsonl")
        # 规整方面数据字段
        norm_items = []
        for it in items:
            aspects = [{"term": a[0], "polarity": a[1]} for a in it["aspects"]]
            norm_items.append({"text": it["text"], "aspects": aspects})
        write_jsonl(norm_items, sub)
        result[name] = norm_items
        print(f"[semeval2014] {name}: {len(norm_items)} 条 -> {sub}")
    return result


if __name__ == "__main__":
    import sys
    raw = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "data", "raw")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "data", "processed")
    build_classification_datasets(raw, out)
    build_aspect_datasets(raw, out)
