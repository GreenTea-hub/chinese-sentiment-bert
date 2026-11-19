# -*- coding: utf-8 -*-
"""豆瓣短评数据转换：把爬取的 {text, star} 原始数据转为标准分类格式。

星级弱标签：50/40 -> positive，30 -> neutral，20/10 -> negative。
输出 data/processed/douban/douban.jsonl（{text, label, label_id}）。

用法：
    .venv\\Scripts\\python.exe scripts/convert_douban.py
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data.preprocess import clean_text, is_valid_text, normalize_label  # noqa: E402

STAR_TO_LABEL = {"50": "positive", "40": "positive", "30": "neutral",
                 "20": "negative", "10": "negative"}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    raw_files = glob.glob(os.path.join(here, "..", "data", "processed",
                                       "douban", "douban_raw_*.jsonl"))
    out_dir = os.path.join(here, "..", "data", "processed", "douban")
    items = []
    for path in raw_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                label = STAR_TO_LABEL.get(str(row.get("star", "")))
                if not label:
                    continue
                text = clean_text(row.get("text", ""))
                if not is_valid_text(text):
                    continue
                _, label_id = normalize_label(label)
                items.append({"text": text, "label": label, "label_id": label_id})

    out_path = os.path.join(out_dir, "douban.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    from collections import Counter
    dist = Counter(it["label"] for it in items)
    print(f"共 {len(items)} 条 -> {out_path}")
    print(f"标签分布: {dict(dist)}")


if __name__ == "__main__":
    main()
