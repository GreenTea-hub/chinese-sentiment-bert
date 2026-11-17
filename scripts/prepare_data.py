# -*- coding: utf-8 -*-
"""数据准备脚本：将 data/raw 的原始数据集整理为 data/processed 的统一格式。

用法：
    python scripts/prepare_data.py

输出（data/processed/）：
    chnsenticorp/{train,val,test}.jsonl     中文酒店评论（8:1:1 划分）
    weibo/{train,val,test}.jsonl             微博情感（官方 train/test 划分）
    semeval2014/semeval2014_{rest,laptop}_{train,test}.jsonl   方面级情感
"""

import os
import sys

# 保证可以从任意工作目录运行
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from data.dataset import (  # noqa: E402
    build_aspect_datasets,
    build_classification_datasets,
)


def main():
    raw_dir = os.path.join(ROOT, "data", "raw")
    out_dir = os.path.join(ROOT, "data", "processed")
    print(f"raw dir : {raw_dir}")
    print(f"out dir : {out_dir}")
    print("=" * 60)
    build_classification_datasets(raw_dir, out_dir)
    build_aspect_datasets(raw_dir, out_dir)
    print("=" * 60)
    print("done.")


if __name__ == "__main__":
    main()
