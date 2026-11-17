# -*- coding: utf-8 -*-
"""数据准备脚本：自动下载原始数据集并整理为统一格式。

用法：
    python scripts/prepare_data.py

流程：
    1. 若 data/raw 中缺少数据集，自动调用 download_data.py 从 ModelScope 下载
       （SHA256 校验，保证团队成员数据完全一致）；
    2. 将原始数据清洗、标签映射并划分 train/val/test，
       输出到 data/processed/（JSONL 格式）。

输出（data/processed/）：
    chnsenticorp/{train,val,test}.jsonl     中文酒店评论（8:1:1 划分）
    weibo/{train,val,test}.jsonl             微博情感（官方 train/test 划分）
    semeval2014/semeval2014_{rest,laptop}_{train,test}.jsonl   方面级情感
"""

import os
import subprocess
import sys

# 保证可以从任意工作目录运行
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from data.dataset import (  # noqa: E402
    build_aspect_datasets,
    build_classification_datasets,
)

# prepare_data.py 所需的 raw 文件清单（与 download_data.py 保持一致）
RAW_FILES = [
    "ChnSentiCorp/ChnSentiCorp_htl_all.csv",
    "weibo_senti_100k/train.csv",
    "weibo_senti_100k/test.csv",
    "semeval2014/Restaurants_Train_v2.xml",
    "semeval2014/Restaurants_Test_Data_PhaseA.xml",
    "semeval2014/Restaurants_Test_Data_phaseB.xml",
    "semeval2014/Laptop_Train_v2.xml",
    "semeval2014/Laptops_Test_Data_PhaseA.xml",
    "semeval2014/Laptops_Test_Data_phaseB.xml",
]


def _raw_complete(raw_dir) -> bool:
    return all(os.path.exists(os.path.join(raw_dir, rel)) for rel in RAW_FILES)


def main():
    raw_dir = os.path.join(ROOT, "data", "raw")
    out_dir = os.path.join(ROOT, "data", "processed")
    print(f"raw dir : {raw_dir}")
    print(f"out dir : {out_dir}")
    print("=" * 60)

    if not _raw_complete(raw_dir):
        print("检测到 data/raw 数据集缺失，正在自动下载 ...")
        script = os.path.join(HERE, "download_data.py")
        ret = subprocess.call([sys.executable, script])
        if ret != 0:
            print("数据集下载失败，请检查网络后重试。")
            sys.exit(1)
        if not _raw_complete(raw_dir):
            print("数据集仍不完整，终止。")
            sys.exit(1)
    else:
        print("data/raw 数据集完整，跳过下载。")

    build_classification_datasets(raw_dir, out_dir)
    build_aspect_datasets(raw_dir, out_dir)
    print("=" * 60)
    print("done.")


if __name__ == "__main__":
    main()

