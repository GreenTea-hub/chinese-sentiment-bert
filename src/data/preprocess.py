# -*- coding: utf-8 -*-
"""数据预处理：文本清洗、标签映射、各数据集的加载与解析。

支持的数据集：
- ChnSentiCorp_htl_all（中文酒店评论，二分类 0/1）
- weibo_senti_100k（微博情感，positive/negative 二分类）
- SemEval-2014 Task 4（方面级情感分析，Restaurant / Laptop，XML 格式）

统一内部标签约定（英文，便于后续模型训练与指标计算）：
- "positive"  -> 1
- "negative"  -> 0
- "neutral"   -> 2（若数据集中出现）
"""

import csv
import json
import re
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# 文本清洗
# ---------------------------------------------------------------------------

# 常见全角/半角标点统一
_FULLWIDTH_TO_ASCII = {
    "，": ",", "。": ".", "！": "!", "？": "?", "：": ":", "；": ";",
    "（": "(", "）": ")", "【": "[", "】": "]", "“": '"', "”": '"',
    "‘": "'", "’": "'", "、": ",", "—": "-", "…": "...", "　": " ",
}


def clean_text(text: str) -> str:
    """基础文本清洗：去空白、统一全角标点、压缩连续空格。"""
    if not isinstance(text, str):
        text = str(text)
    # 去掉首尾空白
    text = text.strip()
    # 统一全角标点为半角
    for full, half in _FULLWIDTH_TO_ASCII.items():
        text = text.replace(full, half)
    # 多个空白字符（含换行制表）压缩为单个空格
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_valid_text(text: str, min_len: int = 2) -> bool:
    """过滤过短或空白的文本。"""
    return len(text.strip()) >= min_len


def tokenize(text: str):
    """中文分词（轻量实现，兼容无 jieba 的环境）。

    优先使用 jieba 精确模式；不可用时退化为按空白与标点切分。
    """
    try:
        import jieba
        return [w for w in jieba.lcut(text) if w.strip()]
    except ImportError:
        # 简单分词：以空白切分，并保留标点两侧的汉字片段
        parts = re.split(r"([,.;:!?()\"'\[\]\-])", text)
        tokens = []
        for p in parts:
            p = p.strip()
            if p:
                tokens.append(p)
        return tokens


# ---------------------------------------------------------------------------
# 标签映射
# ---------------------------------------------------------------------------

_LABEL_TO_ID = {"negative": 0, "positive": 1, "neutral": 2}
_ID_TO_LABEL = {v: k for k, v in _LABEL_TO_ID.items()}


def normalize_label(label) -> tuple:
    """把任意来源的标签规整为 (规范标签, 数值 id)。

    支持的输入：0/1/2、positive/negative/neutral（大小写不敏感）、
    1.0/2.0（weibo 数据常见数值标签）。
    返回 (label_str, label_id)。
    """
    if isinstance(label, (int, float)):
        val = int(label)
        if val in _ID_TO_LABEL:
            return _ID_TO_LABEL[val], val
        raise ValueError(f"未知数值标签: {label}")
    s = str(label).strip().lower()
    if s in _LABEL_TO_ID:
        return s, _LABEL_TO_ID[s]
    # 兼容 1.0 / 2.0 字符串
    try:
        val = int(float(s))
        if val in _ID_TO_LABEL:
            return _ID_TO_LABEL[val], val
    except ValueError:
        pass
    raise ValueError(f"未知文本标签: {label}")


# ---------------------------------------------------------------------------
# 数据集加载
# ---------------------------------------------------------------------------

def load_chnsenticorp(path: str) -> list:
    """加载 ChnSentiCorp_htl_all.csv。

    列：label(0/1), review。返回 [{text, label, label_id}, ...]。
    注：label=1 为正面，label=0 为负面。
    """
    items = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # label,review
        for row in reader:
            if len(row) < 2:
                continue
            label_raw, review = row[0], row[1]
            text = clean_text(review)
            if not is_valid_text(text):
                continue
            label_str, label_id = normalize_label(label_raw)
            items.append({"text": text, "label": label_str, "label_id": label_id})
    return items


def load_weibo(path: str) -> list:
    """加载 weibo_senti_100k 的 train/test.csv（swift 格式）。

    列：query(微博文本), response(标签)。response 可为 positive/negative
    或 1.0/2.0（1=正面, 2=负面）。返回 [{text, label, label_id}, ...]。
    """
    items = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # query,response
        for row in reader:
            if len(row) < 2:
                continue
            query, response = row[0], row[1]
            text = clean_text(query)
            if not is_valid_text(text):
                continue
            # 兼容 1.0=positive, 2.0=negative
            resp = response.strip()
            if resp in ("1.0", "2.0"):
                resp = "positive" if resp == "1.0" else "negative"
            label_str, label_id = normalize_label(resp)
            items.append({"text": text, "label": label_str, "label_id": label_id})
    return items


def load_semeval2014(path: str) -> list:
    """解析 SemEval-2014 Task 4 的 XML 数据。

    每句一个样本：{text, aspects: [(term, polarity), ...]}，
    其中 polarity 为 'positive' / 'negative' / 'neutral' / 'conflict'。
    """
    tree = ET.parse(path)
    root = tree.getroot()
    items = []
    for sentence in root.iter("sentence"):
        text_el = sentence.find("text")
        if text_el is None or not text_el.text:
            continue
        text = clean_text(text_el.text)
        if not is_valid_text(text):
            continue
        aspects = []
        for aspect in sentence.iter("aspectTerm"):
            term = aspect.get("term", "").strip()
            polarity = aspect.get("polarity", "").strip()
            if term and polarity:
                aspects.append((term, polarity))
        items.append({"text": text, "aspects": aspects})
    return items


# ---------------------------------------------------------------------------
# 汇总加载
# ---------------------------------------------------------------------------

def load_all(data_dir: str) -> dict:
    """加载 data/raw 下全部数据集，返回 {dataset_name: items}。"""
    import os
    datasets = {}
    chn_path = os.path.join(data_dir, "ChnSentiCorp", "ChnSentiCorp_htl_all.csv")
    if os.path.exists(chn_path):
        datasets["chnsenticorp"] = load_chnsenticorp(chn_path)

    weibo_train = os.path.join(data_dir, "weibo_senti_100k", "train.csv")
    weibo_test = os.path.join(data_dir, "weibo_senti_100k", "test.csv")
    if os.path.exists(weibo_train):
        datasets["weibo_train"] = load_weibo(weibo_train)
    if os.path.exists(weibo_test):
        datasets["weibo_test"] = load_weibo(weibo_test)

    semeval_dir = os.path.join(data_dir, "semeval2014")
    semeval_map = {
        "semeval2014_rest_train": "Restaurants_Train_v2.xml",
        "semeval2014_rest_test": "Restaurants_Test_Data_PhaseA.xml",
        "semeval2014_laptop_train": "Laptop_Train_v2.xml",
        "semeval2014_laptop_test": "Laptops_Test_Data_PhaseA.xml",
    }
    for key, fname in semeval_map.items():
        p = os.path.join(semeval_dir, fname)
        if os.path.exists(p):
            datasets[key] = load_semeval2014(p)
    return datasets


if __name__ == "__main__":
    # 简单自检：读取原始数据目录并打印统计
    import os
    raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "..", "data", "raw")
    ds = load_all(raw_dir)
    for name, items in ds.items():
        print(f"{name}: {len(items)} 条")
