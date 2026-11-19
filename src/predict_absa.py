# -*- coding: utf-8 -*-
"""ABSA 推理：对输入文本做方面抽取 + 方面情感分类（对接 Flask 界面）。

用法：
    .venv\\Scripts\\python.exe src/predict_absa.py --text "The food is great but the service was horrible."
"""

import argparse
import os
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR  # noqa: E402

# 中文极性显示映射
POLARITY_CN = {0: "积极", 1: "消极", 2: "中性", 3: "冲突"}


def load_absa(domain="restaurant"):
    ckpt_path = os.path.join(OUTPUT_DIR, f"absa_joint_{domain}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到 ABSA 模型 {ckpt_path}")
    from models.absa_joint import BertAbsaJoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = BertAbsaJoint(pretrained_name=ckpt["pretrained"], num_polarities=4)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ckpt["pretrained"])
    return model, tokenizer


def predict_aspects(text, domain="restaurant", device=None):
    """抽取方面词并判断情感，返回 [{term, polarity}, ...]。"""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_absa(domain)
    model.to(dev)

    enc = tokenizer(text, return_tensors="pt",
                    return_offsets_mapping=True,
                    truncation=True, max_length=128)
    ids = enc["input_ids"].to(dev)
    mask = enc["attention_mask"].to(dev)
    offsets = enc["offset_mapping"][0].tolist()

    with torch.no_grad():
        ex_logits, pol_logits = model(ids, mask, return_spans=True)

    pred = ex_logits.argmax(dim=-1)[0].cpu().tolist()
    max_len = int(mask[0].sum().item())

    # 解码 span
    spans, cur = [], None
    for i in range(max_len):
        if pred[i] == 0:
            cur = i
        elif pred[i] == 2 and cur is not None:
            spans.append((cur, i))
            cur = None
    if cur is not None:
        spans.append((cur, max_len))

    aspects = []
    for idx, (s, e) in enumerate(spans):
        # 还原方面词文本
        term = tokenizer.decode(ids[0][s:e], skip_special_tokens=True)
        # 该 span 的情感
        pol_id = int(pol_logits[idx].argmax().item()) if pol_logits is not None else 2
        aspects.append({"term": term, "polarity": POLARITY_CN.get(pol_id, "未知")})
    return aspects


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABSA 推理")
    parser.add_argument("--domain", choices=["restaurant", "laptop"],
                        default="restaurant")
    parser.add_argument("--text", default=None)
    args = parser.parse_args()

    if args.text:
        print(predict_aspects(args.text, domain=args.domain))
    else:
        print("交互模式（输入 exit 退出）：")
        while True:
            line = input("> ").strip()
            if not line or line.lower() in ("exit", "quit"):
                break
            print("  ", predict_aspects(line, domain=args.domain))
