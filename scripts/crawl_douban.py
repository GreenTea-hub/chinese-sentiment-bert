# -*- coding: utf-8 -*-
"""豆瓣电影短评爬虫：自建电影评论情感数据集（报告 5.1 计划之一）。

策略：
    - 抓取指定电影的短评页（每页 20 条，支持多页）
    - 用豆瓣星级做弱标签：>=4 星 -> positive，<=2 星 -> negative，3 星 -> neutral
    - 输出 JSONL 到 data/processed/douban/，格式与现有分类数据集一致
      {text, label, label_id}

用法：
    .venv\\Scripts\\python.exe scripts/crawl_douban.py --movie-id 1292052 --pages 8
    .venv\\Scripts\\python.exe scripts/crawl_douban.py --movie-id 1291546 --pages 8

已知电影 id（可自行在豆瓣页面 URL 中获取）：
    1292052 肖申克的救赎 | 1291546 霸王别姬 | 1295644 让子弹飞
    3541415 流浪地球 | 26363254 我不是药神
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data.preprocess import clean_text, is_valid_text, normalize_label  # noqa: E402

BASE = "https://movie.douban.com/subject/{mid}/comments"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# 星级 -> 弱标签
STAR_TO_LABEL = {"50": "positive", "40": "positive", "30": "neutral",
                 "20": "negative", "10": "negative"}


def fetch_page(mid, start, retries=3):
    """抓取一页短评 HTML；失败重试，间隔限速。"""
    url = f"{BASE.format(mid=mid)}?start={start}&limit=20&status=P&sort=new_score"
    for attempt in range(retries):
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=20) as resp:
                html = resp.read().decode("utf-8", "ignore")
            return html
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [warn] 页 {start} 抓取失败: {e}")
                return None
            time.sleep(3 * (attempt + 1))


def parse_comments(html):
    """从短评页 HTML 解析 (文本, 星级) 列表。"""
    items = []
    # 每条短评：<li class="comment-item" ...> ... <span class="allstarXX rating" ...> ... <p class="comment-content"><span class="short">文本</span></p>
    pattern = re.compile(
        r'<li class="comment-item".*?'
        r'class="allstar(\d+) rating".*?'
        r'<span class="short">(.*?)</span>',
        re.S)
    for m in pattern.finditer(html):
        star, text = m.group(1), m.group(2)
        text = re.sub(r"<.*?>", "", text)
        text = text.strip()
        items.append((text, star))
    return items


def main():
    parser = argparse.ArgumentParser(description="爬取豆瓣电影短评")
    parser.add_argument("--movie-id", default="1292052",
                        help="豆瓣电影 id（URL 中 subject/ 后的一串数字）")
    parser.add_argument("--pages", type=int, default=8,
                        help="抓取页数（每页 20 条）")
    parser.add_argument("--sleep", type=float, default=1.5,
                        help="每页间隔秒数（防反爬）")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "data", "processed", "douban")
    os.makedirs(out_dir, exist_ok=True)

    collected = []
    for page in range(args.pages):
        start = page * 20
        html = fetch_page(args.movie_id, start)
        if not html:
            continue
        comments = parse_comments(html)
        if not comments:
            print(f"页 {start}: 无评论（可能被风控，稍后重试该页）")
            time.sleep(3)
            html2 = fetch_page(args.movie_id, start)
            if html2:
                comments = parse_comments(html2)
        for text, star in comments:
            label = STAR_TO_LABEL.get(star)
            if not label:
                continue
            text = clean_text(text)
            if not is_valid_text(text):
                continue
            _, label_id = normalize_label(label)
            collected.append({"text": text, "label": label, "label_id": label_id})
        print(f"页 {start}: 累计 {len(collected)} 条")
        time.sleep(args.sleep)

    # 保存
    out_path = os.path.join(out_dir, f"douban_{args.movie_id}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for it in collected:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    from collections import Counter
    dist = Counter(it["label"] for it in collected)
    print(f"\n完成：共 {len(collected)} 条 -> {out_path}")
    print(f"标签分布: {dict(dist)}")


if __name__ == "__main__":
    main()
