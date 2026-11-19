# -*- coding: utf-8 -*-
"""情感词典规则法基线（报告 6.2 基线之一）。

实现：内置精简中文情感词典（积极/消极词）+ 否定词 + 程度副词，
统计情感分值并用规则修正。词典可扩展为大连理工情感词汇本体库
（本实现内置常用词表保证开箱即用、结果可复现）。

输出与统一标签体系一致：positive / negative / neutral。
"""

import re


# 常用积极情感词（可扩充）
POSITIVE_WORDS = {
    "好", "棒", "赞", "喜欢", "爱", "满意", "不错", "优秀", "出色", "精彩",
    "舒服", "舒适", "干净", "漂亮", "美观", "实惠", "划算", "超值", "惊喜",
    "开心", "高兴", "快乐", "幸福", "温暖", "贴心", "周到", "热情", "耐心",
    "推荐", "值得", "好评", "满分", "给力", "完美", "一流", "顶级", "惊艳",
    "感动", "经典", "良心", "性价比高", "物美价廉", "好吃", "美味", "正宗",
    "专业", "高效", "迅速", "方便", "便捷", "免费", "大气", "豪华", "精致",
    "期待", "感谢", "谢谢", "加油", "支持", "赞不绝口", "赏心悦目",
}

# 常用消极情感词（可扩充）
NEGATIVE_WORDS = {
    "差", "烂", "渣", "讨厌", "失望", "不满意", "糟糕", "垃圾", "差劲",
    "难吃", "难闻", "脏", "乱", "吵", "慢", "贵", "坑", "骗", "后悔",
    "生气", "愤怒", "伤心", "难过", "郁闷", "烦躁", "恶心", "无语", "崩溃",
    "糟糕透顶", "浪费时间", "浪费钱", "差评", "退款", "投诉", "客服差",
    "态度差", "质量差", "服务差", "环境差", "一般般", "将就", "勉强",
    "失望透顶", "再也不来", "别来", "黑心", "无良", "敷衍", "糊弄",
}

# 否定词：翻转情感极性
NEGATION_WORDS = {"不", "没", "没有", "无", "非", "莫", "别", "未", "不太", "不很", "毫不"}

# 程度副词：加倍情感分值
INTENSIFIERS = {
    "非常": 2.0, "十分": 2.0, "特别": 2.0, "极其": 2.5, "太": 2.0,
    "很": 1.8, "挺": 1.5, "真": 1.6, "超": 2.0, "超级": 2.2,
    "有点": 0.6, "稍微": 0.6, "略": 0.7, "比较": 1.2, "相当": 1.5,
}

NEGATION_RE = re.compile("|".join(sorted(NEGATION_WORDS, key=len, reverse=True)))
INTENSIFIER_RE = re.compile("|".join(sorted(INTENSIFIERS, key=len, reverse=True)))


def _score_window(tokens, idx, window=3):
    """计算第 idx 个情感词窗口内的否定/程度修正系数。"""
    neg = 1.0
    amp = 1.0
    for j in range(max(0, idx - window), idx):
        if NEGATION_RE.search(tokens[j]):
            neg = -1.0
        m = INTENSIFIER_RE.search(tokens[j])
        if m:
            amp *= INTENSIFIERS[m.group(0)]
    return neg * amp


def predict(text: str, threshold=0.5):
    """对文本做情感词典规则法预测。

    返回 (label, score)。score>0 且 >threshold 为 positive，
    score<0 且 < -threshold 为 negative，否则 neutral。
    """
    # 简单分词：按标点和空白切分（词典匹配不需要精确分词）
    tokens = [t for t in re.split(r"[\s,.;:!?()\"'\[\]-]+", text) if t]
    pos_score = 0.0
    neg_score = 0.0
    for idx, tok in enumerate(tokens):
        if tok in POSITIVE_WORDS:
            pos_score += _score_window(tokens, idx)
        elif tok in NEGATIVE_WORDS:
            neg_score += _score_window(tokens, idx)
    score = pos_score + neg_score
    if score >= threshold:
        return "positive", score
    if score <= -threshold:
        return "negative", score
    return "neutral", score


def predict_batch(texts):
    return [predict(t) for t in texts]


if __name__ == "__main__":
    tests = ["这家店的食物很好吃，服务也很热情。",
             "房间太脏了，服务态度差，再也不来了。",
             "位置一般，价格还算可以。"]
    for t in tests:
        label, score = predict(t)
        print(f"{label:>8} (score={score:+.2f}) | {t}")
