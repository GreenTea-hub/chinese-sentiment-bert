# -*- coding: utf-8 -*-
"""基线实验：情感词典规则法 + TF-IDF+SVM（报告 6.2 的两条基线）。

在 data/processed 的分类数据集（chnsenticorp / weibo）上评测，
输出 Accuracy / Macro-F1 / 混淆矩阵，结果打印 + 存 JSON（outputs/results/）。

用法：
    .venv\\Scripts\\python.exe scripts/run_baselines.py --dataset chnsenticorp
    .venv\\Scripts\\python.exe scripts/run_baselines.py --dataset weibo
    .venv\\Scripts\\python.exe scripts/run_baselines.py --dataset all
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from config import DATA_DIR, OUTPUT_DIR  # noqa: E402
from data.dataset import read_jsonl  # noqa: E402
from models.lexicon import predict  # noqa: E402
from utils.metrics import evaluate_classification, get_confusion_matrix  # noqa: E402

DATASETS = ["chnsenticorp", "weibo"]


def run_lexicon(items):
    """词典法：无需训练，直接预测测试集。

    词典法只能对命中情感词的文本给出判定（其余为 neutral）。
    返回：判定样本的指标 + 判定覆盖率（供报告中解释）。
    """
    y_true, y_pred = [], []
    judged = 0
    for it in items:
        label, _ = predict(it["text"])
        if label == "neutral":
            continue  # 未判定：不计入指标
        judged += 1
        y_true.append(it["label_id"])
        y_pred.append(1 if label == "positive" else 0)
    if not y_pred:
        return None, None, 0.0
    res = evaluate_classification(y_true, y_pred, labels=[0, 1])
    cm = get_confusion_matrix(y_true, y_pred, labels=[0, 1])
    coverage = judged / len(items)
    return res, cm, coverage


def run_svm(train_items, test_items):
    """TF-IDF + SVM（LinearSVC）。"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    import jieba

    def seg(text):
        return " ".join(jieba.lcut(text))

    vectorizer = TfidfVectorizer(tokenizer=seg, preprocessor=None,
                                 ngram_range=(1, 2), max_features=20000)
    X_tr = vectorizer.fit_transform([it["text"] for it in train_items])
    X_te = vectorizer.transform([it["text"] for it in test_items])
    y_tr = [it["label_id"] for it in train_items]
    y_te = [it["label_id"] for it in test_items]
    clf = LinearSVC(class_weight="balanced")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    res = evaluate_classification(y_te, y_pred, labels=[0, 1])
    cm = get_confusion_matrix(y_te, y_pred, labels=[0, 1])
    return res, cm


def main():
    parser = argparse.ArgumentParser(description="词典法 + SVM 基线实验")
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    args = parser.parse_args()

    targets = DATASETS if args.dataset == "all" else [args.dataset]
    results = {}
    for ds in targets:
        base = os.path.join(DATA_DIR, "processed", ds)
        train = read_jsonl(os.path.join(base, "train.jsonl"))
        test = read_jsonl(os.path.join(base, "test.jsonl"))
        print(f"\n===== {ds} (test {len(test)} 条) =====")

        lex_res, lex_cm, lex_cov = run_lexicon(test)
        svm_res, svm_cm = run_svm(train, test)
        if lex_res:
            print(f"[词典法]  判定覆盖率 {lex_cov:.2%} | 判定样本 Acc {lex_res['accuracy']:.4f}  Macro-F1 {lex_res['macro_f1']:.4f}")
            print(f"         混淆矩阵:\n{lex_cm}")
            results[ds] = {"lexicon": {**lex_res, "coverage": round(lex_cov, 4)}, "svm": svm_res}
        else:
            print("[词典法]  无判定样本")
            results[ds] = {"lexicon": {"coverage": 0.0}, "svm": svm_res}

        print(f"[SVM]     Acc {svm_res['accuracy']:.4f}  Macro-F1 {svm_res['macro_f1']:.4f}")
        print(f"         混淆矩阵:\n{svm_cm}")

    os.makedirs(os.path.join(OUTPUT_DIR, "results"), exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "results", "baselines.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
