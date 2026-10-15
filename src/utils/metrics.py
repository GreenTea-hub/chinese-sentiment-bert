# -*- coding: utf-8 -*-
"""评价指标：Accuracy、Macro-Precision/Recall/F1、混淆矩阵等。"""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_classification(y_true, y_pred, labels=None):
    """计算情感分类的 Accuracy 与宏平均 Precision/Recall/F1。"""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=labels
    )
    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def get_confusion_matrix(y_true, y_pred, labels=None):
    """返回混淆矩阵，便于可视化与误差分析。"""
    return confusion_matrix(y_true, y_pred, labels=labels)

