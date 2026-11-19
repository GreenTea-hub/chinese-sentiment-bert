# -*- coding: utf-8 -*-
"""评测入口：在测试集上评估模型，输出指标与图表数据。

用法：
    .venv\\Scripts\\python.exe src/evaluate.py --model textcnn --dataset chnsenticorp
    .venv\\Scripts\\python.exe src/evaluate.py --model bert --dataset chnsenticorp --save-plot outputs/plots

输出：Accuracy / Macro-Precision / Macro-Recall / Macro-F1 / 混淆矩阵，
可选保存混淆矩阵与训练曲线图。
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (  # noqa: E402
    DATA_DIR,
    EVAL_BATCH_SIZE,
    MAX_SEQ_LEN,
    OUTPUT_DIR,
    PRETRAINED_MODEL_NAME,
)
from data.dataset import read_jsonl  # noqa: E402
from utils.metrics import evaluate_classification, get_confusion_matrix  # noqa: E402


def load_ckpt(model_name, dataset):
    ckpt_path = os.path.join(OUTPUT_DIR, f"{model_name}_{dataset}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"未找到模型 {ckpt_path}，请先运行 src/train.py")
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def build_model(model_name, ckpt):
    from models.bilstm import BiLSTM
    from models.textcnn import TextCNN
    num_labels = ckpt["num_labels"]
    if model_name == "textcnn":
        model = TextCNN(vocab_size=len(ckpt["vocab"]), num_labels=num_labels,
                        max_len=ckpt["max_len"])
    elif model_name == "bilstm":
        model = BiLSTM(vocab_size=len(ckpt["vocab"]), num_labels=num_labels,
                       max_len=ckpt["max_len"])
    else:
        from models.bert_sentiment import BertSentiment
        model = BertSentiment(pretrained_name=ckpt["pretrained"],
                              num_labels=num_labels)
    model.load_state_dict(ckpt["state_dict"])
    return model


def predict_labels(model, items, model_name, ckpt, device):
    """对样本列表返回预测 label_id 列表。"""
    from train import tokenize_texts  # noqa: F401
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(items), EVAL_BATCH_SIZE):
            chunk = items[i:i + EVAL_BATCH_SIZE]
            texts = [it["text"] for it in chunk]
            if model_name == "bert":
                from models.bert_sentiment import get_bert_tokenizer
                tokenizer = get_bert_tokenizer(ckpt["pretrained"])
                enc = tokenizer(texts, padding="max_length", truncation=True,
                                max_length=ckpt["max_len"], return_tensors="pt")
                ids = enc["input_ids"].to(device)
                mask = enc["attention_mask"].to(device)
            else:
                ids = torch.tensor(
                    tokenize_texts(texts, ckpt["vocab"], ckpt["max_len"]),
                    dtype=torch.long).to(device)
                mask = torch.ones_like(ids)
            logits = model(ids, mask)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
    return preds


def main():
    parser = argparse.ArgumentParser(description="评测情感分类模型")
    parser.add_argument("--model", choices=["textcnn", "bilstm", "bert"],
                        required=True)
    parser.add_argument("--dataset", choices=["chnsenticorp", "weibo"],
                        required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-plot", default=None,
                        help="保存混淆矩阵/训练曲线到指定目录")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_ckpt(args.model, args.dataset)
    model = build_model(args.model, ckpt)

    base = os.path.join(DATA_DIR, "processed", args.dataset)
    test = read_jsonl(os.path.join(base, "test.jsonl"))
    y_true = [ckpt["label_to_idx"][it["label_id"]] for it in test]
    y_pred = predict_labels(model, test, args.model, ckpt, device)

    labels = [str(i) for i in range(ckpt["num_labels"])]
    res = evaluate_classification(y_true, y_pred, labels=labels)
    cm = get_confusion_matrix(y_true, y_pred, labels=labels)
    print(f"[{args.model} / {args.dataset}] 测试集 {len(test)} 条")
    print(f"  Accuracy      : {res['accuracy']:.4f}")
    print(f"  Macro-Precision: {res['macro_precision']:.4f}")
    print(f"  Macro-Recall   : {res['macro_recall']:.4f}")
    print(f"  Macro-F1       : {res['macro_f1']:.4f}")
    print(f"  混淆矩阵:\n{cm}")

    if args.save_plot:
        os.makedirs(args.save_plot, exist_ok=True)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            # 混淆矩阵图
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(cm, cmap="Blues")
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    ax.text(c, r, cm[r, c], ha="center", va="center")
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xlabel("预测")
            ax.set_ylabel("真实")
            fig.colorbar(im)
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_plot,
                                     f"cm_{args.model}_{args.dataset}.png"),
                        dpi=150)
            plt.close()
            # 训练曲线图
            hist = ckpt.get("history", [])
            if hist:
                fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4))
                ax2[0].plot([h["epoch"] for h in hist],
                            [h["train_loss"] for h in hist], marker="o")
                ax2[0].set_title("train loss")
                ax2[1].plot([h["epoch"] for h in hist],
                            [h["val_acc"] for h in hist], marker="o")
                ax2[1].set_title("val acc")
                plt.tight_layout()
                plt.savefig(os.path.join(args.save_plot,
                                         f"curve_{args.model}_{args.dataset}.png"),
                            dpi=150)
                plt.close()
            print(f"图表已保存到 {args.save_plot}")
        except Exception as e:
            print(f"图表保存失败（跳过）: {e}")


if __name__ == "__main__":
    main()
