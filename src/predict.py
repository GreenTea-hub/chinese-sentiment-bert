# -*- coding: utf-8 -*-
"""推理入口：对输入文本进行情感预测，可对接 Web 演示界面。

用法：
    .venv\\Scripts\\python.exe src/predict.py --model bert --dataset chnsenticorp "这家酒店很舒适"
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR  # noqa: E402


def load_model(model_name, dataset):
    """加载已训练模型，返回 (model, tokenizer_or_vocab, ckpt)。"""
    ckpt_path = os.path.join(OUTPUT_DIR, f"{model_name}_{dataset}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到模型 {ckpt_path}，请先运行 src/train.py")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if model_name == "bert":
        from models.bert_sentiment import BertSentiment, get_bert_tokenizer
        model = BertSentiment(pretrained_name=ckpt["pretrained"],
                              num_labels=ckpt["num_labels"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, get_bert_tokenizer(ckpt["pretrained"]), ckpt
    else:
        from models.bilstm import BiLSTM
        from models.textcnn import TextCNN
        if model_name == "textcnn":
            model = TextCNN(vocab_size=len(ckpt["vocab"]),
                            num_labels=ckpt["num_labels"],
                            max_len=ckpt["max_len"])
        else:
            model = BiLSTM(vocab_size=len(ckpt["vocab"]),
                           num_labels=ckpt["num_labels"],
                           max_len=ckpt["max_len"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, ckpt["vocab"], ckpt


def predict(text: str, model_name="bert", dataset="chnsenticorp",
            device=None):
    """输入一句中文文本，返回情感极性及其置信度。

    返回 {"text": ..., "label": "积极"/"消极"/..., "label_id": int,
          "confidence": float}
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer_or_vocab, ckpt = load_model(model_name, dataset)
    model.to(dev)

    idx_to_label = {v: k for k, v in ckpt["label_to_idx"].items()}
    # 统一内部标签到中文显示
    cn_map = {0: "消极", 1: "积极", 2: "中性"}

    with torch.no_grad():
        if model_name == "bert":
            enc = tokenizer_or_vocab(text, padding="max_length",
                                     truncation=True,
                                     max_length=ckpt["max_len"],
                                     return_tensors="pt")
            ids = enc["input_ids"].to(dev)
            mask = enc["attention_mask"].to(dev)
        else:
            import jieba
            ids = [tokenizer_or_vocab.get(w, 1) for w in jieba.lcut(text)]
            ids = (ids[:ckpt["max_len"]] +
                   [0] * max(0, ckpt["max_len"] - len(ids)))
            ids = torch.tensor([ids], dtype=torch.long).to(dev)
            mask = torch.ones_like(ids)
        logits = model(ids, mask)
        probs = torch.softmax(logits, dim=1)[0]
        label_id = int(probs.argmax().item())
        confidence = float(probs.max().item())

    label = cn_map.get(label_id, f"label_{label_id}")
    return {"text": text, "label": label, "label_id": label_id,
            "confidence": confidence}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="情感预测")
    parser.add_argument("--model", choices=["textcnn", "bilstm", "bert"],
                        default="bert")
    parser.add_argument("--dataset", choices=["chnsenticorp", "weibo"],
                        default="chnsenticorp")
    parser.add_argument("--device", default=None)
    parser.add_argument("text", nargs="?", default=None,
                        help="要预测的中文文本；缺省进入交互模式")
    args = parser.parse_args()

    if args.text:
        r = predict(args.text, model_name=args.model, dataset=args.dataset,
                    device=args.device)
        print(f"文本: {r['text']}")
        print(f"情感: {r['label']} (置信度 {r['confidence']:.4f})")
    else:
        print("交互模式：输入文本回车预测，输入 exit 退出")
        while True:
            line = input("> ").strip()
            if not line or line.lower() in ("exit", "quit", "q"):
                break
            r = predict(line, model_name=args.model, dataset=args.dataset,
                        device=args.device)
            print(f"  -> {r['label']} (置信度 {r['confidence']:.4f})")
