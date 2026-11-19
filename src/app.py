# -*- coding: utf-8 -*-
"""Flask 演示界面：情感分类 + 方面级情感分析（报告 6.1 推理封装层）。

用法（在项目根目录）：
    .venv\\Scripts\\python.exe src/app.py
然后浏览器打开 http://127.0.0.1:5000

需要先训练好模型（src/train.py / src/train_absa.py）。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, render_template_string, request  # noqa: E402

from config import OUTPUT_DIR  # noqa: E402

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>中文文本情感分析系统</title>
<style>
  body { font-family: "Microsoft YaHei", sans-serif; max-width: 760px; margin: 40px auto; padding: 0 20px; color: #333; }
  h1 { color: #1a5fb4; }
  textarea { width: 100%; height: 120px; font-size: 15px; padding: 10px; box-sizing: border-box; }
  button { margin-top: 12px; padding: 10px 26px; font-size: 15px; background: #1a5fb4; color: #fff; border: none; border-radius: 6px; cursor: pointer; }
  .result { margin-top: 20px; padding: 16px; background: #f5f8ff; border-radius: 8px; border: 1px solid #dbe4ff; }
  .pos { color: #2e7d32; font-weight: bold; } .neg { color: #c62828; font-weight: bold; } .neu { color: #757575; font-weight: bold; }
  table { border-collapse: collapse; margin-top: 10px; } td, th { border: 1px solid #ccc; padding: 6px 14px; }
</style>
</head>
<body>
<h1>中文文本情感分析系统</h1>
<p>输入一段评论文本，自动分析整体情感极性，并抽取方面级情感。</p>
<textarea id="text" placeholder="例如：这家店的食物很好吃，但服务很一般。"></textarea><br>
<button onclick="analyze()">分析</button>
<div class="result" id="result" style="display:none"></div>
<script>
async function analyze() {
  const text = document.getElementById('text').value.trim();
  if (!text) return alert('请输入文本');
  const r = await fetch('/analyze', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: text})
  });
  const data = await r.json();
  const cls = data.label === '积极' ? 'pos' : data.label === '消极' ? 'neg' : 'neu';
  let html = '<p>整体情感：<span class="' + cls + '">' + data.label + '</span>' +
             '（置信度 ' + (data.confidence * 100).toFixed(1) + '%）</p>';
  if (data.aspects && data.aspects.length) {
    html += '<p>方面级情感：</p><table><tr><th>方面</th><th>情感</th></tr>';
    for (const a of data.aspects) {
      const ac = a.polarity === '积极' ? 'pos' : a.polarity === '消极' ? 'neg' : 'neu';
      html += '<tr><td>' + a.term + '</td><td class="' + ac + '">' + a.polarity + '</td></tr>';
    }
    html += '</table>';
  } else {
    html += '<p>未抽取到方面（当前未加载 ABSA 模型时仅返回整体情感）</p>';
  }
  const el = document.getElementById('result');
  el.innerHTML = html;
  el.style.display = 'block';
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400

    result = {"text": text, "label": "未知", "confidence": 0.0, "aspects": []}

    # 1) 整体情感分类（优先 bert 模型，其次任意已训练分类模型）
    from predict import predict as predict_sentiment
    for m in ("bert", "bilstm", "textcnn"):
        try:
            r = predict_sentiment(text, model_name=m, dataset="chnsenticorp")
            result["label"] = r["label"]
            result["confidence"] = r["confidence"]
            break
        except FileNotFoundError:
            continue

    # 2) 方面级情感分析（若已训练 ABSA 模型）
    try:
        from predict_absa import predict_aspects
        result["aspects"] = predict_aspects(text)
    except (ImportError, FileNotFoundError):
        pass

    return jsonify(result)


if __name__ == "__main__":
    print("启动演示界面: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
