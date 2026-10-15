# -*- coding: utf-8 -*-
"""推理入口：对输入文本进行情感预测，可对接 Web 演示界面。"""


def predict(text: str):
    """输入一句中文文本，返回情感极性及其置信度。"""
    # TODO: 实现推理（加载模型 -> 预测 -> 返回结果）
    raise NotImplementedError


if __name__ == "__main__":
    print(predict("这家店的食物非常好吃，但服务很一般。"))

