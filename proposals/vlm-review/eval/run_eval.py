#!/usr/bin/env python3
"""VLM 复核问句考卷跑分：对照 eval_labels.csv 的人工标注，评测各问句方案的违规判定准确率。

考卷构成：eval_list.csv（sheet,cell,图名,规则,conf,桶,verdict）+ eval_labels.csv
（sheet,cell,真人?,戴帽?,穿背心?,吸烟?；"?"=人眼也拿不准的样本，不计分）。
图来自旁路压测落盘的 temp2/ 裁剪（= 生产复核同分布输入）。

方案代号：
  A  复合条件问句（旧线上版，"未戴→yes"）——历史教训：模型按直觉极性反答
  G  复合+few-shot 示例
  E2  v3 部件问句：全图红框判真人 + 头肩/躯干特写判穿戴，违规判定在代码侧（现行）

在 helmet-dev 容器内运行（需 /tmp/temp2 图、/tmp/eval_*.csv、vlm_review.py）：
    python3 run_eval.py E2
"""
import base64
import csv
import json
import sys
import time
import urllib.request

import cv2

import vlm_review

URL, M = "http://vllm:8013/v1/chat/completions", "/model/Qwen2.5-VL-7B-Instruct-AWQ"
A_H = ("下图为一次告警目标的裁剪图。若目标是真实人员且头上未佩戴安全帽，答 yes；"
       "若已佩戴安全帽、或目标不是真实人员（物体/阴影/车辆等），答 no。只回答一个词：yes 或 no。")
A_V = ("下图为一次告警目标的裁剪图。若目标是真实人员且未穿着香港工程反光衣（高可视反光背心），答 yes；"
       "若已穿着、或目标不是真实人员（物体/阴影/车辆等），答 no。只回答一个词：yes 或 no。")
G_H = A_H.replace("只回答一个词：yes 或 no。",
                  "示例：戴黄色安全帽的工人→no；黑头发没戴帽的工人→yes；一堆护栏→no。只回答一个词：yes 或 no。")
G_V = A_V.replace("只回答一个词：yes 或 no。",
                  "示例：穿带反光条背心的工人→no；穿普通橙色短袖的工人→yes；一堆护栏→no。只回答一个词：yes 或 no。")


def call(img, prompt):
    body = {"model": M, "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"
                                            + base64.b64encode(img).decode()}}]}],
        "max_tokens": 1, "temperature": 0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=30))["choices"][0]["message"]["content"]
    return "yes" if r.strip().lower().startswith("y") else "no"


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "E2"
    labels = {(r["sheet"], r["idx"]): r for r in csv.DictReader(open("eval_labels.csv"))}
    ok = tot = 0
    wrong, lat = [], []
    for (sh, idx), L in labels.items():
        if sh == "3":
            continue
        row = [l for l in csv.reader(open("eval_list.csv")) if l[0] == sh and l[1] == idx][0]
        img = open(f"/tmp/temp2/{row[2]}.jpg", "rb").read()
        for rule, lab in (("no_helmet", L["helmet"]), ("no_vest", L["vest"])):
            if lab == "?":
                continue
            exp = "no" if L["person"] == "N" else ("yes" if lab == "N" else "no")
            t0 = time.monotonic()
            if variant == "A":
                got = call(img, A_H if rule == "no_helmet" else A_V)
            elif variant == "G":
                got = call(img, G_H if rule == "no_helmet" else G_V)
            else:  # E2：需要全图+box，从文件名还原
                import re
                from ultralytics import YOLO
                m = re.match(r"(s\d+f\d+)p(\d+)_", row[2])
                full = cv2.imread(f"/tmp/temp2/{m[1]}_full.jpg")
                yolo = globals().setdefault("_yolo", YOLO("/tmp/p.pt"))
                d = yolo(full, conf=0.15, classes=[0], verbose=False)[0]
                bs = [tuple(map(int, b.xyxy[0])) for b in sorted(d.boxes, key=lambda b: -float(b.conf))]
                k = int(m[2])
                if k >= len(bs):
                    continue
                person, wearing, violated, _ = vlm_review.judge_ppe(
                    URL, M, full, bs[k], rule, timeout=15)
                got = "yes" if violated else "no"
            dt = time.monotonic() - t0
            lat.append(dt)
            tot += 1
            if got == exp:
                ok += 1
            else:
                wrong.append(f"{idx}-{rule}:{got}!={exp}")
    lat = sorted(lat)
    print(f"{variant}: {ok}/{tot} = {ok/max(1,tot):.1%}  p50={lat[len(lat)//2]*1000:.0f}ms")
    print("错题:", "; ".join(wrong) or "无")


if __name__ == "__main__":
    main()
