#!/usr/bin/env python3
"""取 smoke-recheck 裁图的 VLM 判别评分：用 logprobs 还原 P(yes)/P(no)。

仅标准库，宿主机直跑（vLLM 在 127.0.0.1:8013）：
    python3 sandbox/smoke_scores.py --dir temp/smoke-recheck
"""
import argparse
import base64
import glob
import json
import math
import os
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8013/v1/chat/completions"
MODEL = "/model/Qwen2.5-VL-7B-Instruct-AWQ"
PROMPT = ("下图为一次抽烟告警的目标裁剪图。若此人在吸烟（手持香烟/嘴中含烟，点燃与否、"
          "夹在指间均算），答 yes；否则（只是抬手、拿工具、喝水、非真人等）答 no。"
          "只回答一个词：yes 或 no。")


def score(img_bytes, url=URL):
    body = {"model": MODEL, "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"
                                            + base64.b64encode(img_bytes).decode()}}]}],
        "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    last = None
    for _ in range(3):        # vLLM 偶发 500，重试即可（很罕见）
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=30).read())
            break
        except urllib.error.HTTPError as e:
            last = e
    else:
        raise last
    c = r["choices"][0]
    word = c["message"]["content"].strip().lower()
    cands = c["logprobs"]["content"][0]["top_logprobs"]
    py = sum(math.exp(x["logprob"]) for x in cands if x["token"].strip().lower().startswith("y"))
    pn = sum(math.exp(x["logprob"]) for x in cands if x["token"].strip().lower().startswith("n"))
    norm = py / (py + pn) if (py + pn) > 0 else float("nan")
    return word, py, pn, norm, cands[:5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="temp/smoke-recheck")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, "*_crop.jpg")))
    print("| 图 | 输出词 | P(yes) 原始 | P(no) 原始 | P(yes) 归一 | 置信度 |")
    print("| :--- | :---: | ---: | ---: | ---: | :--- |")
    for f in files:
        word, py, pn, norm, top = score(open(f, "rb").read())
        conf = f"{abs(norm - 0.5) * 2 * 100:.1f}%"
        print(f"| {os.path.basename(f)} | **{word}** | {py:.2%} | {pn:.2%} | "
              f"**{norm:.2%}** | {conf} |")
        print(f"  top tokens: " + ", ".join(f"{t['token']!r}:{math.exp(t['logprob']):.3f}"
                                             for t in top))


if __name__ == "__main__":
    main()
