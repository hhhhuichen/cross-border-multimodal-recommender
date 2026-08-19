# -*- coding: utf-8 -*-
"""
第 6a 步：LaBSE 抽取多语种文本特征 -> item_text_feat.npy (n_items, V, 768)。

冻结、离线、缓存——与 ACMR 的设计契约一致（模型侧只吃 .npy，不做端到端微调）。
断点续跑：进度存 text_feat_done.txt，重跑从断点继续。

    python extract_text_feat.py [--batch 64]
"""
import argparse
import json

import numpy as np
import torch
from off_common import PROC_DIR

MODEL = "sentence-transformers/LaBSE"
DIM = 768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer   # 延迟导入，给出清晰报错
    meta = json.loads((PROC_DIR / "meta.json").read_text(encoding="utf-8"))
    n_items, V = meta["n_items"], meta["n_views"]

    texts = []
    for line in (PROC_DIR / "items.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        texts += [v["text"] or " " for v in r["views"]]
    assert len(texts) == n_items * V

    feat_path = PROC_DIR / "item_text_feat.npy"
    done_path = PROC_DIR / "text_feat_done.txt"
    if feat_path.exists() and done_path.exists():
        feat = np.load(feat_path)
        done = int(done_path.read_text())
    else:
        feat = np.zeros((n_items * V, DIM), dtype=np.float32)
        done = 0
    if done >= len(texts):
        print("已全部完成")
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[设备] {dev} | 待处理 {len(texts) - done}/{len(texts)} 条文本")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval()

    with torch.no_grad():
        for s in range(done, len(texts), args.batch):
            chunk = texts[s:s + args.batch]
            enc = tok(chunk, padding=True, truncation=True, max_length=128,
                      return_tensors="pt").to(dev)
            out = model(**enc)
            emb = torch.nn.functional.normalize(out.pooler_output, dim=-1)
            feat[s:s + len(chunk)] = emb.cpu().numpy()
            if (s // args.batch) % 20 == 0:
                np.save(feat_path, feat)
                done_path.write_text(str(s + len(chunk)))
                print(f"  {s + len(chunk)}/{len(texts)}", flush=True)
    np.save(feat_path, feat.reshape(n_items, V, DIM))
    done_path.write_text(str(len(texts)))
    print(f"完成 -> {feat_path}  形状 {(n_items, V, DIM)}")


if __name__ == "__main__":
    main()
