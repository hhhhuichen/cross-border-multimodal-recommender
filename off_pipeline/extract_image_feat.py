# -*- coding: utf-8 -*-
"""
第 6b 步：CLIP ViT-B/32 抽取图像特征 -> item_image_feat.npy (n_items, 512)
          + item_image_mask.npy (n_items,)。

缺图 / 下载失败 / 解码失败的商品：特征置零、mask=0——这正是模型
`use_modality_completion` 模块要接住的真实缺失，不要在这里做任何填补。

    python extract_image_feat.py [--batch 64]
"""
import argparse
import json

import numpy as np
import torch
from off_common import IMG_DIR, PROC_DIR

MODEL = "openai/clip-vit-base-patch32"
DIM = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    meta = json.loads((PROC_DIR / "meta.json").read_text(encoding="utf-8"))
    n_items = meta["n_items"]

    feat_path = PROC_DIR / "item_image_feat.npy"
    mask_path = PROC_DIR / "item_image_mask.npy"
    done_path = PROC_DIR / "image_feat_done.txt"
    if feat_path.exists() and done_path.exists():
        feat, mask = np.load(feat_path), np.load(mask_path)
        done = int(done_path.read_text())
    else:
        feat = np.zeros((n_items, DIM), dtype=np.float32)
        mask = np.zeros(n_items, dtype=np.float32)
        done = 0
    if done >= n_items:
        print("已全部完成")
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[设备] {dev} | 从第 {done} 个商品继续（共 {n_items}）")
    proc = CLIPImageProcessor.from_pretrained(MODEL)
    model = CLIPVisionModelWithProjection.from_pretrained(MODEL).to(dev).eval()

    def load(idx):
        p = IMG_DIR / f"{idx}.jpg"
        if not p.exists() or p.stat().st_size == 0:
            return None
        try:
            return Image.open(p).convert("RGB")
        except Exception:
            return None

    with torch.no_grad():
        for s in range(done, n_items, args.batch):
            ids = list(range(s, min(s + args.batch, n_items)))
            imgs = [(i, load(i)) for i in ids]
            valid = [(i, im) for i, im in imgs if im is not None]
            if valid:
                px = proc(images=[im for _, im in valid],
                          return_tensors="pt").to(dev)
                emb = model(**px).image_embeds
                emb = torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()
                for (i, _), e in zip(valid, emb):
                    feat[i], mask[i] = e, 1.0
            if (s // args.batch) % 20 == 0:
                np.save(feat_path, feat)
                np.save(mask_path, mask)
                done_path.write_text(str(ids[-1] + 1))
                print(f"  {ids[-1] + 1}/{n_items} | 有图率 "
                      f"{mask[:ids[-1] + 1].mean():.3f}", flush=True)
    np.save(feat_path, feat)
    np.save(mask_path, mask)
    done_path.write_text(str(n_items))
    print(f"完成 -> {feat_path} | 有图率 {mask.mean():.3f}")


if __name__ == "__main__":
    main()
