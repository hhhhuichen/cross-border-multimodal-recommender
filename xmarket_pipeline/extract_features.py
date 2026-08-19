#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用冻结 LaBSE/CLIP 为 XMarket schema-v2 商品抽取离线特征。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


TEXT_MODEL = "sentence-transformers/LaBSE"
IMAGE_MODEL = "openai/clip-vit-base-patch32"


def extract_text(processed, rows, batch_size, device):
    from transformers import AutoModel, AutoTokenizer
    texts = [view["text"] or " " for row in rows for view in row["views"]]
    n_items, n_views = len(rows), len(rows[0]["views"])
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    model = AutoModel.from_pretrained(TEXT_MODEL).to(device).eval()
    output = np.zeros((len(texts), 768), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            encoded = tokenizer(
                chunk, padding=True, truncation=True, max_length=128,
                return_tensors="pt",
            ).to(device)
            embedding = F.normalize(model(**encoded).pooler_output, dim=-1)
            output[start:start + len(chunk)] = embedding.cpu().numpy()
            if start % (20 * batch_size) == 0:
                print(f"text {start + len(chunk)}/{len(texts)}", flush=True)
    np.save(processed / "item_text_feat.npy",
            output.reshape(n_items, n_views, 768))


def extract_image(processed, image_dir, rows, batch_size, device):
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    processor = CLIPImageProcessor.from_pretrained(IMAGE_MODEL)
    model = CLIPVisionModelWithProjection.from_pretrained(IMAGE_MODEL).to(device).eval()
    feature = np.zeros((len(rows), 512), dtype=np.float32)
    mask = np.zeros(len(rows), dtype=np.float32)

    def read(index):
        path = image_dir / f"{index}.jpg"
        if not path.is_file():
            return None
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None

    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            indices = list(range(start, min(start + batch_size, len(rows))))
            loaded = [(index, read(index)) for index in indices]
            valid = [(index, image) for index, image in loaded if image is not None]
            if valid:
                pixels = processor(
                    images=[image for _, image in valid], return_tensors="pt"
                ).to(device)
                embedding = F.normalize(model(**pixels).image_embeds, dim=-1)
                for (index, _), value in zip(valid, embedding.cpu().numpy()):
                    feature[index], mask[index] = value, 1.0
            if start % (20 * batch_size) == 0:
                print(f"image {indices[-1] + 1}/{len(rows)}", flush=True)
    np.save(processed / "item_image_feat.npy", feature)
    np.save(processed / "item_image_mask.npy", mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data_xmarket/processed")
    parser.add_argument("--image-dir", default="data_xmarket/images")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--modality", choices=["all", "text", "image"],
                        default="all")
    args = parser.parse_args()
    processed, image_dir = Path(args.processed_dir), Path(args.image_dir)
    rows = [json.loads(line) for line in
            (processed / "items.jsonl").open(encoding="utf-8")]
    if not rows:
        raise ValueError("items.jsonl 为空")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.modality in {"all", "text"}:
        extract_text(processed, rows, args.batch, device)
    if args.modality in {"all", "image"}:
        extract_image(processed, image_dir, rows, args.batch, device)
    manifest = {
        "feature_manifest_version": 1,
        "text_model": TEXT_MODEL,
        "image_model": IMAGE_MODEL,
        "normalized": True,
        "device": str(device),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (processed / "feature_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

