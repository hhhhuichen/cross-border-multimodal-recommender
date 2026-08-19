#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""缓存 XMarket 官方元数据中的一个商品图像 URL，不修改原始数据。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen


def fetch(row, image_dir, timeout):
    index, url = int(row["idx"]), str(row.get("image_url") or "")
    destination = image_dir / f"{index}.jpg"
    if destination.is_file() and destination.stat().st_size:
        return index, "cached"
    if not url:
        return index, "missing_url"
    request = Request(url, headers={"User-Agent": "ACMR-research/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response, tempfile.NamedTemporaryFile(
            prefix=".image-", suffix=".part", dir=image_dir, delete=False
        ) as handle:
            tmp = handle.name
            handle.write(response.read())
        if not os.path.getsize(tmp):
            os.remove(tmp)
            return index, "empty"
        os.replace(tmp, destination)
        return index, "downloaded"
    except Exception as exc:  # 下载失败保留状态，特征层按真实缺图处理。
        if "tmp" in locals() and os.path.exists(tmp):
            os.remove(tmp)
        return index, f"error:{type(exc).__name__}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data_xmarket/processed")
    parser.add_argument("--image-dir", default="data_xmarket/images")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    processed, image_dir = Path(args.processed_dir), Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in
            (processed / "items.jsonl").open(encoding="utf-8")]
    status = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, row, image_dir, args.timeout) for row in rows]
        for done, future in enumerate(as_completed(futures), 1):
            index, value = future.result()
            status[str(index)] = value
            if done % 500 == 0:
                print(f"{done}/{len(rows)}", flush=True)
    (image_dir / "fetch_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

