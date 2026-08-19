#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 XMarket 官方站点下载 Electronics 评分与元数据并记录校验和。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen


OFFICIAL_PAGE = "https://xmrec.github.io"
BASE = "https://ciir.cs.umass.edu/downloads/XMarket/FULL"
DEFAULT_MARKETS = ("us", "cn", "in", "sg")
CATEGORY = "Electronics"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        return
    request = Request(url, headers={"User-Agent": "ACMR-research/1.0"})
    with urlopen(request, timeout=120) as response, tempfile.NamedTemporaryFile(
        prefix=".xmarket-", suffix=".part", dir=destination.parent, delete=False
    ) as handle:
        tmp = handle.name
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    try:
        os.replace(tmp, destination)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data_xmarket/raw")
    parser.add_argument("--markets", nargs="+", default=list(DEFAULT_MARKETS),
                        choices=list(DEFAULT_MARKETS))
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    records = []
    for market in args.markets:
        names = (
            f"ratings_{market}_{CATEGORY}.txt.gz",
            f"metadata_{market}_{CATEGORY}.json.gz",
        )
        for name in names:
            url = f"{BASE}/{market}/{CATEGORY}/{name}"
            destination = raw_dir / name
            print(f"[下载] {url}", flush=True)
            download(url, destination)
            records.append({
                "market": market,
                "category": CATEGORY,
                "kind": "ratings" if name.startswith("ratings") else "metadata",
                "path": str(destination),
                "url": url,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            })
    manifest = {
        "manifest_version": 1,
        "official_page": OFFICIAL_PAGE,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "files": records,
        "license_note": (
            "官方分发页未提供可由本项目确认的显式 LICENSE；原始文件仅用于研究，"
            "不得随本仓库再分发。"
        ),
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"清单已写入 {raw_dir / 'source_manifest.json'}")


if __name__ == "__main__":
    main()

