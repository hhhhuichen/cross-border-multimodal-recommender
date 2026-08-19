# -*- coding: utf-8 -*-
"""可复现实验清单与原子结果写出。"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_snapshot_hash(root=PROJECT_DIR):
    """Git 不可用时，对可执行 Python 源码作稳定内容哈希。"""
    root = Path(root)
    digest = hashlib.sha256()
    files = sorted(
        p for p in root.rglob("*.py")
        if "__pycache__" not in p.parts and "data_" not in p.parts
    )
    for path in files:
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(rel).to_bytes(4, "big"))
        digest.update(rel)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def dataset_source_hashes(dataset):
    recorded = {
        str(name): str(value)
        for name, value in getattr(dataset, "raw_source_hashes", {}).items()
    }
    paths = [Path(value) for value in getattr(dataset, "source_files", ())]
    if recorded:
        out = dict(recorded)
        for (name, expected), path in zip(recorded.items(), paths):
            if path.is_file():
                actual = sha256_file(path)
                if actual != expected:
                    raise ValueError(
                        f"原始数据哈希不一致：{name}"
                    )
        return out
    out = {}
    for path in paths:
        if path.is_file():
            out[str(path)] = sha256_file(path)
    return out


def dataset_artifact_hashes(dataset):
    out = {}
    for value in getattr(dataset, "artifact_files", ()):
        path = Path(value)
        if path.is_file():
            out[str(path)] = sha256_file(path)
    return out


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def build_run_manifest(cfg, dataset, model, *, validation=None, test=None,
                       duration_seconds=None, status="completed"):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    raw_source_hashes = dataset_source_hashes(dataset)
    artifact_hashes = dataset_artifact_hashes(dataset)
    manifest = {
        "manifest_version": 1,
        "status": status,
        "data_schema_version": int(getattr(dataset, "schema_version", 2)),
        "data_schema": getattr(dataset, "data_schema", None),
        "model": cfg.train.model,
        "residual": cfg.model.residual,
        "split_seed": int(getattr(dataset, "split_seed", cfg.data.seed)),
        "train_seed": int(cfg.train.train_seed),
        "parameter_count": {"trainable": int(trainable), "total": int(total)},
        "config": {
            "data": asdict(cfg.data),
            "model": asdict(cfg.model),
            "train": asdict(cfg.train),
        },
        "dataset": {
            "n_users": int(dataset.n_users),
            "n_items": int(dataset.n_items),
            "n_entities": int(dataset.n_entities),
            "n_train": int(len(dataset.train_pairs)),
            "n_validation": int(len(dataset.val_pairs)),
            "n_test": int(len(dataset.test_pairs)),
            "raw_source_hashes": raw_source_hashes,
            # 保留旧键，使已有汇总脚本继续可读。
            "source_hashes": raw_source_hashes,
            "artifact_hashes": artifact_hashes,
            "bpr_sampling": getattr(dataset, "bpr_sampling_stats", None),
        },
        "source_snapshot_sha256": source_snapshot_hash(),
        "duration_seconds": duration_seconds,
        "validation": validation,
        "test": test,
    }
    return _jsonable(manifest)


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=".result-", suffix=".tmp",
        dir=path.parent, delete=False,
    ) as handle:
        json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2,
                  sort_keys=True)
        handle.write("\n")
        tmp = handle.name
    try:
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
