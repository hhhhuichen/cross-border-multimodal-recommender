# -*- coding: utf-8 -*-
"""带数据语义校验的 checkpoint 保存与加载。"""
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
import tempfile
import warnings

import numpy as np
import torch


CHECKPOINT_FORMAT_VERSION = 2

_FINGERPRINT_ARRAYS = (
    "user_country", "item_country", "country_adj", "item_market_mask",
    "train_pairs", "val_pairs", "test_pairs",
    "cold_val_items", "cold_test_items", "transfer_cold_items", "triples",
    "edge_index", "edge_type",
    "item_text_feat", "item_text_lang",
    "item_text_source", "item_text_role", "item_text_valid",
    "item_text_is_fallback", "item_text_content_hash",
    "item_text_dedup_mask", "item_text_language_confidence",
    "item_text_pair_valid", "item_text_market",
    "item_image_feat", "item_image_mask",
    "item_image_available", "item_image_observed",
    "item_train_degree", "zero_train_items",
    "mm_item_edge_index", "mm_item_edge_weight",
)

_FINGERPRINT_SCALARS = (
    "n_users", "n_items", "n_entities", "n_nodes",
    "n_relations", "n_relations_total", "n_countries", "n_languages",
    "schema_version", "split_seed",
)


def _config_payload(cfg):
    def convert(value):
        return asdict(value) if is_dataclass(value) else value

    return {"data": convert(cfg.data), "model": convert(cfg.model)}


def _update_array(digest, name, value):
    digest.update(name.encode("utf-8"))
    if value is None:
        digest.update(b"<none>")
        return

    array = np.asarray(value)
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    if array.dtype.hasobject:
        payload = json.dumps(
            array.tolist(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), default=str,
        ).encode("utf-8")
        digest.update(payload)
    else:
        contiguous = np.ascontiguousarray(array)
        if contiguous.nbytes:
            digest.update(memoryview(contiguous).cast("B"))
        else:
            digest.update(b"<empty>")


def semantic_fingerprint(dataset, cfg):
    """哈希数据映射、划分、特征和会改变前向语义的配置。"""
    digest = hashlib.sha256()
    config_json = json.dumps(
        _config_payload(cfg), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), default=str,
    )
    digest.update(config_json.encode("utf-8"))

    for name in _FINGERPRINT_SCALARS:
        digest.update(f"{name}={getattr(dataset, name, None)!r}".encode("utf-8"))
    for name in _FINGERPRINT_ARRAYS:
        _update_array(digest, name, getattr(dataset, name, None))
    return digest.hexdigest()


def save_checkpoint(path, model, dataset, cfg, extra=None):
    """原子保存模型，并写入可复算的数据/配置语义指纹。"""
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state": model.state_dict(),
        "metadata": {
            "semantic_fingerprint": semantic_fingerprint(dataset, cfg),
            "config": _config_payload(cfg),
            "extra": extra or {},
        },
    }

    handle = tempfile.NamedTemporaryFile(
        prefix=".acmr-checkpoint-", suffix=".tmp", dir=directory, delete=False
    )
    tmp_path = handle.name
    handle.close()
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _load_model_state(model, state_dict, strict):
    upgrade = getattr(model, "upgrade_state_dict", None)
    if upgrade is not None:
        upgraded = upgrade(state_dict)
        if upgraded is not state_dict:
            warnings.warn(
                "checkpoint 使用旧的含 interaction 关系参数布局；"
                "已只保留正反 KG 关系行",
                RuntimeWarning,
                stacklevel=2,
            )
        state_dict = upgraded
    model.load_state_dict(state_dict, strict=strict)
    # 关系注意力是由当前参数计算出的 epoch 级缓存，不属于 state_dict。对已经
    # 前向过的模型加载新权重后必须失效，否则下一次 propagate 会复用旧权重。
    if hasattr(model, "_att_cache"):
        model._att_cache = None


def load_checkpoint(path, model, dataset, cfg, map_location=None,
                    strict=True, allow_legacy=False):
    """加载 checkpoint；新版在加载权重前拒绝数据或配置语义错位。"""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if (isinstance(payload, dict)
            and payload.get("format_version") == CHECKPOINT_FORMAT_VERSION
            and "model_state" in payload):
        metadata = payload.get("metadata", {})
        expected = metadata.get("semantic_fingerprint")
        actual = semantic_fingerprint(dataset, cfg)
        if not expected or expected != actual:
            raise ValueError(
                "checkpoint 与当前数据映射/划分或模型配置不一致："
                f"saved={expected or '<missing>'}, current={actual}"
            )
        _load_model_state(model, payload["model_state"], strict)
        return metadata

    if isinstance(payload, dict) and "format_version" in payload:
        raise ValueError(
            f"不支持的 checkpoint 格式版本：{payload.get('format_version')!r}"
        )

    if not allow_legacy:
        raise ValueError(
            "旧版 checkpoint 不含语义指纹，已拒绝加载；"
            "确认数据映射后可显式传 allow_legacy=True"
        )
    warnings.warn(
        "正在加载旧版 bare state_dict；只能检查参数形状，无法验证数据语义。",
        RuntimeWarning,
        stacklevel=2,
    )
    _load_model_state(model, payload, strict)
    return None
