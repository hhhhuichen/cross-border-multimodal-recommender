# -*- coding: utf-8 -*-
"""ACMR 多模态数据契约。

schema v2 将文本视图的来源、角色与回退状态作为一等数据，避免把复制的
fallback 文本误当成平行语料。持久化数据必须显式声明版本；运行时派生数组
则统一由本模块校验，避免 OFF 与合成数据采用不同语义。
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np


SCHEMA_NAME = "acmr_multimodal"
SCHEMA_VERSION = 2
TEXT_META_FILE = "item_text_meta.npz"
IMAGE_AVAILABLE_FILE = "item_image_available.npy"

TEXT_META_FIELDS = (
    "language",
    "source",
    "role",
    "valid",
    "is_fallback",
    "content_hash",
    "dedup_mask",
    "language_confidence",
)
IMAGE_FIELDS = (
    "available",
    "observed",
    "completion_confidence",
)


def schema_descriptor():
    """返回可直接写入 ``meta.json`` 的稳定 schema 描述。"""
    return {
        "name": SCHEMA_NAME,
        "version": SCHEMA_VERSION,
        "text_view_fields": list(TEXT_META_FIELDS),
        "image_fields": list(IMAGE_FIELDS),
        "normalization": "Unicode NFKC + casefold + whitespace collapse",
        "content_hash": "sha256(normalized UTF-8 text)",
    }


def require_schema_v2(meta):
    """拒绝旧版或含糊的持久化数据，返回规范化的 schema 描述。"""
    schema = meta.get("data_schema")
    top_version = meta.get("schema_version")
    if not isinstance(schema, dict) or schema.get("name") != SCHEMA_NAME:
        raise ValueError(
            "数据不是 ACMR schema v2；请显式重跑数据构建管线，禁止从旧字段推断"
        )
    if schema.get("version") != SCHEMA_VERSION or top_version != SCHEMA_VERSION:
        raise ValueError(
            f"仅支持 ACMR schema v{SCHEMA_VERSION}，检测到 "
            f"data_schema.version={schema.get('version')!r}, "
            f"schema_version={top_version!r}；请显式重建数据"
        )
    missing_text = set(TEXT_META_FIELDS) - set(schema.get("text_view_fields", ()))
    missing_image = set(IMAGE_FIELDS) - set(schema.get("image_fields", ()))
    if missing_text or missing_image:
        raise ValueError(
            "schema v2 字段声明不完整："
            f"缺文本字段 {sorted(missing_text)}，缺图像字段 {sorted(missing_image)}"
        )
    return schema


def normalize_text(text):
    """生成用于内容身份判定的文本；不改变模型实际编码文本。"""
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.sub(r"\s+", " ", value).strip()


def text_content_hash(text):
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def make_text_view(text, language, source, role, *, is_fallback=False,
                   language_confidence=1.0):
    """构造一个 JSON 可序列化的 schema-v2 文本视图。"""
    text = str(text or "").strip()
    confidence = float(language_confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("language_confidence 必须在 [0, 1] 内")
    return {
        "text": text,
        "language": str(language),
        # ``lang`` 保留给现有特征抽取脚本和人工检查工具。
        "lang": str(language),
        "source": str(source),
        "role": str(role),
        "valid": bool(normalize_text(text)),
        "is_fallback": bool(is_fallback),
        "content_hash": text_content_hash(text),
        "language_confidence": confidence,
    }


def dedup_mask(content_hash, valid):
    """逐商品仅保留每个有效内容哈希的第一次出现。"""
    hashes = np.asarray(content_hash)
    valid = np.asarray(valid, dtype=bool)
    if hashes.shape != valid.shape or hashes.ndim != 2:
        raise ValueError("content_hash 与 valid 必须是同形状二维数组")
    out = np.zeros(valid.shape, dtype=bool)
    for row in range(hashes.shape[0]):
        seen = set()
        for view in range(hashes.shape[1]):
            value = str(hashes[row, view])
            if valid[row, view] and value not in seen:
                out[row, view] = True
                seen.add(value)
    return out


def validate_text_metadata(metadata, n_items, n_views, n_languages):
    """校验从 NPZ 或运行时对象取得的文本元数据，并返回普通字典。"""
    missing = set(TEXT_META_FIELDS) - set(metadata)
    if missing:
        raise ValueError(f"文本视图元数据缺字段 {sorted(missing)}")
    arrays = {name: np.asarray(metadata[name]) for name in TEXT_META_FIELDS}
    shape = (int(n_items), int(n_views))
    for name, array in arrays.items():
        if array.shape != shape:
            raise ValueError(f"文本元数据 {name} 形状 {array.shape} != {shape}")

    language = arrays["language"]
    if not np.issubdtype(language.dtype, np.integer):
        raise ValueError("文本元数据 language 必须是整数 dtype")
    if not ((0 <= language) & (language < int(n_languages))).all():
        raise ValueError("文本元数据 language 编号越界")

    for name in ("valid", "is_fallback", "dedup_mask"):
        arrays[name] = arrays[name].astype(bool, copy=False)
    confidence = arrays["language_confidence"]
    if (not np.issubdtype(confidence.dtype, np.number)
            or not np.isfinite(confidence).all()
            or not ((0.0 <= confidence) & (confidence <= 1.0)).all()):
        raise ValueError("language_confidence 必须是 [0,1] 内有限值")
    arrays["language_confidence"] = confidence.astype(np.float32, copy=False)

    hashes = arrays["content_hash"].astype(str)
    hash_ok = np.vectorize(
        lambda value: bool(re.fullmatch(r"[0-9a-f]{64}", value)),
        otypes=[bool],
    )(hashes)
    if not hash_ok.all():
        raise ValueError("content_hash 必须是小写 SHA-256 十六进制字符串")
    expected_dedup = dedup_mask(hashes, arrays["valid"])
    if not np.array_equal(arrays["dedup_mask"], expected_dedup):
        raise ValueError("dedup_mask 与 valid/content_hash 不一致")
    if np.any(arrays["dedup_mask"] & ~arrays["valid"]):
        raise ValueError("无效文本视图不能进入 dedup_mask")

    arrays["content_hash"] = hashes
    arrays["source"] = arrays["source"].astype(str)
    arrays["role"] = arrays["role"].astype(str)
    if np.any(np.char.str_len(arrays["source"]) == 0):
        raise ValueError("文本视图 source 不能为空")
    if np.any(np.char.str_len(arrays["role"]) == 0):
        raise ValueError("文本视图 role 不能为空")
    return arrays


def alignment_pair_mask(language, role, valid, is_fallback, content_hash,
                        deduplicated, language_confidence, min_confidence=0.8):
    """返回 ``(item, source_view, target_view)`` 的可信平行文本对掩码。"""
    language = np.asarray(language)
    role = np.asarray(role).astype(str)
    valid = np.asarray(valid, dtype=bool)
    is_fallback = np.asarray(is_fallback, dtype=bool)
    content_hash = np.asarray(content_hash).astype(str)
    deduplicated = np.asarray(deduplicated, dtype=bool)
    confidence = np.asarray(language_confidence, dtype=np.float32)
    shape = language.shape
    for name, array in (
        ("role", role),
        ("valid", valid),
        ("is_fallback", is_fallback),
        ("content_hash", content_hash),
        ("deduplicated", deduplicated),
        ("language_confidence", confidence),
    ):
        if array.shape != shape:
            raise ValueError(f"{name} 与 language 形状不一致")
    genuine = (
        valid
        & ~is_fallback
        & deduplicated
        & (confidence >= float(min_confidence))
        & (role == "product_name")
    )
    pair = genuine[:, :, None] & genuine[:, None, :]
    pair &= language[:, :, None] != language[:, None, :]
    pair &= content_hash[:, :, None] != content_hash[:, None, :]
    return genuine, pair


def runtime_image_metadata(available, observed):
    """从目录可用性与实际特征掩码生成无泄漏的运行时图像元数据。"""
    available = np.asarray(available, dtype=bool)
    observed = np.asarray(observed, dtype=bool)
    if available.shape != observed.shape or available.ndim != 1:
        raise ValueError("图像 available/observed 必须是同形状一维数组")
    # 已观测原图是完全可靠的输入；缺图的置信度只能由训练期置信度头预测，
    # 数据层初始值必须为 0，不能从验证/测试重建质量反推。
    return {
        "available": available,
        "observed": observed,
        "completion_confidence": observed.astype(np.float32),
    }
