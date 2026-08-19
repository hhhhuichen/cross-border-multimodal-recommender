# -*- coding: utf-8 -*-
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from config import Config
from data_contract import (
    IMAGE_AVAILABLE_FILE,
    SCHEMA_VERSION,
    TEXT_META_FILE,
    dedup_mask,
    normalize_text,
    schema_descriptor,
    text_content_hash,
)
from data_utils import BPRSampler, CKGDataset, KGSampler, build_dataset
from off_data import OFFRealData
from train import make_batch


def tiny_config():
    cfg = Config()
    cfg.data.n_users = 30
    cfg.data.n_items = 40
    cfg.data.n_entities = 80
    cfg.data.n_interactions = 300
    cfg.data.n_triples = 180
    cfg.data.text_dim = 16
    cfg.data.image_dim = 12
    cfg.data.split_seed = 11
    cfg.train.train_seed = 22
    cfg.train.batch_size = 32
    cfg.train.kg_batch_size = 32
    cfg.model.use_mm_item_graph = False
    return cfg


def test_normalization_hash_and_dedup_contract():
    assert normalize_text("  ＡBC\tStraße  ") == "abc strasse"
    assert text_content_hash("ＡBC  ") == text_content_hash("abc")
    hashes = np.array([
        [text_content_hash("same"), text_content_hash(" SAME ")],
        [text_content_hash(""), text_content_hash("different")],
    ])
    valid = np.array([[True, True], [False, True]])
    assert dedup_mask(hashes, valid).tolist() == [[True, False], [False, True]]


def test_synthetic_schema_seed_and_sampler_contracts():
    cfg = tiny_config()
    dataset = build_dataset(cfg)
    assert dataset.schema_version == SCHEMA_VERSION
    assert dataset.split_seed == 11
    assert dataset.train_seed == 22
    assert dataset.item_text_pair_valid.shape == (
        dataset.n_items, cfg.data.n_text_views, cfg.data.n_text_views
    )
    assert dataset.item_text_pair_valid[:, 0, 1].all()
    assert np.array_equal(
        dataset.zero_train_items, np.flatnonzero(dataset.item_train_degree == 0)
    )

    batch = make_batch(dataset, torch.device("cpu"))
    assert batch["item_text_dedup_mask"].dtype == torch.bool
    assert batch["item_text_pair_valid"].dtype == torch.bool
    assert batch["item_text_content_hash"].dtype == torch.long
    assert dataset.item_text_content_hash.dtype.kind in {"U", "S"}

    relations = dataset.ckg_triples[:, 1]
    assert not np.isin(relations, [0, dataset.n_relations + 1]).any()
    assert dataset.ckg_triples[:, [0, 2]].max() < dataset.n_entities

    users, _, negatives = next(iter(BPRSampler(dataset, 32, seed=3)))
    assert all(
        int(negative) not in dataset.train_user_pos[int(user)]
        for user, negative in zip(users, negatives)
    )
    h, r, _, negative_tails = next(iter(KGSampler(dataset, 32, seed=3)))
    truths = {tuple(map(int, row)) for row in dataset.ckg_triples}
    assert all(
        (int(head), int(relation), int(tail)) not in truths
        for head, relation, tail in zip(h, r, negative_tails)
    )

    same_split = tiny_config()
    same_split.train.train_seed = 999
    dataset_again = build_dataset(same_split)
    assert np.array_equal(dataset.train_pairs, dataset_again.train_pairs)


def _write_valid_off_v2(root):
    n_items, n_views = 3, 2
    meta = {
        "schema_version": SCHEMA_VERSION,
        "data_schema": schema_descriptor(),
        "n_items": n_items,
        "n_entities": 4,
        "n_relations": 5,
        "countries": ["M0"],
        "languages": ["en", "zh"],
        "n_markets": 1,
        "n_views": n_views,
        "entity_layout": {"countries": [3, 4]},
    }
    (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    np.save(root / "triples.npy", np.array([
        [0, 5, 3], [1, 5, 3], [2, 5, 3],
    ], dtype=np.int64))
    np.save(root / "item_country.npy", np.zeros(n_items, dtype=np.int64))

    language = np.array([[0, 1], [0, 0], [1, 0]], dtype=np.int64)
    source = np.array([
        ["title_en", "title_zh"],
        ["title_en", "title_en"],
        ["title_zh", "title_en"],
    ])
    role = np.full((n_items, n_views), "product_name")
    valid = np.ones((n_items, n_views), dtype=bool)
    fallback = np.array([[False, False], [False, True], [False, False]])
    content_hash = np.array([
        [text_content_hash("apple"), text_content_hash("苹果")],
        [text_content_hash("banana"), text_content_hash("banana")],
        [text_content_hash("茶"), text_content_hash("tea")],
    ])
    unique = dedup_mask(content_hash, valid)
    confidence = np.ones((n_items, n_views), dtype=np.float32)
    np.save(root / "item_text_lang.npy", language)
    np.savez_compressed(
        root / TEXT_META_FILE,
        language=language,
        source=source,
        role=role,
        valid=valid,
        is_fallback=fallback,
        content_hash=content_hash,
        dedup_mask=unique,
        language_confidence=confidence,
    )
    np.save(root / IMAGE_AVAILABLE_FILE, np.array([True, False, True]))
    np.save(root / "item_text_feat.npy", np.zeros((n_items, n_views, 2), np.float32))
    np.save(root / "item_image_feat.npy", np.zeros((n_items, 2), np.float32))
    np.save(root / "item_image_mask.npy", np.array([1, 0, 1], np.float32))
    np.savez(
        root / "users.npz",
        user_country=np.zeros(1, dtype=np.int64),
        interactions=np.array([[0, 0]], dtype=np.int64),
    )


def test_off_loader_rejects_legacy_and_loads_v2():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        legacy = tmp_path / "legacy"
        legacy.mkdir()
        (legacy / "meta.json").write_text("{}", encoding="utf-8")
        cfg = tiny_config()
        cfg.data.off_dir = str(legacy)
        try:
            OFFRealData(cfg)
        except ValueError as exc:
            assert "schema v2" in str(exc)
        else:
            raise AssertionError("OFF 旧 schema 未被拒绝")

        current = tmp_path / "current"
        current.mkdir()
        _write_valid_off_v2(current)
        cfg = tiny_config()
        cfg.data.off_dir = str(current)
        cfg.data.text_dim = 2
        cfg.data.image_dim = 2
        raw = OFFRealData(cfg)
        assert raw.item_text_pair_valid[0, 0, 1]
        assert not raw.item_text_pair_valid[1].any()
        assert raw.item_image_observed.tolist() == [True, False, True]

        cfg.data.val_ratio = 0.0
        cfg.data.test_ratio = 0.0
        cfg.data.cold_item_ratio = 0.0
        dataset = CKGDataset(raw, cfg)
        assert dataset.zero_train_items.tolist() == [1, 2]
        assert not np.isin(
            dataset.ckg_triples[:, 1], [0, dataset.n_relations + 1]
        ).any()


if __name__ == "__main__":
    test_normalization_hash_and_dedup_contract()
    test_synthetic_schema_seed_and_sampler_contracts()
    test_off_loader_rejects_legacy_and_loads_v2()
    print("PASS: schema-v2 data contract")
