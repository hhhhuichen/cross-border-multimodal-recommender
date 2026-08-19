# -*- coding: utf-8 -*-
"""XMarket Electronics schema-v2 加载器。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_contract import (
    IMAGE_AVAILABLE_FILE,
    SCHEMA_VERSION,
    TEXT_META_FILE,
    alignment_pair_mask,
    require_schema_v2,
    runtime_image_metadata,
    validate_text_metadata,
)


PROJECT_DIR = Path(__file__).resolve().parent


class XMarketRealData:
    def __init__(self, cfg):
        directory = Path(cfg.data.xmarket_dir).expanduser()
        if not directory.is_absolute():
            directory = (PROJECT_DIR / directory).resolve()
        cfg.data.xmarket_dir = str(directory)
        required = (
            "meta.json", "triples.npy", "item_country.npy",
            "item_text_lang.npy", "item_text_market.npy", TEXT_META_FILE,
            IMAGE_AVAILABLE_FILE, "item_market_mask.npy", "splits.npz",
            "item_text_feat.npy", "item_image_feat.npy", "item_image_mask.npy",
        )
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "XMarket processed 缺文件：" + ", ".join(missing)
                + "；请按 xmarket_pipeline/README.md 显式构建"
            )

        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        self.data_schema = require_schema_v2(meta)
        self.schema_version = SCHEMA_VERSION
        if meta.get("dataset") != "XMarket" or meta.get("category") != "Electronics":
            raise ValueError("当前加载器只接受 XMarket Electronics")
        if meta.get("supports_item_origin") is not False:
            raise ValueError("XMarket 必须显式声明 supports_item_origin=false")
        self.market_names = list(meta.get("markets", ()))
        if self.market_names != ["us", "cn", "in", "sg"]:
            raise ValueError("XMarket markets 必须显式声明为 us/cn/in/sg")
        self.catalog_protocol = meta.get("catalog_protocol")
        if not isinstance(self.catalog_protocol, dict):
            raise ValueError("XMarket 缺 catalog_protocol；请显式重建旧处理数据")
        if (int(self.catalog_protocol.get("version", -1)) != 1
                or self.catalog_protocol.get("candidate_source")
                != "market_metadata_before_interaction_filter"
                or self.catalog_protocol.get("interaction_filter")
                != "iterative_bipartite_5_core"):
            raise ValueError("XMarket catalog_protocol 非法；请按当前协议重建")
        built_seed = int(meta["split_seed"])
        configured = getattr(cfg.data, "split_seed", None)
        if configured is not None and int(configured) != built_seed:
            raise ValueError(
                f"XMarket split_seed={built_seed}，配置为 {configured}；请重建或改配置"
            )
        cfg.data.split_seed = built_seed
        self.supports_item_origin = False

        self.n_items = int(meta["n_items"])
        self.n_entities = int(meta["n_entities"])
        self.n_relations = int(meta["n_relations"])
        self.countries = list(meta["countries"])
        self.languages = list(meta["languages"])
        self.n_countries = len(self.countries)
        self.n_languages = len(self.languages)
        self.n_markets = int(meta["n_markets"])

        self.triples = np.load(directory / "triples.npy")
        self.item_country = np.load(directory / "item_country.npy")
        self.item_text_lang = np.load(directory / "item_text_lang.npy")
        self.item_text_market = np.load(directory / "item_text_market.npy")
        self.item_market_mask = np.load(directory / "item_market_mask.npy")
        self.item_text_feat = np.load(directory / "item_text_feat.npy")
        self.item_image_feat = np.load(directory / "item_image_feat.npy")
        self.item_image_mask = np.load(directory / "item_image_mask.npy")
        available = np.load(directory / IMAGE_AVAILABLE_FILE)
        with np.load(directory / TEXT_META_FILE, allow_pickle=False) as payload:
            text_meta = validate_text_metadata(
                payload, self.n_items, int(meta["n_views"]), self.n_languages
            )
        if not np.array_equal(text_meta["language"], self.item_text_lang):
            raise ValueError("XMarket language 数组不一致")
        self.item_text_source = text_meta["source"]
        self.item_text_role = text_meta["role"]
        self.item_text_valid = text_meta["valid"]
        self.item_text_is_fallback = text_meta["is_fallback"]
        self.item_text_content_hash = text_meta["content_hash"]
        self.item_text_dedup_mask = text_meta["dedup_mask"]
        self.item_text_language_confidence = text_meta["language_confidence"]
        self.item_text_genuine, self.item_text_pair_valid = alignment_pair_mask(
            self.item_text_lang, self.item_text_role, self.item_text_valid,
            self.item_text_is_fallback, self.item_text_content_hash,
            self.item_text_dedup_mask, self.item_text_language_confidence,
        )
        image_meta = runtime_image_metadata(
            available, self.item_image_mask > 0.5
        )
        self.item_image_available = image_meta["available"]
        self.item_image_observed = image_meta["observed"]
        self.item_image_completion_confidence = image_meta["completion_confidence"]

        with np.load(directory / "splits.npz") as splits:
            self.user_country = splits["user_country"]
            self.predefined_splits = {
                "train": splits["train_pairs"],
                "val": splits["val_pairs"],
                "test": splits["test_pairs"],
                "cold_val": splits["cold_val_pairs"],
                "cold_test": splits["cold_test_pairs"],
            }
        self.n_users = int(len(self.user_country))
        self.interactions = np.concatenate(
            list(self.predefined_splits.values()), axis=0
        ).astype(np.int64, copy=False)
        self.country_adj = None
        self.raw_source_hashes = dict(meta.get("raw_source_hashes", {}))
        self.source_files = list(self.raw_source_hashes)
        self.artifact_files = [directory / name for name in required]

        if self.item_market_mask.shape != (self.n_markets, self.n_items):
            raise ValueError("XMarket item_market_mask 形状非法")
        catalog_counts = self.catalog_protocol.get("market_item_counts", {})
        expected_catalog_counts = np.asarray([
            int(catalog_counts.get(market, -1)) for market in self.market_names
        ])
        if not np.array_equal(
                self.item_market_mask.sum(axis=1), expected_catalog_counts):
            raise ValueError("XMarket item_market_mask 与 catalog_protocol 不一致")
        if int(self.catalog_protocol.get("union_item_count", -1)) != self.n_items:
            raise ValueError("XMarket n_items 与 catalog_protocol 不一致")
        if self.item_text_market.shape != self.item_text_lang.shape:
            raise ValueError("XMarket item_text_market 形状非法")
        if self.item_text_feat.shape != (
            self.n_items, int(meta["n_views"]), cfg.data.text_dim
        ):
            raise ValueError("XMarket item_text_feat 形状与配置不一致")
        if self.item_image_feat.shape != (self.n_items, cfg.data.image_dim):
            raise ValueError("XMarket item_image_feat 形状与配置不一致")
        if self.item_image_mask.shape != (self.n_items,):
            raise ValueError("XMarket item_image_mask 形状非法")
        if not ((0 <= self.user_country) &
                (self.user_country < self.n_markets)).all():
            raise ValueError("XMarket user_country 越界")
        if not ((0 <= self.item_country) &
                (self.item_country < self.n_countries)).all():
            raise ValueError("XMarket item_country 越界")
        for name, pairs in self.predefined_splits.items():
            if pairs.ndim != 2 or pairs.shape[1] != 2:
                raise ValueError(f"XMarket {name} split 必须为 (N,2)")
            if len(pairs) and not (
                ((0 <= pairs[:, 0]) & (pairs[:, 0] < self.n_users)).all()
                and ((0 <= pairs[:, 1]) & (pairs[:, 1] < self.n_items)).all()
            ):
                raise ValueError(f"XMarket {name} split 编号越界")

        cfg.data.n_users = self.n_users
        cfg.data.n_items = self.n_items
        cfg.data.n_entities = self.n_entities
        cfg.data.n_relations = self.n_relations
        cfg.data.n_text_views = int(meta["n_views"])
        cfg.data.countries = self.countries
        cfg.data.languages = self.languages
