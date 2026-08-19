# -*- coding: utf-8 -*-
"""Compact end-to-end regressions for the XMarket schema-v2 pipeline."""
from __future__ import annotations

from datetime import date, timedelta
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from config import Config
from data_utils import CKGDataset
from evaluation import evaluate
from xmarket_data import XMarketRealData
from xmarket_pipeline.build_xmarket_dataset import chronological_splits, detector


MARKETS = ("us", "cn", "in", "sg")
TARGETS = ("cn", "in", "sg")


def test_seeded_leave_two_out_without_timestamps():
    records = [("u", f"i{index}", "-") for index in range(8)]
    first = chronological_splits(records, "cn", set(), set(), 20260801)
    repeated = chronological_splits(records, "cn", set(), set(), 20260801)
    changed = chronological_splits(records, "cn", set(), set(), 20260802)
    assert first == repeated
    assert first[1:3] != changed[1:3]
    assert len(first[0]) == 6 and len(first[1]) == len(first[2]) == 1


def test_fasttext_numpy2_prediction_fallback():
    class Binding:
        @staticmethod
        def predict(text, k, threshold, on_unicode_error):
            assert text.endswith("\n")
            return [(1.00004, "__label__zh")]

    class Model:
        f = Binding()

        @staticmethod
        def predict(text, k):
            raise ValueError("Unable to avoid copy while creating an array")

    previous = sys.modules.get("fasttext")
    sys.modules["fasttext"] = SimpleNamespace(load_model=lambda path: Model())
    try:
        with tempfile.NamedTemporaryFile() as model_file:
            detect, model_hash = detector(Path(model_file.name))
            assert detect("测试商品") == ("zh", 1.0)
            assert len(model_hash) == 64
    finally:
        if previous is None:
            sys.modules.pop("fasttext", None)
        else:
            sys.modules["fasttext"] = previous


def _write_raw_fixture(root: Path):
    """Write 5-core interactions plus sparse items in the market catalogs."""
    shared = [f"S{item:03d}" for item in range(40)]
    target_unique = {
        market: [f"{market.upper()}{item:03d}" for item in range(20)]
        for market in TARGETS
    }
    us_only = [f"U{item:03d}" for item in range(5)]
    core_catalogs = {
        market: shared + target_unique[market] for market in TARGETS
    }
    core_catalogs["us"] = (
        shared
        + [asin for market in TARGETS for asin in target_unique[market]]
        + us_only
    )
    sparse = {market: f"{market.upper()}-SPARSE" for market in MARKETS}
    catalogs = {
        market: core_catalogs[market] + [sparse[market]] for market in MARKETS
    }
    rows_by_market = {}

    for market in MARKETS:
        rows = []
        # Two degree strata (10 and 5) exercise proportional cold allocation;
        # modulo assignment keeps every user and item in 5-core.
        for item, asin in enumerate(core_catalogs[market]):
            stamp = (date(2020, 1, 1) + timedelta(days=item)).isoformat()
            degree = 10 if item < 20 else 5
            for offset in range(degree):
                user = f"P{(item + offset) % 10:02d}"
                rows.append((user, asin, "5.0", stamp))
        # Existing user, real catalog item, but item degree one: it must be
        # removed from interactions by 5-core while remaining rankable.
        rows.append(("P00", sparse[market], "5.0", "2025-01-01"))
        rows_by_market[market] = rows

        ratings_path = root / f"ratings_{market}_Electronics.txt.gz"
        with gzip.open(ratings_path, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(" ".join(row) + "\n")

        metadata_path = root / f"metadata_{market}_Electronics.json.gz"
        catalog = catalogs[market]
        with gzip.open(metadata_path, "wt", encoding="utf-8") as handle:
            for position, asin in enumerate(catalog):
                record = {
                    "asin": asin,
                    "title": f"{market.upper()} title {asin}",
                    "description": f"{market.upper()} description {asin}",
                    "features": [f"feature {market} {asin}"],
                    "categories": ["Electronics", "Synthetic fixture"],
                    "productDetails": {"Brand": f"Brand-{position % 3}"},
                    "related": {"alsoBought": [catalog[(position + 1) % len(catalog)]]},
                    "imgUrl": ({f"https://example.invalid/{asin}.jpg": [32, 32]}
                               if position % 2 == 0 else ""),
                }
                handle.write(json.dumps(record) + "\n")
    return rows_by_market, catalogs, shared, us_only, sparse


def _build_fixture(raw_dir: Path, processed_dir: Path):
    rows, catalogs, shared, us_only, sparse = _write_raw_fixture(raw_dir)
    script = Path(__file__).resolve().parent / "xmarket_pipeline" / "build_xmarket_dataset.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--raw-dir",
            str(raw_dir),
            "--output-dir",
            str(processed_dir),
            "--split-seed",
            "20260801",
            "--disable-alignment",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    # Feature extraction is deliberately represented by deterministic tiny arrays;
    # this test targets preprocessing/split/evaluation semantics, not encoders.
    items = [json.loads(line) for line in
             (processed_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    n_items = len(items)
    rng = np.random.default_rng(7)
    np.save(
        processed_dir / "item_text_feat.npy",
        rng.normal(size=(n_items, len(MARKETS), 6)).astype(np.float32),
    )
    np.save(
        processed_dir / "item_image_feat.npy",
        rng.normal(size=(n_items, 5)).astype(np.float32),
    )
    image_available = np.load(processed_dir / "item_image_available.npy")
    image_observed = image_available & np.asarray(
        [position % 3 != 0 for position in range(n_items)], dtype=bool
    )
    np.save(processed_dir / "item_image_mask.npy", image_observed.astype(np.float32))
    return rows, catalogs, shared, us_only, sparse, items


class _CatalogScoreModel(torch.nn.Module):
    """Minimal scorer exposing the production full-ranking interface."""

    def __init__(self, n_users, n_items):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items

    def get_embeddings(self, batch):
        del batch
        return torch.zeros(self.n_users, 1), torch.arange(
            self.n_items, dtype=torch.float32
        ).unsqueeze(-1)

    def full_score(self, batch, user_e, item_e, users):
        del batch, user_e
        return item_e[:, 0].unsqueeze(0).expand(len(users), -1)


def test_xmarket_predefined_splits_and_complete_market_evaluation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw_dir = root / "raw"
        processed_dir = root / "processed"
        raw_dir.mkdir()
        rows, catalogs, shared, us_only, sparse, item_records = _build_fixture(
            raw_dir, processed_dir
        )

        cfg = Config()
        cfg.data.source = "xmarket"
        cfg.data.xmarket_dir = str(processed_dir)
        cfg.data.text_dim = 6
        cfg.data.image_dim = 5
        cfg.data.split_seed = 20260801
        cfg.model.use_mm_item_graph = False
        raw = XMarketRealData(cfg)
        dataset = CKGDataset(raw, cfg)

        item_id = {record["asin"]: int(record["idx"]) for record in item_records}
        asin_by_id = {value: key for key, value in item_id.items()}
        user_keys = sorted((market, f"P{user:02d}")
                           for market in MARKETS for user in range(10))
        user_id = {key: index for index, key in enumerate(user_keys)}
        cold_val_pairs = raw.predefined_splits["cold_val"]
        cold_test_pairs = raw.predefined_splits["cold_test"]
        cold_val_by_market = {}
        cold_test_by_market = {}
        for market in TARGETS:
            market_id = MARKETS.index(market)
            val_mask = dataset.user_country[cold_val_pairs[:, 0]] == market_id
            test_mask = dataset.user_country[cold_test_pairs[:, 0]] == market_id
            cold_val_by_market[market] = set(map(
                int, cold_val_pairs[val_mask, 1]
            ))
            cold_test_by_market[market] = set(map(
                int, cold_test_pairs[test_mask, 1]
            ))
        transfer_cold = set().union(
            *(cold_val_by_market[m] | cold_test_by_market[m] for m in TARGETS)
        )

        assert raw.supports_item_origin is False
        assert dataset.supports_item_origin is False
        assert dataset.val_user_pos_cross == {}
        assert dataset.test_user_pos_cross == {}
        assert len(dataset.source_files) == 8
        assert all(Path(path).is_file() for path in dataset.source_files)
        assert np.all(raw.item_country == raw.countries.index("UNKNOWN_ORIGIN"))
        expected_union = set().union(*(set(values) for values in catalogs.values()))
        assert len(shared) == 40 and dataset.n_items == len(expected_union)
        meta = json.loads(
            (processed_dir / "meta.json").read_text(encoding="utf-8")
        )
        split_meta = meta["cold_item_counts"]
        assert split_meta == {
            "validation": {market: 3 for market in TARGETS},
            "test": {market: 3 for market in TARGETS},
        }
        protocol = meta["catalog_protocol"]
        assert protocol["candidate_source"] == (
            "market_metadata_before_interaction_filter"
        )
        assert protocol["interaction_filter"] == "iterative_bipartite_5_core"
        assert protocol["market_item_counts"] == {
            market: len(catalogs[market]) for market in MARKETS
        }
        assert protocol["filtered_item_counts"] == {
            "us": 105, "cn": 60, "in": 60, "sg": 60,
        }
        for market in TARGETS:
            assert len(cold_val_by_market[market]) == 3
            assert len(cold_test_by_market[market]) == 3
            assert cold_val_by_market[market].isdisjoint(
                cold_test_by_market[market]
            )
            assert int(raw.item_market_mask[MARKETS.index(market)].sum()) == 61
            degree = {}
            for _, asin, _, _ in rows[market]:
                degree[asin] = degree.get(asin, 0) + 1
            for selected in (
                cold_val_by_market[market], cold_test_by_market[market]
            ):
                buckets = sorted(
                    int(np.log2(degree[asin_by_id[item]])) for item in selected
                )
                assert buckets == [2, 2, 3]
        assert int(raw.item_market_mask[MARKETS.index("us")].sum()) == 106

        sparse_ids = {item_id[asin] for asin in sparse.values()}
        split_item_ids = set(map(int, np.concatenate([
            pairs[:, 1] for pairs in raw.predefined_splits.values()
        ])))
        assert sparse_ids <= set(map(int, dataset.zero_train_items))
        assert sparse_ids.isdisjoint(split_item_ids)
        for market, asin in sparse.items():
            assert raw.item_market_mask[MARKETS.index(market), item_id[asin]]

        # Target-market cold feedback is held out, while the same ASIN retains US
        # feedback and therefore must not be treated as a globally unseen ID.
        source_users = np.flatnonzero(dataset.user_country == MARKETS.index("us"))
        source_train = dataset.train_pairs[
            np.isin(dataset.train_pairs[:, 0], source_users)
        ]
        for market in TARGETS:
            market_users = np.flatnonzero(
                dataset.user_country == MARKETS.index(market)
            )
            market_train = dataset.train_pairs[
                np.isin(dataset.train_pairs[:, 0], market_users)
            ]
            local_cold = cold_val_by_market[market] | cold_test_by_market[market]
            assert not np.isin(market_train[:, 1], list(local_cold)).any()
        assert transfer_cold <= set(map(int, source_train[:, 1]))
        assert transfer_cold.isdisjoint(set(map(int, dataset.zero_train_items)))
        assert set(map(int, dataset.transfer_cold_items)) == transfer_cold
        assert len(dataset.cold_items) == 0

        # For every target user, warm validation/test are the last two dated warm
        # interactions after that market's cold ASINs have been removed.
        val_pairs = set(map(tuple, dataset.val_pairs.tolist()))
        test_pairs = set(map(tuple, dataset.test_pairs.tolist()))
        for market in TARGETS:
            local_cold = cold_val_by_market[market] | cold_test_by_market[market]
            cold_asins = {record["asin"] for record in item_records
                          if int(record["idx"]) in local_cold}
            by_user = {}
            for user, asin, _, stamp in rows[market]:
                if asin == sparse[market]:
                    continue
                by_user.setdefault(user, []).append((stamp, asin))
            for user, interactions in by_user.items():
                warm = sorted(pair for pair in interactions if pair[1] not in cold_asins)
                uid = user_id[(market, user)]
                assert (uid, item_id[warm[-2][1]]) in val_pairs
                assert (uid, item_id[warm[-1][1]]) in test_pairs

        # US-only items exercise the schema-v2 fallback/dedup contract.
        fallback_item = item_id[us_only[0]]
        assert raw.item_text_is_fallback[fallback_item].tolist() == [
            False, True, True, True
        ]
        assert raw.item_text_dedup_mask[fallback_item].tolist() == [
            True, False, False, False
        ]
        assert not raw.item_text_pair_valid.any()

        model = _CatalogScoreModel(dataset.n_users, dataset.n_items).train()
        report = evaluate(
            model,
            {},
            dataset,
            [10],
            dataset.val_user_pos_cold,
            aggregation="macro",
            return_report=True,
        )
        evaluated_users = np.asarray(sorted(dataset.val_user_pos_cold), dtype=np.int64)
        expected_minimum = min(
            int(dataset.item_market_mask[dataset.user_country[user]].sum())
            - len(dataset.train_user_pos[user])
            for user in evaluated_users
        )
        coverage = report.coverage["eligible"]["10"]
        assert coverage["markets"] == ["CN", "IN", "SG"], report.coverage
        assert coverage["excluded_markets"] == []
        assert coverage["minimum_candidates"] == expected_minimum
        assert expected_minimum > max(map(len, cold_val_by_market.values()))
        assert raw.n_items > int(dataset.item_market_mask[MARKETS.index("cn")].sum())
        assert model.training, "evaluation must restore the caller's train/eval state"

        meta_path = processed_dir / "meta.json"
        stale_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stale_meta.pop("catalog_protocol")
        meta_path.write_text(json.dumps(stale_meta), encoding="utf-8")
        try:
            XMarketRealData(cfg)
        except ValueError as exc:
            assert "catalog_protocol" in str(exc)
        else:
            raise AssertionError("旧 XMarket 目录协议必须显式重建")


if __name__ == "__main__":
    test_seeded_leave_two_out_without_timestamps()
    test_fasttext_numpy2_prediction_fallback()
    test_xmarket_predefined_splits_and_complete_market_evaluation()
    print("PASS: XMarket schema-v2 predefined splits and complete-market evaluation")
