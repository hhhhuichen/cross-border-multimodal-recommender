# -*- coding: utf-8 -*-
"""ASEAN 商品采集范围、Parquet 过滤和清单契约的离线回归测试。"""
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = PROJECT_DIR / "off_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from fetch_off_dump import extract_parquet  # noqa: E402
from build_off_dataset import build_country_vocabulary  # noqa: E402
from build_country_adj import (  # noqa: E402
    M49,
    finalize_trade_shares,
    get_reported_imports,
)
from off_common import (  # noqa: E402
    ASEAN_MEMBERSHIP,
    MARKETS,
    MARKET_ISO,
    TAG2ISO,
    resolve_markets,
    write_collection_manifest,
)
from simulate_users import align_trade_matrix, origin_probabilities  # noqa: E402


def fixture_row(code, markets, language="en", name="fixture"):
    return {
        "code": code,
        "lang": language,
        "product_name": [
            {"lang": "main", "text": name},
            {"lang": language, "text": name},
        ],
        "ingredients_text": [],
        "brands": "fixture-brand",
        "brands_tags": ["fixture-brand"],
        "categories_tags": ["en:foods"],
        "labels_tags": [],
        "origins_tags": [],
        "manufacturing_places": "",
        "countries_tags": markets,
        "images": [],
        "quantity": "1 unit",
        "created_t": 1,
        "last_modified_t": 2,
    }


def test_membership_contract():
    assert ASEAN_MEMBERSHIP["member_count"] == 11
    assert MARKET_ISO == [
        "TH", "SG", "PH", "ID", "MY", "TL", "VN", "KH", "BN", "MM", "LA",
    ]
    assert "CN" not in MARKET_ISO
    assert [market[1] for market in resolve_markets(["vn,TL", "id"])] == [
        "ID", "TL", "VN",
    ]
    try:
        resolve_markets(["CN"])
    except ValueError as exc:
        assert "非 ASEAN-11 市场" in str(exc)
    else:
        raise AssertionError("中国被错误接受为 ASEAN 成员市场")


def test_bulk_collector_filters_to_asean_members():
    rows = [
        fixture_row("TH-1", ["en:thailand"], "th", "ชาไทย"),
        fixture_row("TL-1", ["en:timor-leste"], "tet", "Kafe Timor"),
        fixture_row("CN-1", ["en:china"], "zh", "中国商品"),
        fixture_row("BOTH-1", ["en:thailand", "en:timor-leste"]),
        fixture_row("", ["en:thailand"]),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        parquet_path = root / "food.parquet"
        raw_dir = root / "raw"
        pq.write_table(pa.Table.from_pylist(rows), parquet_path)

        selected = resolve_markets(["TH", "TL"])
        stats = extract_parquet(
            parquet_path, selected, raw_dir=raw_dir, progress=False
        )
        assert stats["scanned_rows"] == 5
        assert stats["matched_unique_rows"] == 3
        assert stats["market_record_counts"] == {"TH": 2, "TL": 2}

        thailand = [
            json.loads(line)
            for line in (raw_dir / "thailand.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        timor = [
            json.loads(line)
            for line in (raw_dir / "timor-leste.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        assert {row["code"] for row in thailand} == {"TH-1", "BOTH-1"}
        assert {row["code"] for row in timor} == {"TL-1", "BOTH-1"}
        assert timor[0]["product_name_tet"] == "Kafe Timor"
        assert not (raw_dir / "china.jsonl").exists()

        manifest = write_collection_manifest(
            selected,
            "fixture-parquet",
            raw_dir=raw_dir,
            source_snapshot={"file": parquet_path.name},
        )
        assert manifest["scope"]["member_iso"] == MARKET_ISO
        assert manifest["complete_market_count"] == 2
        assert not manifest["formal_snapshot_ready"]
        assert {row["iso"] for row in manifest["markets"]} == set(MARKET_ISO)
        assert "CN" not in {row["iso"] for row in manifest["markets"]}


def test_comtrade_queries_non_asean_origins_and_other_residual():
    requested = []

    def fake_fetch(url):
        requested.append(url)
        return {
            "data": [
                {"partnerCode": 0, "primaryValue": 1000},
                {"partnerCode": M49["CN"], "primaryValue": 300},
                {"partnerCode": str(M49["US"]), "primaryValue": 100},
            ]
        }

    origins = ["TH", "CN", "US", "OTHER"]
    imports, covered, _ = get_reported_imports(
        2024,
        ["TH"],
        origins,
        fetcher=fake_fetch,
        request_delay=0,
    )
    assert covered == ["TH"]
    assert imports.shape == (1, 4)
    assert np.allclose(imports[0], [0, 300, 100, 600])
    assert str(M49["CN"]) in requested[0]
    assert str(M49["US"]) in requested[0]
    assert "partner2Code=0" in requested[0]
    assert "customsCode=C00" in requested[0]
    assert "motCode=0" in requested[0]

    provenance = {0: "self-reported imports 2024"}
    shares, pooled = finalize_trade_shares(
        np.vstack([imports, np.zeros_like(imports)]),
        provenance,
        pool_prior=0.0,
    )
    assert np.allclose(shares[0], [0, 0.3, 0.1, 0.6])
    assert np.allclose(shares[1], pooled)
    assert provenance[1] == "ASEAN pooled imports fallback"

    def expanded_fetch(_url):
        return {"count": 500, "data": [{"partnerCode": 0,
                                          "primaryValue": 1000}]}

    try:
        get_reported_imports(
            2024, ["TH"], origins,
            fetcher=expanded_fetch, request_delay=0,
        )
    except RuntimeError as exc:
        assert "未按 partner2Code=0" in str(exc)
    else:
        raise AssertionError("Comtrade 截断明细被错误当作聚合贸易额")


def test_trade_axis_alignment_and_non_asean_sampling_weights():
    # 文件轴故意乱序，加载后必须恢复到 processed meta 的市场/原产国顺序。
    raw = np.array([
        [0.10, 0.20, 0.50, 0.05, 0.15],  # SG
        [0.20, 0.10, 0.15, 0.45, 0.10],  # TH
    ])
    trade_meta = {
        "trade_schema_version": 2,
        "markets": ["SG", "TH"],
        "trade_origins": ["OTHER", "US", "CN", "TH", "SG"],
    }
    countries = ["TH", "SG", "CN", "US", "OTHER"]
    aligned = align_trade_matrix(raw, trade_meta, ["TH", "SG"], countries)
    assert np.allclose(aligned[0], [0.45, 0.10, 0.15, 0.10, 0.20])
    assert np.allclose(aligned[1], [0.05, 0.15, 0.50, 0.20, 0.10])

    probabilities, fallback = origin_probabilities(
        aligned, market=0, origins=[2, 3], pool_sizes=[9, 1]
    )
    assert not fallback
    assert np.allclose(probabilities, [0.6, 0.4])

    zero_trade = aligned.copy()
    zero_trade[0, [2, 3]] = 0
    probabilities, fallback = origin_probabilities(
        zero_trade, market=0, origins=[2, 3], pool_sizes=[9, 1]
    )
    assert fallback
    assert np.allclose(probabilities, [0.9, 0.1])

    try:
        align_trade_matrix(raw, {"markets": ["SG", "TH"]}, ["TH"], countries)
    except ValueError as exc:
        assert "旧契约" in str(exc)
    else:
        raise AssertionError("旧 11x11 贸易矩阵被静默接受")


def test_all_observed_foreign_origins_are_retained():
    foreign = [
        "CN", "JP", "KR", "US", "AU", "NZ", "FR", "DE", "IT", "ES",
        "GB", "NL", "BE", "CH", "PL", "TR", "CA", "BR", "IN", "ZA",
    ]
    vocabulary = build_country_vocabulary(
        Counter({country: len(foreign) - index for index, country in enumerate(foreign)})
    )
    assert vocabulary[:len(MARKET_ISO)] == MARKET_ISO
    assert vocabulary[-1] == "OTHER"
    assert set(foreign) <= set(vocabulary)
    assert len([country for country in vocabulary if country in foreign]) == 20
    parsed_iso = set(TAG2ISO.values()) - {None}
    assert parsed_iso <= set(M49), "可解析原产国必须具备 UN M49 查询编码"


if __name__ == "__main__":
    test_membership_contract()
    test_bulk_collector_filters_to_asean_members()
    test_comtrade_queries_non_asean_origins_and_other_residual()
    test_trade_axis_alignment_and_non_asean_sampling_weights()
    test_all_observed_foreign_origins_are_retained()
    print("PASS: ASEAN-11 product collection contract")
