# -*- coding: utf-8 -*-
"""预注册 B0-B3 筛选/验收的结构化回归测试。"""
from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import numpy as np

from research.validate_experiments import (
    FINAL_SEED_PAIRS,
    SCREENING_SEED_PAIRS,
    acceptance,
    markdown_report,
    resample_hierarchical_difference,
    screening,
)


OFFSETS = {"B0": 0.00, "B1": 0.01, "B2": 0.02, "B3": 0.04}
RESIDUAL = {
    "B0": "none", "B1": "fused", "B2": "decoupled",
    "B3": "market_reliable",
}


def _report(value, markets):
    per_user = {}
    for index, market in enumerate(markets):
        per_user[str(index)] = {
            "market": index,
            "ndcg@10": value,
            "recall@10": value,
        }
    metrics = {
        "ndcg@10": value, "recall@10": value,
        "ndcg@20": value, "recall@20": value,
    }
    eligibility = {
        "markets": list(markets), "excluded_markets": [],
        "users": len(markets), "excluded_users": 0,
        "user_fraction": 1.0, "positives": len(markets),
        "excluded_positives": 0, "positive_fraction": 1.0,
        "minimum_candidates": 20,
    }
    return {
        "micro": dict(metrics), "macro": dict(metrics),
        "per_user": per_user,
        "coverage": {
            "eligible": {"10": deepcopy(eligibility), "20": deepcopy(eligibility)},
            "sampled_users": len(markets), "total_users": len(markets),
            "total_positives": len(markets),
        },
    }


def _runs(source, seed_pairs):
    markets = ["CN", "IN", "SG"] if source == "xmarket" else ["TH", "SG"]
    out = []
    for variant, offset in OFFSETS.items():
        for split_seed, train_seed in seed_pairs:
            value = 0.10 + offset + (split_seed % 10) * 0.0001
            out.append({
                "_path": f"{source}-{variant}-{split_seed}.json",
                "status": "completed",
                "variant": variant,
                "model": "acmr",
                "residual": RESIDUAL[variant],
                "split_seed": split_seed,
                "train_seed": train_seed,
                "data_schema_version": 2,
                "source_snapshot_sha256": "source-hash",
                "parameter_count": {
                    "trainable": 1000 if variant == "B0" else 1100,
                },
                "config": {
                    "data": {"source": source},
                    "train": {"topk": [10, 20]},
                },
                "dataset": {"raw_source_hashes": {"raw": "data-hash"}},
                "test": {
                    "overall": _report(value, markets),
                    "cold": _report(value, markets),
                    "subsets": {
                        "missing_image": _report(value, markets),
                        "genuine_multilingual": _report(value, markets),
                        "long_tail": _report(value, markets),
                    },
                },
            })
    if source == "xmarket":
        for model, variant, offset in (
            ("bpr_mf", "BPR-MF", -0.03),
            ("lightgcn", "LightGCN", -0.02),
            ("vbpr", "VBPR", -0.01),
        ):
            for split_seed, train_seed in seed_pairs:
                value = 0.10 + offset + (split_seed % 10) * 0.0001
                out.append({
                    "_path": f"{source}-{variant}-{split_seed}.json",
                    "status": "completed", "variant": variant,
                    "model": model, "residual": "none",
                    "split_seed": split_seed, "train_seed": train_seed,
                    "data_schema_version": 2,
                    "source_snapshot_sha256": "source-hash",
                    "parameter_count": {"trainable": 500},
                    "config": {
                        "data": {"source": source},
                        "train": {"topk": [10, 20]},
                    },
                    "dataset": {"raw_source_hashes": {"raw": "data-hash"}},
                    "test": {
                        "overall": _report(value, markets),
                        "cold": _report(value, markets),
                        "subsets": {
                            "missing_image": _report(value, markets),
                            "genuine_multilingual": _report(value, markets),
                            "long_tail": _report(value, markets),
                        },
                    },
                })
    return out


def test_off_screening_cannot_unlock_default():
    runs = _runs("off", SCREENING_SEED_PAIRS)
    result = screening(runs)
    assert result["default_residual"] == "none"
    assert result["comparisons"]["B3-B0"]["mean_difference"] > 0
    for subset in ("cold", "missing_image", "genuine_multilingual", "long_tail"):
        endpoint = f"{subset}.macro.ndcg@10"
        assert endpoint in result["variants"]["B3"]
        assert endpoint in result["diagnostic_comparisons"]
        assert subset in result["coverage"]
    report = markdown_report(result)
    assert "genuine_multilingual" in report
    assert "Coverage: 11/11 checked" in report
    try:
        acceptance(runs, 100)
    except ValueError as exc:
        assert "xmarket" in str(exc) or "种子对" in str(exc)
    else:
        raise AssertionError("OFF 筛选结果被错用于最终验收")


def test_xmarket_acceptance_and_parameter_guard():
    runs = _runs("xmarket", FINAL_SEED_PAIRS)
    result = acceptance(runs, 200)
    assert result["default_residual"] == "market_reliable"
    assert set(result["baseline_descriptives"]) == {
        "BPR-MF", "LightGCN", "VBPR",
    }
    assert set(result["baseline_comparisons"]) == {
        "B0-BPR-MF", "B0-LightGCN", "B0-VBPR",
        "B3-BPR-MF", "B3-LightGCN", "B3-VBPR",
    }
    assert result["checks"]["primary_mean_positive"]
    assert result["checks"]["primary_ci_above_zero"]

    missing_baselines = [
        run for run in runs if run["variant"] not in {"BPR-MF", "LightGCN", "VBPR"}
    ]
    try:
        acceptance(missing_baselines, 10)
    except ValueError as exc:
        assert "BPR-MF" in str(exc) or "LightGCN" in str(exc) or "VBPR" in str(exc)
    else:
        raise AssertionError("最终验收未强制要求独立基线")

    invalid = deepcopy(runs)
    for run in invalid:
        if run["variant"] == "B3":
            run["parameter_count"]["trainable"] = 1200
    try:
        acceptance(invalid, 10)
    except ValueError as exc:
        assert "1%" in str(exc)
    else:
        raise AssertionError("B1-B3 参数量不匹配未被拒绝")


def test_hierarchical_micro_respects_user_counts():
    class FixedRng:
        def integers(self, _low, _high, size):
            if size == 2:
                return np.asarray([0, 1])
            return np.zeros(size, dtype=np.int64)

    strata = (np.ones(3), -np.ones(1))
    macro = resample_hierarchical_difference(strata, "macro", FixedRng())
    micro = resample_hierarchical_difference(strata, "micro", FixedRng())
    assert macro == 0.0
    assert micro == 0.5


def test_legacy_ablation_uses_complete_catalog_for_cold_evaluation():
    source = Path("research/summarize_ablation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None)
        if name == "evaluate":
            keywords = {keyword.arg for keyword in node.keywords}
            assert "candidate_items" not in keywords


if __name__ == "__main__":
    test_off_screening_cannot_unlock_default()
    test_xmarket_acceptance_and_parameter_guard()
    test_hierarchical_micro_respects_user_counts()
    test_legacy_ablation_uses_complete_catalog_for_cold_evaluation()
    print("PASS: experiment screening and final acceptance protocol")
