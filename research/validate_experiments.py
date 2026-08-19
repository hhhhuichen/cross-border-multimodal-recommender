#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总配对 B0-B3 结果并执行预注册的 bootstrap/非劣验收。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
from pathlib import Path

import numpy as np


VARIANT = {
    "none": "B0",
    "fused": "B1",
    "decoupled": "B2",
    "market_reliable": "B3",
}
REQUIRED_VARIANTS = ("B0", "B1", "B2", "B3")
BASELINE_VARIANT = {
    "bpr_mf": "BPR-MF", "lightgcn": "LightGCN", "vbpr": "VBPR",
}
REQUIRED_BASELINES = tuple(BASELINE_VARIANT.values())
REPORT_SUBSETS = (
    "overall", "cold", "missing_image", "genuine_multilingual", "long_tail",
)
REPORT_AGGREGATES = ("macro", "micro")
REPORT_METRICS = ("recall@10", "ndcg@10", "recall@20", "ndcg@20")
FINAL_SEED_PAIRS = tuple(
    (20260801 + offset, 20260901 + offset) for offset in range(5)
)
SCREENING_SEED_PAIRS = FINAL_SEED_PAIRS[:3]


def load_runs(patterns):
    paths = []
    for pattern in patterns:
        matched = glob.glob(pattern)
        paths.extend(matched or ([pattern] if Path(pattern).is_file() else []))
    runs = []
    for path in sorted(set(paths)):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            continue
        payload["_path"] = path
        model = payload.get("model")
        payload["variant"] = (
            VARIANT.get(payload.get("residual"), payload.get("residual"))
            if model == "acmr" else BASELINE_VARIANT.get(model, model)
        )
        runs.append(payload)
    if not runs:
        raise ValueError("没有找到 completed 实验清单")
    return runs


def pair_key(run):
    return int(run["split_seed"]), int(run["train_seed"])


def report_at(run, subset):
    test = run.get("test", {})
    value = test.get(subset)
    if value is None:
        value = test.get("subsets", {}).get(subset)
    if value is None:
        raise KeyError(f"{run['_path']} 缺 test.{subset}")
    return value


def aggregate_value(run, subset, metric, aggregate="macro"):
    return float(report_at(run, subset)[aggregate][metric])


def paired_runs(runs, candidate, baseline):
    by_variant = defaultdict(dict)
    for run in runs:
        key = pair_key(run)
        if key in by_variant[run["variant"]]:
            raise ValueError(f"重复实验：variant={run['variant']} seed={key}")
        by_variant[run["variant"]][key] = run
    keys = sorted(set(by_variant[candidate]) & set(by_variant[baseline]))
    if not keys:
        raise ValueError(f"{candidate} 与 {baseline} 没有共同种子")
    return [(by_variant[candidate][key], by_variant[baseline][key]) for key in keys]


def validate_protocol(runs, *, source, expected_pairs):
    """在统计计算前锁定数据、种子、模型和参数公平性。"""
    expected_pairs = tuple(expected_pairs)
    by_variant = defaultdict(dict)
    source_hashes, snapshots = [], set()
    for run in runs:
        variant = run.get("variant")
        if variant not in REQUIRED_VARIANTS:
            continue
        if run.get("model") != "acmr":
            raise ValueError(f"{run['_path']} 不是 ACMR B0-B3 实验")
        if int(run.get("data_schema_version", -1)) != 2:
            raise ValueError(f"{run['_path']} 不是 data schema v2")
        configured_source = run.get("config", {}).get("data", {}).get("source")
        if configured_source != source:
            raise ValueError(
                f"{run['_path']} data.source={configured_source!r}，期望 {source!r}"
            )
        topk = run.get("config", {}).get("train", {}).get("topk", [])
        if not {10, 20}.issubset(set(map(int, topk))):
            raise ValueError(f"{run['_path']} 未同时评测 @10/@20")
        key = pair_key(run)
        if key in by_variant[variant]:
            raise ValueError(f"重复实验：variant={variant} seed={key}")
        by_variant[variant][key] = run
        hashes = run.get("dataset", {}).get("raw_source_hashes")
        if hashes is None:
            hashes = run.get("dataset", {}).get("source_hashes", {})
        if source != "synthetic" and not hashes:
            raise ValueError(f"{run['_path']} 缺原始数据哈希")
        source_hashes.append(hashes)
        snapshots.add(run.get("source_snapshot_sha256"))

    for variant in REQUIRED_VARIANTS:
        keys = tuple(sorted(by_variant[variant]))
        if keys != tuple(sorted(expected_pairs)):
            raise ValueError(
                f"{variant} 种子对 {keys} 与预注册 {expected_pairs} 不一致"
            )
    if len({json.dumps(value, sort_keys=True) for value in source_hashes}) != 1:
        raise ValueError("B0-B3 使用的原始数据哈希不一致")
    if None in snapshots or len(snapshots) != 1:
        raise ValueError("B0-B3 源码快照哈希不一致")

    residual_counts = []
    for variant in ("B1", "B2", "B3"):
        residual_counts.extend(
            int(run["parameter_count"]["trainable"])
            for run in by_variant[variant].values()
        )
    if max(residual_counts) / min(residual_counts) > 1.01:
        raise ValueError("B1-B3 可训练参数量差异超过 1%")

    # 候选集和市场资格只由 split 决定，不得随模型变体改变。
    for key in expected_pairs:
        for subset in REPORT_SUBSETS:
            reference = report_at(by_variant["B0"][key], subset).get("coverage")
            if reference is None:
                raise ValueError(f"B0 种子 {key} 的 {subset} 缺覆盖率清单")
            for variant in REQUIRED_VARIANTS[1:]:
                current = report_at(by_variant[variant][key], subset).get("coverage")
                if current != reference:
                    raise ValueError(
                        f"{subset} 市场资格在 {variant} 与 B0 间不一致：seed={key}"
                    )
    return by_variant


def validate_baseline_protocol(runs, *, source, expected_pairs, acmr_runs):
    """独立基线必须与 B0-B3 共享 split、数据、源码和评测资格。"""
    expected_pairs = tuple(expected_pairs)
    expected_models = {value: key for key, value in BASELINE_VARIANT.items()}
    by_variant = defaultdict(dict)
    for run in runs:
        variant = run.get("variant")
        if variant not in REQUIRED_BASELINES:
            continue
        if run.get("model") != expected_models[variant]:
            raise ValueError(f"{run['_path']} 的基线标签与 model 不一致")
        if run.get("residual") != "none":
            raise ValueError(f"{run['_path']} 独立基线不得启用 residual")
        if int(run.get("data_schema_version", -1)) != 2:
            raise ValueError(f"{run['_path']} 不是 data schema v2")
        configured_source = run.get("config", {}).get("data", {}).get("source")
        if configured_source != source:
            raise ValueError(
                f"{run['_path']} data.source={configured_source!r}，期望 {source!r}"
            )
        topk = run.get("config", {}).get("train", {}).get("topk", [])
        if not {10, 20}.issubset(set(map(int, topk))):
            raise ValueError(f"{run['_path']} 未同时评测 @10/@20")
        key = pair_key(run)
        if key in by_variant[variant]:
            raise ValueError(f"重复实验：variant={variant} seed={key}")
        by_variant[variant][key] = run

    for variant in REQUIRED_BASELINES:
        keys = tuple(sorted(by_variant[variant]))
        if keys != tuple(sorted(expected_pairs)):
            raise ValueError(
                f"{variant} 种子对 {keys} 与预注册 {expected_pairs} 不一致"
            )
        for key in expected_pairs:
            run = by_variant[variant][key]
            reference = acmr_runs["B0"][key]
            hashes = run.get("dataset", {}).get("raw_source_hashes")
            if hashes is None:
                hashes = run.get("dataset", {}).get("source_hashes", {})
            reference_hashes = reference.get("dataset", {}).get("raw_source_hashes")
            if reference_hashes is None:
                reference_hashes = reference.get("dataset", {}).get(
                    "source_hashes", {}
                )
            if hashes != reference_hashes:
                raise ValueError(f"{variant} 与 B0 的原始数据哈希不一致: {key}")
            if run.get("source_snapshot_sha256") != reference.get(
                    "source_snapshot_sha256"):
                raise ValueError(f"{variant} 与 B0 的源码快照不一致: {key}")
            for subset in REPORT_SUBSETS:
                report = report_at(run, subset)
                current = report.get("coverage")
                expected = report_at(reference, subset).get("coverage")
                if current != expected:
                    raise ValueError(
                        f"{variant} 与 B0 的 {subset} 资格清单不一致: {key}"
                    )
                if subset == "overall" and not report.get("per_user"):
                    raise ValueError(
                        f"{variant} 种子 {key} 缺配对 bootstrap 所需 per_user"
                    )
    return by_variant


def sample_sd(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def hierarchical_difference_strata(pair, subset, metric):
    cand_users = report_at(pair[0], subset).get("per_user", {})
    base_users = report_at(pair[1], subset).get("per_user", {})
    if not cand_users or not base_users:
        return None
    if set(cand_users) != set(base_users):
        raise ValueError(
            f"{pair[0]['_path']} 与 {pair[1]['_path']} 的 {subset} 用户不一致"
        )
    by_market = defaultdict(list)
    for user in sorted(cand_users):
        c, b = cand_users[user], base_users[user]
        if metric not in c or metric not in b:
            raise KeyError(f"用户 {user} 缺 {subset}.{metric}")
        market = int(c["market"])
        if market != int(b["market"]):
            raise ValueError(f"用户 {user} 在配对运行中的市场不一致")
        by_market[market].append(float(c[metric]) - float(b[metric]))
    return tuple(
        np.asarray(by_market[market], dtype=np.float64)
        for market in sorted(by_market)
    ) or None


def resample_hierarchical_difference(strata, aggregate, rng):
    selected = rng.integers(0, len(strata), size=len(strata))
    market_means, total, count = [], 0.0, 0
    for index in selected:
        values = strata[int(index)]
        sampled = values[rng.integers(0, len(values), size=len(values))]
        market_means.append(float(sampled.mean()))
        total += float(sampled.sum())
        count += int(sampled.size)
    if aggregate == "macro":
        return float(np.mean(market_means))
    if aggregate == "micro":
        return total / count
    raise ValueError(f"未知聚合方式: {aggregate!r}")


def paired_summary(pairs, subset, metric, aggregate="macro", n_boot=10000,
                   seed=20260801):
    cand = np.asarray([aggregate_value(c, subset, metric, aggregate) for c, _ in pairs])
    base = np.asarray([aggregate_value(b, subset, metric, aggregate) for _, b in pairs])
    diff = cand - base
    strata = [hierarchical_difference_strata(pair, subset, metric)
              for pair in pairs]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for iteration in range(n_boot):
        selected = rng.integers(0, len(pairs), size=len(pairs))
        hierarchy = []
        for index in selected:
            current = strata[int(index)]
            value = (diff[int(index)] if current is None else
                     resample_hierarchical_difference(current, aggregate, rng))
            hierarchy.append(value)
        boot[iteration] = float(np.mean(hierarchy))
    sd_diff = sample_sd(diff)
    return {
        "n_pairs": int(len(pairs)),
        "candidate_mean": float(cand.mean()),
        "candidate_sd": sample_sd(cand),
        "baseline_mean": float(base.mean()),
        "baseline_sd": sample_sd(base),
        "mean_difference": float(diff.mean()),
        "difference_sd": sd_diff,
        "paired_effect_dz": float(diff.mean() / sd_diff) if sd_diff else None,
        "bootstrap_95_ci": [float(np.quantile(boot, 0.025)),
                            float(np.quantile(boot, 0.975))],
        "bootstrap_iterations": int(n_boot),
    }


def comparison(runs, candidate, baseline, subset, metric, aggregate, n_boot):
    pairs = paired_runs(runs, candidate, baseline)
    out = paired_summary(pairs, subset, metric, aggregate, n_boot)
    out.update({"candidate": candidate, "baseline": baseline,
                "subset": subset, "metric": metric, "aggregate": aggregate,
                "seed_pairs": [list(pair_key(c)) for c, _ in pairs]})
    return out


def variant_summaries(by_variant, seed_pairs, variant_order=REQUIRED_VARIANTS):
    variants = {}
    for variant in variant_order:
        ordered = [by_variant[variant][key] for key in seed_pairs]
        summaries = {}
        for subset in REPORT_SUBSETS:
            for aggregate in REPORT_AGGREGATES:
                for metric in REPORT_METRICS:
                    values = np.asarray([
                        aggregate_value(run, subset, metric, aggregate)
                        for run in ordered
                    ])
                    summaries[f"{subset}.{aggregate}.{metric}"] = {
                        "mean": float(values.mean()),
                        "sample_sd": sample_sd(values),
                        "values": values.tolist(),
                    }
        variants[variant] = summaries
    return variants


def paired_descriptive(variants, candidate, baseline, endpoint):
    candidate_values = np.asarray(variants[candidate][endpoint]["values"])
    baseline_values = np.asarray(variants[baseline][endpoint]["values"])
    difference = candidate_values - baseline_values
    difference_sd = sample_sd(difference)
    return {
        "candidate": candidate,
        "baseline": baseline,
        "candidate_mean": float(candidate_values.mean()),
        "candidate_sample_sd": sample_sd(candidate_values),
        "baseline_mean": float(baseline_values.mean()),
        "baseline_sample_sd": sample_sd(baseline_values),
        "mean_difference": float(difference.mean()),
        "difference_sample_sd": difference_sd,
        "paired_effect_dz": (
            float(difference.mean() / difference_sd) if difference_sd else None
        ),
    }


def coverage_summary(by_variant, seed_pairs):
    """资格清单在 B0-B3 间已校验一致，因此只存一份 B0 副本。"""
    output = {}
    for subset in REPORT_SUBSETS:
        output[subset] = []
        for key in seed_pairs:
            output[subset].append({
                "split_seed": int(key[0]),
                "train_seed": int(key[1]),
                "coverage": report_at(by_variant["B0"][key], subset)["coverage"],
            })
    return output


def acceptance(runs, n_boot):
    by_variant = validate_protocol(
        runs, source="xmarket", expected_pairs=FINAL_SEED_PAIRS
    )
    baseline_runs = validate_baseline_protocol(
        runs, source="xmarket", expected_pairs=FINAL_SEED_PAIRS,
        acmr_runs=by_variant,
    )
    for run in runs:
        if run.get("variant") not in REQUIRED_VARIANTS:
            continue
        coverage = report_at(run, "overall")["coverage"]["eligible"]["10"]
        if set(coverage["markets"]) != {"CN", "IN", "SG"}:
            raise ValueError(
                f"{run['_path']} 主指标未覆盖 CN/IN/SG 三个目标市场"
            )
        for subset in ("overall", "cold"):
            if not report_at(run, subset).get("per_user"):
                raise ValueError(
                    f"{run['_path']} 缺分层 bootstrap 所需 {subset}.per_user"
                )
    primary = comparison(runs, "B3", "B0", "overall", "ndcg@10", "macro", n_boot)
    recall = comparison(runs, "B3", "B0", "overall", "recall@10", "macro", n_boot)
    overall_micro = comparison(runs, "B3", "B0", "overall", "ndcg@10", "micro", n_boot)
    cold = comparison(runs, "B3", "B0", "cold", "ndcg@10", "macro", n_boot)
    decoupling = comparison(runs, "B2", "B1", "overall", "ndcg@10", "macro", n_boot)
    conditioning = comparison(runs, "B3", "B2", "overall", "ndcg@10", "macro", n_boot)
    baseline_comparisons = {}
    for candidate in ("B0", "B3"):
        for baseline in REQUIRED_BASELINES:
            baseline_comparisons[f"{candidate}-{baseline}"] = comparison(
                runs, candidate, baseline, "overall", "ndcg@10", "macro", n_boot
            )

    checks = {
        "primary_mean_positive": primary["mean_difference"] > 0.0,
        "primary_ci_above_zero": primary["bootstrap_95_ci"][0] > 0.0,
        "macro_recall_mean_positive": recall["mean_difference"] > 0.0,
        "overall_micro_noninferior": overall_micro["bootstrap_95_ci"][0] >= -0.005,
        "cold_noninferior": cold["bootstrap_95_ci"][0] >= -0.005,
        "decoupling_supported": decoupling["bootstrap_95_ci"][0] > 0.0,
        "market_conditioning_supported": conditioning["bootstrap_95_ci"][0] > 0.0,
    }
    default_enabled = all(checks[name] for name in (
        "primary_mean_positive", "primary_ci_above_zero",
        "macro_recall_mean_positive",
        "overall_micro_noninferior", "cold_noninferior",
    ))
    return {
        "acceptance_version": 2,
        "noninferiority_margin": -0.005,
        "comparisons": {
            "primary": primary, "recall": recall,
            "overall_micro": overall_micro, "cold": cold,
            "decoupling": decoupling, "conditioning": conditioning,
        },
        "baseline_comparisons": baseline_comparisons,
        "descriptives": variant_summaries(by_variant, FINAL_SEED_PAIRS),
        "baseline_descriptives": variant_summaries(
            baseline_runs, FINAL_SEED_PAIRS, REQUIRED_BASELINES
        ),
        "coverage": coverage_summary(by_variant, FINAL_SEED_PAIRS),
        "checks": checks,
        "default_residual": "market_reliable" if default_enabled else "none",
        "interpretation": (
            "验收通过，可默认启用 B3。" if default_enabled else
            "验收未通过，B3 保持关闭；结果仅作为零收益或负结果报告。"
        ),
        "statistical_note": (
            "五种子结果以配对效应量和分层 bootstrap 区间描述；"
            "不据此宣称普适统计显著性。"
        ),
    }


def screening(runs):
    """OFF 只做三组配对机制筛选，永不改变默认残差开关。"""
    by_variant = validate_protocol(
        runs, source="off", expected_pairs=SCREENING_SEED_PAIRS
    )
    variants = variant_summaries(by_variant, SCREENING_SEED_PAIRS)

    comparisons = {}
    for candidate, baseline in (("B1", "B0"), ("B2", "B1"), ("B3", "B2"),
                                ("B3", "B0")):
        comparisons[f"{candidate}-{baseline}"] = paired_descriptive(
            variants, candidate, baseline, "overall.macro.ndcg@10"
        )
    diagnostic_comparisons = {}
    for subset in REPORT_SUBSETS:
        for aggregate in REPORT_AGGREGATES:
            for metric in REPORT_METRICS:
                endpoint = f"{subset}.{aggregate}.{metric}"
                diagnostic_comparisons[endpoint] = paired_descriptive(
                    variants, "B3", "B0", endpoint
                )
    return {
        "screening_version": 1,
        "data": "OFF semi-synthetic feedback",
        "seed_pairs": [list(value) for value in SCREENING_SEED_PAIRS],
        "variants": variants,
        "comparisons": comparisons,
        "diagnostic_comparisons": diagnostic_comparisons,
        "coverage": coverage_summary(by_variant, SCREENING_SEED_PAIRS),
        "default_residual": "none",
        "interpretation": (
            "OFF 只能筛选机制，不能通过 XMarket 外部验收；"
            "因此无论差值方向，B3 均保持默认关闭。"
        ),
    }


def append_descriptive_tables(lines, variants, *, variant_order=None, title=None):
    variant_order = tuple(variant_order or variants)
    first = variants[variant_order[0]]["overall.macro.recall@10"]["values"]
    lines.extend(["", f"### {title or 'Variant Metrics'}", "",
                  f"Values are {len(first)}-seed mean ± sample SD."])
    for subset in REPORT_SUBSETS:
        lines.extend([
            "", f"#### {subset}", "",
            "| Variant | Aggregate | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for variant in variant_order:
            for aggregate in REPORT_AGGREGATES:
                cells = []
                for metric in REPORT_METRICS:
                    value = variants[variant][f"{subset}.{aggregate}.{metric}"]
                    cells.append(f"{value['mean']:.6f} ± {value['sample_sd']:.6f}")
                lines.append(
                    f"| {variant} | {aggregate} | " + " | ".join(cells) + " |"
                )


def append_coverage_table(lines, coverage):
    lines.extend([
        "", "### Eligibility And Coverage", "",
        "Eligibility is data-defined and was verified identical across B0-B3.", "",
        "| Subset | Split seed | K | Eligible markets | Excluded markets | "
        "Users | User coverage | Positives | Interaction coverage | Min candidates |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ])
    for subset in REPORT_SUBSETS:
        for record in coverage[subset]:
            raw = record["coverage"]
            for k in ("10", "20"):
                value = raw["eligible"][k]
                markets = ",".join(map(str, value["markets"])) or "None"
                excluded = ",".join(map(str, value["excluded_markets"])) or "None"
                lines.append(
                    f"| {subset} | {record['split_seed']} | {k} | {markets} | "
                    f"{excluded} | {value['users']} | {value['user_fraction']:.6f} | "
                    f"{value['positives']} | {value['positive_fraction']:.6f} | "
                    f"{value['minimum_candidates']} |"
                )


def append_fallacy_scan(lines, *, screening_mode):
    if screening_mode:
        context = [
            ("Simpson's paradox", "NOTE", "同时报告 user micro 与 market macro，不用聚合值隐藏市场方向。"),
            ("Ecological fallacy", "CAUTION", "市场宏平均不可外推到个体用户。"),
            ("Berkson's paradox", "CAUTION", "OFF 商品样本与模拟用户筛选限制外推。"),
            ("Collider bias", "NOTE", "未在因果模型中调整后处理变量。"),
            ("Base-rate neglect", "NOTE", "报告 Recall/NDCG 及覆盖分母，不解读为诊断准确率。"),
            ("Regression to mean", "NOTE", "不是按极端基线选组的 pre-post 设计。"),
            ("Survivorship bias", "CAUTION", "真实商品元数据可用性与模拟交互生成限制覆盖。"),
            ("Look-elsewhere effect", "CAUTION", "多子集仅作诊断，不用于解锁默认残差。"),
            ("Garden of forking paths", "CAUTION", "模型、配对种子和主指标已锁定；OFF 仍只是筛选。"),
            ("Correlation != causation", "CAUTION", "半合成离线评测不支持真实用户因果效果。"),
            ("Reverse causality", "NOTE", "模拟反馈不用于用户偏好的方向因果声明。"),
        ]
    else:
        context = [
            ("Simpson's paradox", "NOTE", "已同时保留 user micro、market macro 和 per-market 方向。"),
            ("Ecological fallacy", "CAUTION", "市场宏平均不用于推断任一个体用户。"),
            ("Berkson's paradox", "CAUTION", "5-core 及有训练交互用户筛选限制外推。"),
            ("Collider bias", "NOTE", "未在因果模型中调整后处理市场或交互变量。"),
            ("Base-rate neglect", "NOTE", "报告 Recall/NDCG 及覆盖分母，不解读为诊断准确率。"),
            ("Regression to mean", "NOTE", "不是按极端基线选组的 pre-post 设计。"),
            ("Survivorship bias", "CAUTION", "5-core 只覆盖活跃用户和商品。"),
            ("Look-elsewhere effect", "CAUTION", "多子集仅作诊断，验收只使用预注册主/非劣指标。"),
            ("Garden of forking paths", "CAUTION", "锁定模型、种子、阈值后才允许读取测试集。"),
            ("Correlation != causation", "CAUTION", "离线观察评测只支持关联性迁移效果表述。"),
            ("Reverse causality", "NOTE", "不对用户反馈与市场属性做方向因果声明。"),
        ]
    lines.extend(["", "### Fallacy Scan", "", "- Coverage: 11/11 checked", "",
                  "| Fallacy | Severity | Detail |", "|---|---|---|"])
    lines.extend(f"| {name} | {severity} | {detail} |"
                 for name, severity, detail in context)


def markdown_report(result):
    if "screening_version" in result:
        lines = [
            "## Material Passport", "",
            "- Origin Skill: experiment-agent", "- Origin Mode: validate",
            "- Verification Status: ANALYZED", "- Version Label: off_screening_v1",
            "", "## OFF Mechanism Screening", "",
            f"- Default residual: `{result['default_residual']}`",
            f"- Boundary: {result['interpretation']}", "",
            "| Comparison | Mean NDCG@10 difference | Sample SD | Paired dz |",
            "|---|---:|---:|---:|",
        ]
        for name, value in result["comparisons"].items():
            effect = value["paired_effect_dz"]
            effect_text = "N/A" if effect is None else f"{effect:.4f}"
            lines.append(
                f"| {name} | {value['mean_difference']:.6f} | "
                f"{value['difference_sample_sd']:.6f} | {effect_text} |"
            )
        append_descriptive_tables(lines, result["variants"])
        lines.extend([
            "", "### B3-B0 Diagnostic Differences", "",
            "| Subset | Market-macro NDCG@10 difference | Sample SD | Paired dz |",
            "|---|---:|---:|---:|",
        ])
        for subset in REPORT_SUBSETS:
            value = result["diagnostic_comparisons"][
                f"{subset}.macro.ndcg@10"
            ]
            effect = value["paired_effect_dz"]
            effect_text = "N/A" if effect is None else f"{effect:.4f}"
            lines.append(
                f"| {subset} | {value['mean_difference']:.6f} | "
                f"{value['difference_sample_sd']:.6f} | {effect_text} |"
            )
        append_coverage_table(lines, result["coverage"])
        append_fallacy_scan(lines, screening_mode=True)
        lines.extend([
            "", "### Reproducibility", "",
            "- Method: locked manifest/data/source hash audit; no independent full rerun.",
            "- Verdict: CANNOT_VERIFY for independent reproducibility.",
            "- CAUTION: OFF uses real product metadata with simulated users and feedback.",
        ])
        return "\n".join(lines) + "\n"
    lines = [
        "## Material Passport", "",
        "- Origin Skill: experiment-agent", "- Origin Mode: validate",
        "- Verification Status: ANALYZED", "- Version Label: validation_v1",
        "", "## Validation Report", "",
        f"- Default residual: `{result['default_residual']}`",
        f"- Verdict: {result['interpretation']}", "",
        "| Comparison | Mean difference | 95% CI | Paired dz |",
        "|---|---:|---:|---:|",
    ]
    for name, value in result["comparisons"].items():
        lo, hi = value["bootstrap_95_ci"]
        effect = value["paired_effect_dz"]
        effect_text = "N/A" if effect is None else f"{effect:.4f}"
        lines.append(f"| {name} | {value['mean_difference']:.6f} | "
                     f"[{lo:.6f}, {hi:.6f}] | {effect_text} |")
    lines.extend([
        "", "### Independent Baseline Comparisons", "",
        "| Comparison | Mean NDCG@10 difference | 95% CI | Paired dz |",
        "|---|---:|---:|---:|",
    ])
    for name, value in result["baseline_comparisons"].items():
        lo, hi = value["bootstrap_95_ci"]
        effect = value["paired_effect_dz"]
        effect_text = "N/A" if effect is None else f"{effect:.4f}"
        lines.append(
            f"| {name} | {value['mean_difference']:.6f} | "
            f"[{lo:.6f}, {hi:.6f}] | {effect_text} |"
        )
    append_descriptive_tables(
        lines, result["descriptives"], variant_order=REQUIRED_VARIANTS,
        title="ACMR B0-B3 Metrics",
    )
    append_descriptive_tables(
        lines, result["baseline_descriptives"],
        variant_order=REQUIRED_BASELINES, title="Independent Baseline Metrics",
    )
    append_coverage_table(lines, result["coverage"])
    append_fallacy_scan(lines, screening_mode=False)
    lines.extend(["", "### Reproducibility", "",
                  "- Method: manifest/data/source hash protocol audit; no full five-seed rerun.",
                  "- Verdict: CANNOT_VERIFY until an independent rerun reproduces the locked results.",
                  "- CAUTION: five paired seeds yield descriptive intervals, not a universal significance claim.",
                  "- CAUTION: XMarket supports market-transfer claims, not product-origin causal claims."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="结果清单路径或 glob")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--mode", choices=["final", "screening"], default="final")
    parser.add_argument("--output", default="research/validation_report.json")
    parser.add_argument("--markdown", default="research/VALIDATION_REPORT.md")
    args = parser.parse_args()
    if args.bootstrap <= 0:
        parser.error("--bootstrap 必须为正整数")
    runs = load_runs(args.results)
    result = (acceptance(runs, args.bootstrap)
              if args.mode == "final" else screening(runs))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path(args.markdown).write_text(markdown_report(result), encoding="utf-8")
    print(result["interpretation"])


if __name__ == "__main__":
    main()
