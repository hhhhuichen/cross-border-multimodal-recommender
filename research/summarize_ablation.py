# -*- coding: utf-8 -*-
"""从语义校验后的 checkpoint 重算并汇总配对消融结果。"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from checkpoint_utils import load_checkpoint
from config import Config
from data_utils import build_dataset
from model import ACMR
from train import evaluate, make_batch


SEEDS = (2024, 2025, 2026)
CASES = {
    "baseline": {
        "checkpoint": "audit_postsplit_baseline_{seed}.pt",
        "config": {
            "use_degree_sensitive_pruning": False,
            "market_sampling_alpha": 1.0,
            "use_mm_item_graph": False,
        },
    },
    "degree_sensitive_pruning_rho_0_6": {
        "checkpoint": "audit_postsplit_prune06_{seed}.pt",
        "config": {
            "use_degree_sensitive_pruning": True,
            "interaction_prune_ratio": 0.6,
        },
    },
    "market_sampling_alpha_0_3": {
        "checkpoint": "audit_postsplit_market03_{seed}.pt",
        "config": {"market_sampling_alpha": 0.3},
    },
    "id_item_graph": {
        "checkpoint": "audit_postsplit_idgraph_{seed}.pt",
        "implementation": (
            "positive symmetrized per-modality binary kNN; "
            "ID propagation side branch"
        ),
        "config": {
            "use_mm_item_graph": True,
            "mm_graph_k": 10,
            "mm_graph_layers": 1,
            "mm_graph_beta": 0.1,
        },
    },
}


def configure(case, seed):
    cfg = Config()
    cfg.data.seed = seed
    for name, value in case["config"].items():
        if not hasattr(cfg.model, name):
            raise ValueError(f"未知 ModelConfig 字段：{name}")
        setattr(cfg.model, name, value)
    return cfg


def user_count(user_pos, limit):
    return min(len(user_pos), limit) if limit else len(user_pos)


def evaluate_checkpoint(case, seed, checkpoint_dir, device):
    cfg = configure(case, seed)
    dataset = build_dataset(cfg)
    batch = make_batch(dataset, device)
    model = ACMR(dataset, cfg).to(device)
    checkpoint = checkpoint_dir / case["checkpoint"].format(seed=seed)
    metadata = load_checkpoint(
        checkpoint, model, dataset, cfg, map_location=device
    )
    model.refresh_attention(batch)
    limit = cfg.train.eval_max_users or None

    valid = evaluate(
        model, batch, dataset, cfg.train.topk, dataset.val_user_pos,
        max_users=limit,
    )
    valid_cross = evaluate(
        model, batch, dataset, cfg.train.topk, dataset.val_user_pos_cross,
        max_users=limit,
    )
    valid_cold = evaluate(
        model, batch, dataset, cfg.train.topk, dataset.val_user_pos_cold,
        max_users=limit,
    )
    overall = evaluate(
        model, batch, dataset, cfg.train.topk, dataset.test_user_pos,
        max_users=limit, exclude_user_pos=dataset.train_val_user_pos,
    )
    cross = evaluate(
        model, batch, dataset, cfg.train.topk, dataset.test_user_pos_cross,
        max_users=limit, exclude_user_pos=dataset.train_val_user_pos,
    )
    cold = evaluate(
        model, batch, dataset, cfg.train.topk, dataset.test_user_pos_cold,
        max_users=limit, exclude_user_pos=dataset.train_val_user_pos,
    )
    result = {
        "validation": [
            valid["recall@10"], valid_cross["recall@10"],
            valid_cold["recall@10"],
        ],
        "test": [
            overall["recall@10"], cross["recall@10"], cold["recall@10"],
        ],
        "evaluated_users": {
            "validation": [
                user_count(dataset.val_user_pos, limit),
                user_count(dataset.val_user_pos_cross, limit),
                user_count(dataset.val_user_pos_cold, limit),
            ],
            "test": [
                user_count(dataset.test_user_pos, limit),
                user_count(dataset.test_user_pos_cross, limit),
                user_count(dataset.test_user_pos_cold, limit),
            ],
        },
        "selected_epoch": metadata["extra"]["epoch"],
        "checkpoint": str(checkpoint.relative_to(PROJECT_DIR)),
    }
    del model, batch, dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def summarize(per_seed, baseline=None):
    test = np.asarray([per_seed[str(seed)]["test"] for seed in SEEDS])
    result = {
        "test_mean": test.mean(axis=0).tolist(),
        "test_sample_sd": test.std(axis=0, ddof=1).tolist(),
    }
    if baseline is not None:
        base = np.asarray([baseline[str(seed)]["test"] for seed in SEEDS])
        result["paired_mean_delta"] = (test - base).mean(axis=0).tolist()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=PROJECT_DIR / "research" / "ablation_ckpt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_DIR / "research" / "ablation_results.json",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiments = {}
    for name, case in CASES.items():
        per_seed = {
            str(seed): evaluate_checkpoint(
                case, seed, args.checkpoint_dir.resolve(), device
            )
            for seed in SEEDS
        }
        experiments[name] = {
            "config": case["config"],
            "per_seed": per_seed,
        }
        if "implementation" in case:
            experiments[name]["implementation"] = case["implementation"]

    baseline = experiments["baseline"]["per_seed"]
    for name, experiment in experiments.items():
        experiment.update(summarize(
            experiment["per_seed"], None if name == "baseline" else baseline
        ))
        if name != "baseline":
            experiment["default_enabled"] = False

    payload = {
        "protocol": {
            "data": "synthetic",
            "seeds": list(SEEDS),
            "epochs": 10,
            "warm_item_coverage": (
                "every non-cold validation/test item has a training interaction"
            ),
            "cold_item_pool": (
                "sampled from items with positive interactions; at least one "
                "interacted item remains warm"
            ),
            "cold_evaluation_candidates": (
                "complete target-market catalog; training positives excluded "
                "and held-out positives always retained"
            ),
            "evaluation_max_users_per_group": 2000,
            "user_sampling": "fixed NumPy seed 0 without replacement when capped",
            "checkpoint_selection": "cross validation Recall@10",
            "test_policy": (
                "training evaluates test once after checkpoint selection; "
                "the summarizer only recomputes fixed-checkpoint metrics and "
                "never selects or tunes"
            ),
            "metrics": [
                "overall Recall@10", "cross-border Recall@10",
                "cold-start Recall@10",
            ],
            "standard_deviation": "sample SD",
        },
        "experiments": experiments,
        "interpretation": (
            "Synthetic data do not support enabling any literature-inspired "
            "switch by default: no candidate improves overall, cross-border, "
            "and cold-start Recall@10 together. These are engineering checks, "
            "not evidence of real-world effectiveness."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    for name, experiment in experiments.items():
        print(name, np.round(experiment["test_mean"], 6).tolist())


if __name__ == "__main__":
    main()
