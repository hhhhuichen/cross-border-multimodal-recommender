#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""顺序执行锁定的 OFF B0-B3 三组配对机制筛选。"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from research.validate_experiments import SCREENING_SEED_PAIRS


VARIANTS = ("none", "fused", "decoupled", "market_reliable")


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output-dir", default="results/off_screening")
    parser.add_argument("--checkpoint-dir", default="/tmp")
    parser.add_argument("--report-json", default="research/off_screening_report.json")
    parser.add_argument("--report-markdown", default="research/OFF_SCREENING_REPORT.md")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="显式覆盖同名结果和 checkpoint",
    )
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs 必须为正整数")

    output_dir = project_path(args.output_dir)
    checkpoint_dir = project_path(args.checkpoint_dir)
    report_json = project_path(args.report_json)
    report_markdown = project_path(args.report_markdown)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for split_seed, train_seed in SCREENING_SEED_PAIRS:
        for residual in VARIANTS:
            stem = f"off-{split_seed}-{train_seed}-{residual}"
            result = output_dir / f"{stem}.json"
            checkpoint = checkpoint_dir / f"off-{split_seed}-{residual}.pt"
            if not args.overwrite and (result.exists() or checkpoint.exists()):
                raise FileExistsError(
                    f"{result} 或 {checkpoint} 已存在；重跑需显式传 --overwrite"
                )
            command = [
                sys.executable, str(PROJECT_DIR / "train.py"),
                "--data", "off", "--model", "acmr", "--residual", residual,
                "--epochs", str(args.epochs),
                "--split-seed", str(split_seed),
                "--train-seed", str(train_seed),
                "--eval-k", "10", "20", "--market-aggregate", "macro",
                "--eval-max-users", "0",
                "--ckpt-path", str(checkpoint),
                "--result-path", str(result),
            ]
            subprocess.run(command, cwd=PROJECT_DIR, check=True)
            results.append(result)

    subprocess.run([
        sys.executable, str(PROJECT_DIR / "research/validate_experiments.py"),
        "--mode", "screening", *map(str, results),
        "--output", str(report_json), "--markdown", str(report_markdown),
    ], cwd=PROJECT_DIR, check=True)


if __name__ == "__main__":
    main()
