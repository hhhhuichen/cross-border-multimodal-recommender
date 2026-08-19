# -*- coding: utf-8 -*-
"""训练 / 评测入口。

用法:
    python train.py                      # 跑完整模型
    python train.py --no-kg              # 消融：去掉知识图谱
    python train.py --no-mm              # 消融：去掉多模态
    python train.py --no-align --no-adv  # 消融：去掉跨语言对齐 / 语种对抗
"""
import argparse
from dataclasses import asdict
import os
import sys
import time

# Windows 下控制台/管道常是 GBK 编码，中文日志会乱码，统一成 UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from checkpoint_utils import load_checkpoint, save_checkpoint
from config import Config, resolved_split_seed, validate_config
from data_utils import (
    build_dataset, BPRSampler, KGSampler,
    sample_degree_sensitive_interactions,
)
from evaluation import evaluate, diagnostic_subsets
from experiment_utils import build_run_manifest, write_json_atomic
try:
    from baselines import build_model
except ImportError:
    build_model = None
from model import ACMR


def select_validation_metrics(
        target, overall, cross, cold, *, overall_available, cross_available):
    """按声明的 validation 目标选 checkpoint；目标缺失时禁止静默降级。"""
    if target == "cross":
        if not cross_available:
            raise ValueError("跨境验证集为空，无法使用 selection_target=cross")
        return cross
    if target == "cold":
        if cold is None:
            raise ValueError("冷启动验证集不足，无法使用 selection_target=cold")
        return cold
    if target == "overall":
        if not overall_available:
            raise ValueError("整体验证集为空，无法使用 selection_target=overall")
        return overall
    raise ValueError(f"未知 selection_target={target!r}")


# --------------------------------------------------------------------------- #
# 张量准备
# --------------------------------------------------------------------------- #
def make_batch(dataset, device):
    t = lambda x, dt=torch.long: torch.as_tensor(x, dtype=dt, device=device)
    text_shape = dataset.item_text_lang.shape
    batch = {
        "edge_index": t(dataset.edge_index),
        "edge_type": t(dataset.edge_type),
        "item_text": t(dataset.item_text_feat, torch.float),
        "item_image": t(dataset.item_image_feat, torch.float),
        "item_image_mask": t(dataset.item_image_mask, torch.float),
        "item_lang": t(dataset.item_text_lang),
        "user_country": t(dataset.user_country),
        "item_country": t(dataset.item_country),
        "cold_items": t(dataset.cold_items),
        "zero_train_items": t(getattr(dataset, "zero_train_items",
                                       dataset.cold_items)),
        "mm_item_edge_index": t(dataset.mm_item_edge_index),
        "mm_item_edge_weight": t(dataset.mm_item_edge_weight, torch.float),
    }
    optional = {
        "item_text_valid": (np.ones(text_shape, dtype=bool), torch.bool),
        "item_text_is_fallback": (np.zeros(text_shape, dtype=bool), torch.bool),
        "item_text_content_hash": (
            np.full(text_shape, -1, dtype=np.int64), torch.long
        ),
        "item_text_source": (np.zeros(text_shape, dtype=np.int64), torch.long),
        "item_text_role": (np.zeros(text_shape, dtype=np.int64), torch.long),
        "item_text_dedup_mask": (np.ones(text_shape, dtype=bool), torch.bool),
        "item_text_pair_valid": (
            np.zeros(text_shape + (text_shape[1],), dtype=bool), torch.bool
        ),
        "item_text_language_confidence": (
            np.ones(text_shape, dtype=np.float32), torch.float
        ),
        "item_text_genuine": (np.ones(text_shape, dtype=bool), torch.bool),
        "item_text_market": (np.full(text_shape, -1, dtype=np.int64), torch.long),
        "item_image_available": (dataset.item_image_mask, torch.float),
        "item_image_observed": (dataset.item_image_mask, torch.float),
        "item_image_completion_confidence": (
            np.zeros(dataset.n_items, dtype=np.float32), torch.float
        ),
        "item_train_degree": (
            np.zeros(dataset.n_items, dtype=np.int64), torch.long
        ),
    }
    aliases = {
        "item_text_is_fallback": ("item_text_fallback",),
        "item_text_language_confidence": ("item_text_lang_confidence",),
    }
    for name, (default, dtype) in optional.items():
        dataset_name = {
            "item_text_content_hash": "item_text_content_hash_id",
            "item_text_source": "item_text_source_id",
            "item_text_role": "item_text_role_id",
        }.get(name, name)
        value = getattr(dataset, dataset_name, None)
        if value is None:
            for alias in aliases.get(name, ()):
                value = getattr(dataset, alias, None)
                if value is not None:
                    break
        batch[name] = t(default if value is None else value, dtype)
    # 核心模型兼容短别名；它们引用同一张量，不产生额外显存。
    batch["item_text_fallback"] = batch["item_text_is_fallback"]
    batch["item_text_hash"] = batch["item_text_content_hash"]
    batch["item_text_lang_confidence"] = batch["item_text_language_confidence"]
    return batch


def iter_cf_macro_batches(sampler, macro_batch_size):
    """拼接普通采样批次，使每个输出只触发一次完整图传播。"""
    users, pos, neg, size = [], [], [], 0
    for u, i, j in sampler:
        users.append(u); pos.append(i); neg.append(j)
        size += len(u)
        if size >= macro_batch_size:
            yield np.concatenate(users), np.concatenate(pos), np.concatenate(neg)
            users, pos, neg, size = [], [], [], 0
    if size:
        yield np.concatenate(users), np.concatenate(pos), np.concatenate(neg)


def refresh_model(model, batch):
    """ACMR 使用关系注意力缓存，标准基线无需该步骤。"""
    refresh = getattr(model, "refresh_attention", None)
    if refresh is not None:
        refresh(batch)


def report_payload(report):
    if report is None:
        return None
    return {
        "selected_aggregation": report.metrics,
        "micro": report.micro,
        "macro": report.macro,
        "per_market": report.per_market,
        "coverage": report.coverage,
        "per_user": report.per_user,
    }


def print_report(label, report):
    print(f"   [{label}] " + " ".join(
        f"{key}={value:.4f}" for key, value in report.metrics.items()
    ))
    for k, coverage in report.coverage["eligible"].items():
        print(f"      @{k} markets={coverage['markets']} "
              f"excluded={coverage['excluded_markets']} "
              f"coverage={coverage['user_fraction']:.3f}/"
              f"{coverage['positive_fraction']:.3f}")


def make_training_graph_batch(full_batch, dataset, cfg, rng):
    """构造本 epoch 的训练传播图；held-out 评测始终继续使用 full_batch。"""
    m = cfg.model
    if not m.use_degree_sensitive_pruning or m.interaction_prune_ratio <= 0.0:
        return full_batch

    keep_pair = sample_degree_sensitive_interactions(
        dataset.train_pairs, m.interaction_prune_ratio, rng
    )
    n_interactions = len(dataset.train_pairs)
    n_edges = full_batch["edge_index"].size(1)
    if n_edges < 2 * n_interactions:
        raise ValueError("CKG 边顺序不满足交互正向/反向成对布局")
    keep_pair_t = torch.as_tensor(
        keep_pair, dtype=torch.long, device=full_batch["edge_index"].device
    )
    tail = torch.arange(
        2 * n_interactions, n_edges, dtype=torch.long,
        device=keep_pair_t.device,
    )
    positions = torch.cat([
        keep_pair_t, keep_pair_t + n_interactions, tail
    ])
    train_batch = dict(full_batch)
    train_batch["edge_index"] = full_batch["edge_index"][:, positions]
    train_batch["edge_type"] = full_batch["edge_type"][positions]
    return train_batch


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    run_started = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["bpr_mf", "lightgcn", "vbpr", "acmr"],
                    default=None, help="独立基线或 ACMR")
    ap.add_argument("--residual",
                    choices=["none", "fused", "decoupled", "market_reliable"],
                    default=None, help="B0/B1/B2/B3 个性化内容残差")
    ap.add_argument("--no-kg", action="store_true")
    ap.add_argument("--no-mm", action="store_true")
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--no-adv", action="store_true")
    ap.add_argument("--no-market", action="store_true")
    ap.add_argument("--no-country-graph", action="store_true")
    ap.add_argument("--no-mm-complete", action="store_true")
    ap.add_argument("--no-mm-graph", action="store_true",
                    help="关闭冻结多模态 item-item 图")
    ap.add_argument("--use-mm-graph", action="store_true",
                    help="启用实验性的冻结多模态 item-item 图")
    ap.add_argument("--no-collab-cl", action="store_true",
                    help="关闭 CLCRec 式协同-内容对比损失")
    ap.add_argument("--use-collab-cl", action="store_true",
                    help="启用实验性的 CLCRec 式协同-内容对比损失")
    ap.add_argument("--no-cold", action="store_true",
                    help="关闭冷启动 ID dropout（保留集与冷启动评测仍在）")
    ap.add_argument(
        "--align-source", choices=["parallel"], default=None,
        help="仅允许 schema-v2 真实平行文本；保留该参数供旧脚本显式固定协议",
    )
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--kg-gamma", type=float, default=None,
                    help="KG 边注意力叠加系数 kg_att_scale")
    ap.add_argument("--mm-graph-beta", type=float, default=None,
                    help="冻结 item 图残差传播强度，取值 [0,1]")
    ap.add_argument("--lambda-collab-cl", type=float, default=None,
                    help="协同-内容对比损失权重，必须为非负数")
    ap.add_argument("--use-edge-pruning", action="store_true",
                    help="启用 FREEDOM 式度敏感训练交互边剪枝")
    ap.add_argument("--interaction-prune-ratio", type=float, default=None,
                    help="训练传播图的交互边剪枝比例，取值 [0,1)")
    ap.add_argument("--market-sampling-alpha", type=float, default=None,
                    help="市场采样平滑指数 [0,1]；0 等市场，1 原始交互分布")
    ap.add_argument("--seed", type=int, default=None,
                    help="兼容入口：同时设置 split-seed 和 train-seed")
    ap.add_argument("--split-seed", type=int, default=None,
                    help="仅控制数据生成、冷商品和 train/val/test 划分")
    ap.add_argument("--train-seed", type=int, default=None,
                    help="仅控制初始化、采样和训练随机性")
    ap.add_argument("--eval-k", type=int, nargs="+", default=None,
                    help="固定评测截断，例如 --eval-k 10 20")
    ap.add_argument("--market-aggregate", choices=["micro", "macro"], default=None)
    ap.add_argument("--cf-macro-batch-size", type=int, default=None)
    ap.add_argument("--selection-target", choices=["overall", "cross", "cold"],
                    default=None, help="用总体或跨境验证 Recall 选择 checkpoint")
    ap.add_argument("--eval-max-users", type=int, default=None,
                    help="每组评测的用户上限；0 表示全量，默认 2000")
    ap.add_argument("--data", choices=["synthetic", "off", "xmarket"], default=None,
                    help="数据源：synthetic / off / xmarket")
    ap.add_argument("--off-dir", default=None,
                    help="OFF processed 目录；相对路径按项目根解析")
    ap.add_argument("--xmarket-dir", default=None,
                    help="XMarket schema-v2 processed 目录")
    ap.add_argument("--ckpt-path", default=None,
                    help="checkpoint 保存路径；用于临时验证时避免覆盖正式权重")
    ap.add_argument("--result-path", default=None,
                    help="实验指标、覆盖率与源码/数据哈希 JSON")
    args = ap.parse_args()

    cfg = Config()
    if args.model:    cfg.train.model = args.model
    if args.residual: cfg.model.residual = args.residual
    if args.no_kg:     cfg.model.use_kg = False
    if args.no_mm:     cfg.model.use_multimodal = False
    if args.no_align:  cfg.model.use_cross_lingual_align = False
    if args.no_adv:    cfg.model.use_lang_adversarial = False
    if args.no_market: cfg.model.use_market_gate = False
    if args.no_country_graph: cfg.model.use_country_graph = False
    if args.no_mm_complete:   cfg.model.use_modality_completion = False
    if args.no_mm_graph:      cfg.model.use_mm_item_graph = False
    if args.no_collab_cl:     cfg.model.use_collab_content_cl = False
    if args.use_mm_graph:     cfg.model.use_mm_item_graph = True
    if args.use_collab_cl:    cfg.model.use_collab_content_cl = True
    if args.use_edge_pruning: cfg.model.use_degree_sensitive_pruning = True
    if args.no_cold:          cfg.model.cold_id_dropout = 0.0
    if args.align_source:     cfg.model.align_source = args.align_source
    if args.data:             cfg.data.source = args.data
    if args.off_dir:          cfg.data.off_dir = args.off_dir
    if args.xmarket_dir:      cfg.data.xmarket_dir = args.xmarket_dir
    if args.ckpt_path:        cfg.train.ckpt_path = args.ckpt_path
    if args.result_path:      cfg.train.result_path = args.result_path
    if args.epochs:    cfg.train.epochs = args.epochs
    if args.layers:    cfg.model.n_gnn_layers = args.layers
    if args.kg_gamma is not None:
        cfg.model.kg_att_scale = args.kg_gamma
    if args.mm_graph_beta is not None:
        if not 0.0 <= args.mm_graph_beta <= 1.0:
            ap.error("--mm-graph-beta 必须在 [0,1] 内")
        cfg.model.mm_graph_beta = args.mm_graph_beta
    if args.lambda_collab_cl is not None:
        if args.lambda_collab_cl < 0.0:
            ap.error("--lambda-collab-cl 必须为非负数")
        cfg.model.lambda_collab_cl = args.lambda_collab_cl
    if args.interaction_prune_ratio is not None:
        if not 0.0 <= args.interaction_prune_ratio < 1.0:
            ap.error("--interaction-prune-ratio 必须在 [0,1) 内")
        cfg.model.interaction_prune_ratio = args.interaction_prune_ratio
    if args.market_sampling_alpha is not None:
        if not 0.0 <= args.market_sampling_alpha <= 1.0:
            ap.error("--market-sampling-alpha 必须在 [0,1] 内")
        cfg.model.market_sampling_alpha = args.market_sampling_alpha
    if args.seed is not None:
        cfg.data.seed = args.seed
        cfg.data.split_seed = args.seed
        cfg.train.train_seed = args.seed
    if args.split_seed is not None:
        cfg.data.split_seed = args.split_seed
    if args.train_seed is not None:
        cfg.train.train_seed = args.train_seed
    if args.eval_k is not None:
        cfg.train.topk = args.eval_k
    if args.market_aggregate is not None:
        cfg.train.market_aggregate = args.market_aggregate
    if args.cf_macro_batch_size is not None:
        cfg.train.cf_macro_batch_size = args.cf_macro_batch_size
    if args.selection_target is not None:
        cfg.train.selection_target = args.selection_target
    elif cfg.data.source == "xmarket":
        # 无商品原产地真值的跨市场数据只能按 overall 选模。
        cfg.train.selection_target = "overall"
    if args.eval_max_users is not None:
        if args.eval_max_users < 0:
            ap.error("--eval-max-users 必须为非负整数")
        cfg.train.eval_max_users = args.eval_max_users

    # 标准基线的数据图只能包含训练交互，不能继承 ACMR 的 KG 边。
    if cfg.train.model in {"bpr_mf", "lightgcn", "vbpr"}:
        cfg.model.use_kg = False
    validate_config(cfg)
    device = torch.device(
        cfg.train.device if torch.cuda.is_available() and cfg.train.device == "cuda"
        else "cpu"
    )
    torch.manual_seed(cfg.train.train_seed)
    np.random.seed(cfg.train.train_seed)

    print(f"[设备] {device}")
    print("[数据] 构建协同知识图谱 ...")
    dataset = build_dataset(cfg)
    split_seed = int(getattr(dataset, "split_seed", resolved_split_seed(cfg)))
    print(f"[实验] model={cfg.train.model} residual={cfg.model.residual} "
          f"split_seed={split_seed} train_seed={cfg.train.train_seed}")

    print(f"  用户 {dataset.n_users} | 商品 {dataset.n_items} | 实体 {dataset.n_entities}")
    print(f"  训练交互 {len(dataset.train_pairs)} | 验证交互 {len(dataset.val_pairs)} | 测试交互 {len(dataset.test_pairs)}")
    print(f"  CKG 边数 {dataset.edge_index.shape[1]} | 关系数 {dataset.n_relations_total}")
    print(f"  冻结多模态 item 图边数 {dataset.mm_item_edge_index.shape[1]}")
    eval_max_users = cfg.train.eval_max_users or None
    if eval_max_users is None:
        print("  评测用户：全量")
    else:
        print(f"  评测用户上限：每组 {eval_max_users}（超出时固定 seed 抽样）")

    batch = make_batch(dataset, device)
    if cfg.train.model == "acmr":
        model = ACMR(dataset, cfg).to(device)
    else:
        if build_model is None:
            raise RuntimeError("baselines.py 不可用，无法构造独立基线")
        model = build_model(cfg.train.model, dataset, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                           weight_decay=cfg.train.weight_decay)

    cf_sampler = BPRSampler(
        dataset, cfg.train.batch_size, cfg.train.train_seed,
        market_alpha=cfg.model.market_sampling_alpha,
    )
    dataset.bpr_sampling_stats = {
        "eligible_pairs": int(len(cf_sampler.pairs)),
        "excluded_users_without_negatives": int(len(cf_sampler.excluded_users)),
        "excluded_pairs_without_negatives": int(cf_sampler.excluded_pair_count),
    }
    if len(cf_sampler.excluded_users):
        print(
            "  BPR 严格负采样排除 "
            f"{len(cf_sampler.excluded_users)} 个目录饱和用户 / "
            f"{cf_sampler.excluded_pair_count} 条正交互；完整边仍参与图传播"
        )
    kg_sampler = (KGSampler(dataset, cfg.train.kg_batch_size,
                            cfg.train.train_seed)
                  if cfg.train.model == "acmr" and cfg.model.use_kg else None)
    T = lambda x: torch.as_tensor(x, dtype=torch.long, device=device)
    rng_aux = np.random.default_rng(cfg.train.train_seed + 7)
    rng_graph = np.random.default_rng(cfg.train.train_seed + 11)

    best, patience, best_metrics = -1.0, 0, None
    os.makedirs(os.path.dirname(cfg.train.ckpt_path) or ".", exist_ok=True)

    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.time()
        model.train()
        train_batch = make_training_graph_batch(
            batch, dataset, cfg, rng_graph
        )
        refresh_model(model, train_batch)
        stat = {"cf": 0.0, "kg": 0.0, "align": 0.0, "ccl": 0.0,
                "adv": 0.0, "mmc": 0.0}
        n_cf = 0

        # 对抗强度按 DANN 的调度逐步升温，避免早期训练不稳
        p = epoch / cfg.train.epochs
        lambd = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        # ---- (a) 推荐主任务 ----
        for users, pos, neg in iter_cf_macro_batches(
                cf_sampler, cfg.train.cf_macro_batch_size):
            u, i, j = T(users), T(pos), T(neg)
            loss = model.cf_loss(train_batch, u, i, j)

            iu = i.unique()
            if iu.numel() > cfg.model.auxiliary_item_batch_size:
                pick = rng_aux.choice(
                    iu.numel(), cfg.model.auxiliary_item_batch_size, replace=False
                )
                iu = iu[T(pick)]
            is_acmr = cfg.train.model == "acmr"
            if (is_acmr and cfg.model.use_multimodal
                    and cfg.model.use_cross_lingual_align):
                l_align = model.alignment_loss(train_batch, iu)
                loss = loss + cfg.model.lambda_align * l_align
                stat["align"] += float(l_align.detach())
            if (is_acmr and cfg.model.use_multimodal
                    and cfg.model.use_collab_content_cl):
                l_ccl = model.collaborative_content_loss(train_batch, iu)
                loss = loss + cfg.model.lambda_collab_cl * l_ccl
                stat["ccl"] += float(l_ccl.detach())
            if (is_acmr and cfg.model.use_multimodal
                    and cfg.model.use_lang_adversarial):
                l_adv = model.adversarial_loss(train_batch, iu, lambd)
                loss = loss + cfg.model.lambda_adv * l_adv
                stat["adv"] += float(l_adv.detach())
            if (is_acmr and cfg.model.use_multimodal
                    and cfg.model.use_modality_completion):
                l_mmc = model.modality_completion_loss(train_batch, iu)
                loss = loss + cfg.model.lambda_mm_complete * l_mmc
                stat["mmc"] += float(l_mmc.detach())

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            stat["cf"] += float(loss.detach())
            n_cf += 1

        # ---- (b) KG 表示学习任务（交替优化）----
        n_kg = 0
        if cfg.train.model == "acmr" and cfg.model.use_kg:
            for h, r, t, tn in kg_sampler:
                l_kg = model.kg_loss(T(h), T(r), T(t), T(tn))
                opt.zero_grad()
                (cfg.model.lambda_kg * l_kg).backward()
                opt.step()
                stat["kg"] += float(l_kg.detach())
                n_kg += 1

        kept_interactions = int(
            ((train_batch["edge_type"] == 0).sum()).item()
        )
        msg = (f"Epoch {epoch:3d} | CF {stat['cf']/max(n_cf,1):.4f}"
               f" | KG {stat['kg']/max(n_kg,1):.4f}"
               f" | Align {stat['align']/max(n_cf,1):.4f}"
               f" | CCL {stat['ccl']/max(n_cf,1):.4f}"
               f" | Adv {stat['adv']/max(n_cf,1):.4f}"
               f" | MMC {stat['mmc']/max(n_cf,1):.4f}"
               f" | UI-Edges {kept_interactions}/{len(dataset.train_pairs)}"
               f" | {time.time()-t0:.1f}s")
        print(msg)

        # ---- (c) 评测 ----
        if epoch % cfg.train.eval_every == 0 or epoch == cfg.train.epochs:
            refresh_model(model, batch)
            valid_report = evaluate(
                model, batch, dataset, cfg.train.topk, dataset.val_user_pos,
                max_users=eval_max_users,
                aggregation=cfg.train.market_aggregate, return_report=True,
            )
            valid = valid_report.metrics
            print_report("验证", valid_report)
            valid_cross_report = evaluate(
                model, batch, dataset, cfg.train.topk,
                dataset.val_user_pos_cross, max_users=eval_max_users,
                aggregation=cfg.train.market_aggregate, return_report=True,
            )
            valid_cross = valid_cross_report.metrics
            print_report("验证-跨境", valid_cross_report)

            valid_cold = None
            valid_cold_report = None
            if dataset.val_user_pos_cold:
                # 冷正例与完整目标市场目录排名；不再限定 cold-only 候选池。
                valid_cold_report = evaluate(
                    model, batch, dataset, cfg.train.topk,
                    dataset.val_user_pos_cold,
                    max_users=eval_max_users,
                    aggregation=cfg.train.market_aggregate,
                    return_report=True,
                )
                valid_cold = valid_cold_report.metrics
                print_report("验证-冷启动", valid_cold_report)

            first_k = str(min(cfg.train.topk))
            has_eligible = lambda report: bool(
                report.coverage["eligible"][first_k]["users"]
            )
            if valid_cold_report is not None and not has_eligible(valid_cold_report):
                valid_cold = None

            selection_metrics = select_validation_metrics(
                cfg.train.selection_target,
                valid,
                valid_cross,
                valid_cold,
                overall_available=has_eligible(valid_report),
                cross_available=has_eligible(valid_cross_report),
            )
            key = selection_metrics[f"ndcg@{min(cfg.train.topk)}"]
            if key > best:
                best, patience = key, 0
                best_metrics = {
                    "overall": valid_report,
                    "cross": valid_cross_report,
                    "cold": valid_cold_report,
                }
                save_checkpoint(
                    cfg.train.ckpt_path, model, dataset, cfg,
                    extra={
                        "epoch": epoch,
                        "selection_target": cfg.train.selection_target,
                        "selection_ndcg": float(key),
                        "market_aggregate": cfg.train.market_aggregate,
                        "evaluation_max_users": cfg.train.eval_max_users,
                    },
                )
            else:
                patience += 1
                if patience >= cfg.train.early_stop_patience:
                    print("早停触发。")
                    break

    print("\n===== 最优结果 =====")
    if best_metrics:
        print("验证:", {k: round(v, 4)
                        for k, v in best_metrics["overall"].metrics.items()})
        print("验证-跨境:", {
            k: round(v, 4) for k, v in best_metrics["cross"].metrics.items()
        })
        if best_metrics["cold"] is not None:
            print("验证-冷启动:", {
                k: round(v, 4) for k, v in best_metrics["cold"].metrics.items()
            })
        load_checkpoint(
            cfg.train.ckpt_path, model, dataset, cfg, map_location=device
        )
        refresh_model(model, batch)
        overall_report = evaluate(
            model, batch, dataset, cfg.train.topk, dataset.test_user_pos,
            max_users=eval_max_users,
            exclude_user_pos=dataset.train_val_user_pos,
            aggregation=cfg.train.market_aggregate, return_report=True,
        )
        cross_report = evaluate(
            model, batch, dataset, cfg.train.topk, dataset.test_user_pos_cross,
            max_users=eval_max_users,
            exclude_user_pos=dataset.train_val_user_pos,
            aggregation=cfg.train.market_aggregate, return_report=True,
        )
        print_report("测试", overall_report)
        print_report("测试-跨境", cross_report)
        cold_report = None
        if dataset.test_user_pos_cold:
            cold_report = evaluate(
                model, batch, dataset, cfg.train.topk,
                dataset.test_user_pos_cold, max_users=eval_max_users,
                exclude_user_pos=dataset.train_val_user_pos,
                aggregation=cfg.train.market_aggregate, return_report=True,
            )
            print_report("测试-冷启动", cold_report)

        subset_reports = {}
        for name, positives in diagnostic_subsets(
                dataset, dataset.test_user_pos).items():
            if positives:
                subset_reports[name] = evaluate(
                    model, batch, dataset, cfg.train.topk, positives,
                    max_users=eval_max_users,
                    exclude_user_pos=dataset.train_val_user_pos,
                    aggregation=cfg.train.market_aggregate,
                    return_report=True,
                )
                print_report(f"测试-{name}", subset_reports[name])

        manifest = build_run_manifest(
            cfg, dataset, model,
            validation={name: report_payload(report)
                        for name, report in best_metrics.items()},
            test={
                "overall": report_payload(overall_report),
                "cross": report_payload(cross_report),
                "cold": report_payload(cold_report),
                "subsets": {name: report_payload(report)
                            for name, report in subset_reports.items()},
            },
            duration_seconds=time.time() - run_started,
        )
        write_json_atomic(cfg.train.result_path, manifest)
        print(f"结果清单已写入 {cfg.train.result_path}")
    print(f"权重已保存至 {cfg.train.ckpt_path}")


if __name__ == "__main__":
    main()
