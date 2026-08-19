# -*- coding: utf-8 -*-
"""统一的全目录排序评测、市场资格预注册与诊断子集。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import torch


@dataclass
class EvaluationReport:
    """同时携带选定聚合指标、两种聚合结果和覆盖分母。"""

    metrics: Dict[str, float]
    micro: Dict[str, float]
    macro: Dict[str, float]
    per_market: Dict[int, Dict[str, float]]
    coverage: Dict[str, object]
    per_user: Dict[int, Dict[str, float]]


def filter_user_positives(
    user_pos: Mapping[int, np.ndarray], item_mask: np.ndarray
) -> Dict[int, np.ndarray]:
    """按商品布尔掩码生成新的用户正例集合，不读取任何测试外信息。"""
    item_mask = np.asarray(item_mask, dtype=bool)
    out = {}
    for user, items in user_pos.items():
        kept = np.asarray(items, dtype=np.int64)
        kept = kept[item_mask[kept]]
        if len(kept):
            out[int(user)] = kept
    return out


def diagnostic_subsets(dataset, user_pos: Mapping[int, np.ndarray]):
    """构造缺图、真实多语和长尾诊断集合；所有分组仅依赖训练/元数据。"""
    n_items = int(dataset.n_items)
    observed = np.asarray(
        getattr(dataset, "item_image_observed", dataset.item_image_mask),
        dtype=bool,
    )
    valid = np.asarray(
        getattr(dataset, "item_text_valid", np.ones_like(dataset.item_text_lang)),
        dtype=bool,
    )
    fallback = np.asarray(
        getattr(dataset, "item_text_is_fallback", np.zeros_like(valid)),
        dtype=bool,
    )
    langs = np.asarray(dataset.item_text_lang)
    genuine_multilingual = np.zeros(n_items, dtype=bool)
    for item in range(n_items):
        keep = valid[item] & ~fallback[item]
        genuine_multilingual[item] = len(np.unique(langs[item][keep])) >= 2

    degree = np.asarray(
        getattr(dataset, "item_train_degree", np.zeros(n_items)), dtype=np.int64
    )
    positive_degree = degree[degree > 0]
    threshold = int(np.quantile(positive_degree, 0.2)) if len(positive_degree) else 0
    long_tail = (degree > 0) & (degree <= threshold)
    masks = {
        "missing_image": ~observed,
        "genuine_multilingual": genuine_multilingual,
        "long_tail": long_tail,
    }
    return {name: filter_user_positives(user_pos, mask)
            for name, mask in masks.items()}


def _market_ids(dataset, users):
    if hasattr(dataset, "user_country"):
        return np.asarray(dataset.user_country, dtype=np.int64)[users]
    return np.zeros(len(users), dtype=np.int64)


def _candidate_base_mask(dataset, user, candidate_mask):
    allowed = candidate_mask.copy()
    market_mask = getattr(dataset, "item_market_mask", None)
    if market_mask is not None:
        market = int(dataset.user_country[int(user)])
        if market < 0 or market >= len(market_mask):
            raise ValueError(f"用户 {user} 的市场编号 {market} 越界")
        allowed &= np.asarray(market_mask[market], dtype=bool)
    return allowed


def _prepare_protocol(dataset, users, user_pos, topks, candidate_items,
                      exclude_user_pos):
    n_items = int(dataset.n_items)
    candidate_mask = np.ones(n_items, dtype=bool)
    if candidate_items is not None:
        candidate_mask[:] = False
        candidate_items = np.asarray(candidate_items, dtype=np.int64)
        if ((candidate_items < 0) | (candidate_items >= n_items)).any():
            raise ValueError("candidate_items 含越界商品")
        candidate_mask[candidate_items] = True

    available = np.zeros(len(users), dtype=np.int64)
    markets = _market_ids(dataset, users)
    for row, user in enumerate(users):
        allowed = _candidate_base_mask(dataset, int(user), candidate_mask)
        seen = np.asarray(exclude_user_pos.get(int(user), ()), dtype=np.int64)
        if len(seen):
            allowed[seen] = False
        gt = np.asarray(user_pos[int(user)], dtype=np.int64)
        if len(gt) and not allowed[gt].all():
            bad = gt[~allowed[gt]][:5].tolist()
            raise ValueError(
                f"用户 {int(user)} 的 held-out 正例不在有效候选目录中：{bad}"
            )
        available[row] = int(allowed.sum())

    eligible = {}
    market_values = sorted(map(int, np.unique(markets)))
    for k in sorted(set(map(int, topks))):
        eligible[k] = [
            market for market in market_values
            if bool((available[markets == market] >= k).all())
        ]
    return candidate_mask, markets, available, eligible


def _metric_values(ranked, truth, k):
    hits = np.isin(ranked[:k], truth).astype(np.float64)
    recall = float(hits.sum() / len(truth))
    hit = float(hits.sum() > 0)
    dcg = float((hits / np.log2(np.arange(2, k + 2))).sum())
    ideal = float((1.0 / np.log2(
        np.arange(2, min(len(truth), k) + 2)
    )).sum())
    return recall, dcg / ideal if ideal else 0.0, hit


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def evaluate_report(model, batch, dataset, topks, user_pos_dict, *,
                    max_users=None, bs=512, candidate_items=None,
                    exclude_user_pos=None, aggregation="macro"):
    """在预注册的合格市场上评测，不按用户动态缩小 K。

    冷启动调用方应保留 ``candidate_items=None``，使冷商品与完整目标市场目录
    共同排名。该参数只为显式的受限目录研究保留。
    """
    if aggregation not in {"micro", "macro"}:
        raise ValueError("aggregation 必须是 micro 或 macro")
    ks = sorted(set(map(int, topks)))
    if not ks or ks[0] <= 0:
        raise ValueError("topks 必须是非空正整数集合")
    if int(dataset.n_items) < ks[-1]:
        raise ValueError(f"完整商品数 {dataset.n_items} 小于最大 K={ks[-1]}")
    if exclude_user_pos is None:
        exclude_user_pos = dataset.train_user_pos

    all_users = np.asarray(sorted(map(int, user_pos_dict.keys())), dtype=np.int64)
    if len(all_users) == 0:
        zero = {f"{name}@{k}": 0.0 for k in ks
                for name in ("recall", "ndcg", "hit")}
        empty_eligibility = {
            "markets": [],
            "excluded_markets": [],
            "users": 0,
            "excluded_users": 0,
            "positives": 0,
            "excluded_positives": 0,
            "user_fraction": 0.0,
            "positive_fraction": 0.0,
            "minimum_candidates": 0,
        }
        coverage = {
            "total_users": 0,
            "sampled_users": 0,
            "total_positives": 0,
            "eligible": {str(k): dict(empty_eligibility) for k in ks},
        }
        return EvaluationReport(zero, dict(zero), dict(zero), {}, coverage, {})

    candidate_mask, all_markets, available, eligible = _prepare_protocol(
        dataset, all_users, user_pos_dict, ks, candidate_items, exclude_user_pos
    )

    # 资格由完整受评人群预注册，抽样只降低计算量，不改变市场纳入规则。
    users = all_users
    if max_users is not None and len(users) > int(max_users):
        users = np.sort(np.random.default_rng(0).choice(
            users, int(max_users), replace=False
        ))
    row_of = {int(u): idx for idx, u in enumerate(all_users)}
    sampled_rows = np.asarray([row_of[int(u)] for u in users], dtype=np.int64)
    markets = all_markets[sampled_rows]

    was_training = bool(model.training)
    model.eval()
    ranked_by_user = {}
    try:
        with torch.no_grad():
            user_e, item_e = model.get_embeddings(batch)
            item_cache = None
            if hasattr(model, "precompute_item_tables"):
                item_cache = model.precompute_item_tables(batch, item_e)
            market_t = getattr(dataset, "item_market_mask", None)
            market_t = (torch.as_tensor(market_t, dtype=torch.bool,
                                        device=item_e.device)
                        if market_t is not None else None)
            candidate_t = torch.as_tensor(
                candidate_mask, dtype=torch.bool, device=item_e.device
            )

            for start in range(0, len(users), bs):
                ub = users[start:start + bs]
                ut = torch.as_tensor(ub, dtype=torch.long, device=item_e.device)
                if item_cache is not None and hasattr(model, "full_score_cached"):
                    scores = model.full_score_cached(
                        batch, user_e, item_e, ut, item_cache
                    )
                else:
                    scores = model.full_score(batch, user_e, item_e, ut)
                allowed = candidate_t.unsqueeze(0).expand_as(scores).clone()
                if market_t is not None:
                    market_ids = torch.as_tensor(
                        dataset.user_country[ub], dtype=torch.long,
                        device=scores.device,
                    )
                    allowed &= market_t[market_ids]
                for row, user in enumerate(ub):
                    seen = exclude_user_pos.get(int(user))
                    if seen is not None and len(seen):
                        allowed[row, torch.as_tensor(
                            seen, dtype=torch.long, device=scores.device
                        )] = False
                scores = scores.masked_fill(~allowed, float("-inf"))
                ranked = torch.topk(scores, ks[-1], dim=-1).indices.cpu().numpy()
                ranked_by_user.update({int(u): ranked[row]
                                       for row, u in enumerate(ub)})
    finally:
        model.train(was_training)

    by_metric = {"micro": defaultdict(list), "market": defaultdict(lambda: defaultdict(list))}
    per_user = {int(u): {"market": int(m)} for u, m in zip(users, markets)}
    for k in ks:
        eligible_markets = set(eligible[k])
        for user, market in zip(users, markets):
            if int(market) not in eligible_markets:
                continue
            truth = np.asarray(user_pos_dict[int(user)], dtype=np.int64)
            values = _metric_values(ranked_by_user[int(user)], truth, k)
            for name, value in zip(("recall", "ndcg", "hit"), values):
                key = f"{name}@{k}"
                by_metric["micro"][key].append(value)
                by_metric["market"][int(market)][key].append(value)
                per_user[int(user)][key] = value

    micro = {f"{name}@{k}": _mean(by_metric["micro"][f"{name}@{k}"])
             for k in ks for name in ("recall", "ndcg", "hit")}
    per_market = {
        market: {f"{name}@{k}": _mean(values[f"{name}@{k}"])
                 for k in ks for name in ("recall", "ndcg", "hit")}
        for market, values in by_metric["market"].items()
    }
    macro = {}
    for k in ks:
        for name in ("recall", "ndcg", "hit"):
            key = f"{name}@{k}"
            values = [per_market[m][key] for m in eligible[k]
                      if m in per_market and by_metric["market"][m][key]]
            macro[key] = _mean(values)

    names = getattr(dataset, "countries", None)
    total_positives = sum(len(user_pos_dict[int(u)]) for u in all_users)
    coverage = {"total_users": int(len(all_users)),
                "sampled_users": int(len(users)),
                "total_positives": int(total_positives), "eligible": {}}
    for k in ks:
        market_set = set(eligible[k])
        row_mask = np.isin(all_markets, list(market_set))
        covered_users = all_users[row_mask]
        positives = sum(len(user_pos_dict[int(u)]) for u in covered_users)
        market_label = lambda m: names[m] if names is not None and m < len(names) else str(m)
        all_market_set = set(map(int, np.unique(all_markets)))
        coverage["eligible"][str(k)] = {
            "markets": [market_label(m) for m in sorted(market_set)],
            "excluded_markets": [market_label(m) for m in sorted(all_market_set - market_set)],
            "users": int(len(covered_users)),
            "excluded_users": int(len(all_users) - len(covered_users)),
            "positives": int(positives),
            "excluded_positives": int(total_positives - positives),
            "user_fraction": float(len(covered_users) / len(all_users)),
            "positive_fraction": float(positives / total_positives) if total_positives else 0.0,
            "minimum_candidates": int(available[row_mask].min()) if row_mask.any() else 0,
        }
    selected = macro if aggregation == "macro" else micro
    return EvaluationReport(dict(selected), micro, macro, per_market, coverage, per_user)


def evaluate(model, batch, dataset, topks, user_pos_dict, max_users=None, bs=512,
             candidate_items=None, exclude_user_pos=None, aggregation="macro",
             return_report=False):
    """向后兼容的指标入口；需要覆盖率时设置 ``return_report=True``。"""
    report = evaluate_report(
        model, batch, dataset, topks, user_pos_dict,
        max_users=max_users, bs=bs, candidate_items=candidate_items,
        exclude_user_pos=exclude_user_pos, aggregation=aggregation,
    )
    return report if return_report else report.metrics
