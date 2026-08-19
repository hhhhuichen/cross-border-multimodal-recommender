# -*- coding: utf-8 -*-
"""
第 4 步：在真实商品全集上生成校准过的用户交互层。

诚实性边界（论文方法学章节的核心）：
  * item_country 是观测值（第 2 步），本脚本绝不触碰；
  * 被模拟的只有用户偏好与购买行为；
  * 两个校准量都来自公开可引用的外部数据，且与模型输入信息源分离：
      - 各市场跨境购买基准率：Lazada CIKM AnalytiCup 2017 实测
        （MY 71.0% / PH 53.3% / SG 78.2%，其余市场用全体池化值 67.3%）；
      - 跨境原产伙伴分布：UN Comtrade 市场×完整原产国进口份额
        （trade_shares.npy，只喂模拟器；模型侧 country_adj 用地理邻接）。
  * 用户口味 = 品类偏好（从真实品类三元组读出），与合成数据生成器同构。

    python simulate_users.py --n-users 6000 --n-inter 120000
"""
import argparse
import json
from collections import defaultdict

import numpy as np
from off_common import DATA_DIR, PROC_DIR

# Lazada CIKM AnalytiCup 2017 实测跨境（international）占比；其余市场用池化值
LAZADA_INTL_RATE = {"MY": 0.710, "PH": 0.533, "SG": 0.782}
POOLED_INTL_RATE = 0.673
TRADE_SCHEMA_VERSION = 2


def align_trade_matrix(trade, trade_meta, market_iso, countries):
    """按显式轴标签重排贸易矩阵，拒绝旧 11x11 隐式列契约。"""
    if trade_meta.get("trade_schema_version") != TRADE_SCHEMA_VERSION:
        raise ValueError(
            "trade_shares 使用旧契约；请重新运行 build_country_adj.py，"
            "生成包含非 ASEAN 原产国列的 v2 权重"
        )
    trade_markets = list(trade_meta.get("markets", ()))
    trade_origins = list(trade_meta.get("trade_origins", ()))
    if trade.shape != (len(trade_markets), len(trade_origins)):
        raise ValueError("trade_shares.npy 形状与 country_adj_meta.json 轴标签不一致")
    try:
        row_take = [trade_markets.index(market) for market in market_iso]
        column_take = [trade_origins.index(origin) for origin in countries]
    except ValueError as exc:
        raise ValueError(
            "trade_shares 缺 processed 数据中的市场或原产国；"
            "商品数据重建后必须重跑 build_country_adj.py"
        ) from exc
    aligned = np.asarray(trade[np.ix_(row_take, column_take)], dtype=np.float64)
    if (aligned.shape != (len(market_iso), len(countries))
            or not np.isfinite(aligned).all() or (aligned < 0).any()):
        raise ValueError("对齐后的 trade_shares 必须是有限非负市场×原产国矩阵")
    row_sums = aligned.sum(axis=1)
    if (row_sums <= 0).any():
        missing = [market_iso[i] for i in np.flatnonzero(row_sums <= 0)]
        raise ValueError(f"trade_shares 存在无贸易覆盖市场: {missing}")
    return aligned / row_sums[:, None]


def origin_probabilities(trade, market, origins, pool_sizes):
    """返回可售外国原产池的概率；零覆盖时显式退回目录规模权重。"""
    origins = np.asarray(origins, dtype=np.int64)
    if origins.ndim != 1 or len(origins) != len(pool_sizes):
        raise ValueError("origins 与 pool_sizes 必须是一一对应的一维序列")
    if not len(origins):
        return np.empty(0, dtype=np.float64), False
    if not 0 <= market < trade.shape[0]:
        raise ValueError("市场索引超出 trade_shares 行轴")
    if (origins < 0).any() or (origins >= trade.shape[1]).any():
        raise ValueError("商品原产国索引超出 trade_shares 列轴")
    weights = np.asarray(trade[market, origins], dtype=np.float64)
    used_catalog_fallback = weights.sum() <= 0
    if used_catalog_fallback:
        weights = np.asarray(pool_sizes, dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("外国原产池无法构造有效采样概率")
    return weights / weights.sum(), used_catalog_fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=6000)
    ap.add_argument("--n-inter", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    meta = json.loads((PROC_DIR / "meta.json").read_text(encoding="utf-8"))
    item_country = np.load(PROC_DIR / "item_country.npy")
    triples = np.load(PROC_DIR / "triples.npy")
    trade = np.load(DATA_DIR / "trade_shares.npy")
    n_items = meta["n_items"]
    countries = meta["countries"]
    n_markets = meta["n_markets"]
    market_iso = list(meta.get("markets", countries[:n_markets]))
    if len(market_iso) != n_markets:
        raise ValueError("meta.json 的 markets 与 n_markets 不一致")

    adj_meta_path = DATA_DIR / "country_adj_meta.json"
    if not adj_meta_path.is_file():
        raise FileNotFoundError(
            "缺 country_adj_meta.json；请重新运行 build_country_adj.py"
        )
    adj_meta = json.loads(adj_meta_path.read_text(encoding="utf-8"))
    trade = align_trade_matrix(
        trade, adj_meta, market_iso=market_iso, countries=countries
    )

    # ---- 商品可购市场（sold_in，关系 5）与品类（关系 1） ----
    country_ent_base = meta["entity_layout"]["countries"][0]
    sold_in = defaultdict(list)                  # market_idx -> [item...]
    for h, r, t in triples:
        if r == 5:
            m = t - country_ent_base
            if 0 <= m < n_markets:
                sold_in[int(m)].append(int(h))
    item_cats = defaultdict(list)
    for h, r, t in triples:
        if r == 1:
            item_cats[int(h)].append(int(t))

    # ---- 用户国别：按各市场在售商品数加权（目录加权，论文里写明） ----
    m_sizes = np.array([len(sold_in[m]) for m in range(n_markets)], dtype=float)
    if not (m_sizes > 0).any():
        raise ValueError("所有 ASEAN 市场目录均为空，无法模拟用户")
    user_country = rng.choice(n_markets, size=args.n_users,
                              p=m_sizes / m_sizes.sum())

    # ---- 用户口味：偏好 1~3 个真实品类 + 少量噪声 ----
    all_cats = sorted({c for cs in item_cats.values() for c in cs})
    cat_pos = {c: k for k, c in enumerate(all_cats)}
    K = len(all_cats)
    item_vec = np.zeros((n_items, 1), dtype=np.int64)   # 主品类
    for i in range(n_items):
        cs = item_cats.get(i)
        item_vec[i] = cat_pos[cs[-1]] if cs else rng.integers(0, max(K, 1))
    item_pop = 0.5 * rng.normal(0, 1, n_items)

    n_pref = rng.integers(1, 4, size=args.n_users)
    user_prefs = [rng.choice(K, size=k, replace=False) if K >= 3 else
                  np.array([0]) for k in n_pref]

    # ---- 每条交互：市场基准率定域内/跨境，贸易份额定目的地，口味定商品 ----
    domestic_pool = {m: [] for m in range(n_markets)}
    foreign_pool_by_origin = {m: defaultdict(list) for m in range(n_markets)}
    for m in range(n_markets):
        for i in sold_in[m]:
            oc = int(item_country[i])
            if oc == m:
                domestic_pool[m].append(i)
            else:
                foreign_pool_by_origin[m][oc].append(i)

    origin_sampling = {}
    zero_trade_fallback_markets = []
    for market in range(n_markets):
        pools = foreign_pool_by_origin[market]
        origins = sorted(origin for origin, pool in pools.items() if pool)
        probabilities, used_fallback = origin_probabilities(
            trade,
            market,
            origins,
            [len(pools[origin]) for origin in origins],
        )
        origin_sampling[market] = (origins, probabilities, used_fallback)
        if used_fallback:
            zero_trade_fallback_markets.append(market_iso[market])

    intl_rate = {m: LAZADA_INTL_RATE.get(market_iso[m], POOLED_INTL_RATE)
                 for m in range(n_markets)}

    def pick(u, pool):
        pool = np.asarray(pool)
        cat_of = item_vec[pool, 0]
        pref = user_prefs[u]
        score = np.where(np.isin(cat_of, pref), 3.0, 0.0) + item_pop[pool] \
            + rng.normal(0, 0.5, len(pool))
        ex = np.exp(score - score.max())
        return int(rng.choice(pool, p=ex / ex.sum()))

    users_out, items_out, n_forced_intl = [], [], 0
    n_catalog_weight_fallback = 0
    per_user = rng.multinomial(args.n_inter, np.full(args.n_users,
                                                     1.0 / args.n_users))
    for u in range(args.n_users):
        m = int(user_country[u])
        fpools = foreign_pool_by_origin[m]
        origins, origin_probs, used_trade_fallback = origin_sampling[m]
        for _ in range(int(per_user[u])):
            go_intl = rng.random() < intl_rate[m]
            if not go_intl and not domestic_pool[m]:
                go_intl, n_forced_intl = True, n_forced_intl + 1
            if go_intl and origins:
                o = origins[int(rng.choice(len(origins), p=origin_probs))]
                items_out.append(pick(u, fpools[o]))
                if used_trade_fallback:
                    n_catalog_weight_fallback += 1
            elif domestic_pool[m]:
                items_out.append(pick(u, domestic_pool[m]))
            else:
                continue
            users_out.append(u)

    pairs = np.unique(np.stack([users_out, items_out], 1), axis=0)
    cross = (user_country[pairs[:, 0]] != item_country[pairs[:, 1]]).mean()
    np.savez(PROC_DIR / "users.npz", user_country=user_country,
             interactions=pairs)
    calib = {"lazada_intl_rate": LAZADA_INTL_RATE,
             "pooled_intl_rate": POOLED_INTL_RATE,
             "n_users": args.n_users, "n_interactions": int(len(pairs)),
             "realized_cross_border_share": float(cross),
             "forced_international": n_forced_intl, "seed": args.seed,
             "markets": market_iso,
             "trade_schema_version": TRADE_SCHEMA_VERSION,
             "trade_origin_count": len(countries),
             "catalog_weight_fallback_selections": n_catalog_weight_fallback,
             "zero_trade_fallback_markets": zero_trade_fallback_markets,
             "empty_catalog_markets": [
                 market_iso[index] for index in np.flatnonzero(m_sizes == 0)
             ]}
    (PROC_DIR / "simulation_meta.json").write_text(
        json.dumps(calib, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"用户 {args.n_users} | 交互 {len(pairs)}（去重后）"
          f" | 实际跨境占比 {cross:.3f}")
    print(f"域内池空而强制跨境的次数: {n_forced_intl}")
    print(f"已写入 {PROC_DIR}/users.npz")


if __name__ == "__main__":
    main()
