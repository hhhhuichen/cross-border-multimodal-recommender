# -*- coding: utf-8 -*-
"""
数据层：
  1) 合成一份东盟跨境多语种场景的玩具数据（真实数据接入方式见 README）
  2) 构建协同知识图谱 CKG = 用户-商品交互图 ∪ 多语种知识图谱
  3) BPR 采样器 / KG 三元组采样器
  4) 训练测试划分（额外区分"跨境交互"子集，用于跨境迁移评测）

图建模约定
---------
节点编号统一到一个空间：
    实体节点  : [0, n_entities)          其中 [0, n_items) 是与商品对齐的实体
    用户节点  : [n_entities, n_entities + n_users)
关系编号：
    0                 : interact (user -> item)
    1 .. R            : KG 正向关系
    R+1 .. 2R+1       : 上述关系的反向关系
"""
import numpy as np
from collections import defaultdict

from data_contract import (
    SCHEMA_VERSION,
    alignment_pair_mask,
    dedup_mask,
    runtime_image_metadata,
    schema_descriptor,
    text_content_hash,
)


def _configured_seed(data_cfg, name):
    """读取新 seed 字段；未配置或为 None 时兼容旧 ``data.seed``。"""
    value = getattr(data_cfg, name, None)
    if value is None:
        value = getattr(data_cfg, "seed", 2026)
    return int(value)


# --------------------------------------------------------------------------- #
# 冻结多模态 item-item 图（LATTICE / FREEDOM 风格）
# --------------------------------------------------------------------------- #
def _row_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, np.finfo(np.float32).eps)


def _modality_knn_pairs(features, valid_mask, k, chunk_size):
    """返回无向 top-k 近邻对及其余弦权重，不构造完整 N x N 矩阵。"""
    valid = np.flatnonzero(np.asarray(valid_mask, dtype=bool))
    if len(valid) <= 1 or k <= 0:
        return {}

    z = _row_normalize(np.asarray(features, dtype=np.float32)[valid])
    k = min(int(k), len(valid) - 1)
    chunk_size = max(1, int(chunk_size))
    pairs = {}

    for start in range(0, len(valid), chunk_size):
        stop = min(start + chunk_size, len(valid))
        sim = z[start:stop] @ z.T
        local_rows = np.arange(stop - start)
        global_rows = np.arange(start, stop)
        sim[local_rows, global_rows] = -np.inf
        nbr = np.argpartition(sim, -k, axis=1)[:, -k:]
        nbr_score = np.take_along_axis(sim, nbr, axis=1)

        for row, (cols, scores) in enumerate(zip(nbr, nbr_score)):
            dst = int(valid[start + row])
            for col, score in zip(cols, scores):
                score = float(score)
                if not np.isfinite(score) or score <= 0.0:
                    continue
                src = int(valid[int(col)])
                key = (src, dst) if src < dst else (dst, src)
                # kNN 通常是有向的；用 union + max 对称化，避免双向命中被重复计权。
                if score > pairs.get(key, 0.0):
                    pairs[key] = score
    return pairs


def build_frozen_mm_item_graph(text_feat, image_feat, image_mask, k=10,
                               text_weight=0.5, chunk_size=512):
    """构建 FREEDOM 式冻结图：各模态 top-k 二值化、分别归一化后融合。"""
    if not 0.0 <= text_weight <= 1.0:
        raise ValueError("mm_graph_text_weight 必须在 [0, 1] 内")

    n_items = int(len(text_feat))
    text_base = np.asarray(text_feat, dtype=np.float32).mean(axis=1)
    text_pairs = _modality_knn_pairs(
        text_base, np.ones(n_items, dtype=bool), k, chunk_size
    ) if text_weight > 0.0 else {}
    image_pairs = _modality_knn_pairs(
        image_feat, np.asarray(image_mask) > 0.5, k, chunk_size
    ) if text_weight < 1.0 else {}

    def normalized_binary_edges(pairs):
        if not pairs:
            return {}
        pair = np.asarray(list(pairs), dtype=np.int64)
        src = np.concatenate([pair[:, 0], pair[:, 1]])
        dst = np.concatenate([pair[:, 1], pair[:, 0]])
        degree = np.bincount(src, minlength=n_items).astype(np.float32)
        denom = np.sqrt(np.maximum(
            degree[src] * degree[dst], np.finfo(np.float32).eps
        ))
        return {
            (int(s), int(d)): float(1.0 / z)
            for s, d, z in zip(src, dst, denom)
        }

    # FREEDOM 先把每个模态的 cosine top-k 图二值化，再分别做
    # D^-1/2 A D^-1/2；cosine 只决定邻接关系，不继续充当传播权重。
    directed_weight = defaultdict(float)
    for edge, weight in normalized_binary_edges(text_pairs).items():
        directed_weight[edge] += text_weight * weight
    for edge, weight in normalized_binary_edges(image_pairs).items():
        directed_weight[edge] += (1.0 - text_weight) * weight

    if not directed_weight:
        return (np.empty((2, 0), dtype=np.int64),
                np.empty(0, dtype=np.float32))
    edge = np.asarray(list(directed_weight), dtype=np.int64)
    weight = np.asarray(list(directed_weight.values()), dtype=np.float32)
    return edge.T, weight


def sample_degree_sensitive_interactions(train_pairs, prune_ratio, rng):
    """按 FREEDOM 的 1/sqrt(deg(u)deg(i)) 概率无放回保留交互边索引。"""
    pairs = np.asarray(train_pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("train_pairs 必须为 (N, 2)")
    if not 0.0 <= prune_ratio < 1.0:
        raise ValueError("interaction_prune_ratio 必须在 [0, 1) 内")
    n_edges = len(pairs)
    if n_edges == 0 or prune_ratio == 0.0:
        return np.arange(n_edges, dtype=np.int64)

    user_deg = np.bincount(pairs[:, 0])
    item_deg = np.bincount(pairs[:, 1])
    weight = 1.0 / np.sqrt(
        user_deg[pairs[:, 0]].astype(np.float64)
        * item_deg[pairs[:, 1]].astype(np.float64)
    )
    weight /= weight.sum()
    n_keep = max(1, int(np.ceil(n_edges * (1.0 - prune_ratio))))
    keep = rng.choice(n_edges, size=n_keep, replace=False, p=weight)
    return np.sort(keep.astype(np.int64, copy=False))


# --------------------------------------------------------------------------- #
# 合成数据
# --------------------------------------------------------------------------- #
class ASEANSyntheticData:
    """生成带国家/语种/多模态特征的玩具数据集，用于跑通全流程。

    设计要点：交互、KG 三元组、图文特征全部派生自同一套「潜在品类结构」——
    每个商品属于一个品类簇，用户偏好 1~2 个品类，交互按 口味相似度 + 流行度
    的 softmax 采样（本国/跨境保持 70/30 偏斜）。这样：
      * 跨境交互与本国交互共享同一口味信号，跨境推荐是可学习的（指标不恒为 0）；
      * KG 携带 品类/品牌/原产国/属性 结构，消融 KG 会真实掉点；
      * 文本/图像特征是 item 潜向量的随机投影 + 噪声，多模态模块有信号可用。
    若交互/特征全用均匀噪声，跨境指标恒为 0，消融表也拉不开差距。
    """

    LATENT_DIM = 16          # 潜在口味空间维度（品类，图文特征可见）
    BRAND_DIM = 8            # 品牌偏好空间维度（只有 KG 可见，图文特征不编码）
    DOMESTIC_RATIO = 0.7     # 本国交互占比

    def __init__(self, cfg):
        self.cfg = cfg
        d = cfg.data
        # 合成数据内容与划分同属数据随机性；改变 train_seed 不应改变样本。
        rng = np.random.default_rng(_configured_seed(d, "split_seed"))
        self.rng = rng
        self.schema_version = SCHEMA_VERSION
        self.data_schema = schema_descriptor()

        self.n_users = d.n_users
        self.n_items = d.n_items
        self.n_entities = d.n_entities
        self.n_relations = d.n_relations
        self.countries = d.countries
        self.languages = d.languages
        self.n_countries = len(d.countries)
        self.n_languages = len(d.languages)
        assert self.n_relations >= 4, "结构化 KG 至少需要 4 种关系（品类/品牌/原产国/属性）"

        # ---- 非商品实体布局：品类 / 品牌 / 国家 / 属性 ----
        spare = self.n_entities - self.n_items
        assert spare >= self.n_countries + 8, "n_entities 需大于 n_items + 国家数 + 8"
        self.n_cats = min(32, max(4, spare // 8))
        self.n_brands = min(300, max(4, spare // 4))
        self.n_brands = min(self.n_brands, spare - self.n_cats - self.n_countries)
        self.n_attrs = spare - self.n_cats - self.n_brands - self.n_countries
        base = self.n_items
        self.cat_entity = base + np.arange(self.n_cats);        base += self.n_cats
        self.brand_entity = base + np.arange(self.n_brands);    base += self.n_brands
        self.country_entity = base + np.arange(self.n_countries); base += self.n_countries
        self.attr_entity = base + np.arange(self.n_attrs)

        # ---- 潜在品类结构：一切信号的共同来源 ----
        k = self.LATENT_DIM
        cat_centers = rng.normal(0, 1, size=(self.n_cats, k))
        self.item_cat = rng.integers(0, self.n_cats, size=self.n_items)
        self.item_latent = cat_centers[self.item_cat] + 0.5 * rng.normal(0, 1, (self.n_items, k))
        self.item_pop = 0.5 * rng.normal(0, 1, self.n_items)    # 商品流行度（长尾）
        # 用户偏好：主品类 + 可选次品类的混合
        prim = rng.integers(0, self.n_cats, size=self.n_users)
        sec = rng.integers(0, self.n_cats, size=self.n_users)
        w = 0.35 * rng.random((self.n_users, 1))
        self.user_pref = ((1 - w) * cat_centers[prim] + w * cat_centers[sec]
                          + 0.3 * rng.normal(0, 1, (self.n_users, k)))

        # ---- 品牌层信号：品牌归属品类，用户对品牌有独立偏好 ----
        # 该信号刻意不编码进图文特征，只通过 KG 的 item->brand 边可达，
        # 用于保证「消融 KG 会真实掉点」：没有 KG 时品牌偏好只能靠共现间接学
        kb = self.BRAND_DIM
        brand_cat = rng.integers(0, self.n_cats, size=self.n_brands)
        self.item_brand = self._choice_by_cat(self.item_cat, self.n_brands, brand_cat)
        self.brand_latent = rng.normal(0, 1, size=(self.n_brands, kb))
        self.user_brand_pref = rng.normal(0, 1, size=(self.n_users, kb))

        # ---- 用户 / 商品的国家归属 ----
        self.user_country = rng.integers(0, self.n_countries, size=self.n_users)
        self.item_country = rng.integers(0, self.n_countries, size=self.n_items)

        # ---- 国家关系图：地理邻接 + 主要贸易通道（对称 0/1，自环由模型侧加）----
        self.country_adj = self._build_country_adj()

        # ---- 交互：70% 本国、30% 跨境，组内按口味 softmax 采样 ----
        self.interactions = self._gen_interactions(d.n_interactions)

        # ---- 多语种 KG 三元组（品类/品牌/原产国/属性 + 噪声填充）----
        self.triples = self._gen_triples(d.n_triples)

        # ---- 多模态特征：item 潜向量的随机投影 + 噪声 ----
        # 文本：每个 item 有 n_text_views 个语种视图（同一商品的不同语言描述），
        # 共享同一语义 base，模拟 LaBSE 的跨语言语义空间
        Wt = rng.normal(0, 1, size=(k, d.text_dim)) / np.sqrt(k)
        text_base = (self.item_latent @ Wt)[:, None, :]                     # (N, 1, text_dim)
        view_noise = rng.normal(0, 1, size=(self.n_items, d.n_text_views, d.text_dim))
        self.item_text_feat = (0.8 * text_base + 0.2 * view_noise).astype(np.float32)
        self.item_text_lang = rng.integers(
            0, self.n_languages, size=(self.n_items, d.n_text_views)
        )
        # 前两视图模拟可信平行商品名；语言数允许时强制异语言，避免合成数据
        # 把同语言对误计为跨语言监督。
        if d.n_text_views >= 2 and self.n_languages >= 2:
            offset = rng.integers(1, self.n_languages, size=self.n_items)
            self.item_text_lang[:, 1] = (
                self.item_text_lang[:, 0] + offset
            ) % self.n_languages
        shape = (self.n_items, d.n_text_views)
        self.item_text_source = np.empty(shape, dtype="<U32")
        self.item_text_role = np.empty(shape, dtype="<U16")
        self.item_text_valid = np.ones(shape, dtype=bool)
        self.item_text_is_fallback = np.zeros(shape, dtype=bool)
        self.item_text_content_hash = np.empty(shape, dtype="<U64")
        self.item_text_language_confidence = np.ones(shape, dtype=np.float32)
        for view in range(d.n_text_views):
            self.item_text_source[:, view] = f"synthetic_view_{view}"
            self.item_text_role[:, view] = (
                "product_name" if view < 2 else "description"
            )
            self.item_text_content_hash[:, view] = [
                text_content_hash(f"synthetic item {item} view {view}")
                for item in range(self.n_items)
            ]
        self.item_text_dedup_mask = dedup_mask(
            self.item_text_content_hash, self.item_text_valid
        )
        self.item_text_genuine, self.item_text_pair_valid = alignment_pair_mask(
            self.item_text_lang,
            self.item_text_role,
            self.item_text_valid,
            self.item_text_is_fallback,
            self.item_text_content_hash,
            self.item_text_dedup_mask,
            self.item_text_language_confidence,
        )
        self.item_text_market = np.full(shape, -1, dtype=np.int64)
        Wv = rng.normal(0, 1, size=(k, d.image_dim)) / np.sqrt(k)
        self.item_image_feat = (
            0.8 * self.item_latent @ Wv
            + 0.6 * rng.normal(0, 1, size=(self.n_items, d.image_dim))
        ).astype(np.float32)
        # 部分商品缺图（跨境场景常见），用掩码标记
        self.item_image_mask = (rng.random(self.n_items) > 0.15).astype(np.float32)
        image_meta = runtime_image_metadata(
            self.item_image_mask > 0.5, self.item_image_mask > 0.5
        )
        self.item_image_available = image_meta["available"]
        self.item_image_observed = image_meta["observed"]
        self.item_image_completion_confidence = image_meta[
            "completion_confidence"
        ]

    # ASEAN-11 的陆地邻接与主要海上贸易通道；接真实数据时可换成
    # 双边贸易额归一化后的加权矩阵，形状 (n_countries, n_countries) 即可
    _COUNTRY_LINKS = [
        ("TH", "MM"), ("TH", "LA"), ("TH", "KH"), ("TH", "MY"),
        ("VN", "LA"), ("VN", "KH"), ("KH", "LA"), ("MM", "LA"),
        ("MY", "SG"), ("MY", "ID"), ("MY", "BN"), ("MY", "PH"),
        ("ID", "SG"), ("ID", "PH"), ("ID", "TL"),
    ]

    def _build_country_adj(self):
        idx = {c: i for i, c in enumerate(self.countries)}
        A = np.zeros((self.n_countries, self.n_countries), dtype=np.float32)
        for a, b in self._COUNTRY_LINKS:
            if a in idx and b in idx:
                A[idx[a], idx[b]] = A[idx[b], idx[a]] = 1.0
        return A

    def _gen_interactions(self, n):
        rng = self.rng
        k = self.LATENT_DIM
        # 每条交互先归属到用户，再决定本国/跨境
        user_of = rng.integers(0, self.n_users, size=n)
        is_dom = rng.random(n) < self.DOMESTIC_RATIO
        dom_cnt = np.zeros(self.n_users, dtype=np.int64)
        cross_cnt = np.zeros(self.n_users, dtype=np.int64)
        np.add.at(dom_cnt, user_of[is_dom], 1)
        np.add.at(cross_cnt, user_of[~is_dom], 1)

        # 口味打分：品类偏好 + 品牌偏好 + 流行度；跨境与本国共享同一口味信号。
        # 品牌权重要足够大：它是「只有 KG 可见」的信号，权重太小会被噪声淹没，
        # 消融 KG 就看不出差距
        taste = (self.user_pref @ self.item_latent.T * (2.0 / np.sqrt(k))
                 + self.user_brand_pref @ self.brand_latent[self.item_brand].T
                   * (2.5 / np.sqrt(self.BRAND_DIM))
                 + self.item_pop)
        same_pool = [np.where(self.item_country == c)[0] for c in range(self.n_countries)]
        diff_pool = [np.where(self.item_country != c)[0] for c in range(self.n_countries)]
        all_items = np.arange(self.n_items)

        us, its = [], []
        for u in range(self.n_users):
            nd, nx = int(dom_cnt[u]), int(cross_cnt[u])
            if nd + nx == 0:
                continue
            ex = np.exp(taste[u] - taste[u].max())
            c = self.user_country[u]
            # 跨境流量沿贸易走廊分布：邻接/贸易紧密国家的商品被优先选中
            # （这正是国家关系图模块要捕捉的信号）
            adj_w = np.exp(1.5 * self.country_adj[c, self.item_country])
            for pool, cnt, w in ((same_pool[c], nd, None),
                                 (diff_pool[c], nx, adj_w)):
                if cnt == 0:
                    continue
                if len(pool) == 0:
                    pool = all_items
                p = ex[pool] if w is None else ex[pool] * w[pool]
                its.append(rng.choice(pool, size=cnt, p=p / p.sum()))
                us.append(np.full(cnt, u, dtype=np.int64))
        pairs = np.unique(
            np.stack([np.concatenate(us), np.concatenate(its)], axis=1), axis=0
        )
        return pairs

    def _choice_by_cat(self, owner_cat, n_choices, choice_cat):
        """为每个 item 从「与其品类相同」的候选实体池中采样，池空则全池兜底。"""
        rng = self.rng
        out = np.empty(len(owner_cat), dtype=np.int64)
        for c in range(self.n_cats):
            m = owner_cat == c
            if not m.any():
                continue
            pool = np.where(choice_cat == c)[0]
            if len(pool) == 0:
                pool = np.arange(n_choices)
            out[m] = rng.choice(pool, size=m.sum())
        return out

    def _gen_triples(self, n):
        """关系 0 留给 interact；1=所属品类 2=品牌 3=原产国 4=属性，5..R 为噪声关系。"""
        rng = self.rng
        R = self.n_relations
        items = np.arange(self.n_items)
        hs, rs, ts = [], [], []

        def add(h, r, t):
            hs.append(h); rs.append(np.full(len(h), r, dtype=np.int64)); ts.append(t)

        add(items, 1, self.cat_entity[self.item_cat])
        add(items, 2, self.brand_entity[self.item_brand])
        add(items, 3, self.country_entity[self.item_country])
        if self.n_attrs > 0:
            attr_cat = rng.integers(0, self.n_cats, size=self.n_attrs)
            for _ in range(2):   # 每个 item 挂 2 个与品类相关的属性
                item_attr = self._choice_by_cat(self.item_cat, self.n_attrs, attr_cat)
                add(items, 4, self.attr_entity[item_attr])

        h = np.concatenate(hs); r = np.concatenate(rs); t = np.concatenate(ts)
        # 噪声三元组补足到 n 条，模拟真实 KG 里与推荐无关的边。
        # 只在非商品实体之间采样：真实 KG 的无关事实多挂在属性类实体上，
        # 若直接挂到商品节点会把随机邻居注入商品表示，KG 反而变成负资产
        n_noise = max(0, n - len(h))
        if n_noise:
            lo = 5 if R >= 5 else 1
            h2 = rng.integers(self.n_items, self.n_entities, size=n_noise)
            r2 = rng.integers(lo, R + 1, size=n_noise)
            t2 = rng.integers(self.n_items, self.n_entities, size=n_noise)
            keep = h2 != t2
            h = np.concatenate([h, h2[keep]])
            r = np.concatenate([r, r2[keep]])
            t = np.concatenate([t, t2[keep]])
        return np.stack([h, r, t], axis=1)


# --------------------------------------------------------------------------- #
# 数据集封装
# --------------------------------------------------------------------------- #
class CKGDataset:
    """把原始数据整理成模型可直接消费的形式。"""

    def __init__(self, raw, cfg):
        self.cfg = cfg
        self.n_users = raw.n_users
        self.n_items = raw.n_items
        self.n_entities = raw.n_entities
        self.n_relations = raw.n_relations
        self.n_countries = raw.n_countries
        self.n_languages = raw.n_languages
        self.countries = list(getattr(raw, "countries", range(self.n_countries)))
        self.languages = list(getattr(raw, "languages", range(self.n_languages)))
        self.raw_source_hashes = dict(
            getattr(raw, "raw_source_hashes", {})
        )
        self.source_files = list(getattr(raw, "source_files", ()))
        self.artifact_files = list(getattr(raw, "artifact_files", ()))
        if len(self.countries) != self.n_countries:
            raise ValueError("countries 词表长度与 n_countries 不一致")
        if len(self.languages) != self.n_languages:
            raise ValueError("languages 词表长度与 n_languages 不一致")
        self.n_nodes = self.n_entities + self.n_users
        self.split_seed = _configured_seed(cfg.data, "split_seed")
        self.train_seed = int(
            getattr(cfg.train, "train_seed", getattr(cfg.data, "seed", 2026))
        )
        self.schema_version = getattr(raw, "schema_version", None)
        self.data_schema = getattr(raw, "data_schema", None)

        self.user_country = raw.user_country
        self.item_country = raw.item_country
        self.country_adj = getattr(raw, "country_adj", None)
        self.item_text_feat = raw.item_text_feat
        self.item_text_lang = raw.item_text_lang
        self.item_image_feat = raw.item_image_feat
        self.item_image_mask = raw.item_image_mask
        self.item_market_mask = getattr(raw, "item_market_mask", None)
        self.triples = np.asarray(raw.triples, dtype=np.int64).reshape(-1, 3)
        self._attach_multimodal_contract(raw)

        m = cfg.model
        if (m.use_multimodal and m.use_mm_item_graph
                and m.mm_graph_beta > 0.0 and m.mm_graph_k > 0):
            self.mm_item_edge_index, self.mm_item_edge_weight = (
                build_frozen_mm_item_graph(
                    self.item_text_feat,
                    self.item_image_feat,
                    self.item_image_mask,
                    k=m.mm_graph_k,
                    text_weight=m.mm_graph_text_weight,
                    chunk_size=m.mm_graph_chunk_size,
                )
            )
        else:
            self.mm_item_edge_index = np.empty((2, 0), dtype=np.int64)
            self.mm_item_edge_weight = np.empty(0, dtype=np.float32)

        # XMarket 等真实反馈管线可提供预定义时间/cold 切分。此时 cold 表示
        # “目标市场无训练反馈”，同一 ASIN 仍可在 US 有训练反馈，不能把全局 ID
        # 强制置零。OFF/合成数据则继续使用全局新商品保留集。
        predefined = getattr(raw, "predefined_splits", None)
        if predefined is not None:
            required = {"train", "val", "test", "cold_val", "cold_test"}
            if set(predefined) != required:
                raise ValueError(
                    f"predefined_splits 必须恰含 {sorted(required)}"
                )
            splits = {
                name: np.asarray(predefined[name], dtype=np.int64).reshape(-1, 2)
                for name in required
            }
            seen_pairs = set()
            for name in ("train", "val", "test", "cold_val", "cold_test"):
                pairs_here = {tuple(map(int, pair)) for pair in splits[name]}
                overlap = seen_pairs & pairs_here
                if overlap:
                    raise ValueError(
                        f"predefined_splits 的 {name} 与先前 split 重叠"
                    )
                seen_pairs |= pairs_here
            self.train_pairs = splits["train"]
            self.val_pairs = splits["val"]
            self.test_pairs = splits["test"]
            self.val_user_pos_cold = self._to_dict(splits["cold_val"])
            self.test_user_pos_cold = self._to_dict(splits["cold_test"])
            self.cold_val_items = np.unique(splits["cold_val"][:, 1])
            self.cold_test_items = np.unique(splits["cold_test"][:, 1])
            self.transfer_cold_items = np.union1d(
                self.cold_val_items, self.cold_test_items
            ).astype(np.int64, copy=False)
            # 仅“全局零训练度”商品在下方 zero_train_items 中屏蔽 ID。
            self.cold_items = np.array([], dtype=np.int64)
        else:
            pairs = raw.interactions
            cold_ratio = getattr(cfg.data, "cold_item_ratio", 0.0)
            inter_items = np.unique(pairs[:, 1])
            n_cold = min(
                int(len(inter_items) * cold_ratio),
                max(len(inter_items) - 1, 0),
            )
            if n_cold > 0:
                rng = np.random.default_rng(self.split_seed + 1)
                self.cold_items = rng.choice(
                    inter_items, size=n_cold, replace=False
                )
                cold_val_ratio = float(
                    getattr(cfg.data, "cold_val_item_ratio", 0.25)
                )
                if not 0.0 < cold_val_ratio < 1.0:
                    raise ValueError("cold_val_item_ratio 必须在 (0, 1) 内")
                n_cold_val = int(round(n_cold * cold_val_ratio))
                if n_cold > 1:
                    n_cold_val = min(max(n_cold_val, 1), n_cold - 1)
                else:
                    n_cold_val = n_cold
                self.cold_val_items = np.sort(self.cold_items[:n_cold_val])
                self.cold_test_items = np.sort(self.cold_items[n_cold_val:])
                cold_val_mask = np.isin(pairs[:, 1], self.cold_val_items)
                cold_test_mask = np.isin(pairs[:, 1], self.cold_test_items)
                self.val_user_pos_cold = self._to_dict(pairs[cold_val_mask])
                self.test_user_pos_cold = self._to_dict(pairs[cold_test_mask])
                pairs = pairs[~(cold_val_mask | cold_test_mask)]
            else:
                self.cold_items = np.array([], dtype=np.int64)
                self.cold_val_items = np.array([], dtype=np.int64)
                self.cold_test_items = np.array([], dtype=np.int64)
                self.val_user_pos_cold = {}
                self.test_user_pos_cold = {}
            self.transfer_cold_items = self.cold_items.copy()
            self.train_pairs, self.val_pairs, self.test_pairs = self._split_three(
                pairs, cfg.data.val_ratio, cfg.data.test_ratio, self.split_seed
            )
        self.train_user_pos = self._to_dict(self.train_pairs)
        self.val_user_pos = self._to_dict(self.val_pairs)
        self.test_user_pos = self._to_dict(self.test_pairs)
        self.train_val_user_pos = self._merge_user_pos(
            self.train_user_pos, self.val_user_pos, self.val_user_pos_cold
        )
        # 仅供最终评测/诊断查看，训练代码不得读取这个 held-out-inclusive 集合。
        # 若用它过滤训练负样本，负样本分布会随 validation/test 标签改变，形成泄漏。
        self.eval_user_pos_all = self._merge_user_pos(
            self.train_user_pos, self.val_user_pos, self.test_user_pos,
            self.val_user_pos_cold, self.test_user_pos_cold
        )
        self.item_train_degree = np.bincount(
            self.train_pairs[:, 1], minlength=self.n_items
        ).astype(np.int64, copy=False)
        self.zero_train_items = np.flatnonzero(
            self.item_train_degree == 0
        ).astype(np.int64, copy=False)
        self.zero_train_item_mask = self.item_train_degree == 0

        self.supports_item_origin = bool(
            getattr(raw, "supports_item_origin", True)
        )
        if self.supports_item_origin:
            val_cross_mask = (
                self.user_country[self.val_pairs[:, 0]]
                != self.item_country[self.val_pairs[:, 1]]
            )
            self.val_user_pos_cross = self._to_dict(
                self.val_pairs[val_cross_mask]
            )
            cross_mask = (
                self.user_country[self.test_pairs[:, 0]]
                != self.item_country[self.test_pairs[:, 1]]
            )
            self.test_user_pos_cross = self._to_dict(
                self.test_pairs[cross_mask]
            )
        else:
            self.val_user_pos_cross = {}
            self.test_user_pos_cross = {}

        self.edge_index, self.edge_type = self._build_ckg()
        # TransR 只拟合知识关系。交互边已经由 BPR/LightGCN 主干学习；把它们
        # 再塞入平移关系会引入从未用于传播的 interaction-TransR 参数。
        kg = np.unique(self.triples, axis=0)
        reverse = np.column_stack([
            kg[:, 2], kg[:, 1] + self.n_relations + 1, kg[:, 0]
        ])
        self.ckg_triples = np.concatenate([kg, reverse], axis=0).astype(
            np.int64, copy=False
        )
        if np.any(
            (self.ckg_triples[:, 1] == 0)
            | (self.ckg_triples[:, 1] == self.n_relations + 1)
        ):
            raise ValueError("TransR 三元组不得包含 interaction 正反关系")

    def _attach_multimodal_contract(self, raw):
        """把 schema-v2 元数据复制到数据集；仅为内存测试对象提供保守兜底。"""
        shape = self.item_text_lang.shape
        if self.item_text_feat.shape[:2] != shape:
            raise ValueError("item_text_feat 与 item_text_lang 的 item/view 维不一致")

        required = (
            "item_text_source",
            "item_text_role",
            "item_text_valid",
            "item_text_is_fallback",
            "item_text_content_hash",
            "item_text_dedup_mask",
            "item_text_language_confidence",
        )
        if all(hasattr(raw, name) for name in required):
            for name in required:
                setattr(self, name, np.asarray(getattr(raw, name)))
        else:
            # 仅兼容直接构造的旧内存 fixture；置信度置 0，确保它不会被误当成
            # 真实平行文本。OFF 持久化加载器在到达这里前已经严格拒绝旧 schema。
            self.item_text_source = np.full(
                shape, "legacy_in_memory", dtype="<U32"
            )
            self.item_text_role = np.full(shape, "product_name", dtype="<U16")
            self.item_text_valid = np.ones(shape, dtype=bool)
            self.item_text_is_fallback = np.ones(shape, dtype=bool)
            self.item_text_content_hash = np.empty(shape, dtype="<U64")
            for item in range(shape[0]):
                for view in range(shape[1]):
                    self.item_text_content_hash[item, view] = text_content_hash(
                        f"legacy in-memory item {item} view {view}"
                    )
            self.item_text_dedup_mask = dedup_mask(
                self.item_text_content_hash, self.item_text_valid
            )
            self.item_text_language_confidence = np.zeros(
                shape, dtype=np.float32
            )
        if self.item_text_source.shape != shape:
            raise ValueError("item_text_source 形状必须与 item_text_lang 一致")

        self.item_text_genuine, self.item_text_pair_valid = alignment_pair_mask(
            self.item_text_lang,
            self.item_text_role,
            self.item_text_valid,
            self.item_text_is_fallback,
            self.item_text_content_hash,
            self.item_text_dedup_mask,
            self.item_text_language_confidence,
        )
        supplied_pair = getattr(raw, "item_text_pair_valid", None)
        if supplied_pair is not None and not np.array_equal(
                np.asarray(supplied_pair, dtype=bool), self.item_text_pair_valid):
            raise ValueError("raw.item_text_pair_valid 与 schema-v2 字段不一致")
        # JSON/NPZ 与 Dataset 均保留可审计字符串。核心模型只接收预计算的
        # dedup/pair mask；以下整数仅作为可选分析索引，不进入默认 batch。
        self.item_text_source_name = self.item_text_source.astype(str)
        self.item_text_role_name = self.item_text_role.astype(str)
        self.item_text_content_hash_hex = self.item_text_content_hash.astype(str)
        source_values, source_id = np.unique(
            self.item_text_source_name, return_inverse=True
        )
        role_values, role_id = np.unique(
            self.item_text_role_name, return_inverse=True
        )
        self.item_text_source_vocab = source_values.tolist()
        self.item_text_role_vocab = role_values.tolist()
        self.item_text_source_id = source_id.reshape(shape).astype(np.int64)
        self.item_text_role_id = role_id.reshape(shape).astype(np.int64)
        self.item_text_content_hash_id = np.fromiter(
            (
                int(value[:16], 16) & np.iinfo(np.int64).max
                for value in self.item_text_content_hash_hex.flat
            ),
            dtype=np.int64,
            count=self.item_text_content_hash_hex.size,
        ).reshape(shape)
        self.item_text_market = np.asarray(
            getattr(raw, "item_text_market", np.full(shape, -1, dtype=np.int64)),
            dtype=np.int64,
        )
        if self.item_text_market.shape != shape:
            raise ValueError("item_text_market 形状必须与 item_text_lang 一致")

        image_meta = runtime_image_metadata(
            getattr(raw, "item_image_available", self.item_image_mask > 0.5),
            getattr(raw, "item_image_observed", self.item_image_mask > 0.5),
        )
        self.item_image_available = image_meta["available"]
        self.item_image_observed = image_meta["observed"]
        supplied_confidence = getattr(
            raw, "item_image_completion_confidence", None
        )
        self.item_image_completion_confidence = (
            image_meta["completion_confidence"]
            if supplied_confidence is None
            else np.asarray(supplied_confidence, dtype=np.float32)
        )
        if self.item_image_completion_confidence.shape != (self.n_items,):
            raise ValueError("item_image_completion_confidence 形状非法")
        if (not np.isfinite(self.item_image_completion_confidence).all()
                or not ((0.0 <= self.item_image_completion_confidence)
                        & (self.item_image_completion_confidence <= 1.0)).all()):
            raise ValueError("item_image_completion_confidence 必须位于 [0,1]")

    def kg_positive_items(self, item_ids, rng):
        """兼容旧调用方；品牌/品类邻居不再冒充跨语言等价商品。"""
        del rng
        return np.full(len(item_ids), -1, dtype=np.int64)

    # ---------------- 内部工具 ---------------- #
    @staticmethod
    def _to_dict(pairs):
        d = defaultdict(list)
        for u, i in pairs:
            d[int(u)].append(int(i))
        return {k: np.array(v) for k, v in d.items()}

    @staticmethod
    def _merge_user_pos(*dicts):
        merged = defaultdict(set)
        for d in dicts:
            for u, items in d.items():
                merged[int(u)].update(int(i) for i in items)
        return {
            u: np.array(sorted(items), dtype=np.int64)
            for u, items in merged.items()
        }

    @staticmethod
    def _split_three(pairs, val_ratio, test_ratio, seed):
        rng = np.random.default_rng(seed)
        by_user = defaultdict(list)
        for u, i in pairs:
            by_user[int(u)].append(int(i))
        tr, va, te = [], [], []
        for u, items in by_user.items():
            items = np.array(items)
            rng.shuffle(items)
            n_test = int(len(items) * test_ratio)
            n_val = int(len(items) * val_ratio)
            while len(items) - n_test - n_val < 1:
                if n_test >= n_val and n_test > 0:
                    n_test -= 1
                elif n_val > 0:
                    n_val -= 1
                else:
                    break
            te += [(u, i) for i in items[:n_test]]
            va += [(u, i) for i in items[n_test:n_test + n_val]]
            tr += [(u, i) for i in items[n_test + n_val:]]

        # 用户级划分只保证每个用户有训练交互，仍可能让某个非 cold 商品只出现
        # 在 validation/test。此类商品若继续当作 warm item，会把随机、未训练的
        # ID 混入 warm 指标。每个缺失商品确定性地移一条 held-out 交互回训练集；
        # 优先移动 validation，只有 validation 中不存在时才移动 test。
        train_items = {i for _, i in tr}
        all_items = train_items | {i for _, i in va} | {i for _, i in te}
        missing = all_items - train_items
        for heldout in (va, te):
            kept = []
            for pair in heldout:
                if pair[1] in missing:
                    tr.append(pair)
                    missing.remove(pair[1])
                else:
                    kept.append(pair)
            heldout[:] = kept
        if missing:
            raise RuntimeError("无法保证所有非 cold 商品至少有一条训练交互")

        return (
            np.asarray(tr, dtype=np.int64).reshape(-1, 2),
            np.asarray(va, dtype=np.int64).reshape(-1, 2),
            np.asarray(te, dtype=np.int64).reshape(-1, 2),
        )

    def _build_ckg(self):
        """拼接交互边 + KG 边 + 各自的反向边，返回 (2, E) 的 edge_index 与 (E,) 的 edge_type。"""
        R = self.n_relations
        u = self.train_pairs[:, 0] + self.n_entities
        i = self.train_pairs[:, 1]

        src = [u, i]                               # user->item, item->user
        dst = [i, u]
        typ = [np.zeros(len(u), dtype=np.int64),
               np.full(len(u), R + 1, dtype=np.int64)]

        # no-KG 是数据集本身的语义，不应依赖 train.py 对 edge_index 做隐式二次
        # 修改。这样 notebook、服务推理和 checkpoint 指纹都得到同一个图。
        if self.cfg.model.use_kg:
            h, r, t = self.triples[:, 0], self.triples[:, 1], self.triples[:, 2]
            src += [h, t]
            dst += [t, h]
            typ += [r, r + R + 1]

        edge_index = np.stack(
            [np.concatenate(src), np.concatenate(dst)], axis=0
        ).astype(np.int64)
        edge_type = np.concatenate(typ).astype(np.int64)
        return edge_index, edge_type

    @property
    def n_relations_total(self):
        """含 interact、反向关系在内的全部关系数。"""
        return 2 * (self.n_relations + 1)


# --------------------------------------------------------------------------- #
# 采样器
# --------------------------------------------------------------------------- #
class BPRSampler:
    """为每条正样本采一个用户未交互过的负样本。

    负样本只从「训练集中出现过的商品」里抽：冷启动保留商品在训练期不可见，
    若被反复采为负样本会被系统性压分，冷启动评测直接归零。
    """

    def __init__(self, dataset, batch_size, seed=0, market_alpha=1.0):
        self.ds = dataset
        self.bs = batch_size
        self.rng = np.random.default_rng(seed)
        source_pairs = dataset.train_pairs
        self.market_alpha = float(market_alpha)
        if not 0.0 <= self.market_alpha <= 1.0:
            raise ValueError("market_sampling_alpha 必须在 [0, 1] 内")
        # 训练时只能使用训练交互判定正/负样本。validation/test 正反馈在真实部署
        # 时不可知，拿它们过滤负样本会让优化轨迹依赖测试标签。
        self.pos_set = {
            u: set(v.tolist()) for u, v in dataset.train_user_pos.items()
        }
        self._positive_keys = np.unique(
            source_pairs[:, 0] * dataset.n_items + source_pairs[:, 1]
        )
        self.warm_items = np.unique(source_pairs[:, 1])
        self.item_market_mask = getattr(dataset, "item_market_mask", None)
        self._market_warm_items = None
        if self.item_market_mask is not None:
            self._market_warm_items = {
                int(m): self.warm_items[self.item_market_mask[int(m), self.warm_items]]
                for m in np.unique(dataset.user_country)
            }

        # 极小市场中，个别用户可能已与该市场全部 warm 商品交互。此时严格
        # BPR 不存在合法负例；保留这些边参与图传播，但不为其构造伪负样本。
        excluded = []
        for user in np.unique(source_pairs[:, 0]):
            pool = self.warm_items
            if self._market_warm_items is not None:
                market = int(dataset.user_country[int(user)])
                pool = self._market_warm_items.get(
                    market, np.empty(0, dtype=np.int64)
                )
            blocked = self.pos_set.get(int(user), set())
            if not len(pool) or all(int(item) in blocked for item in pool):
                excluded.append(int(user))
        self.excluded_users = np.asarray(excluded, dtype=np.int64)
        keep = ~np.isin(source_pairs[:, 0], self.excluded_users)
        self.pairs = source_pairs[keep]
        self.excluded_pair_count = int((~keep).sum())
        if not len(self.pairs):
            raise ValueError("所有训练用户均没有可用 BPR 负样本")

        pair_market = dataset.user_country[self.pairs[:, 0]]
        self._market_pair_indices = {
            int(m): np.flatnonzero(pair_market == m)
            for m in np.unique(pair_market)
        }

    def __len__(self):
        return int(np.ceil(len(self.pairs) / self.bs))

    def _draw(self, size=None):
        return self.warm_items[self.rng.integers(0, len(self.warm_items), size=size)]

    def _epoch_indices(self):
        """按 n_market^alpha 分配本轮正交互；alpha=1 是原始全量随机排列。"""
        groups = list(self._market_pair_indices.values())
        sizes = np.asarray([len(g) for g in groups], dtype=np.int64)
        mass = sizes.astype(np.float64) ** self.market_alpha
        raw = len(self.pairs) * mass / mass.sum()
        counts = np.floor(raw).astype(np.int64)
        remainder = len(self.pairs) - int(counts.sum())
        if remainder:
            order = np.argsort(-(raw - counts), kind="stable")
            counts[order[:remainder]] += 1

        sampled = []
        for group, count in zip(groups, counts):
            if count == 0:
                continue
            sampled.append(self.rng.choice(
                group, size=int(count), replace=int(count) > len(group)
            ))
        return self.rng.permutation(np.concatenate(sampled))

    def _draw_for_user(self, u):
        u = int(u)
        pool = self.warm_items
        if self._market_warm_items is not None:
            market = int(self.ds.user_country[u])
            pool = self._market_warm_items.get(market, np.empty(0, dtype=np.int64))
        blocked = self.pos_set.get(u, set())
        for _ in range(32):
            if len(pool) == 0:
                break
            candidate = int(pool[self.rng.integers(0, len(pool))])
            if candidate not in blocked:
                return candidate
        candidates = np.asarray(
            [i for i in pool if int(i) not in blocked], dtype=np.int64
        )
        if len(candidates) == 0:
            raise ValueError(f"用户 {u} 没有可用 BPR 负样本")
        return int(candidates[self.rng.integers(0, len(candidates))])

    def _draw_for_users(self, users):
        """批量拒绝采样；仅对极少数高密度用户走精确兜底。"""
        users = np.asarray(users, dtype=np.int64)
        neg = np.full(users.shape, -1, dtype=np.int64)
        user_markets = self.ds.user_country[users]
        for _ in range(32):
            pending = np.flatnonzero(neg < 0)
            if not len(pending):
                break
            for market in np.unique(user_markets[pending]):
                positions = pending[user_markets[pending] == market]
                pool = self.warm_items
                if self._market_warm_items is not None:
                    pool = self._market_warm_items.get(
                        int(market), np.empty(0, dtype=np.int64)
                    )
                if not len(pool):
                    continue
                candidates = pool[
                    self.rng.integers(0, len(pool), size=len(positions))
                ]
                keys = users[positions] * self.ds.n_items + candidates
                valid = ~np.isin(keys, self._positive_keys, assume_unique=False)
                neg[positions[valid]] = candidates[valid]

        for position in np.flatnonzero(neg < 0):
            neg[position] = self._draw_for_user(users[position])
        return neg

    def __iter__(self):
        idx = self._epoch_indices()
        for s in range(0, len(idx), self.bs):
            batch = self.pairs[idx[s:s + self.bs]]
            users, pos = batch[:, 0], batch[:, 1]
            neg = self._draw_for_users(users)
            yield users, pos, neg


class KGSampler:
    """TransR 训练用的三元组采样：破坏尾实体构造负三元组。

    采样源仅含 KG 正向/反向关系，不含用户交互；头尾编号均位于实体空间
    ``[0, n_entities)``。交互几何只由推荐主任务优化。
    """

    def __init__(self, dataset, batch_size, seed=0):
        self.ds = dataset
        self.bs = batch_size
        self.rng = np.random.default_rng(seed)
        self.triples = dataset.ckg_triples
        self.true_triples = {tuple(map(int, x)) for x in self.triples}
        self.forbidden_tails = defaultdict(set)
        for h, r, t in self.triples:
            self.forbidden_tails[(int(h), int(r))].add(int(t))
        self.rel_tail_candidates = {
            int(r): np.unique(self.triples[self.triples[:, 1] == r, 2])
            for r in np.unique(self.triples[:, 1])
        }
        self._relation_forbidden_keys = {
            int(r): np.unique(
                rel[:, 0] * self.ds.n_entities + rel[:, 2]
            )
            for r in np.unique(self.triples[:, 1])
            for rel in [self.triples[self.triples[:, 1] == r]]
        }

    def __len__(self):
        return int(np.ceil(len(self.triples) / self.bs))

    def __iter__(self):
        idx = self.rng.permutation(len(self.triples))
        for s in range(0, len(idx), self.bs):
            b = self.triples[idx[s:s + self.bs]]
            neg_t = self._sample_tails(b)
            keep = neg_t >= 0
            if keep.any():
                kept = b[keep]
                yield (kept[:, 0], kept[:, 1], kept[:, 2],
                       neg_t[keep])

    def _sample_tails(self, triples):
        """按关系分组批量拒绝采样，并对未命中的行做精确兜底。"""
        triples = np.asarray(triples, dtype=np.int64)
        neg = np.full(len(triples), -1, dtype=np.int64)
        for relation in np.unique(triples[:, 1]):
            relation = int(relation)
            relation_rows = np.flatnonzero(triples[:, 1] == relation)
            tails = self.rel_tail_candidates.get(relation)
            if tails is None or not len(tails):
                continue
            forbidden_keys = self._relation_forbidden_keys[relation]
            for _ in range(32):
                pending = relation_rows[neg[relation_rows] < 0]
                if not len(pending):
                    break
                candidates = tails[
                    self.rng.integers(0, len(tails), size=len(pending))
                ]
                keys = (
                    triples[pending, 0] * self.ds.n_entities + candidates
                )
                valid = ~np.isin(keys, forbidden_keys, assume_unique=False)
                neg[pending[valid]] = candidates[valid]
            for row in relation_rows[neg[relation_rows] < 0]:
                sampled = self._sample_tail(
                    int(triples[row, 0]), relation, int(triples[row, 2])
                )
                if sampled is not None:
                    neg[row] = sampled
        return neg

    def _sample_tail(self, h, r, t):
        tails = self.rel_tail_candidates.get(r)
        if tails is None or len(tails) == 0:
            return None
        forbidden = self.forbidden_tails.get((h, r), set())
        if len(forbidden) >= len(tails):
            return None

        # 通常一次即可命中；密集关系最多拒绝采样若干次，再精确构造小型兜底池。
        for _ in range(32):
            candidate = int(tails[self.rng.integers(0, len(tails))])
            if candidate not in forbidden:
                return candidate
        candidates = np.asarray(
            [x for x in tails if int(x) not in forbidden], dtype=np.int64
        )
        if len(candidates) == 0:
            return None
        return int(candidates[self.rng.integers(0, len(candidates))])


def build_dataset(cfg):
    val_ratio = float(cfg.data.val_ratio)
    test_ratio = float(cfg.data.test_ratio)
    cold_ratio = float(cfg.data.cold_item_ratio)
    cold_val_ratio = float(cfg.data.cold_val_item_ratio)
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio 必须在 [0, 1) 内")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("test_ratio 必须在 [0, 1) 内")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio 必须小于 1")
    if not 0.0 <= cold_ratio < 1.0:
        raise ValueError("cold_item_ratio 必须在 [0, 1) 内")
    if not 0.0 < cold_val_ratio < 1.0:
        raise ValueError("cold_val_item_ratio 必须在 (0, 1) 内")

    if getattr(cfg.data, "source", "synthetic") == "off":
        from off_data import OFFRealData
        raw = OFFRealData(cfg)
    elif getattr(cfg.data, "source", "synthetic") == "xmarket":
        from xmarket_data import XMarketRealData
        raw = XMarketRealData(cfg)
    else:
        raw = ASEANSyntheticData(cfg)
    return CKGDataset(raw, cfg)
