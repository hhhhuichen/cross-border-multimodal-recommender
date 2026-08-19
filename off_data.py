# -*- coding: utf-8 -*-
"""
OFF 真实数据加载器：把 off_pipeline 各步的产物组装成与 ASEANSyntheticData
完全同构的接口，供 CKGDataset 直接消费（README 第三节的数组契约）。

用法（在 train.py 里）:
    python train.py --data off
"""
import json
from pathlib import Path

import numpy as np

from data_contract import (
    IMAGE_AVAILABLE_FILE,
    SCHEMA_VERSION,
    TEXT_META_FILE,
    alignment_pair_mask,
    require_schema_v2,
    runtime_image_metadata,
    validate_text_metadata,
)


PROJECT_DIR = Path(__file__).resolve().parent


class OFFRealData:
    """从 data_off/processed 读取全部数组。缺文件时给出指向具体管线步骤的报错。"""

    _STEPS = {
        "triples.npy": "off_pipeline/build_off_dataset.py",
        "item_country.npy": "off_pipeline/build_off_dataset.py",
        "item_text_lang.npy": "off_pipeline/build_off_dataset.py",
        TEXT_META_FILE: "off_pipeline/build_off_dataset.py",
        IMAGE_AVAILABLE_FILE: "off_pipeline/build_off_dataset.py",
        "users.npz": "off_pipeline/simulate_users.py",
        "item_text_feat.npy": "off_pipeline/extract_text_feat.py",
        "item_image_feat.npy": "off_pipeline/extract_image_feat.py",
        "item_image_mask.npy": "off_pipeline/extract_image_feat.py",
    }

    def __init__(self, cfg):
        self.cfg = cfg
        d = Path(cfg.data.off_dir).expanduser()
        if not d.is_absolute():
            d = (PROJECT_DIR / d).resolve()
        cfg.data.off_dir = str(d)
        meta_path = d / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"缺 {meta_path} —— 先运行 off_pipeline/build_off_dataset.py"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.data_schema = require_schema_v2(meta)
        self.schema_version = SCHEMA_VERSION
        for fname, step in self._STEPS.items():
            if not (d / fname).exists():
                raise FileNotFoundError(
                    f"缺 {d / fname} —— 先运行 {step}（管线顺序见 off_pipeline/）")

        self.raw_source_hashes = dict(meta.get("raw_source_hashes", {}))
        self.source_files = [d.parent / name for name in self.raw_source_hashes]
        self.artifact_files = [meta_path, d / "items.jsonl"] + [
            d / name for name in self._STEPS
        ]

        self.n_items = meta["n_items"]
        self.n_entities = meta["n_entities"]
        self.n_relations = meta["n_relations"]
        self.countries = meta["countries"]  # 成员市场在前，外部原产国在后
        self.languages = meta["languages"]
        self.n_countries = len(self.countries)
        self.n_languages = len(self.languages)
        self.n_markets = meta["n_markets"]
        if not (self.n_items > 0 and self.n_entities >= self.n_items
                and self.n_relations >= 5):
            raise ValueError("OFF meta 的商品、实体或关系数量非法")
        if not 0 < self.n_markets <= self.n_countries:
            raise ValueError("OFF meta 的 n_markets 必须位于有效国家范围")
        if meta.get("dataset") == "OpenFoodFacts-ASEAN-products":
            markets = list(meta.get("markets", ()))
            declared = list(meta.get("market_scope", {}).get("member_iso", ()))
            if markets != self.countries[:self.n_markets] or declared != markets:
                raise ValueError("OFF ASEAN 市场顺序或成员范围与 countries 不一致")

        self.triples = np.load(d / "triples.npy")
        self.item_country = np.load(d / "item_country.npy")
        self.item_text_lang = np.load(d / "item_text_lang.npy")
        with np.load(d / TEXT_META_FILE, allow_pickle=False) as text_meta_file:
            text_meta = validate_text_metadata(
                text_meta_file,
                self.n_items,
                int(meta["n_views"]),
                self.n_languages,
            )
        self.item_text_feat = np.load(d / "item_text_feat.npy")
        self.item_image_feat = np.load(d / "item_image_feat.npy")
        self.item_image_mask = np.load(d / "item_image_mask.npy")
        item_image_available = np.load(d / IMAGE_AVAILABLE_FILE)

        if not np.array_equal(text_meta["language"], self.item_text_lang):
            raise ValueError(
                "item_text_lang.npy 与 item_text_meta.npz/language 不一致；"
                "请清空 processed 后显式重建"
            )
        self.item_text_source = text_meta["source"]
        self.item_text_role = text_meta["role"]
        self.item_text_valid = text_meta["valid"]
        self.item_text_is_fallback = text_meta["is_fallback"]
        self.item_text_content_hash = text_meta["content_hash"]
        self.item_text_dedup_mask = text_meta["dedup_mask"]
        self.item_text_language_confidence = text_meta["language_confidence"]
        self.item_text_genuine, self.item_text_pair_valid = alignment_pair_mask(
            self.item_text_lang,
            self.item_text_role,
            self.item_text_valid,
            self.item_text_is_fallback,
            self.item_text_content_hash,
            self.item_text_dedup_mask,
            self.item_text_language_confidence,
        )
        # OFF 的文本字段来自全局商品记录，不宣称是某个在售市场的本地元数据。
        self.item_text_market = np.full(
            self.item_text_lang.shape, -1, dtype=np.int64
        )

        image_meta = runtime_image_metadata(
            item_image_available, self.item_image_mask > 0.5
        )
        self.item_image_available = image_meta["available"]
        self.item_image_observed = image_meta["observed"]
        self.item_image_completion_confidence = image_meta[
            "completion_confidence"
        ]

        with np.load(d / "users.npz") as users:
            self.user_country = users["user_country"]
            self.interactions = users["interactions"]
        if self.user_country.ndim != 1:
            raise ValueError("users.npz/user_country 必须是一维数组")
        self.n_users = int(self.user_country.shape[0])

        def require(condition, message):
            if not condition:
                raise ValueError(message)

        def require_integer(name, array):
            require(
                np.issubdtype(array.dtype, np.integer),
                f"{name} 必须使用整数 dtype",
            )

        def require_numeric(name, array):
            require(
                np.issubdtype(array.dtype, np.number),
                f"{name} 必须使用数值 dtype",
            )

        V = int(meta["n_views"])
        require(self.n_users > 0 and int(V) > 0, "OFF 用户数和文本视图数必须为正")
        require(self.triples.ndim == 2 and self.triples.shape[1] == 3
                and len(self.triples) > 0, "triples.npy 必须是非空 (N, 3)")
        require(self.interactions.ndim == 2 and self.interactions.shape[1] == 2
                and len(self.interactions) > 0,
                "users.npz/interactions 必须是非空 (N, 2)")
        require(self.user_country.shape == (self.n_users,),
                "users.npz/user_country 形状非法")
        require(self.item_country.shape == (self.n_items,),
                "item_country.npy 形状非法")
        require(self.item_text_lang.shape == (self.n_items, V),
                "item_text_lang.npy 形状非法")
        require(
            self.item_text_feat.shape == (self.n_items, V, cfg.data.text_dim),
            f"文本特征形状 {self.item_text_feat.shape} != "
            f"{(self.n_items, V, cfg.data.text_dim)}",
        )
        require(
            self.item_image_feat.shape == (self.n_items, cfg.data.image_dim),
            "item_image_feat.npy 形状非法",
        )
        require(self.item_image_mask.shape == (self.n_items,),
                "item_image_mask.npy 形状非法")
        require(self.item_image_available.shape == (self.n_items,),
                "item_image_available.npy 形状非法")

        for name, array in (
            ("triples", self.triples),
            ("interactions", self.interactions),
            ("user_country", self.user_country),
            ("item_country", self.item_country),
            ("item_text_lang", self.item_text_lang),
        ):
            require_integer(name, array)
        for name, array in (
            ("item_text_feat", self.item_text_feat),
            ("item_image_feat", self.item_image_feat),
            ("item_image_mask", self.item_image_mask),
        ):
            require_numeric(name, array)
        require(
            ((0 <= self.interactions[:, 0])
             & (self.interactions[:, 0] < self.n_users)).all(),
            "interactions 用户编号越界",
        )
        require(
            ((0 <= self.interactions[:, 1])
             & (self.interactions[:, 1] < self.n_items)).all(),
            "interactions 商品编号越界",
        )
        require(
            len(np.unique(self.interactions, axis=0)) == len(self.interactions),
            "users.npz/interactions 包含重复用户-商品对；"
            "请先聚合为唯一隐式反馈",
        )
        require(
            ((0 <= self.triples[:, 0])
             & (self.triples[:, 0] < self.n_entities)
             & (0 <= self.triples[:, 2])
             & (self.triples[:, 2] < self.n_entities)).all(),
            "triples 头/尾实体编号越界",
        )
        require(
            ((1 <= self.triples[:, 1])
             & (self.triples[:, 1] <= self.n_relations)).all(),
            "triples 关系编号越界",
        )
        require(
            ((0 <= self.user_country)
             & (self.user_country < self.n_markets)).all(),
            "用户市场编号越界",
        )
        require(
            ((0 <= self.item_country)
             & (self.item_country < self.n_countries)).all(),
            "商品国家编号越界",
        )
        require(
            ((0 <= self.item_text_lang)
             & (self.item_text_lang < self.n_languages)).all(),
            "文本语言编号越界",
        )
        require(np.isfinite(self.item_text_feat).all(), "文本特征含 NaN/Inf")
        require(np.isfinite(self.item_image_feat).all(), "图像特征含 NaN/Inf")
        require(
            np.isfinite(self.item_image_mask).all()
            and ((0.0 <= self.item_image_mask)
                 & (self.item_image_mask <= 1.0)).all(),
            "图像掩码必须是 [0,1] 内有限值",
        )
        require(
            np.array_equal(self.item_image_observed, self.item_image_mask > 0.5),
            "图像 observed 与 item_image_mask 不一致",
        )

        # 每个市场的真实可售候选集。用户模拟器只会从 sold_in（关系 5）中购买，
        # 因此训练负采样和评测也必须使用同一候选边界；否则 KG 可以靠“是否可售”
        # 这一确定性捷径击败无 KG 模型，而不是学习用户偏好。
        country_ent_base = meta["entity_layout"]["countries"][0]
        require(
            0 <= int(country_ent_base)
            and int(country_ent_base) + self.n_markets <= self.n_entities,
            "OFF meta 的国家实体区间越界",
        )
        self.item_market_mask = np.zeros(
            (self.n_markets, self.n_items), dtype=bool
        )
        sold = self.triples[self.triples[:, 1] == 5]
        for h, _, t in sold:
            market = int(t) - int(country_ent_base)
            if 0 <= market < self.n_markets and int(h) < self.n_items:
                self.item_market_mask[market, int(h)] = True
        # 极小市场可能在产地完整性筛选后没有候选。保留该事实供覆盖报告使用，
        # 但模拟器不得为这类市场生成用户，评测也不会把它动态伪装成有覆盖市场。
        self.empty_catalog_markets = np.flatnonzero(
            ~self.item_market_mask.any(axis=1)
        )
        interaction_markets = self.user_country[self.interactions[:, 0]]
        if not self.item_market_mask[
                interaction_markets, self.interactions[:, 1]].all():
            raise ValueError("OFF 交互中存在用户所在市场不可售的商品")

        # 模型侧国家图：地理邻接（只覆盖 11 个市场；外国产地节点无地理边，
        # 其嵌入靠 KG 的 origin 关系和门控学习）。信息源与模拟器的贸易矩阵分离。
        geo_path = d.parent / "country_adj_geo.npy"
        if geo_path.exists():
            geo = np.load(geo_path)
            geo_meta_path = d.parent / "country_adj_meta.json"
            if (meta.get("dataset") == "OpenFoodFacts-ASEAN-products"
                    and geo_meta_path.is_file()):
                geo_meta = json.loads(
                    geo_meta_path.read_text(encoding="utf-8")
                )
                if list(geo_meta.get("markets", ())) != list(
                        meta.get("markets", ())):
                    raise ValueError(
                        "country_adj_geo.npy 的成员顺序已过期；"
                        "请重新运行 build_country_adj.py"
                    )
            require(
                geo.ndim == 2 and geo.shape[0] == geo.shape[1]
                and geo.shape[0] <= self.n_countries
                and np.issubdtype(geo.dtype, np.number)
                and np.isfinite(geo).all(),
                "country_adj_geo.npy 必须是有限值方阵且不超过国家数",
            )
            adj = np.zeros((self.n_countries, self.n_countries), dtype=np.float32)
            adj[:geo.shape[0], :geo.shape[1]] = geo
            self.country_adj = adj
        else:
            self.country_adj = None

        # 让 config 与真实数据同步（供模型构造使用）
        cfg.data.n_users = self.n_users
        cfg.data.n_items = self.n_items
        cfg.data.n_entities = self.n_entities
        cfg.data.n_relations = self.n_relations
        cfg.data.n_text_views = V
        cfg.data.countries = self.countries
        cfg.data.languages = self.languages
