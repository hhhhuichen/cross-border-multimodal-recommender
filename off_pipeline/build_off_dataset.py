# -*- coding: utf-8 -*-
"""
第 2 步：清洗原始 OFF 记录 -> ACMR 商品全集 + 多语种 KG。

核心决策（与调研报告一致）：
  * 商品全集 = **有可观测产地**的商品（origins_tags 优先，manufacturing_places
    自由文本兜底）。产地不可观测的商品不进入全集——绝不推断、绝不模拟
    item_country（防循环论证）。被排除的数量如实写入 meta 供论文报告。
  * 产地词表 = 当前 ASEAN-11 市场 + 全部可解析的外部原产国 + OTHER。
  * 文本三视图：v0 = 本地语商品名（按市场语言优先级），v1 = 英文名，
    v2 = 本地语成分表；缺失视图回退到已有视图（并记录其真实语言）。
  * KG 关系：1=所属品类 2=品牌 3=原产国 4=标签属性(halal/organic/…) 5=在售市场。
    与 data_utils 的关系编号约定一致（0 留给 interact，反向边由 CKGDataset 生成）。

    python build_off_dataset.py
"""
import argparse
import json
import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from off_common import (
    ASEAN_MEMBERSHIP,
    COLLECTION_MANIFEST,
    MARKETS,
    MARKET_ISO,
    PROC_DIR,
    RAW_DIR,
    TEXT_LANGS,
    freetext_to_iso,
    tag_to_iso,
)

# 该脚本通常以 ``python off_pipeline/build_off_dataset.py`` 运行，此时 Python
# 只把 off_pipeline 放进模块搜索路径；显式加入项目根以复用统一数据契约。
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_contract import (  # noqa: E402
    IMAGE_AVAILABLE_FILE,
    SCHEMA_VERSION,
    TEXT_META_FILE,
    dedup_mask,
    make_text_view,
    schema_descriptor,
)

MIN_TAG_COUNT = 3        # 品类/品牌/标签实体的最低出现次数（去长尾）
N_VIEWS = 3


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_origin(p):
    """返回 (ISO, 来源字段) 或 (None, None)。"""
    votes = []
    for t in p.get("origins_tags") or []:
        iso = tag_to_iso(t)
        if iso:
            votes.append(iso)
    if votes:
        return Counter(votes).most_common(1)[0][0], "origins_tags"
    mp = (p.get("manufacturing_places") or "").strip()
    if mp:
        iso = freetext_to_iso(mp)
        if iso:
            return iso, "manufacturing_places"
    return None, None


def sold_in_markets(p):
    out = []
    for t in p.get("countries_tags") or []:
        iso = tag_to_iso(t)
        if iso in MARKET_ISO:
            out.append(iso)
    return sorted(set(out))


def build_country_vocabulary(origin_counter):
    """保留全部观测原产国；顺序按频次稳定，市场轴固定在最前。"""
    foreign = [
        country for country, _ in origin_counter.most_common()
        if country not in MARKET_ISO and country != "OTHER"
    ]
    return MARKET_ISO + foreign + ["OTHER"]


def clean(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def text_views(p, markets):
    """构造 schema-v2 文本视图，回退视图保留真实来源与语言。"""
    lang_pri = []
    for iso in markets:                       # 该商品所有在售市场的语言优先级
        for _, miso, langs in MARKETS:
            if miso == iso:
                lang_pri += langs
    lang_pri += [p.get("lang") or p.get("lc") or "en", "en"]

    brand = clean((p.get("brands") or "").split(",")[0])

    def named(lc):
        t = clean(p.get(f"product_name_{lc}"))
        return (f"{brand} {t}".strip(), f"product_name_{lc}") if t else ("", "")

    def copied(view, role):
        return make_text_view(
            view["text"], view["language"], view["source"], role,
            is_fallback=True,
            language_confidence=view["language_confidence"],
        )

    # v0 本地语名
    v0 = None
    for lc in lang_pri:
        text, source = named(lc)
        if lc in TEXT_LANGS and text:
            v0 = make_text_view(
                text, lc, source, "product_name", language_confidence=1.0
            )
            break
    if v0 is None:
        t = clean(p.get("product_name"))
        declared = p.get("lang") or p.get("lc")
        lc = declared if declared in TEXT_LANGS else "en"
        text = f"{brand} {t}".strip() if t else brand
        v0 = make_text_view(
            text,
            lc,
            "product_name" if t else "brands",
            "product_name",
            is_fallback=not bool(t),
            # 无可信语言声明时仍可用于推荐池化，但不能作为跨语言正样本。
            language_confidence=1.0 if declared in TEXT_LANGS else 0.0,
        )
    # v1 英文名
    english, english_source = named("en")
    if english:
        v1 = make_text_view(
            english, "en", english_source, "product_name",
            language_confidence=1.0,
        )
    else:
        v1 = copied(v0, "product_name")
    # v2 本地语成分表
    v2 = None
    for lc in [v0["language"], "en", "th", "vi", "id", "ms", "zh"]:
        t = clean(p.get(f"ingredients_text_{lc}"))
        if t:
            v2 = make_text_view(
                t[:1000], lc, f"ingredients_text_{lc}", "ingredients",
                language_confidence=1.0,
            )
            break
    if v2 is None:
        v2 = copied(v0, "ingredients")
    return [v0, v1, v2]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="把 Open Food Facts ASEAN-11 原始商品构建为 ACMR schema v2"
    )
    parser.add_argument(
        "--allow-partial-markets", action="store_true",
        help="允许缺少成员市场原始文件；仅用于调试，正式实验不得使用",
    )
    args = parser.parse_args(argv)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 读入 + 条码去重 ----
    seen, records = {}, []
    n_lines = 0
    raw_paths = [RAW_DIR / f"{country}.jsonl" for country, _, _ in MARKETS]
    missing_raw = [path.name for path in raw_paths if not path.is_file()]
    if missing_raw and not args.allow_partial_markets:
        raise FileNotFoundError(
            "缺 ASEAN 成员市场原始文件: " + ", ".join(missing_raw)
            + "；先运行 fetch_off_dump.py（正式）或显式使用 "
              "--allow-partial-markets 调试"
        )
    for country, iso, _ in MARKETS:
        f = RAW_DIR / f"{country}.jsonl"
        if not f.exists():
            print(f"[警告] 缺 {f.name}（该市场未抓取或未完成）")
            continue
        for line in f.open(encoding="utf-8"):
            n_lines += 1
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue
            code = p.get("code")
            if not code or code in seen:
                continue
            seen[code] = True
            records.append(p)
    print(f"原始 {n_lines} 行 -> 去重后 {len(records)} 个条码")

    # ---- 产地解析 + 全集筛选 ----
    kept, dropped_no_origin, dropped_no_market = [], 0, 0
    origin_counter = Counter()
    for p in records:
        mkts = sold_in_markets(p)
        if not mkts:
            dropped_no_market += 1
            continue
        iso, src = resolve_origin(p)
        if iso is None:
            dropped_no_origin += 1
            continue
        origin_counter[iso] += 1
        kept.append((p, iso, src, mkts))
    print(f"有产地: {len(kept)} | 无产地剔除: {dropped_no_origin} "
          f"| 无目标市场剔除: {dropped_no_market}")
    print("产地分布 Top15:", origin_counter.most_common(15))
    market_item_counts = Counter(
        market for _, _, _, markets in kept for market in markets
    )

    # ---- 产地词表：11 市场 + 全部观测外国 + OTHER ----
    # 原产国数量很小，完整保留可让 Comtrade 为每个实际伙伴提供独立权重。
    countries = build_country_vocabulary(origin_counter)
    c_idx = {c: i for i, c in enumerate(countries)}

    # ---- 实体词表（最低频次过滤） ----
    cat_cnt, brand_cnt, label_cnt = Counter(), Counter(), Counter()
    for p, iso, src, mkts in kept:
        for t in (p.get("categories_tags") or [])[-3:]:
            cat_cnt[t] += 1
        for t in (p.get("brands_tags") or [])[:1]:
            brand_cnt[t] += 1
        for t in (p.get("labels_tags") or [])[:5]:
            label_cnt[t] += 1
    cats = sorted(t for t, c in cat_cnt.items() if c >= MIN_TAG_COUNT)
    brands = sorted(t for t, c in brand_cnt.items() if c >= MIN_TAG_COUNT)
    labels = sorted(t for t, c in label_cnt.items() if c >= MIN_TAG_COUNT)
    print(f"实体: 品类 {len(cats)} | 品牌 {len(brands)} | 标签 {len(labels)} "
          f"| 国家 {len(countries)}")

    n_items = len(kept)
    base = n_items
    cat_ent = {t: base + i for i, t in enumerate(cats)}; base += len(cats)
    brand_ent = {t: base + i for i, t in enumerate(brands)}; base += len(brands)
    country_ent = {c: base + i for i, c in enumerate(countries)}; base += len(countries)
    label_ent = {t: base + i for i, t in enumerate(labels)}; base += len(labels)
    n_entities = base

    lang_idx = {lc: i for i, lc in enumerate(TEXT_LANGS)}

    # ---- 逐商品产出 ----
    triples = []
    item_country = np.zeros(n_items, dtype=np.int64)
    item_text_lang = np.zeros((n_items, N_VIEWS), dtype=np.int64)
    item_text_source = np.empty((n_items, N_VIEWS), dtype="<U64")
    item_text_role = np.empty((n_items, N_VIEWS), dtype="<U32")
    item_text_valid = np.zeros((n_items, N_VIEWS), dtype=bool)
    item_text_fallback = np.zeros((n_items, N_VIEWS), dtype=bool)
    item_text_hash = np.empty((n_items, N_VIEWS), dtype="<U64")
    item_text_language_confidence = np.zeros(
        (n_items, N_VIEWS), dtype=np.float32
    )
    item_image_available = np.zeros(n_items, dtype=bool)
    item_rows = []
    origin_src_counter = Counter()
    for i, (p, iso, src, mkts) in enumerate(kept):
        oc = c_idx.get(iso, c_idx["OTHER"])
        item_country[i] = oc
        origin_src_counter[src] += 1

        for t in (p.get("categories_tags") or [])[-3:]:
            if t in cat_ent:
                triples.append((i, 1, cat_ent[t]))
        for t in (p.get("brands_tags") or [])[:1]:
            if t in brand_ent:
                triples.append((i, 2, brand_ent[t]))
        triples.append((i, 3, country_ent[countries[oc]]))
        for t in (p.get("labels_tags") or [])[:5]:
            if t in label_ent:
                triples.append((i, 4, label_ent[t]))
        for m in mkts:
            triples.append((i, 5, country_ent[m]))

        views = text_views(p, mkts)
        for v, view in enumerate(views):
            item_text_lang[i, v] = lang_idx.get(
                view["language"], lang_idx["en"]
            )
            item_text_source[i, v] = view["source"]
            item_text_role[i, v] = view["role"]
            item_text_valid[i, v] = view["valid"]
            item_text_fallback[i, v] = view["is_fallback"]
            item_text_hash[i, v] = view["content_hash"]
            item_text_language_confidence[i, v] = view["language_confidence"]
        img = p.get("image_front_url") or p.get("image_front_small_url") or ""
        item_image_available[i] = bool(img)
        item_rows.append({
            "idx": i, "code": p.get("code"), "origin": iso, "origin_src": src,
            "markets": mkts, "views": views,
            "image_url": img,  # 兼容现有下载脚本
            "image": {"url": img, "available": bool(img)},
        })

    item_text_dedup = dedup_mask(item_text_hash, item_text_valid)
    triples = np.array(sorted(set(triples)), dtype=np.int64).reshape(-1, 3)
    np.save(PROC_DIR / "triples.npy", triples)
    np.save(PROC_DIR / "item_country.npy", item_country)
    np.save(PROC_DIR / "item_text_lang.npy", item_text_lang)
    np.savez_compressed(
        PROC_DIR / TEXT_META_FILE,
        language=item_text_lang,
        source=item_text_source,
        role=item_text_role,
        valid=item_text_valid,
        is_fallback=item_text_fallback,
        content_hash=item_text_hash,
        dedup_mask=item_text_dedup,
        language_confidence=item_text_language_confidence,
    )
    np.save(PROC_DIR / IMAGE_AVAILABLE_FILE, item_image_available)
    with (PROC_DIR / "items.jsonl").open("w", encoding="utf-8") as f:
        for r in item_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "dataset": "OpenFoodFacts-ASEAN-products",
        "schema_version": SCHEMA_VERSION,
        "data_schema": schema_descriptor(),
        "n_items": n_items, "n_entities": n_entities, "n_relations": 5,
        "relations": {"1": "category", "2": "brand", "3": "origin_country",
                      "4": "label", "5": "sold_in"},
        "countries": countries, "n_markets": len(MARKET_ISO),
        "markets": MARKET_ISO,
        "market_scope": {
            **ASEAN_MEMBERSHIP,
            "member_iso": MARKET_ISO,
            "member_market_item_counts": {
                iso: int(market_item_counts.get(iso, 0)) for iso in MARKET_ISO
            },
            "partial_raw_allowed": bool(args.allow_partial_markets),
            "missing_raw_markets": missing_raw,
        },
        "languages": TEXT_LANGS, "n_views": N_VIEWS,
        "n_triples": int(len(triples)),
        "entity_layout": {"items": [0, n_items],
                          "categories": [n_items, n_items + len(cats)],
                          "brands": [n_items + len(cats),
                                     n_items + len(cats) + len(brands)],
                          "countries": [country_ent[countries[0]],
                                        country_ent[countries[0]] + len(countries)],
                          "labels": [label_ent[labels[0]] if labels else base, base]},
        "provenance": {
            "dropped_no_origin": dropped_no_origin,
            "dropped_no_target_market": dropped_no_market,
            "origin_source_counts": dict(origin_src_counter),
            "origin_distribution": dict(origin_counter),
            "note": ("item_country 全部来自 origins_tags / manufacturing_places "
                     "观测值；无产地商品被剔除而非推断。"),
        },
        # 只纳入成员市场文件；目录中遗留的 china.jsonl 等非成员快照被忽略。
        "raw_source_hashes": {
            f"raw/{path.name}": file_sha256(path)
            for path in raw_paths if path.is_file()
        },
    }
    if COLLECTION_MANIFEST.is_file():
        manifest = json.loads(COLLECTION_MANIFEST.read_text(encoding="utf-8"))
        declared = manifest.get("scope", {}).get("member_iso")
        if declared != MARKET_ISO:
            raise ValueError(
                "collection_manifest.json 的 ASEAN 成员版本与构建器不一致；"
                "请重新运行采集器"
            )
        meta["collection_manifest"] = {
            "path": "raw/collection_manifest.json",
            "sha256": file_sha256(COLLECTION_MANIFEST),
            "collector": manifest.get("collector"),
            "generated_at": manifest.get("generated_at"),
            "formal_snapshot_ready": bool(
                manifest.get("formal_snapshot_ready", False)
            ),
        }
    (PROC_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 语言分布报告
    lang_dist = Counter()
    for r in item_rows:
        lang_dist[r["views"][0]["lang"]] += 1
    print("v0 视图语言分布:", dict(lang_dist.most_common()))
    print(f"三元组 {len(triples)} 条 | 已写入 {PROC_DIR}")


if __name__ == "__main__":
    main()
