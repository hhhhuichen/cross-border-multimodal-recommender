#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 XMarket Electronics 构造成 ACMR schema-v2 真实反馈数据。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

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


MARKETS = ("us", "cn", "in", "sg")
TARGETS = ("cn", "in", "sg")
CATEGORY = "Electronics"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fraction(value, seed):
    payload = f"{seed}:{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64


def timestamp_value(value):
    """解析 ISO 日期或 Unix 时间戳；缺失/非法值返回 None。"""
    value = str(value or "").strip()
    if value.casefold() in {"", "-", "none", "null", "nan"}:
        return None
    try:
        return float(value)
    except ValueError:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None


def load_ratings(path):
    """同一用户-商品重复评分保留日期最新的一条；任意星级均表示交互。"""
    latest = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split()
            if len(fields) != 4:
                raise ValueError(f"{path}:{line_no} 不是四列评分记录")
            user, asin, rating, date = fields
            try:
                float(rating)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no} rating 非数值") from exc
            key = (user, asin)
            previous = latest.get(key)
            if previous is None or date > previous:
                latest[key] = date
    return [(user, asin, date) for (user, asin), date in latest.items()]


def five_core(records, minimum=5):
    active = list(records)
    while True:
        users = Counter(user for user, _, _ in active)
        items = Counter(asin for _, asin, _ in active)
        filtered = [row for row in active
                    if users[row[0]] >= minimum and items[row[1]] >= minimum]
        if len(filtered) == len(active):
            return filtered
        active = filtered


def load_metadata(path, allowed_items=None):
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} JSON 非法") from exc
            asin = str(record.get("asin") or "")
            if asin and (allowed_items is None or asin in allowed_items):
                out[asin] = record
    return out


def clean_text(value):
    return " ".join(str(value or "").split())


def record_text(record):
    parts = [record.get("title"), record.get("description")]
    parts.extend((record.get("features") or [])[:8])
    return clean_text(" ".join(str(part or "") for part in parts))[:6000]


def first_image_url(record):
    raw = record.get("imgUrl")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return ""
    if isinstance(raw, dict) and raw:
        return sorted(raw)[0]
    if isinstance(raw, list) and raw:
        return str(raw[0])
    return ""


def detector(path):
    if path is None:
        return None, None
    try:
        import fasttext
    except ImportError as exc:
        raise RuntimeError(
            "语言检测要求 fasttext；安装后传入固定的 lid.176.ftz"
        ) from exc
    model = fasttext.load_model(str(path))

    def detect(text):
        cleaned = clean_text(text)
        try:
            labels, probabilities = model.predict(cleaned, k=1)
        except ValueError as exc:
            # fasttext-wheel 0.9.2 uses np.array(..., copy=False), which
            # NumPy 2.x rejects. Preserve the public API path for compatible
            # versions and use the binding result only for this known error.
            if "Unable to avoid copy" not in str(exc):
                raise
            predictions = model.f.predict(cleaned + "\n", 1, 0.0, "strict")
            if not predictions:
                return "unknown", 0.0
            probability, label = predictions[0]
            labels, probabilities = (label,), (probability,)
        language = labels[0].removeprefix("__label__")
        confidence = min(1.0, max(0.0, float(probabilities[0])))
        return language, confidence

    return detect, file_sha256(path)


def choose_views(asin, metadata, detect):
    available = {market: records[asin] for market, records in metadata.items()
                 if asin in records}
    if not available:
        available = {"us": {"asin": asin, "title": asin}}
    fallback_market = "us" if "us" in available else sorted(available)[0]
    views = []
    for market in MARKETS:
        source = market if market in available else fallback_market
        text = record_text(available[source]) or asin
        if detect is None:
            language, confidence = "unknown", 0.0
        else:
            language, confidence = detect(text)
        views.append(make_text_view(
            text, language, source, "product_name",
            is_fallback=market not in available,
            language_confidence=confidence,
        ))
    return views, available[fallback_market]


def brand_of(record):
    details = record.get("productDetails") or {}
    for key in ("Brand", "brand", "Manufacturer"):
        value = clean_text(details.get(key))
        if value:
            return value.casefold()
    return ""


def related_asins(record):
    related = record.get("related") or {}
    values = []
    for key in ("alsoBought", "boughtTogether", "alsoViewed"):
        values.extend(related.get(key) or [])
    return list(dict.fromkeys(map(str, values)))[:20]


def chronological_splits(records, market, cold_val, cold_test, split_seed):
    train, val, test, val_cold, test_cold = [], [], [], [], []
    by_user = defaultdict(list)
    for user, asin, date in records:
        by_user[user].append((date, asin))
    for user, interactions in by_user.items():
        warm = []
        for date, asin in interactions:
            row = (market, user, asin)
            if asin in cold_val:
                val_cold.append(row)
            elif asin in cold_test:
                test_cold.append(row)
            else:
                warm.append((date, asin))
        timestamps = [timestamp_value(value[0]) for value in warm]
        if all(value is not None for value in timestamps):
            warm = [pair for _, pair in sorted(
                zip(timestamps, warm), key=lambda value: (value[0], value[1][1])
            )]
        else:
            # 缺时间戳时不伪造顺序；用预注册 split_seed 做
            # leave-two-out，同一数据/种子可完全重现。
            warm.sort(key=lambda value: stable_fraction(
                f"{market}:{user}:{value[1]}", split_seed
            ))
        if len(warm) >= 3:
            train.extend((market, user, asin) for _, asin in warm[:-2])
            val.append((market, user, warm[-2][1]))
            test.append((market, user, warm[-1][1]))
        else:
            train.extend((market, user, asin) for _, asin in warm)
    return train, val, test, val_cold, test_cold


def _proportional_quota(total, bucket_sizes, capacities=None):
    """Allocate an exact stratified quota without exceeding any bucket."""
    keys = sorted(bucket_sizes)
    sizes = {key: int(bucket_sizes[key]) for key in keys}
    capacity = (sizes if capacities is None
                else {key: int(capacities[key]) for key in keys})
    total = int(total)
    if total < 0 or total > sum(capacity.values()):
        raise ValueError("分层 cold quota 超出可用商品数")
    if total == 0:
        return {key: 0 for key in keys}

    denominator = sum(sizes.values())
    ideal = {key: total * sizes[key] / denominator for key in keys}
    counts = {
        key: min(int(math.floor(ideal[key])), capacity[key]) for key in keys
    }
    remaining = total - sum(counts.values())
    order = sorted(keys, key=lambda key: (-(ideal[key] - counts[key]), key))
    while remaining:
        progressed = False
        for key in order:
            if counts[key] >= capacity[key]:
                continue
            counts[key] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise RuntimeError("无法满足分层 cold quota")
    return counts


def market_cold_items(records, market, split_seed):
    """Select 5% validation and 5% test items inside one target market."""
    degree = Counter(asin for _, asin, _ in records)
    by_bucket = defaultdict(list)
    for asin, count in degree.items():
        by_bucket[int(math.log2(max(count, 1)))].append(asin)
    sizes = {bucket: len(items) for bucket, items in by_bucket.items()}
    n_val = int(round(0.05 * len(degree)))
    n_test = int(round(0.05 * len(degree)))
    val_quota = _proportional_quota(n_val, sizes)
    remaining = {bucket: sizes[bucket] - val_quota[bucket] for bucket in sizes}
    test_quota = _proportional_quota(n_test, sizes, remaining)

    cold_val, cold_test = set(), set()
    for bucket, items in by_bucket.items():
        ordered = sorted(
            items,
            key=lambda asin: stable_fraction(f"{market}:{asin}", split_seed),
        )
        nv, nt = val_quota[bucket], test_quota[bucket]
        cold_val.update(ordered[:nv])
        cold_test.update(ordered[nv:nv + nt])
    if cold_val & cold_test:
        raise RuntimeError(f"{market} cold validation/test 商品发生重叠")
    return cold_val, cold_test


def map_pairs(rows, user_index, item_index):
    return np.asarray([
        (user_index[(market, user)], item_index[asin])
        for market, user, asin in rows
    ], dtype=np.int64).reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data_xmarket/raw")
    parser.add_argument("--output-dir", default="data_xmarket/processed")
    parser.add_argument("--split-seed", type=int, default=20260801)
    parser.add_argument("--lid-model", help="固定 fastText lid.176.ftz 路径")
    parser.add_argument("--disable-alignment", action="store_true",
                        help="显式禁用语言检测；置信度置 0，不产生对齐正例")
    args = parser.parse_args()
    if not args.lid_model and not args.disable_alignment:
        parser.error("必须提供 --lid-model；或显式 --disable-alignment")
    if args.lid_model and args.disable_alignment:
        parser.error("--lid-model 与 --disable-alignment 不可同时使用")

    raw_dir, output_dir = Path(args.raw_dir), Path(args.output_dir)
    required = {}
    for market in MARKETS:
        required[(market, "ratings")] = raw_dir / f"ratings_{market}_{CATEGORY}.txt.gz"
        required[(market, "metadata")] = raw_dir / f"metadata_{market}_{CATEGORY}.json.gz"
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺 XMarket 官方文件：\n" + "\n".join(missing))

    raw_ratings, ratings, metadata = {}, {}, {}
    for market in MARKETS:
        raw_ratings[market] = load_ratings(required[(market, "ratings")])
        ratings[market] = five_core(raw_ratings[market], minimum=5)
        metadata[market] = load_metadata(required[(market, "metadata")])
        raw_items = {asin for _, asin, _ in raw_ratings[market]}
        missing_metadata = raw_items - set(metadata[market])
        if missing_metadata:
            raise ValueError(
                f"{market} 有 {len(missing_metadata)} 个评分商品缺少市场元数据"
            )
        print(
            f"[{market}] raw unique={len(raw_ratings[market])} "
            f"catalog={len(metadata[market])} 5-core={len(ratings[market])}"
        )
        if not ratings[market]:
            raise ValueError(f"{market} 经 5-core 后为空")

    # 候选目录来自过滤前的官方市场元数据。5-core 只限定哪些评分交互可进入
    # train/validation/test；它不能把低交互但真实在售的商品从排名目录中删除。
    market_items = {
        market: set(metadata[market]) for market in MARKETS
    }
    all_items = sorted(set().union(*market_items.values()))
    item_index = {asin: index for index, asin in enumerate(all_items)}
    filtered_items = {
        market: {asin for _, asin, _ in ratings[market]} for market in MARKETS
    }

    # 每个目标市场在 5-core 交互商品及其流行度层内独立划分 5%/5%。同一
    # ASIN 在不同市场可以具有不同 cold 状态；用户与候选目录均保持市场本地。
    cold_val = {}
    cold_test = {}
    for market in TARGETS:
        cold_val[market], cold_test[market] = market_cold_items(
            ratings[market], market, args.split_seed
        )

    split_rows = {name: [] for name in
                  ("train", "val", "test", "cold_val", "cold_test")}
    # 源市场全部作为辅助训练反馈；目标市场只在各自本地用户内划分。
    split_rows["train"].extend(("us", u, a) for u, a, _ in ratings["us"])
    for market in TARGETS:
        parts = chronological_splits(
            ratings[market], market, cold_val[market], cold_test[market],
            args.split_seed,
        )
        for name, rows in zip(split_rows, parts):
            split_rows[name].extend(rows)

    train_users = {(market, user) for market, user, _ in split_rows["train"]}
    for name in ("val", "test", "cold_val", "cold_test"):
        split_rows[name] = [row for row in split_rows[name]
                            if (row[0], row[1]) in train_users]

    user_keys = sorted({(market, user) for rows in split_rows.values()
                        for market, user, _ in rows})
    user_index = {key: index for index, key in enumerate(user_keys)}
    market_index = {market: index for index, market in enumerate(MARKETS)}
    user_country = np.asarray(
        [market_index[market] for market, _ in user_keys], dtype=np.int64
    )
    mapped = {name: map_pairs(rows, user_index, item_index)
              for name, rows in split_rows.items()}

    # 任何非 cold held-out 商品若完全没有源/目标训练交互，移回训练以免伪 warm。
    train_item = set(mapped["train"][:, 1].tolist())
    cold_ids = {
        item_index[asin]
        for market in TARGETS
        for asin in cold_val[market] | cold_test[market]
    }
    for name in ("val", "test"):
        kept, moved = [], []
        for pair in mapped[name].tolist():
            if pair[1] not in train_item and pair[1] not in cold_ids:
                moved.append(pair); train_item.add(pair[1])
            else:
                kept.append(pair)
        if moved:
            mapped["train"] = np.concatenate([
                mapped["train"], np.asarray(moved, dtype=np.int64)
            ], axis=0)
        mapped[name] = np.asarray(kept, dtype=np.int64).reshape(-1, 2)

    detect, lid_hash = detector(Path(args.lid_model)) if args.lid_model else (None, None)
    item_views, image_urls, language_names = [], [], set()
    representative = {}
    for asin in all_items:
        views, record = choose_views(asin, metadata, detect)
        item_views.append(views)
        language_names.update(view["language"] for view in views)
        representative[asin] = record
        image_urls.append(first_image_url(record))
    languages = sorted(language_names)
    language_index = {value: index for index, value in enumerate(languages)}

    shape = (len(all_items), len(MARKETS))
    item_text_lang = np.zeros(shape, dtype=np.int64)
    item_text_source = np.empty(shape, dtype="<U8")
    item_text_role = np.empty(shape, dtype="<U32")
    item_text_valid = np.zeros(shape, dtype=bool)
    item_text_fallback = np.zeros(shape, dtype=bool)
    item_text_hash = np.empty(shape, dtype="<U64")
    item_text_confidence = np.zeros(shape, dtype=np.float32)
    item_text_market = np.full(shape, -1, dtype=np.int64)
    for item, views in enumerate(item_views):
        for view, payload in enumerate(views):
            item_text_lang[item, view] = language_index[payload["language"]]
            item_text_source[item, view] = payload["source"]
            item_text_role[item, view] = payload["role"]
            item_text_valid[item, view] = payload["valid"]
            item_text_fallback[item, view] = payload["is_fallback"]
            item_text_hash[item, view] = payload["content_hash"]
            item_text_confidence[item, view] = payload["language_confidence"]
            item_text_market[item, view] = market_index[payload["source"]]
    item_text_dedup = dedup_mask(item_text_hash, item_text_valid)

    # KG：brand、category、co-purchase、sold-in；没有商品原产地关系。
    categories, brands = set(), set()
    for record in representative.values():
        categories.update(clean_text(x).casefold()
                          for x in (record.get("categories") or []) if clean_text(x))
        brand = brand_of(record)
        if brand:
            brands.add(brand)
    base = len(all_items)
    category_entity = {value: base + i for i, value in enumerate(sorted(categories))}
    base += len(category_entity)
    brand_entity = {value: base + i for i, value in enumerate(sorted(brands))}
    base += len(brand_entity)
    market_entity = {market: base + i for i, market in enumerate(MARKETS)}
    base += len(market_entity)
    triples = []
    for asin, item in item_index.items():
        record = representative[asin]
        for category in (record.get("categories") or []):
            key = clean_text(category).casefold()
            if key in category_entity:
                triples.append((item, 1, category_entity[key]))
        brand = brand_of(record)
        if brand in brand_entity:
            triples.append((item, 2, brand_entity[brand]))
        for other in related_asins(record):
            if other in item_index and item_index[other] != item:
                triples.append((item, 3, item_index[other]))
        for market in MARKETS:
            if asin in market_items[market]:
                triples.append((item, 4, market_entity[market]))
    triples = np.asarray(sorted(set(triples)), dtype=np.int64).reshape(-1, 3)

    item_market_mask = np.zeros((len(MARKETS), len(all_items)), dtype=bool)
    for market, items in market_items.items():
        item_market_mask[market_index[market], [item_index[a] for a in items]] = True
    unknown_origin = len(MARKETS)
    item_country = np.full(len(all_items), unknown_origin, dtype=np.int64)
    image_available = np.asarray([bool(url) for url in image_urls], dtype=bool)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "triples.npy", triples)
    np.save(output_dir / "item_country.npy", item_country)
    np.save(output_dir / "item_text_lang.npy", item_text_lang)
    np.save(output_dir / "item_text_market.npy", item_text_market)
    np.save(output_dir / "item_market_mask.npy", item_market_mask)
    np.save(output_dir / IMAGE_AVAILABLE_FILE, image_available)
    np.savez_compressed(
        output_dir / TEXT_META_FILE,
        language=item_text_lang, source=item_text_source, role=item_text_role,
        valid=item_text_valid, is_fallback=item_text_fallback,
        content_hash=item_text_hash, dedup_mask=item_text_dedup,
        language_confidence=item_text_confidence,
    )
    np.savez_compressed(
        output_dir / "splits.npz", user_country=user_country,
        train_pairs=mapped["train"], val_pairs=mapped["val"],
        test_pairs=mapped["test"], cold_val_pairs=mapped["cold_val"],
        cold_test_pairs=mapped["cold_test"],
    )
    with (output_dir / "items.jsonl").open("w", encoding="utf-8") as handle:
        for asin, views, image_url in zip(all_items, item_views, image_urls):
            handle.write(json.dumps({
                "idx": item_index[asin], "asin": asin, "views": views,
                "markets": [m for m in MARKETS if asin in market_items[m]],
                "image_url": image_url,
            }, ensure_ascii=False) + "\n")

    raw_hashes = {str(path): file_sha256(path) for path in required.values()}
    meta = {
        "schema_version": SCHEMA_VERSION,
        "data_schema": schema_descriptor(),
        "dataset": "XMarket", "category": CATEGORY,
        "source_market": "us", "target_markets": list(TARGETS),
        "markets": list(MARKETS),
        "countries": [m.upper() for m in MARKETS] + ["UNKNOWN_ORIGIN"],
        "n_markets": len(MARKETS), "supports_item_origin": False,
        "languages": languages, "n_views": len(MARKETS),
        "n_users": len(user_keys), "n_items": len(all_items),
        "n_entities": base, "n_relations": 4, "n_triples": len(triples),
        "relations": {"1": "category", "2": "brand",
                      "3": "co_purchase", "4": "sold_in"},
        "split_seed": args.split_seed,
        "split_counts": {name: len(value) for name, value in mapped.items()},
        "catalog_protocol": {
            "version": 1,
            "candidate_source": "market_metadata_before_interaction_filter",
            "interaction_filter": "iterative_bipartite_5_core",
            "market_item_counts": {
                market: len(market_items[market]) for market in MARKETS
            },
            "union_item_count": len(all_items),
            "raw_interaction_counts": {
                market: len(raw_ratings[market]) for market in MARKETS
            },
            "filtered_interaction_counts": {
                market: len(ratings[market]) for market in MARKETS
            },
            "filtered_user_counts": {
                market: len({user for user, _, _ in ratings[market]})
                for market in MARKETS
            },
            "filtered_item_counts": {
                market: len(filtered_items[market]) for market in MARKETS
            },
        },
        "cold_item_counts": {
            "validation": {market: len(cold_val[market]) for market in TARGETS},
            "test": {market: len(cold_test[market]) for market in TARGETS},
        },
        "language_detection": {
            "model": str(args.lid_model) if args.lid_model else None,
            "sha256": lid_hash, "minimum_alignment_confidence": 0.8,
            "disabled": bool(args.disable_alignment),
        },
        "raw_source_hashes": raw_hashes,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "XMarket has no verified item origin; origin-based cross-border metrics are disabled.",
            "Users are market-local; user identities are never linked across markets.",
            "Every observed rating is treated as an implicit relevant interaction, matching FOREC.",
        ],
    }
    (output_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"完成：{output_dir} | users={len(user_keys)} items={len(all_items)}")


if __name__ == "__main__":
    main()
