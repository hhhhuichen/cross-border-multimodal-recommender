# -*- coding: utf-8 -*-
"""
第 1 步（批量路线）：从 OFF 官方 Hugging Face 全量 Parquet 导出提取东盟市场商品。

OFF 文档明确建议批量提取用 dump 而非搜索 API（对社区服务器更友好，
也不受 503/401 与 1 万条搜索深度限制）。本脚本产出与 fetch_off_metadata.py
**完全相同格式**的 raw/<market>.jsonl，下游 build_off_dataset.py 无需改动。

  * 下载 openfoodfacts/product-database 的 food.parquet（约 4~5GB，断点续传）；
  * 逐 row-group 流式过滤 countries_tags 含 ASEAN-11 目标市场的行；
  * Parquet 的多语言字段是 [{lang, text}] 结构，翻译回 API 的
    product_name_<lc> 扁平字段；图片 URL 按 OFF 规则从 code + rev 重建。
  * huggingface.co 不可达时自动切 hf-mirror.com 镜像。

    python fetch_off_dump.py
    python fetch_off_dump.py --markets TH ID VN TL
    python fetch_off_dump.py --parquet /path/to/food.parquet
"""
import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from off_common import (
    INGREDIENT_LANGS,
    MARKETS,
    RAW_DIR,
    TEXT_LANGS,
    collection_state,
    file_sha256,
    resolve_markets,
    tag_to_iso,
    utc_now,
    write_collection_manifest,
    write_json_atomic,
)

REPO = "openfoodfacts/product-database"
FILENAME = "food.parquet"

COLS = ["code", "lang", "product_name", "ingredients_text", "brands",
        "brands_tags", "categories_tags", "labels_tags", "origins_tags",
        "manufacturing_places", "countries_tags", "images", "quantity",
        "created_t", "last_modified_t"]


def download():
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(REPO, FILENAME, repo_type="dataset")
    except Exception as e:
        print(f"huggingface.co 直连失败（{type(e).__name__}），切换 hf-mirror.com")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        from importlib import reload
        import huggingface_hub
        reload(huggingface_hub)
        return huggingface_hub.hf_hub_download(REPO, FILENAME,
                                               repo_type="dataset")


def lang_map(cell):
    """[{lang,text}] -> {lang: text}；'main' 条目单独返回。"""
    out, main = {}, ""
    for e in cell or []:
        if not e:
            continue
        lc, tx = e.get("lang"), (e.get("text") or "").strip()
        if not tx:
            continue
        if lc == "main":
            main = tx
        elif lc:
            out[lc] = tx
    return main, out


def image_url(code, images, main_lang):
    """重建正面图 400px URL。选中图（front_*，带 rev）优先，原始图兜底。"""
    if not images:
        return ""
    code = str(code)
    if len(code) > 8:
        code = code.zfill(13)
        split = f"{code[0:3]}/{code[3:6]}/{code[6:9]}/{code[9:]}"
    else:
        split = code
    base = f"https://images.openfoodfacts.org/images/products/{split}"
    fronts = [e for e in images if e and str(e.get("key", "")).startswith("front")]
    for pref in (f"front_{main_lang}", "front_en", None):
        for e in fronts:
            if pref is None or e.get("key") == pref:
                rev = e.get("rev")
                if rev not in (None, ""):
                    return f"{base}/{e['key']}.{rev}.400.jpg"
    for e in images or []:                 # 无选中图 -> 用第一张原始上传图
        imgid = e.get("imgid") or e.get("key")
        if imgid and str(imgid).isdigit():
            return f"{base}/{imgid}.400.jpg"
    return ""


def to_api_record(row):
    """Parquet 行 -> API JSON 同构记录（只含 build_off_dataset 用到的字段）。"""
    main_lang = row.get("lang") or "en"
    name_main, names = lang_map(row.get("product_name"))
    ing_main, ings = lang_map(row.get("ingredients_text"))
    rec = {
        "code": str(row.get("code") or ""),
        "lang": main_lang, "lc": main_lang,
        "product_name": name_main or names.get(main_lang, ""),
        "brands": row.get("brands") or "",
        "brands_tags": list(row.get("brands_tags") or []),
        "categories_tags": list(row.get("categories_tags") or []),
        "labels_tags": list(row.get("labels_tags") or []),
        "origins_tags": list(row.get("origins_tags") or []),
        "manufacturing_places": row.get("manufacturing_places") or "",
        "countries_tags": list(row.get("countries_tags") or []),
        "quantity": row.get("quantity") or "",
        "created_t": row.get("created_t"),
        "last_modified_t": row.get("last_modified_t"),
        "image_front_url": image_url(row.get("code"), row.get("images"),
                                     main_lang),
    }
    for lc in TEXT_LANGS:
        if lc in names:
            rec[f"product_name_{lc}"] = names[lc]
    if main_lang in TEXT_LANGS and main_lang not in names and name_main:
        rec[f"product_name_{main_lang}"] = name_main
    for lc in INGREDIENT_LANGS:
        if lc in ings:
            rec[f"ingredients_text_{lc}"] = ings[lc]
    if main_lang in INGREDIENT_LANGS and main_lang not in ings and ing_main:
        rec[f"ingredients_text_{main_lang}"] = ing_main
    return rec


def selected_markets(args, parser):
    values = args.markets if args.markets is not None else args.market
    try:
        return resolve_markets(values)
    except ValueError as exc:
        parser.error(str(exc))


def extract_parquet(path, markets, raw_dir=RAW_DIR, progress=True):
    """从 OFF Parquet 原子地产出逐 ASEAN 市场 JSONL，返回采集统计。"""
    path = Path(path)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    selected_iso = {market[1] for market in markets}
    country_by_iso = {iso: country for country, iso, _ in markets}

    pf = pq.ParquetFile(path)
    if "countries_tags" not in pf.schema_arrow.names:
        raise ValueError("OFF Parquet 缺必需列 countries_tags")
    have = [column for column in COLS if column in pf.schema_arrow.names]
    missing = sorted(set(COLS) - set(have))
    if missing:
        print(f"[警告] dump 缺列 {missing}，对应字段将为空")

    part_paths = {
        iso: raw_dir / f".{country_by_iso[iso]}.{os.getpid()}.jsonl.part"
        for iso in selected_iso
    }
    writers = {
        iso: part_paths[iso].open("w", encoding="utf-8")
        for iso in selected_iso
    }
    per_market = {iso: 0 for iso in selected_iso}
    n_scan = n_unique_hit = 0
    try:
        for row_group in range(pf.num_row_groups):
            tags = pf.read_row_group(
                row_group, columns=["countries_tags"]
            ).column(0).to_pylist()
            n_scan += len(tags)
            matched = []
            for row_index, row_tags in enumerate(tags):
                hit = {
                    iso for iso in (
                        tag_to_iso(tag)
                        for tag in (row_tags or [])
                    )
                    if iso in selected_iso
                }
                if hit:
                    matched.append((row_index, hit))

            if matched:
                table = pf.read_row_group(row_group, columns=have)
                indices = pa.array(
                    [row_index for row_index, _ in matched],
                    type=pa.int64(),
                )
                rows = table.take(indices).to_pylist()
            else:
                rows = []
            for row, (_, hit) in zip(rows, matched):
                rec = to_api_record(row)
                if not rec["code"]:
                    continue
                line = json.dumps(rec, ensure_ascii=False) + "\n"
                for iso in hit:
                    writers[iso].write(line)
                    per_market[iso] += 1
                n_unique_hit += 1
            if progress:
                print(
                    f"  row-group {row_group + 1}/{pf.num_row_groups} | "
                    f"扫描 {n_scan} | 命中 {n_unique_hit}",
                    flush=True,
                )
    finally:
        for handle in writers.values():
            handle.close()

    for market in markets:
        country, iso, _ = market
        destination = raw_dir / f"{country}.jsonl"
        part_paths[iso].replace(destination)
        state = collection_state(
            market,
            "official-huggingface-parquet",
            done=True,
            catalog_coverage="complete_bulk_snapshot",
            completed_at=utc_now(),
            record_count=per_market[iso],
            source_file=path.name,
        )
        write_json_atomic(raw_dir / f"{country}.state", state)

    return {
        "scanned_rows": n_scan,
        "matched_unique_rows": n_unique_hit,
        "market_record_counts": per_market,
        "missing_columns": missing,
    }


def main():
    parser = argparse.ArgumentParser(
        description="从 Open Food Facts 官方 Parquet 提取 ASEAN-11 商品目录"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--markets", nargs="+", default=None,
                       help="指定 ASEAN 市场；接受 ISO 或 OFF 英文名")
    group.add_argument("--market", action="append", default=None,
                       help="兼容入口：指定一个市场，可重复使用")
    parser.add_argument("--parquet", default=None,
                        help="使用本地 food.parquet，省略时从官方 HF 数据集下载")
    parser.add_argument("--list-markets", action="store_true")
    args = parser.parse_args()
    if args.list_markets:
        for country, iso, languages in MARKETS:
            print(f"{iso}\t{country}\t{','.join(languages)}")
        return

    markets = selected_markets(args, parser)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.parquet:
        path = Path(args.parquet).expanduser().resolve()
        if not path.is_file():
            parser.error(f"--parquet 文件不存在: {path}")
    else:
        print("下载 food.parquet（断点续传，约 4~5GB）...", flush=True)
        path = Path(download())
    print(f"就绪: {path}", flush=True)
    stats = extract_parquet(path, markets)
    snapshot = {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    write_collection_manifest(
        markets,
        "official-huggingface-parquet",
        source_snapshot=snapshot,
    )
    print(
        f"\n完成：扫描 {stats['scanned_rows']} 条，命中 ASEAN 商品 "
        f"{stats['matched_unique_rows']} 条（跨市场重复计数前）"
    )


if __name__ == "__main__":
    main()
