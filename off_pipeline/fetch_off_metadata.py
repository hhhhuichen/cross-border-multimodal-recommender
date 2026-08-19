# -*- coding: utf-8 -*-
"""
第 1 步：从 Open Food Facts 官方 API v2 抓取 ASEAN-11 商品元数据。

  * 官方公开 API + 学术 User-Agent，串行请求 + 6s 间隔，遵守 OFF 的礼貌抓取指引
    （search 类接口建议 ≤10 次/分钟）；503 指数退避。
  * 可断点续跑：每市场一个 .state 文件记录已完成页数，重跑自动跳过。
  * API 搜索深度上限 10,000 条；超限市场用两个排序视图增加覆盖，但清单会
    明确标为 depth-limited，正式快照应改用 fetch_off_dump.py。

    python fetch_off_metadata.py            # 全量（约 1~2 小时，取决于服务器状态）
    python fetch_off_metadata.py --markets TH ID VN
    python fetch_off_metadata.py --market thailand   # 兼容单市场入口
"""
import argparse
import json
import math
import time
from off_common import (
    FIELDS,
    MARKETS,
    RAW_DIR,
    http_get_json,
    load_collection_state,
    resolve_markets,
    utc_now,
    write_collection_manifest,
    write_json_atomic,
)

PAGE_SIZE = 100
PAUSE = 6.0
DEPTH_CAP = 10000        # API 搜索最大可达深度（page * page_size）


def fetch_market(market):
    country, iso, _ = market
    out_path = RAW_DIR / f"{country}.jsonl"
    st_path = RAW_DIR / f"{country}.state"
    st = load_collection_state(st_path, market, "api-v2")
    if st.get("done"):
        if not out_path.is_file():
            raise FileNotFoundError(
                f"{st_path.name} 标记完成但 {out_path.name} 不存在；"
                "请恢复原始文件或移走失效状态后重采"
            )
        write_json_atomic(st_path, st)
        print(f"[{country}/{iso}] 已完成，跳过")
        return
    if not st_path.exists() and out_path.is_file() and out_path.stat().st_size:
        raise RuntimeError(
            f"{out_path.name} 存在但没有断点状态；拒绝直接追加混合快照"
        )

    d = http_get_json({"countries_tags_en": country, "fields": "code",
                       "page_size": 1})
    if "_error" in d:
        print(f"[{country}] 计数失败: {d['_error']}（保留状态，下次续跑）")
        return
    total = int(d.get("count", 0))
    st["reported_catalog_count"] = total
    st["started_at"] = st.get("started_at") or utc_now()
    time.sleep(PAUSE)
    if total == 0:
        empty = out_path.with_name(f".{out_path.name}.tmp")
        empty.write_text("", encoding="utf-8")
        empty.replace(out_path)
        st["done"] = True
        st["completed_at"] = utc_now()
        st["catalog_coverage"] = "complete_api_pagination"
        write_json_atomic(st_path, st)
        print(f"[{country}/{iso}] 0 条商品")
        return

    # 超过深度上限 -> 两趟不同排序维度覆盖（v2 不支持降序语法，
    # 用 last_modified_t 作第二维度，与 created_t 序不同，重叠部分去重）
    passes = (["created_t"] if total <= DEPTH_CAP - 100
              else ["created_t", "last_modified_t"])
    reachable = min(total, DEPTH_CAP)
    print(f"[{country}/{iso}] 共 {total} 条，{len(passes)} 趟抓取")

    with out_path.open("a", encoding="utf-8") as f:
        for sort_by in passes:
            n_pages = math.ceil(min(total, DEPTH_CAP) / PAGE_SIZE)
            if len(passes) == 2 and sort_by == passes[1]:
                # 第二趟只需要补第一趟够不到的深度
                n_pages = math.ceil((total - reachable + 2 * PAGE_SIZE) / PAGE_SIZE)
            done_pages = st["passes"].get(sort_by, 0)
            for page in range(done_pages + 1, n_pages + 1):
                params = {"countries_tags_en": country, "fields": FIELDS,
                          "page_size": PAGE_SIZE, "page": page,
                          "sort_by": sort_by}
                d = http_get_json(params)
                if "_error" in d:
                    print(f"[{country}] 第 {page}/{n_pages} 页失败: "
                          f"{d['_error']}（状态已保存）")
                    write_json_atomic(st_path, st)
                    return
                prods = d.get("products", [])
                for p in prods:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
                f.flush()
                st["passes"][sort_by] = page
                write_json_atomic(st_path, st)
                if page % 10 == 0 or page == n_pages:
                    print(f"[{country}] {sort_by} {page}/{n_pages} 页 "
                          f"(+{len(prods)})", flush=True)
                if not prods:          # 提前到底
                    break
                time.sleep(PAUSE)
    st["done"] = True
    st["completed_at"] = utc_now()
    st["catalog_coverage"] = (
        "complete_api_pagination" if total <= DEPTH_CAP - 100
        else "api_depth_limited_union_of_sorts"
    )
    write_json_atomic(st_path, st)
    print(f"[{country}/{iso}] 完成")


def selected_markets(args, parser):
    values = args.markets if args.markets is not None else args.market
    try:
        return resolve_markets(values)
    except ValueError as exc:
        parser.error(str(exc))


def market_done(market):
    country = market[0]
    state = load_collection_state(
        RAW_DIR / f"{country}.state", market, "api-v2"
    )
    raw_path = RAW_DIR / f"{country}.jsonl"
    if state.get("done") and not raw_path.is_file():
        raise FileNotFoundError(
            f"{country}.state 标记完成但 {raw_path.name} 不存在"
        )
    return bool(state.get("done")) and raw_path.is_file()


def main():
    ap = argparse.ArgumentParser(
        description="从 Open Food Facts API 采集 ASEAN-11 商品元数据"
    )
    group = ap.add_mutually_exclusive_group()
    group.add_argument(
        "--markets", nargs="+", default=None,
        help="采集指定 ASEAN 市场；接受 ISO 或 OFF 英文名，可空格分隔",
    )
    group.add_argument(
        "--market", action="append", default=None,
        help="兼容入口：指定一个市场，可重复使用",
    )
    ap.add_argument("--list-markets", action="store_true",
                    help="列出当前成员契约后退出")
    args = ap.parse_args()
    if args.list_markets:
        for country, iso, languages in MARKETS:
            print(f"{iso}\t{country}\t{','.join(languages)}")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    todo = selected_markets(args, ap)
    for round_ in range(3):            # 整体重试轮：失败的市场再来
        pending = [market for market in todo if not market_done(market)]
        if not pending:
            break
        if round_:
            print(f"== 第 {round_ + 1} 轮，剩余 {len(pending)} 个市场，"
                  f"等待 120s 后继续 ==")
            time.sleep(120)
        for market in pending:
            fetch_market(market)
            time.sleep(PAUSE)

    done = sum(market_done(market) for market in todo)
    write_collection_manifest(todo, "api-v2")
    print(f"\n== 抓取结束：{done}/{len(todo)} 个市场完成 ==")


if __name__ == "__main__":
    main()
