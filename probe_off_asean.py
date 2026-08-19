# -*- coding: utf-8 -*-
"""
Open Food Facts 东盟数据可用性探测（Task 0a：go/no-go 测量）。

回答三个决定方案成立与否的问题：
  1) 当前 ASEAN-11 各成员市场有多少商品？
  2) origins_tags / manufacturing_places 填充率——这是 item_country 的真实来源；
  3) product_name_<lc> 各语种填充率 + 图片覆盖率——决定多语种与多模态前提是否成立。

用官方 API v2（https://openfoodfacts.github.io/openfoodfacts-server/api/），
带正常 User-Agent、串行请求 + 间隔，不做任何并发压测。

    python probe_off_asean.py
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from off_pipeline.off_common import (
    ASEAN_MEMBERSHIP,
    FIELDS,
    MARKETS,
)

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(encoding="utf-8")

API = "https://world.openfoodfacts.org/api/v2/search"
UA = "ACMR-thesis-research/0.1 (academic data feasibility probe; contact: student)"
PAUSE = 6.0          # 秒；OFF 建议 search 类查询 ≤10 次/分钟

COUNTRIES = MARKETS

SAMPLE_FIELDS = FIELDS


def fetch(params, retries=3):
    url = API + "?" + urllib.parse.urlencode(params)
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            if k == retries - 1:
                return {"_error": f"{type(e).__name__}: {e}"}
            time.sleep(PAUSE * (k + 2))
    return {"_error": "unreachable"}


def count_for(country):
    d = fetch({"countries_tags_en": country, "fields": "code", "page_size": 1})
    return d.get("_error") or d.get("count")


def sample_for(country, n=100):
    return fetch({"countries_tags_en": country, "fields": SAMPLE_FIELDS,
                  "page_size": n, "page": 1})


def pct(a, b):
    return f"{100.0 * a / b:5.1f}%" if b else "   n/a"


def main():
    print("=" * 78)
    print("Open Food Facts 东盟数据探测  (API v2, 串行 + %.0fs 间隔)" % PAUSE)
    print("=" * 78)

    counts, samples = {}, {}
    for country, iso, _ in COUNTRIES:
        c = count_for(country)
        counts[iso] = c
        print(f"[计数] {iso:3s} {country:12s} -> {c}")
        time.sleep(PAUSE)

    print("\n" + "-" * 78)
    print("样本字段填充率（每市场取前 100 条；商品数为 0/报错的市场跳过）")
    print("-" * 78)
    hdr = (f"{'市场':<5}{'样本':>5}{'origins':>9}{'manuf':>8}{'图片':>8}"
           f"{'本地语名称':>12}{'英文名称':>10}{'本地语成分':>12}")
    print(hdr)

    rows = []
    for country, iso, langs in COUNTRIES:
        c = counts.get(iso)
        if not isinstance(c, int) or c == 0:
            continue
        d = sample_for(country)
        time.sleep(PAUSE)
        if "_error" in d:
            print(f"{iso:<5}  取样失败: {d['_error']}")
            continue
        prods = d.get("products", [])
        samples[iso] = prods
        n = len(prods)
        if n == 0:
            continue

        def has(p, key):
            v = p.get(key)
            return bool(v) and v != [] and v != ""

        n_org = sum(has(p, "origins_tags") for p in prods)
        n_man = sum(has(p, "manufacturing_places") for p in prods)
        n_img = sum(has(p, "image_front_url") or has(p, "image_url") for p in prods)
        n_loc = sum(any(has(p, f"product_name_{lc}") for lc in langs) for p in prods)
        n_en = sum(has(p, "product_name_en") for p in prods)
        n_ing = sum(any(has(p, f"ingredients_text_{lc}") for lc in langs)
                    for p in prods)
        print(f"{iso:<5}{n:>5}{pct(n_org,n):>9}{pct(n_man,n):>8}{pct(n_img,n):>8}"
              f"{pct(n_loc,n):>12}{pct(n_en,n):>10}{pct(n_ing,n):>12}")
        rows.append(dict(iso=iso, count=c, n=n, origins=n_org, manuf=n_man,
                         image=n_img, local_name=n_loc, en_name=n_en,
                         local_ingredients=n_ing, langs=langs))

    # 关键判据：origins_tags 里出现的国家是否与销售国不同（真正的跨境信号）
    print("\n" + "-" * 78)
    print("跨境信号抽查：origins_tags 与销售国是否不同")
    print("-" * 78)
    for iso, prods in samples.items():
        withorg = [p for p in prods if p.get("origins_tags")]
        if not withorg:
            print(f"{iso}: 样本中无 origins_tags")
            continue
        diff = 0
        for p in withorg:
            orig = {o.split(":")[-1].lower() for o in p["origins_tags"]}
            sold = {c.split(":")[-1].lower() for c in p.get("countries_tags", [])}
            if orig and sold and not (orig & sold):
                diff += 1
        ex = withorg[0]
        print(f"{iso}: 有 origins 的 {len(withorg)} 条中 {diff} 条产地≠销售国 "
              f"({pct(diff, len(withorg)).strip()})  例: {ex['origins_tags']}")

    out = {
        "membership": ASEAN_MEMBERSHIP,
        "member_iso": [iso for _, iso, _ in COUNTRIES],
        "counts": counts,
        "fill_rates": rows,
    }
    output = Path(__file__).resolve().parent / "off_asean_probe.json"
    with output.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {output}")


if __name__ == "__main__":
    main()
