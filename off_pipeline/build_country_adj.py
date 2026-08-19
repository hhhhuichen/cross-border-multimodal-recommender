# -*- coding: utf-8 -*-
"""构建模拟器贸易权重和模型侧国家地理图。

``trade_shares.npy`` 的行是 ASEAN 销售市场，列是 ``processed/meta.json``
中的完整原产国词表（包括全部可解析非 ASEAN 国家和 OTHER）。贸易权重只供用户
交互模拟器使用；模型仍只接收独立构建的 ASEAN 地理邻接图。
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from off_common import ASEAN_MEMBERSHIP, DATA_DIR, MARKET_ISO, PROC_DIR, UA


TRADE_SCHEMA_VERSION = 2

# UN M49。覆盖 off_common.TAG2ISO 能解析出的全部国家；区域标签不会进入产地词表。
M49 = {
    "AE": 784, "AR": 32, "AT": 40, "AU": 36, "BD": 50, "BE": 56,
    "BG": 100, "BN": 96, "BR": 76, "CA": 124, "CH": 756,
    "CL": 152, "CN": 156, "CO": 170, "CZ": 203, "DE": 276,
    "DK": 208, "EC": 218, "EG": 818, "ES": 724, "ET": 231,
    "FI": 246, "FR": 250, "GB": 826, "GR": 300, "HK": 344,
    "HU": 348, "ID": 360, "IE": 372, "IL": 376, "IN": 356,
    "IR": 364, "IT": 380, "JP": 392, "KE": 404, "KH": 116,
    "KR": 410, "LA": 418, "LK": 144, "MA": 504, "MM": 104,
    "MX": 484, "MY": 458, "NL": 528, "NO": 578, "NP": 524,
    "NZ": 554, "PE": 604, "PH": 608, "PK": 586, "PL": 616,
    "PT": 620, "RO": 642, "RU": 643, "SA": 682, "SE": 752,
    "SG": 702, "SK": 703, "TH": 764, "TL": 626, "TR": 792,
    "TW": 158, "UA": 804, "US": 842, "VN": 704, "ZA": 710,
}
COMTRADE = (
    "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
    "?reporterCode={rep}&period={year}&partnerCode={partners}"
    "&partner2Code=0&cmdCode=TOTAL&flowCode={flow}"
    "&customsCode=C00&motCode=0"
)

# 模型侧只使用地理/固定通道先验，不消费上面的 Comtrade 数值。
GEO_LINKS = [
    ("TH", "MM"), ("TH", "LA"), ("TH", "KH"), ("TH", "MY"),
    ("VN", "LA"), ("VN", "KH"), ("KH", "LA"), ("MM", "LA"),
    ("MY", "SG"), ("MY", "ID"), ("MY", "BN"), ("MY", "PH"),
    ("ID", "SG"), ("ID", "PH"), ("ID", "TL"),
]


def fetch(url, retries=4):
    delay = 10.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            if attempt == retries - 1:
                return {"_error": f"{type(exc).__name__}: {exc}"}
            time.sleep(delay)
            delay *= 2
    return {"_error": "unreachable"}


def _code(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_aggregate_row(row):
    """排除接口默认展开的第二伙伴、海关制度和运输方式明细。"""
    return (
        _code(row.get("partner2Code", 0)) == 0
        and row.get("customsCode", "C00") == "C00"
        and _code(row.get("motCode", 0)) == 0
    )


def _response_error(response):
    return response.get("_error") or response.get("error") or ""


def _validate_granularity(response, expected_max):
    """preview 返回明细展开/截断时拒绝继续，避免重复累加。"""
    count = response.get("count")
    if count is not None and int(count) > expected_max:
        raise RuntimeError(
            "UN Comtrade 未按 partner2Code=0/customsCode=C00/motCode=0 "
            f"聚合：请求至多 {expected_max} 个伙伴，却返回 {count} 条"
        )


def _request(reporter, partners, year, flow, fetcher, request_delay):
    partner_codes = ",".join(str(M49[c]) for c in partners)
    url = COMTRADE.format(
        rep=M49[reporter], year=year, partners=partner_codes, flow=flow
    )
    response = fetcher(url)
    if request_delay > 0:
        time.sleep(request_delay)
    return response, url


def load_origin_contract(meta_path=None):
    """读取构建后的市场/产地轴；贸易矩阵不得先于商品词表猜测列。"""
    path = Path(meta_path or (PROC_DIR / "meta.json"))
    if not path.is_file():
        raise FileNotFoundError(
            f"缺 {path}；必须先运行 build_off_dataset.py 再构建贸易权重"
        )
    meta = json.loads(path.read_text(encoding="utf-8"))
    markets = list(meta.get("markets", ()))
    origins = list(meta.get("countries", ()))
    if markets != MARKET_ISO:
        raise ValueError("processed/meta.json 的 ASEAN 市场顺序与当前成员契约不一致")
    if origins[:len(markets)] != markets or not origins:
        raise ValueError("processed/meta.json 的 countries 未以市场轴开头")
    if origins[-1] != "OTHER":
        raise ValueError("processed/meta.json 的 countries 必须以 OTHER 收尾")
    missing_m49 = [
        origin for origin in origins
        if origin != "OTHER" and origin not in M49
    ]
    if missing_m49:
        raise ValueError(
            "以下原产国缺 UN M49 编码，不能静默使用统一权重: "
            + ", ".join(missing_m49)
        )
    return markets, origins


def get_reported_imports(
    year, markets, origins, *, fetcher=fetch, request_delay=3.0
):
    """查询目标市场自报进口；OTHER 为世界总额扣除显式伙伴后的剩余额。"""
    origin_index = {origin: index for index, origin in enumerate(origins)}
    market_index = {market: index for index, market in enumerate(markets)}
    code_to_origin = {M49[origin]: origin for origin in origins if origin != "OTHER"}
    explicit = [origin for origin in origins if origin != "OTHER"]
    values = np.zeros((len(markets), len(origins)), dtype=np.float64)
    covered = []
    urls = []

    for market in markets:
        partners = [origin for origin in explicit if origin != market]
        # partnerCode=0 是世界总额，用于给被折叠到 OTHER 的长尾伙伴分配残差。
        partner_codes = "0," + ",".join(str(M49[c]) for c in partners)
        url = COMTRADE.format(
            rep=M49[market], year=year, partners=partner_codes, flow="M"
        )
        response = fetcher(url)
        if request_delay > 0:
            time.sleep(request_delay)
        urls.append(url)
        if _response_error(response) or not response.get("data"):
            detail = _response_error(response) or "空响应"
            print(f"  [M] {market} {year}: 无数据（{detail}）")
            continue
        _validate_granularity(response, len(partners) + 1)

        world_total = 0.0
        for row in response["data"]:
            if not _is_aggregate_row(row):
                continue
            partner_code = _code(row.get("partnerCode"))
            value = float(row.get("primaryValue") or 0.0)
            if value <= 0:
                continue
            if partner_code == 0:
                world_total += value
                continue
            origin = code_to_origin.get(partner_code)
            if origin and origin != market:
                values[market_index[market], origin_index[origin]] += value

        row = values[market_index[market]]
        named_total = float(row.sum())
        if "OTHER" in origin_index and world_total > 0:
            row[origin_index["OTHER"]] = max(world_total - named_total, 0.0)
        if row.sum() > 0:
            covered.append(market)
        print(
            f"  [M] {market} {year}: 显式伙伴 {int(np.count_nonzero(row))} 个"
            f" | 世界总额 {'有' if world_total > 0 else '无'}"
        )
    return values, covered, urls


def get_mirror_exports(
    year, markets, origins, missing_markets, *, fetcher=fetch, request_delay=3.0
):
    """用原产伙伴自报的出口镜像补目标市场缺失行。"""
    origin_index = {origin: index for index, origin in enumerate(origins)}
    market_index = {market: index for index, market in enumerate(markets)}
    code_to_market = {M49[market]: market for market in missing_markets}
    values = np.zeros((len(markets), len(origins)), dtype=np.float64)
    urls = []
    for origin in origins:
        if origin == "OTHER":
            continue
        partners = [market for market in missing_markets if market != origin]
        if not partners:
            continue
        response, url = _request(
            origin, partners, year, "X", fetcher, request_delay
        )
        urls.append(url)
        if _response_error(response) or not response.get("data"):
            continue
        _validate_granularity(response, len(partners))
        for row in response["data"]:
            if not _is_aggregate_row(row):
                continue
            market = code_to_market.get(_code(row.get("partnerCode")))
            value = float(row.get("primaryValue") or 0.0)
            if market and value > 0:
                values[market_index[market], origin_index[origin]] += value
    return values, urls


def finalize_trade_shares(imports, provenance, pool_prior=0.01):
    """归一化并以 ASEAN 汇总进口分布平滑稀疏伙伴。"""
    imports = np.asarray(imports, dtype=np.float64)
    if imports.ndim != 2 or not np.isfinite(imports).all() or (imports < 0).any():
        raise ValueError("Comtrade 进口矩阵必须是有限非负二维数组")
    if not 0.0 <= pool_prior < 1.0:
        raise ValueError("pool_prior 必须位于 [0, 1)")
    covered = imports.sum(axis=1) > 0
    if not covered.any():
        raise RuntimeError("Comtrade 所有目标市场均无有效贸易值，拒绝生成伪权重")
    pooled = imports[covered].sum(axis=0)
    if pooled.sum() <= 0:
        raise RuntimeError("Comtrade 汇总伙伴分布为空")
    pooled /= pooled.sum()

    shares = np.zeros_like(imports)
    for row_index in range(imports.shape[0]):
        total = imports[row_index].sum()
        if total > 0:
            row = imports[row_index] / total
            shares[row_index] = (1.0 - pool_prior) * row + pool_prior * pooled
        else:
            shares[row_index] = pooled
            provenance[row_index] = "ASEAN pooled imports fallback"
    shares /= shares.sum(axis=1, keepdims=True)
    return shares, pooled


def build_trade_matrix(
    markets, origins, years, *, fetcher=fetch, request_delay=3.0,
    pool_prior=0.01,
):
    imports = np.zeros((len(markets), len(origins)), dtype=np.float64)
    market_index = {market: index for index, market in enumerate(markets)}
    provenance = {}
    query_urls = []

    for year in years:
        reported, covered, urls = get_reported_imports(
            year, markets, origins, fetcher=fetcher,
            request_delay=request_delay,
        )
        query_urls.extend(urls)
        for market in covered:
            if market not in provenance:
                imports[market_index[market]] = reported[market_index[market]]
                provenance[market] = f"self-reported imports {year}"
        if len(provenance) == len(markets):
            break

    missing = [market for market in markets if market not in provenance]
    if missing:
        print(f"缺自报进口: {missing}，查询非 ASEAN 原产伙伴的出口镜像")
        for year in years:
            mirrored, urls = get_mirror_exports(
                year, markets, origins, missing, fetcher=fetcher,
                request_delay=request_delay,
            )
            query_urls.extend(urls)
            for market in list(missing):
                row = mirrored[market_index[market]]
                if row.sum() > 0:
                    imports[market_index[market]] = row
                    provenance[market] = f"partner-reported exports mirror {year}"
                    missing.remove(market)
            if not missing:
                break

    row_provenance = {
        market_index[market]: source for market, source in provenance.items()
    }
    shares, pooled = finalize_trade_shares(
        imports, row_provenance, pool_prior=pool_prior
    )
    provenance = {
        market: row_provenance[market_index[market]] for market in markets
    }
    return shares, pooled, provenance, query_urls


def build_geo(markets):
    index = {market: position for position, market in enumerate(markets)}
    geo = np.zeros((len(markets), len(markets)), dtype=np.float32)
    for left, right in GEO_LINKS:
        if left in index and right in index:
            geo[index[left], index[right]] = 1.0
            geo[index[right], index[left]] = 1.0
    return geo


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="构建 ASEAN 市场到完整商品原产国的 Comtrade 权重"
    )
    parser.add_argument("--years", nargs="+", type=int, default=[2024, 2023])
    parser.add_argument("--request-delay", type=float, default=3.0)
    parser.add_argument(
        "--pool-prior", type=float, default=0.01,
        help="用于稀疏伙伴的 ASEAN 汇总进口先验混合比例",
    )
    args = parser.parse_args(argv)
    if args.request_delay < 0:
        parser.error("--request-delay 不能为负")
    markets, origins = load_origin_contract()
    shares, pooled, provenance, query_urls = build_trade_matrix(
        markets, origins, args.years, request_delay=args.request_delay,
        pool_prior=args.pool_prior,
    )
    geo = build_geo(markets)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / "trade_shares.npy", shares.astype(np.float32))
    np.save(DATA_DIR / "country_adj_geo.npy", geo)
    meta = {
        "trade_schema_version": TRADE_SCHEMA_VERSION,
        "markets": markets,
        "trade_origins": origins,
        "membership": ASEAN_MEMBERSHIP,
        "trade": {
            "source": "UN Comtrade public preview API",
            "cmd_code": "TOTAL",
            "flow": "imports with partner-reported export mirror fallback",
            "years_requested": args.years,
            "shape": list(shares.shape),
            "row_axis": "ASEAN destination market",
            "column_axis": "observed item origin country",
            "other_definition": (
                "world import total minus explicitly retained origin partners"
            ),
            "pool_prior": args.pool_prior,
            "pooled_origin_share": {
                origin: float(pooled[index]) for index, origin in enumerate(origins)
            },
            "provenance": provenance,
            "query_count": len(query_urls),
            "query_urls": query_urls,
        },
        "geo": {
            "source": "fixed geographic adjacency",
            "shape": list(geo.shape),
        },
        "note": (
            "trade_shares 只喂模拟器；country_adj_geo 只喂模型。"
            "两者保持信息源分离，防止循环论证。"
        ),
    }
    (DATA_DIR / "country_adj_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n进口份额（每个市场 Top 8 原产伙伴）：")
    for row_index, market in enumerate(markets):
        order = np.argsort(shares[row_index])[::-1][:8]
        summary = " ".join(
            f"{origins[column]}={100 * shares[row_index, column]:.1f}%"
            for column in order if shares[row_index, column] > 0
        )
        print(f"  {market}: {summary}")
    print(
        f"\n已写入 {DATA_DIR}/trade_shares.npy "
        f"shape={shares.shape} 和 country_adj_geo.npy shape={geo.shape}"
    )
    print(f"来源: {provenance}")


if __name__ == "__main__":
    main()
