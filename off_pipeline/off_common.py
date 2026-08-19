# -*- coding: utf-8 -*-
"""OFF 东盟商品管线的共享常量、成员范围与采集清单工具。"""
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout.reconfigure(encoding="utf-8")

# 管线按源码仓库根目录解析，产物与代码保持相互独立。
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_off"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
IMG_DIR = DATA_DIR / "images"

UA = ("ACMR-thesis-research/0.2 (academic; ASEAN cross-border food recommendation; "
      "single student, serial polite requests)")

# 东帝汶于 2025-10-26 成为第 11 个成员国。成员版本必须进入采集清单，
# 避免把 ASEAN Plus Three 国家误写成东盟成员。
ASEAN_MEMBERSHIP = {
    "name": "Association of Southeast Asian Nations",
    "effective_date": "2025-10-26",
    "member_count": 11,
    "source_url": "https://asean.org/about-asean/",
}

# (OFF countries_tags_en 值, ISO 3166-1 alpha-2, 本地语言优先级)
# 顺序是数据集中的市场编号契约；只能通过显式重建改变。
MARKETS = [
    ("thailand",    "TH", ["th", "en"]),
    ("singapore",   "SG", ["en", "zh", "ms"]),
    ("philippines", "PH", ["tl", "en"]),
    ("indonesia",   "ID", ["id", "en"]),
    ("malaysia",    "MY", ["ms", "en"]),
    ("timor-leste", "TL", ["tet", "pt", "id", "en"]),
    ("vietnam",     "VN", ["vi", "en"]),
    ("cambodia",    "KH", ["km", "en"]),
    ("brunei",      "BN", ["ms", "en"]),
    ("myanmar",     "MM", ["my", "en"]),
    ("laos",        "LA", ["lo", "th", "en"]),
]
MARKET_ISO = [m[1] for m in MARKETS]

TEXT_LANGS = [
    "zh", "th", "vi", "id", "ms", "en", "tl", "my", "km", "lo",
    "tet", "pt",
]
INGREDIENT_LANGS = ["th", "vi", "id", "ms", "en", "zh", "tet", "pt"]

FIELDS = ",".join(
    ["code", "lang", "lc", "product_name", "brands", "brands_tags",
     "categories_tags", "labels_tags", "origins_tags", "manufacturing_places",
     "countries_tags", "image_front_url", "image_front_small_url", "quantity",
     "created_t", "last_modified_t"]
    + [f"product_name_{lc}" for lc in TEXT_LANGS]
    + [f"ingredients_text_{lc}" for lc in INGREDIENT_LANGS]
)

API = "https://world.openfoodfacts.org/api/v2/search"
COLLECTION_MANIFEST = RAW_DIR / "collection_manifest.json"
COLLECTION_SCHEMA_VERSION = 1
OFF_PROVIDER = {
    "name": "Open Food Facts",
    "api_url": API,
    "bulk_dataset": "openfoodfacts/product-database",
    "bulk_file": "food.parquet",
    "data_page": "https://world.openfoodfacts.org/data",
    "terms_url": "https://world.openfoodfacts.org/terms-of-use",
}

# OFF origins/countries 标签 -> ISO。ASEAN-11 + 常见食品原产国。
# 值为 None 表示明确不是国家（区域标签），直接跳过。
TAG2ISO = {
    "thailand": "TH", "singapore": "SG", "philippines": "PH", "indonesia": "ID",
    "malaysia": "MY", "china": "CN", "vietnam": "VN", "viet-nam": "VN",
    "cambodia": "KH", "brunei": "BN", "myanmar": "MM", "burma": "MM", "laos": "LA",
    "timor-leste": "TL", "east-timor": "TL", "timor-leste-democratic-republic-of": "TL",
    "japan": "JP", "south-korea": "KR", "korea": "KR", "taiwan": "TW",
    "hong-kong": "HK", "india": "IN", "united-states": "US",
    "united-states-of-america": "US", "usa": "US", "australia": "AU",
    "new-zealand": "NZ", "france": "FR", "germany": "DE", "italy": "IT",
    "spain": "ES", "united-kingdom": "GB", "uk": "GB", "netherlands": "NL",
    "belgium": "BE", "switzerland": "CH", "poland": "PL", "turkey": "TR",
    "austria": "AT", "denmark": "DK", "sweden": "SE", "norway": "NO",
    "portugal": "PT", "greece": "GR", "ireland": "IE", "canada": "CA",
    "mexico": "MX", "brazil": "BR", "argentina": "AR", "chile": "CL",
    "russia": "RU", "ukraine": "UA", "czech-republic": "CZ", "slovakia": "SK",
    "hungary": "HU", "romania": "RO", "bulgaria": "BG", "finland": "FI",
    "israel": "IL", "saudi-arabia": "SA", "united-arab-emirates": "AE",
    "pakistan": "PK", "bangladesh": "BD", "sri-lanka": "LK", "nepal": "NP",
    "iran": "IR", "egypt": "EG", "morocco": "MA", "south-africa": "ZA",
    "ethiopia": "ET", "kenya": "KE", "colombia": "CO", "peru": "PE",
    "ecuador": "EC", "european-union": None, "europe": None, "asia": None,
    "southeast-asia": None, "world": None, "unknown": None, "various": None,
}

# 自由文本 manufacturing_places 的子串匹配表（小写包含即命中，按长度降序试）
NAME2ISO = {
    "thailand": "TH", "singapore": "SG", "philippines": "PH", "indonesia": "ID",
    "malaysia": "MY", "china": "CN", "vietnam": "VN", "viet nam": "VN",
    "cambodia": "KH", "brunei": "BN", "myanmar": "MM", "laos": "LA",
    "timor-leste": "TL", "timor leste": "TL", "east timor": "TL",
    "japan": "JP", "korea": "KR", "taiwan": "TW", "hong kong": "HK",
    "india": "IN", "united states": "US", "u.s.a": "US", "usa": "US",
    "australia": "AU", "new zealand": "NZ", "france": "FR", "germany": "DE",
    "italy": "IT", "spain": "ES", "united kingdom": "GB", "netherlands": "NL",
    "belgium": "BE", "switzerland": "CH", "poland": "PL", "turkey": "TR",
    "canada": "CA", "brazil": "BR", "russia": "RU",
    "ประเทศไทย": "TH", "ไทย": "TH", "中国": "CN", "中國": "CN", "台湾": "TW",
    "日本": "JP", "韓国": "KR", "한국": "KR", "việt nam": "VN", "ลาว": "LA",
}


def resolve_markets(values=None):
    """把 ISO/英文市场名解析为按成员契约排序的市场元组。"""
    if not values:
        return list(MARKETS)
    tokens = []
    for value in values:
        tokens.extend(part.strip().lower() for part in value.split(",")
                      if part.strip())
    aliases = {}
    for market in MARKETS:
        country, iso, _ = market
        aliases[country] = market
        aliases[country.replace("-", " ")] = market
        aliases[iso.lower()] = market
    aliases["east-timor"] = aliases["timor-leste"]
    aliases["east timor"] = aliases["timor-leste"]
    unknown = sorted({token for token in tokens if token not in aliases})
    if unknown:
        valid = ", ".join(f"{country}/{iso}" for country, iso, _ in MARKETS)
        raise ValueError(f"非 ASEAN-11 市场: {', '.join(unknown)}；可选: {valid}")
    selected = {aliases[token][1] for token in tokens}
    return [market for market in MARKETS if market[1] in selected]


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path):
    with Path(path).open("rb") as handle:
        return sum(1 for _ in handle)


def write_json_atomic(path, payload):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def collection_state(market, via, **extra):
    country, iso, _ = market
    state = {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "scope": "ASEAN member market product catalog",
        "membership_effective_date": ASEAN_MEMBERSHIP["effective_date"],
        "country_tag": country,
        "iso": iso,
        "via": via,
        "query": {"countries_tags_en": country},
        "fields_sha256": hashlib.sha256(FIELDS.encode("utf-8")).hexdigest(),
        "passes": {},
        "done": False,
    }
    state.update(extra)
    return state


def load_collection_state(path, market, via):
    """读取状态；兼容旧状态，但补齐后续清单所需的显式范围字段。"""
    path = Path(path)
    expected = collection_state(market, via)
    if not path.exists():
        return expected
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("country_tag") not in (None, expected["country_tag"]):
        raise ValueError(f"{path} 的 country_tag 与请求市场不一致")
    if state.get("iso") not in (None, expected["iso"]):
        raise ValueError(f"{path} 的 ISO 与请求市场不一致")
    legacy = "schema_version" not in state
    merged = dict(expected)
    merged.update(state)
    merged["schema_version"] = COLLECTION_SCHEMA_VERSION
    merged["country_tag"] = expected["country_tag"]
    merged["iso"] = expected["iso"]
    merged["query"] = expected["query"]
    merged["fields_sha256"] = expected["fields_sha256"]
    if legacy:
        merged["legacy_state_upgraded"] = True
    return merged


def write_collection_manifest(requested_markets, collector, raw_dir=RAW_DIR,
                              source_snapshot=None):
    """记录实际存在的 ASEAN 原始快照；非成员文件不会进入清单。"""
    raw_dir = Path(raw_dir)
    rows = []
    for market in MARKETS:
        country, iso, languages = market
        raw_path = raw_dir / f"{country}.jsonl"
        state_path = raw_dir / f"{country}.state"
        state = (json.loads(state_path.read_text(encoding="utf-8"))
                 if state_path.exists() else {})
        exists = raw_path.is_file()
        rows.append({
            "country_tag": country,
            "iso": iso,
            "languages": languages,
            "raw_file": raw_path.name,
            "available": exists,
            "complete": bool(state.get("done")) and exists,
            "collector": state.get("via"),
            "catalog_coverage": state.get("catalog_coverage", "unknown"),
            "record_count": count_jsonl(raw_path) if exists else 0,
            "sha256": file_sha256(raw_path) if exists else None,
        })
    payload = {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "dataset": "OpenFoodFacts-ASEAN-products",
        "generated_at": utc_now(),
        "scope": {
            **ASEAN_MEMBERSHIP,
            "member_iso": MARKET_ISO,
            "excludes_non_members": True,
        },
        "provider": OFF_PROVIDER,
        "collector": collector,
        "requested_markets": [market[1] for market in requested_markets],
        "source_snapshot": source_snapshot,
        "markets": rows,
        "complete_market_count": sum(row["complete"] for row in rows),
        "formal_snapshot_ready": all(
            row["complete"]
            and row["catalog_coverage"] == "complete_bulk_snapshot"
            for row in rows
        ),
    }
    write_json_atomic(raw_dir / COLLECTION_MANIFEST.name, payload)
    return payload


def http_get_json(params, retries=6, pause=6.0, timeout=90):
    """带指数退避的 GET。返回 dict；彻底失败返回 {"_error": ...}。"""
    url = API + "?" + urllib.parse.urlencode(params)
    delay = 30.0
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if k == retries - 1:
                return {"_error": msg}
            print(f"    重试 {k + 1}/{retries - 1}（{delay:.0f}s 后）: {msg}",
                  flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 300.0)
    return {"_error": "unreachable"}


def tag_to_iso(tag):
    """'en:thailand' / 'th:ลาว' -> ISO 或 None。"""
    key = tag.split(":", 1)[-1].strip().lower()
    if key in TAG2ISO:
        return TAG2ISO[key]
    return None


def freetext_to_iso(text):
    """manufacturing_places 自由文本 -> ISO 或 None。"""
    low = text.lower()
    for name in sorted(NAME2ISO, key=len, reverse=True):
        if name in low:
            return NAME2ISO[name]
    return None
