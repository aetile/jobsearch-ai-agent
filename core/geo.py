from __future__ import annotations

import importlib.util
import re
import time

from .config import IDF_DEPTS, get_geo_api_cfg, get_nominatim_user_agent
from .resilience import GracefulShutdown, interruptible_sleep, log
from .storage import EXPLICIT_IDF_OR_REMOTE_RX, SeenDB, is_unknown, normalize_text

_LAST_NOMINATIM_TS = 0.0

IDF_CITIES_NORM: set[str] = {
    "paris",
    "courbevoie",
    "puteaux",
    "la defense",
    "nanterre",
    "levallois perret",
    "neuilly sur seine",
    "boulogne billancourt",
    "issy les moulineaux",
    "montrouge",
    "malakoff",
    "vanves",
    "clamart",
    "chatillon",
    "fontenay aux roses",
    "bagneux",
    "antony",
    "chateaudun",
    "suresnes",
    "rueil malmaison",
    "asnieres sur seine",
    "clichy",
    "gennevilliers",
    "colombes",
    "bois colombes",
    "saint denis",
    "aubervilliers",
    "pantin",
    "bobigny",
    "montreuil",
    "noisy le grand",
    "creteil",
    "maisons alfort",
    "ivry sur seine",
    "vitry sur seine",
    "villejuif",
    "vincennes",
    "saint mande",
    "charenton le pont",
    "versailles",
    "velizy villacoublay",
    "saint quentin en yvelines",
    "montigny le bretonneux",
    "guyancourt",
    "poissy",
    "saint germain en laye",
    "mantes la jolie",
    "massy",
    "orsay",
    "saclay",
    "gif sur yvette",
    "evry courcouronnes",
    "corbeil essonnes",
    "cergy",
    "pontoise",
    "taverny",
    "franconville",
    "argenteuil",
    "ermont",
    "eaubonne",
    "sannois",
    "sarcelles",
    "garges les gonesse",
    "roissy en france",
    "melun",
    "meaux",
    "serris",
    "marne la vallee",
}


def extract_geo_query(raw_location: str) -> tuple[str | None, str]:
    cleaned = re.sub(
        r"(?i)\b(?:address line \d+|batiment[^:]*:|siege social[^:]*:)\b",
        " ",
        raw_location or "",
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;-/")
    if cp_match := re.search(r"\b(\d{5})\b", cleaned):
        return cp_match.group(1), cleaned
    parts = [p.strip(" ,;-/") for p in re.split(r"[,/]|(?:\s+-\s+)", cleaned) if p.strip(" ,;-/")]
    if not parts:
        return None, ""
    candidates = [
        p
        for p in reversed(parts)
        if normalize_text(p) not in {"france", "idf", "ile-de-france", "ile de france", "non precise"}
    ]
    query = candidates[0] if candidates else parts[-1]
    return None, query


def _lookup_idf_fallback(cp: str | None, query: str, raw_location: str) -> dict | None:
    if cp and len(cp) >= 2 and cp[:2] in IDF_DEPTS:
        return {"country_code": "fr", "dept_code": cp[:2], "is_idf": True}

    norm_query = normalize_text(query)
    if norm_query in IDF_CITIES_NORM:
        return {"country_code": "fr", "dept_code": None, "is_idf": True}

    norm_raw = normalize_text(raw_location)
    for city in IDF_CITIES_NORM:
        if re.search(rf"\b{re.escape(city)}\b", norm_raw):
            return {"country_code": "fr", "dept_code": None, "is_idf": True}

    return None


def lookup_fr_commune_geo_api(
    cp: str | None,
    query: str,
    timeout_s: float = 8.0,
    retries: int = 1,
    retry_backoff_s: float = 1.0,
) -> dict | None:
    if importlib.util.find_spec("httpx") is None:
        return None
    import httpx

    params = {"codePostal": cp} if cp else {"nom": query, "boost": "population", "limit": 1}
    label = cp or query
    max_attempts = max(1, int(retries) + 1)

    for attempt in range(1, max_attempts + 1):
        try:
            with httpx.Client(timeout=timeout_s) as client:
                resp = client.get("https://geo.api.gouv.fr/communes", params=params)
                resp.raise_for_status()
                rows = resp.json()
            if not isinstance(rows, list) or not rows:
                return None
            first = rows[0]
            dept = str(first.get("codeDepartement") or "").strip().upper()
            if not dept:
                return None
            return {
                "country_code": "fr",
                "dept_code": dept,
                "is_idf": dept in IDF_DEPTS,
            }
        except GracefulShutdown:
            raise
        except Exception as exc:
            if attempt < max_attempts:
                log(
                    f"[RETRY] geo.api.gouv.fr ({label}) : {type(exc).__name__} "
                    f"(essai {attempt}/{max_attempts}, attente {retry_backoff_s:.1f}s)"
                )
                interruptible_sleep(retry_backoff_s)
            else:
                log(f"[WARN] geo.api.gouv.fr indisponible pour '{label}' : {type(exc).__name__}")
                return None

    return None


def lookup_country_nominatim(query: str, timeout_s: float = 5.0) -> dict | None:
    global _LAST_NOMINATIM_TS
    if not query or importlib.util.find_spec("httpx") is None:
        return None
    import httpx

    elapsed = time.monotonic() - _LAST_NOMINATIM_TS
    if elapsed < 1.05:
        interruptible_sleep(1.05 - elapsed)

    headers = {"User-Agent": get_nominatim_user_agent(), "Accept-Language": "en,fr"}
    params = {"q": query, "format": "json", "addressdetails": 1, "limit": 1}
    try:
        with httpx.Client(timeout=timeout_s, headers=headers) as client:
            _LAST_NOMINATIM_TS = time.monotonic()
            resp = client.get("https://nominatim.openstreetmap.org/search", params=params)
            resp.raise_for_status()
            rows = resp.json()
    except GracefulShutdown:
        raise
    except Exception as exc:
        log(f"[WARN] Nominatim indisponible pour '{query}' : {type(exc).__name__}")
        return None

    if not isinstance(rows, list) or not rows:
        return None
    address = rows[0].get("address") if isinstance(rows[0], dict) else {}
    cc = str((address or {}).get("country_code") or "").strip().lower() or None
    is_idf = True if cc == "fr" and str((address or {}).get("state") or "").lower().find("ile-de-france") >= 0 else None
    return {
        "country_code": cc,
        "dept_code": None,
        "is_idf": is_idf,
    }


def resolve_location_info(
    raw_location: str,
    store: SeenDB | None = None,
    geo_cfg: dict | None = None,
) -> dict:
    cfg = get_geo_api_cfg(
        geo_cfg if isinstance(geo_cfg, dict) and "geo_api" not in geo_cfg else {"geo_api": geo_cfg or {}}
    )
    norm = normalize_text(raw_location)
    if not norm or is_unknown(raw_location):
        return {"country_code": None, "dept_code": None, "is_idf": None}
    if EXPLICIT_IDF_OR_REMOTE_RX.search(norm):
        return {"country_code": "fr", "dept_code": None, "is_idf": True}

    cache_key = norm
    if store is not None:
        cached = store.get_cached_location(cache_key)
        if cached is not None:
            return cached

    cp, query = extract_geo_query(raw_location)
    timeout_s = float(cfg.get("timeout_s", 8.0))
    retries = int(cfg.get("retries", 1))
    retry_backoff_s = float(cfg.get("retry_backoff_s", 1.0))
    cache_negative = bool(cfg.get("cache_negative_lookup", False))
    idf_fallback = bool(cfg.get("idf_fallback", True))

    info = (
        lookup_fr_commune_geo_api(
            cp,
            query,
            timeout_s=timeout_s,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
        )
        if (cp or query)
        else None
    )

    if info is not None:
        if store is not None:
            store.set_cached_location(
                cache_key,
                info.get("country_code"),
                info.get("dept_code"),
                info.get("is_idf"),
            )
        return info

    if idf_fallback:
        fallback_info = _lookup_idf_fallback(cp, query, raw_location)
        if fallback_info is not None:
            if store is not None:
                store.set_cached_location(
                    cache_key,
                    fallback_info.get("country_code"),
                    fallback_info.get("dept_code"),
                    fallback_info.get("is_idf"),
                )
            return fallback_info

    result = {"country_code": "non_fr", "dept_code": None, "is_idf": False}
    if store is not None and cache_negative:
        store.set_cached_location(
            cache_key,
            result.get("country_code"),
            result.get("dept_code"),
            result.get("is_idf"),
        )
    return result
