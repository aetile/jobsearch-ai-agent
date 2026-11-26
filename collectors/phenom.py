"""Collecteur pour plateformes carrières Phenom (ex. Thales)."""
from __future__ import annotations

from datetime import datetime
import json
import re
import unicodedata
from urllib.parse import urlparse

import httpx

from . import BaseCollector

_DDO_RE = re.compile(r"phApp\.ddo\s*=\s*(\{.*?\});\s*(?:phApp\.|</script>)", re.DOTALL)


def _slugify_title(title: str) -> str:
    norm = unicodedata.normalize("NFKD", title or "")
    ascii_only = "".join(ch for ch in norm if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", ascii_only).strip("-")
    return cleaned or "Job"


def _portal_origin(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return raw_url.rstrip("/")


def _resolve_search_page_url(cfg: dict) -> tuple[str, str]:
    """Retourne (url_page_recherche, locale_path) en évitant d'appeler /widgets en GET."""
    explicit_url = str(cfg.get("url") or cfg.get("portal_url") or "").strip()
    raw_endpoint = str(cfg.get("endpoint") or "").strip()
    locale_raw = str(cfg.get("locale") or "fr_fr").strip().lower().replace("_", "/")
    locale_path = f"/{locale_raw.strip('/')}" if locale_raw.strip("/") else "/fr/fr"

    if explicit_url:
        return explicit_url, locale_path

    if raw_endpoint:
        parsed = urlparse(raw_endpoint)
        path_low = parsed.path.lower().rstrip("/")
        if path_low.endswith("/widgets") or "/widgets/" in path_low:
            origin = _portal_origin(raw_endpoint)
            return f"{origin}{locale_path}/search-results", locale_path
        return raw_endpoint, locale_path

    return "", locale_path


def _build_phenom_url(portal_base: str, req_id: str, title: str, locale_path: str = "/fr/fr") -> str:
    origin = _portal_origin(portal_base)
    prefix = locale_path.strip("/")
    slug = _slugify_title(title)
    if req_id:
        return f"{origin}/{prefix}/job/{req_id}/{slug}"
    return f"{origin}/{prefix}/job/{slug}"


def _extract_ddo_jobs(html: str, default_locale_path: str = "/fr/fr") -> tuple[list[dict], str, dict]:
    """Extrait (jobs, locale_path, ddo) depuis le HTML Phenom."""
    match = _DDO_RE.search(html)
    if not match:
        return [], default_locale_path, {}
    try:
        ddo = json.loads(match.group(1))
    except ValueError:
        return [], default_locale_path, {}
    if not isinstance(ddo, dict):
        return [], default_locale_path, {}
    locale_val = str(ddo.get("locale") or "").strip().lower().replace("_", "/")
    locale_path = f"/{locale_val.strip('/')}" if locale_val.strip("/") else default_locale_path
    jobs = (
        ddo.get("eagerLoadRefineSearch", {}).get("data", {}).get("jobs")
        or ddo.get("refineSearch", {}).get("data", {}).get("jobs")
        or []
    )
    return (jobs if isinstance(jobs, list) else []), locale_path, ddo


def _fetch_paginated_jobs(
    client: httpx.Client,
    page_url: str,
    ddo: dict,
    keywords: list[str],
    max_pages: int = 5,
    page_size: int = 20,
) -> list[dict]:
    """Interroge l'endpoint /widgets de Phenom pour paginer au-delà du lot initial."""
    origin = _portal_origin(page_url)
    widgets_url = f"{origin}/widgets"
    ddo_key = str(ddo.get("ddoKey") or "").strip()
    site_id = str(ddo.get("siteId") or ddo.get("site") or "").strip()
    locale = str(ddo.get("locale") or "fr_fr").strip()

    query_terms = " ".join(str(k).strip() for k in keywords if str(k).strip())
    collected: list[dict] = []

    for page_idx in range(max_pages):
        offset = page_idx * page_size
        payload: dict = {
            "lang": locale,
            "deviceType": "desktop",
            "country": "fr",
            "pageName": "search-results",
            "ddoKey": ddo_key or "refineSearch",
            "sortBy": "",
            "subsearch": "",
            "from": offset,
            "jobs": True,
            "all_jobs": False,
            "customFacetName": "",
            "customFacetValue": [],
            "facetFilters": [],
            "keywords": query_terms,
            "location": "",
            "size": page_size,
            "pageSize": page_size,
        }
        if site_id:
            payload["siteId"] = site_id

        try:
            resp = client.post(widgets_url, json=payload)
            if resp.status_code >= 400:
                break
            data = resp.json()
        except Exception:
            break

        if not isinstance(data, dict):
            break
        batch = (
            data.get("refineSearch", {}).get("data", {}).get("jobs")
            or data.get("eagerLoadRefineSearch", {}).get("data", {}).get("jobs")
            or data.get("jobs")
            or []
        )
        if not isinstance(batch, list) or not batch:
            break
        dict_batch = [item for item in batch if isinstance(item, dict)]
        if not dict_batch:
            break
        collected.extend(dict_batch)
        if len(dict_batch) < page_size:
            break

    return collected


def _matches_source_keywords(entry: dict, html_to_text, keywords: list[str]) -> bool:
    if not keywords:
        return True
    desc_raw = str(entry.get("description") or entry.get("descriptionTeaser") or "")
    desc_txt = html_to_text(desc_raw) if "<" in desc_raw else desc_raw
    skills = " ".join(str(s) for s in (entry.get("ml_skills") or []))
    cats = " ".join(str(c) for c in (entry.get("multi_category") or []))
    haystack = " ".join(
        str(x or "")
        for x in (
            entry.get("title"),
            entry.get("jobTitle"),
            entry.get("category"),
            entry.get("address"),
            entry.get("city"),
            cats,
            skills,
            desc_txt,
        )
    ).lower()
    return any(str(kw).strip().lower() in haystack for kw in keywords if str(kw).strip())


class PhenomCollector(BaseCollector):
    source_type = "phenom"

    def __init__(self, source_cfg: dict, criteria: dict, helpers: dict):
        super().__init__(source_cfg, criteria, helpers)
        self.log = helpers.get("log", print)
        self.should_stop = helpers.get("should_stop", lambda: False)
        self.make_job_id = helpers["make_job_id"]
        self.html_to_text = helpers["html_to_text"]
        self.estimate_age = helpers["estimate_job_age_days"]

    def collect(self, now: datetime | None = None) -> list[dict]:
        page_url, locale_path = _resolve_search_page_url(self.cfg)
        if not page_url:
            return []

        timeout_s = max(5.0, float(self.criteria.get("http_timeout_ms", 20000)) / 1000.0)
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
        }
        keywords = [str(k) for k in (self.cfg.get("keywords") or []) if str(k).strip()]
        source_name = str(self.cfg.get("name") or self.source_type)
        company_default = str(self.cfg.get("company") or source_name)
        max_pages = max(1, int(self.cfg.get("max_pages", 5)))
        page_size = max(1, min(50, int(self.cfg.get("page_size", 20))))

        ddo: dict = {}
        try:
            with httpx.Client(timeout=timeout_s, follow_redirects=True, headers=headers) as client:
                if self.should_stop():
                    return []
                resp = client.get(page_url)
                resp.raise_for_status()
                body = resp.text

                try:
                    data = json.loads(body)
                    raw_jobs = (
                        data.get("refineSearch", {}).get("data", {}).get("jobs")
                        or data.get("eagerLoadRefineSearch", {}).get("data", {}).get("jobs")
                        or data.get("jobs")
                        or []
                    )
                except ValueError:
                    raw_jobs, locale_path, ddo = _extract_ddo_jobs(body, default_locale_path=locale_path)

                if not self.should_stop() and ddo and (len(raw_jobs) <= 10 or keywords):
                    extra_jobs = _fetch_paginated_jobs(
                        client,
                        page_url,
                        ddo,
                        keywords=keywords,
                        max_pages=max_pages,
                        page_size=page_size,
                    )
                    if extra_jobs:
                        raw_jobs = list(raw_jobs) + extra_jobs
        except Exception as exc:
            self.log(f"[WARN] phenom ({source_name}): erreur de collecte : {exc}")
            return []

        if not isinstance(raw_jobs, list):
            return []

        jobs: list[dict] = []
        seen_ids: set[str] = set()
        for entry in raw_jobs:
            if self.should_stop():
                break
            if not isinstance(entry, dict):
                continue
            if not _matches_source_keywords(entry, self.html_to_text, keywords):
                continue

            req_id = str(
                entry.get("reqId")
                or entry.get("jobId")
                or entry.get("jobSeqNo")
                or entry.get("id")
                or ""
            ).strip()
            title = str(entry.get("title") or entry.get("jobTitle") or "").strip()
            url = _build_phenom_url(page_url, req_id, title, locale_path=locale_path)
            job_id = self.make_job_id(source_name, url)
            if job_id in seen_ids:
                continue
            seen_ids.add(job_id)

            loc_raw = (
                entry.get("address")
                or entry.get("cityStateCountry")
                or entry.get("city")
                or entry.get("location")
                or ""
            )
            if isinstance(loc_raw, list):
                location = " / ".join(str(x).strip() for x in loc_raw if str(x).strip()) or "À extraire"
            else:
                location = str(loc_raw).strip() or "À extraire"

            desc_raw = str(entry.get("description") or entry.get("descriptionTeaser") or "")
            description = self.html_to_text(desc_raw) if "<" in desc_raw else desc_raw.strip()
            posted_date = entry.get("postedDate") or entry.get("dateCreated") or entry.get("posted_date")

            age_days, age_detail = self.estimate_age(
                description,
                published_at=posted_date,
                reference_now=now,
                header_chars=int(self.criteria.get("header_chars", 800)),
            )

            jobs.append({
                "id": job_id,
                "source": source_name,
                "category": self.source_type,
                "title": title or "À extraire",
                "company": company_default,
                "location": location,
                "url": url,
                "description": description,
                "age_days": age_days,
                "age_detail": age_detail,
            })

        return jobs
