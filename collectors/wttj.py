"""Collecteur Welcome to the Jungle via l'API de recherche Algolia."""
from __future__ import annotations

from datetime import datetime
import json
import re
from typing import Any

import httpx

from . import BaseCollector

DEFAULT_ALGOLIA_URL = (
    "https://csekhvms53-dsn.algolia.net/1/indexes/wk_cms_organizations_production/query"
)


def _strip_html(raw: Any, html_to_text) -> str:
    if raw is None:
        return ""
    if isinstance(raw, list):
        parts = [_strip_html(item, html_to_text) for item in raw]
        return "\n".join(p for p in parts if p)
    if isinstance(raw, dict):
        parts = [_strip_html(v, html_to_text) for v in raw.values()]
        return "\n".join(p for p in parts if p)
    text = str(raw).strip()
    if not text:
        return ""
    return html_to_text(text) if "<" in text else text


def _format_offices(offices: Any) -> str:
    if not isinstance(offices, list):
        return ""
    places: list[str] = []
    for off in offices:
        if not isinstance(off, dict):
            continue
        city = str(off.get("city") or "").strip()
        zip_code = str(off.get("zip_code") or off.get("postal_code") or "").strip()
        country = str(off.get("country_code") or "").strip()
        chunk = " ".join(x for x in (zip_code, city, country) if x)
        if chunk and chunk not in places:
            places.append(chunk)
    return " / ".join(places)


def _build_job_url(hit: dict[str, Any], default_company_slug: str = "") -> str:
    company_obj = hit.get("organization") if isinstance(hit.get("organization"), dict) else {}
    company_slug = str(
        hit.get("organization_slug")
        or company_obj.get("slug")
        or default_company_slug
        or ""
    ).strip()
    job_slug = str(hit.get("slug") or hit.get("reference") or hit.get("objectID") or "").strip()
    if company_slug and job_slug:
        return f"https://www.welcometothejungle.com/fr/companies/{company_slug}/jobs/{job_slug}"
    if company_slug:
        return f"https://www.welcometothejungle.com/fr/companies/{company_slug}/jobs"
    return "https://www.welcometothejungle.com/fr/jobs"


class WttjAlgoliaCollector(BaseCollector):
    source_type = "wttj"

    def __init__(self, source_cfg: dict, criteria: dict, helpers: dict):
        super().__init__(source_cfg, criteria, helpers)
        self.log = helpers.get("log", print)
        self.should_stop = helpers.get("should_stop", lambda: False)
        self.sleep = helpers.get("sleep", lambda _s: None)
        self.make_job_id = helpers["make_job_id"]
        self.html_to_text = helpers["html_to_text"]
        self.estimate_age = helpers["estimate_job_age_days"]

    def _headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
            "Origin": "https://www.welcometothejungle.com",
            "Referer": "https://www.welcometothejungle.com/",
        }
        app_id = str(self.cfg.get("algolia_app_id") or "").strip()
        api_key = str(self.cfg.get("algolia_api_key") or "").strip()
        if app_id:
            headers["x-algolia-application-id"] = app_id
        if api_key:
            headers["x-algolia-api-key"] = api_key
        extra_headers = self.cfg.get("headers")
        if isinstance(extra_headers, dict):
            for k, v in extra_headers.items():
                if v is not None:
                    headers[str(k)] = str(v)
        return headers

    def _query_algolia(self, client: httpx.Client, endpoint: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        resp = client.post(endpoint, json=payload)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits")
        return hits if isinstance(hits, list) else []

    def _normalize_hit(
        self,
        hit: dict[str, Any],
        now: datetime | None,
        *,
        company_fallback: str = "",
        company_slug_fallback: str = "",
        location_fallback: str = "",
    ) -> dict | None:
        title = re.sub(r"\s+", " ", str(hit.get("name") or hit.get("title") or "").strip())
        if not title:
            return None

        org_obj = hit.get("organization") if isinstance(hit.get("organization"), dict) else {}
        company = str(
            hit.get("organization_name")
            or org_obj.get("name")
            or company_fallback
            or self.cfg.get("company")
            or "Welcome to the Jungle"
        ).strip()

        location = (
            _format_offices(hit.get("offices"))
            or str(hit.get("office_city") or hit.get("location") or "").strip()
            or location_fallback
            or str(self.cfg.get("zone") or self.cfg.get("location") or "À extraire").strip()
        )

        desc_parts = [
            _strip_html(hit.get("description"), self.html_to_text),
            _strip_html(hit.get("descriptions"), self.html_to_text),
            _strip_html(hit.get("profile"), self.html_to_text),
            _strip_html(hit.get("requirements"), self.html_to_text),
        ]
        description = "\n\n".join(p for p in desc_parts if p).strip()
        if not description:
            sectors = hit.get("sectors")
            sector_txt = ", ".join(str(s) for s in sectors) if isinstance(sectors, list) else ""
            description = f"Poste : {title} chez {company} ({location}). {sector_txt}".strip()

        url = _build_job_url(hit, default_company_slug=company_slug_fallback)
        source_name = str(self.cfg.get("name") or self.source_type)
        published_at = hit.get("published_at") or hit.get("updated_at")
        age_days, age_detail = self.estimate_age(
            description,
            published_at=published_at,
            reference_now=now,
            header_chars=int(self.criteria.get("header_chars", 800)),
        )

        job: dict[str, Any] = {
            "id": self.make_job_id(source_name, f"{url}#{hit.get('objectID') or title}"),
            "source": source_name,
            "category": self.source_type,
            "title": title,
            "company": company,
            "location": location,
            "url": url,
            "description": description,
            "age_days": age_days,
            "age_detail": age_detail,
        }
        if self.cfg.get("remote_hint"):
            job["remote_hint"] = self.cfg["remote_hint"]
        return job

    def collect(self, now: datetime | None = None) -> list[dict]:
        endpoint = str(self.cfg.get("endpoint") or self.cfg.get("url") or DEFAULT_ALGOLIA_URL).strip()
        if not endpoint:
            return []

        keywords = [str(k).strip() for k in (self.cfg.get("keywords") or [""]) if str(k).strip() or not self.cfg.get("keywords")]
        if not keywords:
            keywords = [""]

        hits_per_page = int(self.cfg.get("hits_per_page", 20))
        max_pages = int(self.cfg.get("max_pages", 1))
        facet_filters = self.cfg.get("facet_filters")
        if facet_filters is None:
            facet_filters = [["offices.country_code:FR"]]

        timeout_s = max(5.0, float(self.criteria.get("http_timeout_ms", 20000)) / 1000.0)
        pause_s = float(self.cfg.get("request_delay_s", 0.3))
        index_mode = str(self.cfg.get("algolia_mode") or "jobs").lower()

        jobs: list[dict] = []
        seen_ids: set[str] = set()

        with httpx.Client(timeout=timeout_s, follow_redirects=True, headers=self._headers()) as client:
            for kw in keywords:
                for page in range(max_pages):
                    if self.should_stop():
                        return jobs
                    payload: dict[str, Any] = {
                        "query": kw,
                        "hitsPerPage": hits_per_page,
                        "page": page,
                        "analytics": False,
                        "facetFilters": facet_filters,
                    }
                    try:
                        hits = self._query_algolia(client, endpoint, payload)
                    except httpx.HTTPError as exc:
                        self.log(f"[WARN] wttj ({self.cfg.get('name')}): erreur HTTP Algolia : {exc}")
                        return jobs
                    except ValueError as exc:
                        self.log(f"[WARN] wttj ({self.cfg.get('name')}): JSON invalide : {exc}")
                        return jobs

                    if not hits:
                        break

                    if index_mode == "organizations":
                        for org in hits:
                            if int(org.get("jobs_count") or 0) <= 0:
                                continue
                            org_name = str(org.get("name") or "").strip()
                            org_slug = str(org.get("slug") or "").strip()
                            org_loc = _format_offices(org.get("offices"))
                            job = self._normalize_hit(
                                {
                                    "objectID": f"{org.get('objectID') or org_slug}-jobs",
                                    "name": f"Offres ouvertes ({org.get('jobs_count')}) - {org_name}",
                                    "slug": "",
                                    "descriptions": org.get("descriptions"),
                                    "sectors": org.get("sectors"),
                                    "published_at": org.get("published_at"),
                                },
                                now=now,
                                company_fallback=org_name,
                                company_slug_fallback=org_slug,
                                location_fallback=org_loc,
                            )
                            if job and job["id"] not in seen_ids:
                                seen_ids.add(job["id"])
                                jobs.append(job)
                    else:
                        for hit in hits:
                            job = self._normalize_hit(hit, now=now)
                            if job and job["id"] not in seen_ids:
                                seen_ids.add(job["id"])
                                jobs.append(job)

                    if pause_s > 0:
                        self.sleep(pause_s)

        return jobs
