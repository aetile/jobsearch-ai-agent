"""Collecteur générique HTML / JSON-LD pour sites d'offres."""
from __future__ import annotations

from datetime import datetime
import json
import re
from urllib.parse import quote_plus, urljoin, urlparse

import httpx

from . import BaseCollector

_JOB_LINK_RX = re.compile(
    r"(?:/offre|/offres|/emploi|/emplois|/job|/jobs|/career|/careers|/poste|/postes"
    r"|/annonce|/annonces|/opportunites|/recrutement|/candidature|/detail)",
    re.IGNORECASE,
)
_NOISE_URL_RX = re.compile(
    r"(?:/rss|/flux|/candidature-spontanee|/get-started|/login|/connexion|/compte|/favoris|/selection|/nos-offres)",
    re.IGNORECASE,
)
_NOISE_TITLE_RX = re.compile(
    r"(?:trouvez votre prochaine|trouvez votre futur|ma s[ée]lection|les offres d'emploi"
    r"|flux rss|candidature spontan[ée]e)",
    re.IGNORECASE,
)
_SKIP_EXT_RX = re.compile(
    r"\.(?:css|js|png|jpe?g|gif|svg|webp|ico|woff2?|ttf|pdf|zip|xml|json)(?:\?|$)",
    re.IGNORECASE,
)


def _iter_json_ld_objects(raw_json) -> list[dict]:
    found: list[dict] = []
    if isinstance(raw_json, dict):
        graph = raw_json.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                found.extend(_iter_json_ld_objects(item))
        jtype = raw_json.get("@type")
        types_list = [jtype] if isinstance(jtype, str) else (jtype or [])
        if any(str(t).lower() == "jobposting" for t in types_list):
            found.append(raw_json)
    elif isinstance(raw_json, list):
        for item in raw_json:
            found.extend(_iter_json_ld_objects(item))
    return found


def _extract_json_ld_jobposting(html: str) -> dict | None:
    for match in re.finditer(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        re.IGNORECASE | re.DOTALL,
    ):
        block = (match.group(1) or "").strip()
        if not block:
            continue
        try:
            parsed = json.loads(block)
        except ValueError:
            continue
        posts = _iter_json_ld_objects(parsed)
        if posts:
            return posts[0]
    return None


def _extract_ld_location(posting: dict) -> str:
    loc = posting.get("jobLocation")
    places = loc if isinstance(loc, list) else ([loc] if isinstance(loc, dict) else [])
    parts: list[str] = []
    for place in places:
        if not isinstance(place, dict):
            continue
        addr = place.get("address")
        if isinstance(addr, dict):
            postal = str(addr.get("postalCode") or "").strip()
            city = str(addr.get("addressLocality") or "").strip()
            region = str(addr.get("addressRegion") or "").strip()
            chunk = " ".join(x for x in (postal, city, region) if x)
            if chunk:
                parts.append(chunk)
        elif isinstance(place.get("name"), str) and place["name"].strip():
            parts.append(place["name"].strip())
    return " / ".join(dict.fromkeys(parts))


def _extract_ld_company(posting: dict, fallback: str) -> str:
    org = posting.get("hiringOrganization")
    if isinstance(org, dict) and isinstance(org.get("name"), str) and org["name"].strip():
        return org["name"].strip()
    return fallback


class WebGenericCollector(BaseCollector):
    source_type = "web_generic"

    def __init__(self, source_cfg: dict, criteria: dict, helpers: dict):
        super().__init__(source_cfg, criteria, helpers)
        self.log = helpers.get("log", print)
        self.should_stop = helpers.get("should_stop", lambda: False)
        self.sleep = helpers.get("sleep", lambda _s: None)
        self.make_job_id = helpers["make_job_id"]
        self.html_to_text = helpers["html_to_text"]
        self.estimate_age = helpers["estimate_job_age_days"]

    def _collect_target_urls(self) -> list[str]:
        urls: list[str] = []
        for key in ("urls", "search_urls", "careers_urls"):
            val = self.cfg.get(key)
            if isinstance(val, list):
                urls.extend(str(u).strip() for u in val if u)
        if self.cfg.get("url"):
            urls.append(str(self.cfg["url"]).strip())

        templates = self.cfg.get("search_url_templates") or []
        keywords = self.cfg.get("keywords") or [""]
        for tpl in templates:
            for kw in keywords:
                q = str(kw).strip()
                urls.append(
                    str(tpl).format(
                        query=q,
                        query_encoded=q.replace(" ", "%20"),
                        query_plus=quote_plus(q),
                    )
                )

        seen: set[str] = set()
        deduped: list[str] = []
        for u in urls:
            if u and u not in seen:
                seen.add(u)
                deduped.append(u)
        return deduped

    def _fetch(self, client: httpx.Client, url: str) -> str | None:
        try:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPError as exc:
            self.log(f"[WARN] web_generic ({self.cfg.get('name')}): erreur HTTP sur {url} : {exc}")
            return None

    def _extract_links(self, base_url: str, html: str, max_links: int) -> list[str]:
        base_parsed = urlparse(base_url)
        base_host = base_parsed.netloc.lower()
        base_path = base_parsed.path.rstrip("/") or "/"
        seen: set[str] = set()
        links: list[str] = []

        for match in re.finditer(r'<a\b[^>]*href=["\']([^"\']+)["\']', html, re.IGNORECASE):
            href = (match.group(1) or "").strip()
            if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
                continue
            full_url = urljoin(base_url, href).split("#", 1)[0]
            parsed = urlparse(full_url)
            if parsed.scheme not in ("http", "https"):
                continue
            if parsed.netloc.lower() != base_host:
                continue
            if _SKIP_EXT_RX.search(parsed.path) or _NOISE_URL_RX.search(parsed.path):
                continue
            if (parsed.path.rstrip("/") or "/") == base_path and parsed.query == base_parsed.query:
                continue
            if not _JOB_LINK_RX.search(parsed.path):
                continue
            if full_url not in seen:
                seen.add(full_url)
                links.append(full_url)
                if len(links) >= max_links:
                    break
        return links

    def _parse_detail_page(
            self,
            url: str,
            html: str,
            now: datetime | None = None,
            *,
            require_job_posting: bool = False,
        ) -> dict | None:
            posting = _extract_json_ld_jobposting(html)
            if require_job_posting and posting is None:
                return None

            fallback_company = str(self.cfg.get("company") or self.cfg.get("name") or "À extraire")
            fallback_loc = str(self.cfg.get("zone") or self.cfg.get("location") or "À extraire")

            if posting is not None:
                title = str(posting.get("title") or "").strip()
                raw_desc = (
                    posting.get("description")
                    if posting.get("description") is not None
                    else (posting.get("Description") or "")
                )
                desc_raw = str(raw_desc)
                description = self.html_to_text(desc_raw) if "<" in desc_raw else desc_raw.strip()
                company = _extract_ld_company(posting, fallback_company)
                location = _extract_ld_location(posting) or fallback_loc
                date_posted = posting.get("datePosted")
                valid_through = posting.get("validThrough")
            else:
                h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.IGNORECASE | re.DOTALL)
                title = self.html_to_text(h1_match.group(1)).strip() if h1_match else ""
                main_match = re.search(
                    r"<(?:main|article)[^>]*>(.*?)</(?:main|article)>",
                    html,
                    re.IGNORECASE | re.DOTALL,
                )
                body_src = main_match.group(1) if main_match else html
                description = self.html_to_text(body_src)
                company = fallback_company
                location = fallback_loc
                date_posted = None
                valid_through = None

            clean_title = " ".join(title.split())
            if not clean_title or _NOISE_TITLE_RX.search(clean_title) or len(description.strip()) < 80:
                return None

            age_days, age_detail = self.estimate_age(
                description,
                published_at=date_posted,
                reference_now=now,
                header_chars=int(self.criteria.get("header_chars", 800)),
            )
            source_name = str(self.cfg.get("name") or self.source_type)
            job: dict = {
                "id": self.make_job_id(source_name, url),
                "source": source_name,
                "category": self.source_type,
                "title": clean_title,
                "company": company,
                "location": location,
                "url": url,
                "description": description,
                "age_days": age_days,
                "age_detail": age_detail,
            }
            if valid_through:
                job["valid_through"] = valid_through
            if self.cfg.get("remote_hint"):
                job["remote_hint"] = self.cfg["remote_hint"]
            return job

    def collect(self, now: datetime | None = None) -> list[dict]:
        urls = self._collect_target_urls()
        if not urls:
            return []

        timeout_s = max(5.0, float(self.criteria.get("http_timeout_ms", 20000)) / 1000.0)
        max_detail_pages = int(self.cfg.get("max_detail_pages", 25))
        pause_s = float(self.cfg.get("request_delay_s", 0.5))

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        jobs: list[dict] = []
        seen_ids: set[str] = set()

        with httpx.Client(timeout=timeout_s, follow_redirects=True, headers=headers) as client:
            for list_url in urls:
                if self.should_stop():
                    break
                listing_html = self._fetch(client, list_url)
                if not listing_html:
                    continue

                candidate_links = self._extract_links(list_url, listing_html, max_detail_pages)
                if not candidate_links:
                    parsed_direct = self._parse_detail_page(
                        list_url, listing_html, now=now, require_job_posting=True
                    )
                    if parsed_direct and parsed_direct["id"] not in seen_ids:
                        seen_ids.add(parsed_direct["id"])
                        jobs.append(parsed_direct)
                    continue

                for detail_url in candidate_links:
                    if self.should_stop():
                        break
                    if pause_s > 0:
                        self.sleep(pause_s)
                    detail_html = self._fetch(client, detail_url)
                    if not detail_html:
                        continue
                    item = self._parse_detail_page(detail_url, detail_html, now=now)
                    if item and item["id"] not in seen_ids:
                        seen_ids.add(item["id"])
                        jobs.append(item)

        return jobs
