from __future__ import annotations

import os
import threading
import time
from datetime import datetime
import httpx
from . import BaseCollector

MAX_CALLS_PER_SECOND = 10.0


class _RateLimiter:
    """Garantit au maximum `max_per_second` appels HTTP par seconde (0,1 s d'écart min)."""

    def __init__(self, max_per_second: float = MAX_CALLS_PER_SECOND, sleep_fn=time.sleep):
        rate = min(float(max_per_second), MAX_CALLS_PER_SECOND)
        if rate <= 0:
            raise ValueError("max_per_second doit être strictement positif.")
        self._min_interval = 1.0 / rate
        self._sleep = sleep_fn
        self._lock = threading.Lock()
        self._next_allowed_at = 0.0

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next_allowed_at:
                    base = max(now, self._next_allowed_at)
                    self._next_allowed_at = base + self._min_interval
                    return
                wait_s = self._next_allowed_at - now
            if wait_s > 0:
                self._sleep(wait_s)


class FranceTravailCollector(BaseCollector):
    source_type = "francetravail"

    def _normalize(self, item: dict, now: datetime | None, header_chars: int) -> dict:
        offer_id = str(item.get("id", ""))
        lieu = item.get("lieuTravail") or {}
        entreprise = item.get("entreprise") or {}
        salaire = item.get("salaire") or {}
        published_raw = item.get("dateActualisation") or item.get("dateCreation")
        description = item.get("description") or ""
        age_days, age_detail = self.helpers["estimate_job_age_days"](
            description, published_at=published_raw, reference_now=now, header_chars=header_chars
        )
        location = lieu.get("libelle") or "Non précisé"
        postal = str(lieu.get("codePostal") or "").strip()
        if postal and postal not in location:
            location = f"{location} ({postal})"
        salary_text = salaire.get("libelle") or salaire.get("complement1") or "Non précisé"

        return {
            "id": f"FT-{offer_id}",
            "source": self.cfg.get("name", "francetravail"),
            "category": "api",
            "title": item.get("intitule") or "Titre non précisé",
            "company": entreprise.get("nom") or "Entreprise non précisée",
            "location": location,
            "salary_text": salary_text,
            "url": (item.get("origineOffre") or {}).get(
                "urlOrigine", f"https://candidat.francetravail.fr/offres/recherche/detail/{offer_id}"
            ),
            "age_days": age_days,
            "age_detail": age_detail,
            "published_at": published_raw,
            "contract_type": item.get("typeContrat"),
            "description": description,
        }

    def collect(self, now: datetime | None = None) -> list[dict]:
        client_id, client_secret = os.getenv("FT_CLIENT_ID"), os.getenv("FT_CLIENT_SECRET")
        if not client_id or not client_secret:
            self.helpers["log"]("[INFO] France Travail : identifiants absents — source ignorée.")
            return []

        rate_cfg = self.cfg.get("max_calls_per_second", MAX_CALLS_PER_SECOND)
        limiter = _RateLimiter(max_per_second=rate_cfg, sleep_fn=self.helpers.get("sleep", time.sleep))

        token_url = "https://entreprise.francetravail.fr/connexion/oauth2/access_token?realm=/partenaire"
        search_url = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
        common = dict(self.cfg.get("common_params") or {})
        max_results = max(1, min(int(self.cfg.get("max_results_per_query", 50)), 150))
        header_chars = int(self.criteria.get("header_chars", 800))
        jobs: dict[str, dict] = {}

        with httpx.Client(timeout=20.0) as http:
            limiter.acquire()
            token_resp = http.post(
                token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "scope": "api_offresdemploiv2 o2dsoffre",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            token_resp.raise_for_status()
            headers = {"Authorization": f"Bearer {token_resp.json()['access_token']}"}

            for query in self.cfg.get("queries") or []:
                if self.helpers["should_stop"]():
                    break
                params = {**common, **query, "range": f"0-{max_results - 1}"}
                limiter.acquire()
                resp = http.get(search_url, params=params, headers=headers)
                if resp.status_code == 204:
                    continue
                resp.raise_for_status()
                for item in resp.json().get("resultats", []):
                    job = self._normalize(item, now=now, header_chars=header_chars)
                    jobs.setdefault(job["id"], job)
                self.helpers["sleep"](0.5)

        return list(jobs.values())
