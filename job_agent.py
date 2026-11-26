import json
import os
import re
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from markdownify import markdownify as md
from playwright.sync_api import sync_playwright
from pydantic import BaseModel, Field

load_dotenv()

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
DB_PATH = OUTPUT_DIR / "jobs_seen.db"
SOURCES_PATH = BASE_DIR / "sources.json"
CRITERIA_PATH = BASE_DIR / "criteria.json"
CV_HTML_PATH = BASE_DIR / "cv_arnaud_etile_devops_v2.html"
CV_MD_PATH = BASE_DIR / "cv.md"
PROMPT_TXT_PATH = BASE_DIR / "prompt.txt"

DEFAULT_SYSTEM_INSTRUCTION = (
    "Tu es un recruteur technique senior spécialisé en DevOps, Ingénierie Système Linux, "
    "Automatisation/Infrastructure et Bases de Données critiques. "
    "Évalue la convergence entre le CV du candidat et l'offre d'emploi fournie de manière factuelle."
)

DEFAULT_CRITERIA = {
    "model_name": "gemini-3.8-flash",
    "http_timeout_ms": 20000,
    "retry_max_attempts": 3,
    "retry_backoff_base_s": 1.5,
    "retry_backoff_cap_s": 10.0,
    "retry_max_elapsed_s": 60.0,
    "retry_circuit_breaker_503": 3,
    "max_gemini_evaluations_per_run": 15,
    "max_job_age_days": 30,
    "strict_job_age_filter": False,
    "expired_offer_phrases": [
        "offre pourvue",
        "poste pourvu",
        "offre expirée",
        "offre expiree",
        "n'est plus disponible",
        "nest plus disponible",
        "cette offre n'est plus en ligne",
        "cette offre nest plus en ligne",
        "recrutement clos",
        "clôture des candidatures",
        "cloture des candidatures",
        "position filled",
        "job has expired",
        "no longer available",
    ],
    "blocked_link_texts": [
        "offres",
        "inscription",
        "connexion",
        "se connecter",
        "aller au contenu principal",
        "passer au contenu principal",
        "trier par pertinence",
        "trier par alphabétique",
        "trier par alphabetique",
        "trier par date",
    ],
    "min_salary_k": 65,
    "salary_filter_tolerance_k": 5,
    "max_offers_per_page": 8,
    "france_travail_keywords": "DevOps Linux Kubernetes",
    "reference_location": "Taverny",
    "accepted_zones": [
        "full remote",
        "télétravail partiel",
        "télétravail",
        "95",
        "val-d'oise",
        "75",
        "paris",
        "77",
        "seine-et-marne",
        "78",
        "yvelines",
        "91",
        "essonne",
        "92",
        "hauts-de-seine",
        "93",
        "seine-saint-denis",
        "94",
        "val-de-marne",
        "idf hors 95 (75, 77, 78, 91, 92, 93, 94 — télétravail partiel ou full remote)",
        "roissy-cdg",
        "saint-denis",
        "nord-ouest idf",
    ],
    "blocked_title_patterns": [
        r"\bjunior\b",
        r"\balternance\b",
        r"\bapprentissage\b",
        r"\bstage\b",
        r"\bfrontend\b",
        r"\bfront-end\b",
        r"\bfreelance\b",
    ],
    "blocked_text_patterns": [
        r"\bfreelance\b",
        r"\btjm\b",
        r"\bportage salarial\b",
        r"\bindépendant\b",
        r"\bindependant\b",
    ],
    "required_keywords": [
        "devops",
        "sre",
        "platform",
        "kubernetes",
        "openshift",
        "terraform",
        "ansible",
        "saltstack",
        "linux",
        "rhel",
        "azure",
        "dba",
        "oracle",
        "cassandra",
        "infrastructure",
        "système",
        "systemes",
    ],
    "excluded_url_parts": [
        "login",
        "signin",
        "signup",
        "cookie",
        "privacy",
        "legal",
        "mentions",
        "contact",
        "press",
        "presse",
        "news",
        "actualit",
        "faq",
        "about",
        "a-propos",
        "linkedin.com",
        "facebook.com",
        "x.com",
        "twitter.com",
        "youtube.com",
    ],
}

FRENCH_MONTHS = {
    "janvier": 1,
    "jan": 1,
    "février": 2,
    "fevrier": 2,
    "févr": 2,
    "fevr": 2,
    "fév": 2,
    "fev": 2,
    "mars": 3,
    "mar": 3,
    "avril": 4,
    "avr": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "juil": 7,
    "août": 8,
    "aout": 8,
    "aoû": 8,
    "aou": 8,
    "septembre": 9,
    "sept": 9,
    "sep": 9,
    "octobre": 10,
    "oct": 10,
    "novembre": 11,
    "nov": 11,
    "décembre": 12,
    "decembre": 12,
    "déc": 12,
    "dec": 12,
}

_STOP_REQUESTED = False


class GracefulShutdown(Exception):
    """Signal d'interruption propre (SIGINT / SIGTERM)."""


class DailyQuotaExceededError(Exception):
    """Quota journalier Gemini épuisé (429 RESOURCE_EXHAUSTED)."""


class OfferEvaluationTimeoutError(Exception):
    """Budget de temps dépassé pour l'évaluation d'une offre."""


def _handle_stop_signal(signum: int, _frame) -> None:
    global _STOP_REQUESTED
    sig_name = (
        signal.Signals(signum).name
        if signum in signal.Signals._value2member_map_
        else str(signum)
    )
    print(f"\n[STOP] Signal {sig_name} reçu — arrêt propre en cours...")
    _STOP_REQUESTED = True
    raise GracefulShutdown(sig_name)


def install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _handle_stop_signal)
    signal.signal(signal.SIGTERM, _handle_stop_signal)


def should_stop() -> bool:
    return _STOP_REQUESTED


class JobEvaluation(BaseModel):
    job_id: str
    title: str
    company: str
    location: str
    remote_mode: Literal["full_remote", "hybrid", "onsite", "unspecified"]
    location_or_remote_ok: bool = Field(
        description=(
            "True si le poste est dans le Val-d'Oise (95), en full remote, ou en Île-de-France "
            "hors 95 (75 Paris, 77 Seine-et-Marne, 78 Yvelines, 91 Essonne, 92 Hauts-de-Seine, "
            "93 Seine-Saint-Denis, 94 Val-de-Marne) avec télétravail partiel (hybride) ou full remote"
        )
    )
    salary_min_k: int | None = Field(
        default=None, description="Salaire annuel brut minimum indiqué ou estimé en k€"
    )
    salary_max_k: int | None = Field(
        default=None, description="Salaire annuel brut maximum indiqué ou estimé en k€"
    )
    match_score: int = Field(ge=0, le=100, description="Convergence avec le CV de 0 à 100")
    meets_criteria: bool = Field(
        description=(
            "True si le poste correspond aux critères : séniorité, localisation/télétravail "
            "(95 accepté ; tous les départements IDF hors 95 — 75, 77, 78, 91, 92, 93, 94 — acceptés "
            "avec télétravail partiel ou full remote), et statut cohérent (absence de salaire non éliminatoire)"
        )
    )
    matching_skills: list[str]
    missing_skills: list[str]
    red_flags: list[str]
    summary: str = Field(description="Synthèse factuelle en 2-3 phrases")


def load_criteria(path: Path = CRITERIA_PATH) -> dict:
    criteria = {
        key: list(val) if isinstance(val, list) else val
        for key, val in DEFAULT_CRITERIA.items()
    }
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            for key, val in loaded.items():
                if val is not None:
                    criteria[key] = val
    return criteria


def load_system_instruction(path: Path = PROMPT_TXT_PATH) -> str:
    if path.exists():
        content = path.read_text(encoding="utf-8").strip()
        if content:
            return content
    return DEFAULT_SYSTEM_INSTRUCTION


def load_cv_markdown() -> str:
    if CV_MD_PATH.exists():
        return CV_MD_PATH.read_text(encoding="utf-8")

    if not CV_HTML_PATH.exists():
        raise FileNotFoundError(f"CV introuvable : {CV_HTML_PATH}")

    soup = BeautifulSoup(CV_HTML_PATH.read_text(encoding="utf-8"), "html.parser")
    for tag in soup(["style", "script", "head", "meta", "link"]):
        tag.decompose()

    container = soup.find("div", class_="cv-container") or soup.body
    raw_md = md(str(container), heading_style="ATX", strip=["img", "svg"])
    lines = [line.strip() for line in raw_md.splitlines() if line.strip()]
    return "\n".join(lines)


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seen_jobs (
            job_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT,
            match_score INTEGER,
            seen_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def is_job_seen(conn: sqlite3.Connection, job_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM seen_jobs WHERE job_id = ?", (job_id,)).fetchone()
    return row is not None


def mark_job_seen(
    conn: sqlite3.Connection,
    job_id: str,
    source: str,
    title: str,
    url: str,
    match_score: int | None,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO seen_jobs (job_id, source, title, url, match_score, seen_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (job_id, source, title, url, match_score, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def load_sources(sources_path: Path = SOURCES_PATH) -> list[dict]:
    if not sources_path.exists():
        return []

    data = json.loads(sources_path.read_text(encoding="utf-8"))
    targets: list[dict] = []

    for entry in data.get("jobboards", []):
        if entry.get("enabled"):
            for url in entry.get("search_urls", []):
                targets.append(
                    {
                        "source": entry["name"],
                        "category": "jobboard",
                        "zone": entry.get("zone", ""),
                        "url": url,
                    }
                )

    for entry in data.get("companies", []):
        if entry.get("enabled"):
            for url in entry.get("careers_urls", []):
                targets.append(
                    {
                        "source": entry["name"],
                        "category": "company",
                        "zone": entry.get("zone", ""),
                        "url": url,
                    }
                )

    return targets


def is_candidate_offer_url(
    href: str, base_domain: str, excluded_url_parts: list[str] | tuple[str, ...]
) -> bool:
    if not href.startswith(("http://", "https://")):
        return False
    parsed = urlparse(href)
    if parsed.netloc != base_domain:
        return False

    lower_full = href.lower()
    if any(part.lower() in lower_full for part in excluded_url_parts):
        return False

    path_and_query = f"{parsed.path} {parsed.query}".lower()
    return any(
        token in path_and_query
        for token in (
            "job",
            "viewjob",
            "job-listing",
            "offre",
            "emploi",
            "poste",
            "career",
            "vacancy",
            "position",
            "detail",
        )
    )


def _is_blocked_link_text(text: str, blocked_texts: list[str] | tuple[str, ...]) -> bool:
    cleaned = " ".join(text.lower().split())
    if len(cleaned) < 4:
        return True
    for blocked in blocked_texts:
        b = " ".join(str(blocked).lower().split())
        if not b:
            continue
        if cleaned == b or cleaned.startswith(f"{b} "):
            return True
    return False


def _interruptible_sleep(seconds: float) -> None:
    deadline = time.monotonic() + max(0.0, seconds)
    while not should_stop():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.25, remaining))
    if should_stop():
        raise GracefulShutdown("SIGINT")


def detect_expired_phrase(text: str, phrases: list[str] | tuple[str, ...]) -> str | None:
    lower_text = " ".join(text.lower().split())
    for phrase in phrases:
        cleaned = " ".join(str(phrase).lower().split())
        if cleaned and cleaned in lower_text:
            return str(phrase)
    return None


def estimate_job_age_days(
    text: str,
    published_at: str | None = None,
    reference_now: datetime | None = None,
) -> tuple[int | None, str]:
    now = reference_now or datetime.now(timezone.utc)

    if published_at:
        try:
            iso_candidate = published_at.replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso_candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            days = max(0, int((now - dt).total_seconds() // 86400))
            return days, f"date structurée ({published_at[:10]})"
        except Exception:
            pass

    lower_text = text.lower()

    if re.search(r"\b(?:aujourd['’]hui|today|à l['’]instant|just now)\b", lower_text):
        return 0, "aujourd'hui"
    if re.search(r"\b(?:hier|yesterday)\b", lower_text):
        return 1, "hier"

    rel_match = re.search(
        r"\b(?:publi[ée]\s+)?(?:il\s+y\s+a|depuis|posted)\s+(\d+)\s+"
        r"(heures?|h|jours?|j|semaines?|sem|mois|days?|weeks?|months?)\b",
        lower_text,
    )
    if rel_match:
        qty = int(rel_match.group(1))
        unit = rel_match.group(2)
        if unit.startswith(("heure", "h")):
            return max(0, qty // 24), f"il y a {qty} {unit}"
        if unit.startswith(("jour", "j", "day")):
            return qty, f"il y a {qty} {unit}"
        if unit.startswith(("semaine", "sem", "week")):
            return qty * 7, f"il y a {qty} {unit}"
        if unit.startswith(("mois", "month")):
            return qty * 30, f"il y a {qty} {unit}"

    iso_match = re.search(
        r"(?:publi[ée](?:\s+le)?|mise\s+[àa]\s+jour(?:\s+le)?|date(?:\s+de\s+publication)?)\s*:?\s*"
        r"(\d{4})-(\d{2})-(\d{2})",
        lower_text,
    )
    if iso_match:
        try:
            year, month, day = map(int, iso_match.groups())
            dt = datetime(year, month, day, tzinfo=timezone.utc)
            days = max(0, (now - dt).days)
            return days, f"date {year:04d}-{month:02d}-{day:02d}"
        except ValueError:
            pass

    slash_match = re.search(
        r"(?:publi[ée](?:\s+le)?|mise\s+[àa]\s+jour(?:\s+le)?|date(?:\s+de\s+publication)?)\s*:?\s*"
        r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})",
        lower_text,
    )
    if slash_match:
        try:
            day, month, year = map(int, slash_match.groups())
            if year < 100:
                year += 2000
            dt = datetime(year, month, day, tzinfo=timezone.utc)
            days = max(0, (now - dt).days)
            return days, f"date {day:02d}/{month:02d}/{year:04d}"
        except ValueError:
            pass

    month_names_pattern = "|".join(sorted(FRENCH_MONTHS.keys(), key=len, reverse=True))
    fr_date_match = re.search(
        rf"(?:publi[ée](?:\s+le)?|mise\s+[àa]\s+jour(?:\s+le)?|date(?:\s+de\s+publication)?)\s*:?\s*"
        rf"(\d{{1,2}})\s+({month_names_pattern})\s+(\d{{4}})",
        lower_text,
    )
    if fr_date_match:
        try:
            day = int(fr_date_match.group(1))
            month = FRENCH_MONTHS[fr_date_match.group(2)]
            year = int(fr_date_match.group(3))
            dt = datetime(year, month, day, tzinfo=timezone.utc)
            days = max(0, (now - dt).days)
            return days, f"date {day} {fr_date_match.group(2)} {year}"
        except ValueError:
            pass

    return None, "date non détectée"


def scrape_target_pages(
    targets: list[dict],
    criteria: dict,
    run_deadline: float | None = None,
) -> list[dict]:
    jobs: list[dict] = []
    seen_urls: set[str] = set()
    max_per_page = int(criteria.get("max_offers_per_page", 5))
    goto_timeout_ms = int(criteria.get("page_goto_timeout_ms", 12000))
    excluded_url_parts = criteria.get("excluded_url_parts", [])
    blocked_link_texts = criteria.get("blocked_link_texts", DEFAULT_CRITERIA["blocked_link_texts"])
    expired_phrases = criteria.get("expired_offer_phrases", DEFAULT_CRITERIA["expired_offer_phrases"])

    browser = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                ),
                locale="fr-FR",
            )
            for target in targets:
                if should_stop():
                    raise GracefulShutdown("SIGINT")
                if run_deadline is not None and time.monotonic() >= run_deadline:
                    print("[INFO] Temps global de scraping atteint — passage à l'évaluation.")
                    break

                base_url = target["url"]
                source_name = target["source"]
                domain = urlparse(base_url).netloc
                print(f"[SCRAPE] ({source_name}) {base_url}")

                page = context.new_page()
                try:
                    try:
                        page.evaluate("() => { try { window.stop(); } catch (e) {} }")
                    except Exception:
                        pass
                    page.goto(base_url, wait_until="domcontentloaded", timeout=goto_timeout_ms)
                    _interruptible_sleep(1.0)

                    raw_links = page.evaluate(
                        """() => Array.from(document.querySelectorAll('a[href]')).map(a => ({
                            href: a.href,
                            text: (a.innerText || '').trim()
                        }))"""
                    )

                    offer_links: list[tuple[str, str]] = []
                    for item in raw_links:
                        href = (item.get("href") or "").split("#")[0]
                        text = " ".join((item.get("text") or "").split())
                        if not href or href in seen_urls:
                            continue
                        if not is_candidate_offer_url(href, domain, excluded_url_parts):
                            continue
                        if _is_blocked_link_text(text, blocked_link_texts):
                            continue

                        seen_urls.add(href)
                        offer_links.append((href, text[:140]))
                        if len(offer_links) >= max_per_page:
                            break

                    print(f"[SCRAPE] ({source_name}) {len(offer_links)} lien(s) retenu(s)")

                    for offer_url, link_title in offer_links:
                        if should_stop():
                            raise GracefulShutdown("SIGINT")
                        if run_deadline is not None and time.monotonic() >= run_deadline:
                            break

                        offer_page = context.new_page()
                        try:
                            offer_page.evaluate("() => { try { window.stop(); } catch (e) {} }")
                            offer_page.goto(
                                offer_url, wait_until="domcontentloaded", timeout=goto_timeout_ms
                            )
                            _interruptible_sleep(0.5)
                            body_text = offer_page.inner_text("body")
                            trimmed_text = re.sub(r"\n{3,}", "\n\n", body_text).strip()[:12000]

                            if len(trimmed_text) < 200:
                                continue

                            expired_hit = detect_expired_phrase(trimmed_text, expired_phrases)
                            if expired_hit:
                                print(
                                    f"[STALE] Fiche marquée inactive/expirée ({expired_hit}) : {offer_url}"
                                )
                                continue

                            age_days, age_detail = estimate_job_age_days(trimmed_text)
                            job_id = f"{source_name.upper()}-{abs(hash(offer_url)) % 10**8}"
                            jobs.append(
                                {
                                    "id": job_id,
                                    "source": source_name,
                                    "title": link_title,
                                    "company": (
                                        source_name if target["category"] == "company" else "À extraire"
                                    ),
                                    "location": target.get("zone") or "À extraire",
                                    "salary_text": "Non précisé",
                                    "url": offer_url,
                                    "age_days": age_days,
                                    "age_detail": age_detail,
                                    "description": trimmed_text,
                                }
                            )
                        except GracefulShutdown:
                            raise
                        except Exception as exc:
                            print(f"[WARN] Lecture fiche impossible ({offer_url}): {exc}")
                        finally:
                            try:
                                offer_page.close()
                            except Exception:
                                pass

                except GracefulShutdown:
                    raise
                except Exception as exc:
                    print(f"[WARN] Page source inaccessible ({base_url}): {exc}")
                    if "interrupted by another navigation" in str(exc).lower():
                        try:
                            context.close()
                        except Exception:
                            pass
                        context = browser.new_context(
                            user_agent=(
                                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                            ),
                            locale="fr-FR",
                        )
                finally:
                    try:
                        page.close()
                    except Exception:
                        pass

            browser.close()
            browser = None
    except GracefulShutdown:
        print("[STOP] Scraping interrompu — fermeture propre du navigateur...")
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass
        raise

    return jobs


def fetch_france_travail_jobs(keywords: str = "DevOps Linux Kubernetes") -> list[dict]:
    client_id = os.getenv("FT_CLIENT_ID")
    client_secret = os.getenv("FT_CLIENT_SECRET")

    if not client_id or not client_secret:
        print("[INFO] Aucune source active et identifiants France Travail absents : jeu de test.")
        return [
            {
                "id": "SAMPLE-001",
                "source": "sample",
                "title": "Ingénieur Système Linux & Automatisation (Interne)",
                "company": "Groupe Industriel Nord IDF",
                "location": "95 - Val-d'Oise / Télétravail hybride",
                "salary_text": "66 000 € - 72 000 € / an",
                "url": "https://example.org/jobs/sample-001",
                "age_days": 3,
                "age_detail": "il y a 3 jours",
                "description": (
                    "Poste interne sur Linux (RHEL/Debian), Ansible, Terraform, supervision "
                    "Prometheus/Grafana et fiabilisation de bases Oracle. Télétravail hybride ou remote."
                ),
            }
        ]

    token_url = (
        "https://entreprise.francetravail.fr/connexion/oauth2/access_token"
        "?realm=/partenaire"
    )
    search_url = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"

    with httpx.Client(timeout=20.0) as http:
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
        access_token = token_resp.json()["access_token"]

        offers_resp = http.get(
            search_url,
            params={"motsCles": keywords, "range": "0-29"},
            headers={"Authorization": f"Bearer {access_token}"},
        )
        offers_resp.raise_for_status()
        raw_items = offers_resp.json().get("resultats", [])

    normalized: list[dict] = []
    for item in raw_items:
        lieu = item.get("lieuTravail", {})
        entreprise = item.get("entreprise", {})
        salaire = item.get("salaire", {})
        offer_id = str(item.get("id", ""))
        published_raw = item.get("dateActualisation") or item.get("dateCreation")
        description = item.get("description", "")
        age_days, age_detail = estimate_job_age_days(description, published_at=published_raw)
        normalized.append(
            {
                "id": f"FT-{offer_id}",
                "source": "francetravail",
                "title": item.get("intitule", "Titre non précisé"),
                "company": entreprise.get("nom", "Entreprise non précisée"),
                "location": lieu.get("libelle", "Non précisé"),
                "salary_text": salaire.get("libelle", "Non précisé"),
                "url": item.get("origineOffre", {}).get(
                    "urlOrigine",
                    f"https://candidat.francetravail.fr/offres/recherche/detail/{offer_id}",
                ),
                "age_days": age_days,
                "age_detail": age_detail,
                "description": description,
            }
        )
    return normalized


def passes_deterministic_filter(job: dict, criteria: dict) -> tuple[bool, str]:
    title_lower = job["title"].lower()
    full_text_lower = f"{job['title']} {job['location']} {job['description']}".lower()

    expired_phrases = criteria.get("expired_offer_phrases", DEFAULT_CRITERIA["expired_offer_phrases"])
    expired_hit = detect_expired_phrase(full_text_lower, expired_phrases)
    if expired_hit:
        return False, f"Offre marquée inactive/expirée ('{expired_hit}')"

    max_age_raw = criteria.get("max_job_age_days", 30)
    strict_age = bool(criteria.get("strict_job_age_filter", False))
    age_days = job.get("age_days")
    if age_days is None and not job.get("age_detail"):
        age_days, _ = estimate_job_age_days(full_text_lower)

    if max_age_raw is not None:
        max_age_days = int(max_age_raw)
        if max_age_days > 0 and age_days is not None and age_days > max_age_days:
            return False, f"Offre trop ancienne ({age_days}j > {max_age_days}j)"
        if max_age_days > 0 and age_days is None and strict_age:
            return False, "Date de publication introuvable (filtre âge strict activé)"

    blocked_title_patterns = criteria.get("blocked_title_patterns", [])
    blocked_text_patterns = criteria.get("blocked_text_patterns", [])
    required_keywords = [kw.lower() for kw in criteria.get("required_keywords", [])]
    min_salary_k = int(criteria.get("min_salary_k", 65))
    tolerance_k = int(criteria.get("salary_filter_tolerance_k", 5))

    for pattern in blocked_title_patterns:
        if re.search(pattern, title_lower, flags=re.IGNORECASE):
            return False, f"Titre exclu par motif '{pattern}'"

    for pattern in blocked_text_patterns:
        if re.search(pattern, full_text_lower, flags=re.IGNORECASE):
            return False, f"Offre exclue par motif texte '{pattern}'"

    if required_keywords and not any(kw in full_text_lower for kw in required_keywords):
        return False, "Aucun mot-clé technique prioritaire détecté"

    for match in re.findall(r"\b(\d{2})\s?k\s?(?:€|eur|euros)?(?:\s*/\s*an)?\b", full_text_lower):
        val = int(match)
        if 25 <= val < min_salary_k - tolerance_k:
            return False, f"Salaire affiché ({val}k€) inférieur au seuil"

    return True, "OK"


def format_criteria_context(criteria: dict) -> str:
    zones = ", ".join(criteria.get("accepted_zones", [])) or "Non précisées"
    max_age = criteria.get("max_job_age_days")
    age_line = (
        f"Âge maximum souhaité : {max_age} jours"
        if max_age is not None and int(max_age) > 0
        else "Âge maximum souhaité : non limité"
    )
    return (
        f"Salaire cible minimum : {criteria.get('min_salary_k', 65)} k€ brut/an "
        f"(absence de salaire non éliminatoire) | "
        f"Localisation de référence : {criteria.get('reference_location', 'Taverny')} | "
        f"Règle géo : Val-d'Oise (95) accepté ; tous les départements IDF hors 95 "
        f"(75 Paris, 77 Seine-et-Marne, 78 Yvelines, 91 Essonne, 92 Hauts-de-Seine, "
        f"93 Seine-Saint-Denis, 94 Val-de-Marne) sont valides avec télétravail partiel ou full remote | "
        f"Zones : {zones} | {age_line}"
    )


def _resolve_retry_settings(criteria: dict) -> tuple[int, float, float, float]:
    max_attempts = max(1, int(criteria.get("retry_max_attempts", 3)))
    base_s = max(0.1, float(criteria.get("retry_backoff_base_s", 1.5)))
    cap_s = max(base_s, float(criteria.get("retry_backoff_cap_s", 10.0)))
    max_elapsed_s = max(1.0, float(criteria.get("retry_max_elapsed_s", 60.0)))

    env_attempts = os.getenv("GEMINI_RETRY_MAX_ATTEMPTS")
    if env_attempts is not None and env_attempts.strip():
        max_attempts = max(1, int(env_attempts))

    env_base = os.getenv("GEMINI_RETRY_BACKOFF_BASE_S")
    if env_base is not None and env_base.strip():
        base_s = max(0.1, float(env_base))

    env_cap = os.getenv("GEMINI_RETRY_BACKOFF_CAP_S")
    if env_cap is not None and env_cap.strip():
        cap_s = max(base_s, float(env_cap))

    env_elapsed = os.getenv("GEMINI_RETRY_MAX_ELAPSED_S")
    if env_elapsed is not None and env_elapsed.strip():
        max_elapsed_s = max(1.0, float(env_elapsed))

    return max_attempts, base_s, cap_s, max_elapsed_s


def _is_daily_quota_error(exc: ClientError) -> bool:
    msg = str(exc).lower()
    return (
        "per_day" in msg
        or "requests per day" in msg
        or "free_tier_requests" in msg
        or "retry in " in msg
    )


def evaluate_job_with_gemini(
    client: genai.Client,
    cv_markdown: str,
    job: dict,
    system_instruction: str,
    criteria: dict,
) -> JobEvaluation:
    model_name = criteria.get("model_name", "gemini-3.8-flash")
    criteria_context = format_criteria_context(criteria)
    max_attempts, base_s, cap_s, max_elapsed_s = _resolve_retry_settings(criteria)
    http_timeout_ms = max(2000, int(criteria.get("http_timeout_ms", 20000)))
    start_time = time.monotonic()

    age_days = job.get("age_days")
    age_detail = job.get("age_detail") or "non précisée"
    age_info = f"{age_days} jour(s) ({age_detail})" if age_days is not None else f"inconnue ({age_detail})"

    prompt = f"""
--- CRITÈRES DE RECHERCHE ---
{criteria_context}

--- CV DU CANDIDAT ---
{cv_markdown}

--- OFFRE D'EMPLOI ---
ID : {job["id"]}
Source : {job["source"]}
Titre : {job["title"]}
Entreprise : {job["company"]}
Lieu / Zone : {job["location"]}
Ancienneté détectée : {age_info}
Salaire brut affiché : {job["salary_text"]}
URL : {job["url"]}
Contenu extrait :
{job["description"]}
"""

    for attempt in range(max_attempts):
        if should_stop():
            raise GracefulShutdown("SIGINT")
        elapsed = time.monotonic() - start_time
        if elapsed >= max_elapsed_s:
            raise OfferEvaluationTimeoutError(
                f"Timeout ({max_elapsed_s:.1f}s) atteint sur {job['id']}"
            )
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=JobEvaluation,
                    temperature=0.1,
                    http_options=types.HttpOptions(timeout=http_timeout_ms),
                ),
            )
            return JobEvaluation.model_validate_json(response.text)
        except GracefulShutdown:
            raise
        except KeyboardInterrupt:
            raise GracefulShutdown("SIGINT")
        except ClientError as exc:
            if getattr(exc, "code", None) == 429 or "429" in str(exc):
                if _is_daily_quota_error(exc):
                    raise DailyQuotaExceededError(str(exc)) from exc
                if attempt == max_attempts - 1:
                    raise DailyQuotaExceededError(str(exc)) from exc
                remaining = max_elapsed_s - (time.monotonic() - start_time)
                if remaining <= 0:
                    raise OfferEvaluationTimeoutError(
                        f"Timeout ({max_elapsed_s:.1f}s) atteint sur {job['id']}"
                    )
                wait_s = min(cap_s, base_s * (2 ** attempt), remaining)
                print(
                    f"[RETRY] 429 sur {job['id']} (essai {attempt + 1}/{max_attempts}), "
                    f"nouvel essai dans {wait_s:.1f}s"
                )
                _interruptible_sleep(wait_s)
            else:
                raise
        except ServerError as exc:
            remaining = max_elapsed_s - (time.monotonic() - start_time)
            if attempt == max_attempts - 1:
                raise
            if remaining <= 0.2:
                raise OfferEvaluationTimeoutError(
                    f"Timeout ({max_elapsed_s:.1f}s) atteint sur {job['id']}"
                ) from exc
            wait_s = min(cap_s, base_s * (2 ** attempt), remaining)
            print(
                f"[RETRY] 503 sur {job['id']} (essai {attempt + 1}/{max_attempts}), "
                f"nouvel essai dans {wait_s:.1f}s ({exc})"
            )
            _interruptible_sleep(wait_s)

    raise RuntimeError(f"Échec inattendu de l'évaluation pour {job['id']}")


def _atomic_write_text(target_path: Path, content: str) -> None:
    tmp_path = target_path.with_name(f".{target_path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, target_path)


def _format_markdown_report(evaluations: list[dict], partial: bool = False) -> str:
    sorted_evals = sorted(evaluations, key=lambda x: x["match_score"], reverse=True)
    status_suffix = " _(en cours / interrompu — résultats partiels)_" if partial else ""
    lines = [
        "# Rapport d'évaluation des offres d'emploi",
        f"_Mis à jour le {datetime.now().strftime('%Y-%m-%d %H:%M')}{status_suffix}_",
        "",
    ]

    for item in sorted_evals:
        age_str = (
            f"{item['age_days']} j"
            if item.get("age_days") is not None
            else "non précisé"
        )
        lines.extend(
            [
                f"## {item['title']} — {item['company']} ({item['match_score']}/100)",
                (
                    f"- **ID :** `{item['job_id']}` | **Télétravail :** `{item['remote_mode']}` | "
                    f"**Ancienneté :** `{age_str}` | "
                    f"**Zone/Remote OK :** `{item['location_or_remote_ok']}` | "
                    f"**Critères OK :** `{item['meets_criteria']}`"
                ),
                f"- **Lieu :** {item['location']} | **Lien :** {item['url']}",
                f"- **Salaire estimé/indiqué (k€) :** {item['salary_min_k']} – {item['salary_max_k']}",
                f"- **Résumé :** {item['summary']}",
                f"- **Compétences alignées :** {', '.join(item['matching_skills']) or 'Aucune'}",
                f"- **Points manquants :** {', '.join(item['missing_skills']) or 'Aucun'}",
                f"- **Points de vigilance :** {', '.join(item['red_flags']) or 'Aucun'}",
                "",
            ]
        )

    return "\n".join(lines)


def write_outputs(evaluations: list[dict], partial: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "offres_evaluees.json"
    md_path = OUTPUT_DIR / "rapport_offres.md"

    json_content = json.dumps(evaluations, indent=2, ensure_ascii=False)
    md_content = _format_markdown_report(evaluations, partial=partial)

    _atomic_write_text(json_path, json_content)
    _atomic_write_text(md_path, md_content)


def _load_existing_evaluations(json_path: Path) -> dict[str, dict]:
    if not json_path.exists():
        return {}
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return {
                str(item["job_id"]): item
                for item in raw
                if isinstance(item, dict) and "job_id" in item
            }
    except Exception:
        pass
    return {}


def main() -> int:
    install_signal_handlers()

    if not os.getenv("GEMINI_API_KEY"):
        raise EnvironmentError("La variable d'environnement GEMINI_API_KEY est manquante.")

    conn: sqlite3.Connection | None = None
    evaluations_by_id: dict[str, dict] = {}
    interrupted = False
    quota_exceeded = False

    try:
        criteria = load_criteria(CRITERIA_PATH)
        system_instruction = load_system_instruction(PROMPT_TXT_PATH)
        cv_markdown = load_cv_markdown()
        conn = init_db(DB_PATH)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        evaluations_by_id = _load_existing_evaluations(OUTPUT_DIR / "offres_evaluees.json")

        write_outputs(list(evaluations_by_id.values()), partial=True)

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

        targets = load_sources()
        ft_keywords = str(criteria.get("france_travail_keywords", "DevOps Linux Kubernetes"))
        max_run_duration_s = float(criteria.get("max_run_duration_s", 90.0))
        run_deadline = time.monotonic() + max_run_duration_s if max_run_duration_s > 0 else None
        jobs = (
            scrape_target_pages(targets, criteria, run_deadline=run_deadline)
            if targets
            else fetch_france_travail_jobs(ft_keywords)
        )

        max_evals_per_run = int(criteria.get("max_gemini_evaluations_per_run", 15))
        circuit_breaker_503 = int(criteria.get("retry_circuit_breaker_503", 3))
        consecutive_503 = 0
        evals_this_run = 0

        for job in jobs:
            if should_stop():
                raise GracefulShutdown("SIGINT")

            if max_evals_per_run > 0 and evals_this_run >= max_evals_per_run:
                print(
                    f"[INFO] Plafond d'évaluations par exécution atteint ({max_evals_per_run}) — arrêt."
                )
                break

            job_id = job["id"]
            if is_job_seen(conn, job_id):
                print(f"[SKIP] Déjà traitée : {job_id} - {job['title']}")
                continue

            passed, reason = passes_deterministic_filter(job, criteria)
            if not passed:
                print(f"[FILTER] {job_id} ({job['title']}) -> {reason}")
                mark_job_seen(conn, job_id, job["source"], job["title"], job["url"], 0)
                continue

            print(f"[GEMINI] Évaluation de {job_id} - {job['title']}...")
            try:
                evaluation = evaluate_job_with_gemini(
                    client, cv_markdown, job, system_instruction, criteria
                )
            except GracefulShutdown:
                raise
            except DailyQuotaExceededError:
                quota_exceeded = True
                print(
                    f"[QUOTA] Quota Gemini atteint (429 RESOURCE_EXHAUSTED) — "
                    f"{job_id} non marquée comme vue, arrêt propre."
                )
                break
            except OfferEvaluationTimeoutError as exc:
                consecutive_503 += 1
                print(f"[WARN] {exc} — offre ignorée pour ce run.")
                if circuit_breaker_503 > 0 and consecutive_503 >= circuit_breaker_503:
                    print(
                        f"[STOP] Coupe-circuit activé après {consecutive_503} échecs consécutifs — arrêt."
                    )
                    break
                continue
            except ServerError as exc:
                consecutive_503 += 1
                print(f"[WARN] 503 persistant sur {job_id} ({job['title']}) : {exc}")
                if circuit_breaker_503 > 0 and consecutive_503 >= circuit_breaker_503:
                    print(
                        f"[STOP] Coupe-circuit 503 activé après {consecutive_503} échecs consécutifs — arrêt."
                    )
                    break
                continue

            consecutive_503 = 0
            evals_this_run += 1
            mark_job_seen(
                conn,
                job_id,
                job["source"],
                job["title"],
                job["url"],
                evaluation.match_score,
            )

            record = evaluation.model_dump()
            record["url"] = job["url"]
            record["age_days"] = job.get("age_days")
            record["age_detail"] = job.get("age_detail")
            evaluations_by_id[job_id] = record

            write_outputs(list(evaluations_by_id.values()), partial=True)
            print(f"[SAVE] {job_id} ajouté à output/ ({len(evaluations_by_id)} évaluée(s))")

    except (GracefulShutdown, KeyboardInterrupt):
        interrupted = True
        print("[STOP] Interruption détectée — finalisation de la sauvegarde...")
    finally:
        try:
            write_outputs(
                list(evaluations_by_id.values()),
                partial=(interrupted or quota_exceeded),
            )
            print(f"[SAVE] Sorties synchronisées dans {OUTPUT_DIR}")
        except Exception as exc:
            print(f"[WARN] Échec de l'écriture des sorties : {exc}")
        if conn is not None:
            try:
                conn.close()
                print("[CLEANUP] Connexion SQLite fermée.")
            except Exception:
                pass

    if quota_exceeded:
        print(
            f"[DONE] Quota journalier atteint ({len(evaluations_by_id)} offre(s) évaluée(s) sauvegardée(s))."
        )
        return 75

    if interrupted:
        print(
            f"[DONE] Arrêt propre ({len(evaluations_by_id)} offre(s) évaluée(s) sauvegardée(s))."
        )
        return 130

    print(f"[DONE] {len(evaluations_by_id)} offre(s) évaluée(s) -> {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
