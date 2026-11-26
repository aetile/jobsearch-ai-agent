from __future__ import annotations

import contextlib
import hashlib
import html as html_lib
import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .config import IDF_DEPTS, compiled_regex, get_prompt_cfg, get_runtime_cfg, atomic_write_text
from .resilience import critical_section, log

FRENCH_MONTHS = {
    "janvier": 1, "jan": 1, "fevrier": 2, "fevr": 2, "fev": 2, "mars": 3, "mar": 3,
    "avril": 4, "avr": 4, "mai": 5, "juin": 6, "juillet": 7, "juil": 7, "aout": 8, "aou": 8,
    "septembre": 9, "sept": 9, "sep": 9, "octobre": 10, "oct": 10, "novembre": 11, "nov": 11,
    "decembre": 12, "dec": 12,
}

TRACKING_PARAMS = {
    "tk", "vjs", "from", "fccid", "xkcb", "advn", "adid", "sjdu", "pp", "ref", "referrer",
    "gclid", "fbclid", "mc_cid", "mc_eid", "trk", "trackingid", "origin", "_ga", "sid", "src",
    "source", "campaign", "cmp",
}

EXPLICIT_IDF_OR_REMOTE_RX = re.compile(
    r"\b(?:75|77|78|91|92|93|94|95|ile[\s-]*de[\s-]*france|idf|paris|taverny"
    r"|full[\s-]*remote|teletravail|remote)\b",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    text = (text or "").replace("’", "'").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.split())


def is_unknown(value: str | None) -> bool:
    return normalize_text(value or "") in {"", "a extraire", "non precise", "non precisee", "n/a"}


def html_to_text(raw: str) -> str:
    if not raw:
        return ""
    if "&lt;" in raw:
        raw = html_lib.unescape(raw)
    raw = re.sub(r"(?i)<\s*li[^>]*>", "\n- ", raw)
    raw = re.sub(r"(?i)<\s*(?:br|/p|/div|/li|/ul|/ol|/h[1-6]|/tr)\s*/?>", "\n", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html_lib.unescape(raw)
    return "\n".join(line for line in (" ".join(line.split()) for line in raw.splitlines()) if line)


def parse_iso_datetime(value) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        match = re.match(r"(\d{4}-\d{2}-\d{2})", text)
        if not match:
            return None
        try:
            dt = datetime.fromisoformat(match.group(1))
        except ValueError:
            return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def canonicalize_url(url: str) -> str:
    parts = urlsplit((url or "").strip())
    host = parts.netloc.lower().removeprefix("www.")
    path = re.sub(r"/{2,}", "/", parts.path).rstrip("/") or "/"
    query = sorted(
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in TRACKING_PARAMS and not k.lower().startswith("utm_")
    )
    return urlunsplit(((parts.scheme or "https").lower(), host, path, urlencode(query), ""))


def make_job_id(source: str, url: str) -> str:
    prefix = re.sub(r"[^A-Z0-9]+", "", normalize_text(source).upper()) or "SRC"
    digest = hashlib.sha1(canonicalize_url(url).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def detect_expired_phrase(text: str, phrases: Iterable[str]) -> str | None:
    low = normalize_text(text)
    for phrase in phrases or []:
        cleaned = normalize_text(str(phrase))
        if cleaned and cleaned in low:
            return str(phrase)
    return None


_DATE_LABEL = (
    r"(?:publie(?:e)?(?:\s+le)?|mise?\s+a\s+jour(?:\s+le)?|mis\s+a\s+jour(?:\s+le)?"
    r"|date(?:\s+de\s+publication)?|posted(?:\s+on)?)\s*:?\s*"
)
_REL_UNITS = r"(?P<unit>heures?|hours?|jours?|days?|semaines?|weeks?|sem|mois|months?|h|j)"
_REL_RX = re.compile(
    r"(?:il y a|publie(?:e)?|posted|mise?\s+a\s+jour|mis\s+a\s+jour|actualise(?:e)?)\s+"
    r"(?:il y a\s+)?(?:plus de\s+|over\s+)?(?P<n>\d+)\+?\s*" + _REL_UNITS + r"\b"
)
_AGO_RX = re.compile(r"(?P<n>\d+)\+?\s*(?P<unit>days?|weeks?|months?|hours?)\s+ago\b")
_LABEL_WORD_RX = re.compile(
    r"(?:publie(?:e)?|mise?\s+a\s+jour|mis\s+a\s+jour|actualise(?:e)?|ajoute(?:e)?|posted|updated)"
    r"\s*(?:le\s*)?:?\s*(?P<w>aujourd'?hui|hier|today|yesterday)\b"
)
_ISO_RX = re.compile(_DATE_LABEL + r"(\d{4})-(\d{2})-(\d{2})")
_SLASH_RX = re.compile(_DATE_LABEL + r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})")
_FR_DATE_RX = re.compile(
    _DATE_LABEL + r"(\d{1,2})\s+(" + "|".join(sorted(FRENCH_MONTHS, key=len, reverse=True)) + r")\.?\s+(\d{4})"
)


def _relative_to_days(qty: int, unit: str) -> int:
    first = unit[0]
    if first == "h":
        return max(0, qty // 24)
    if first in ("j", "d"):
        return qty
    if first in ("s", "w"):
        return qty * 7
    return qty * 30


def estimate_job_age_days(
    text: str,
    published_at: str | None = None,
    reference_now: datetime | None = None,
    header_chars: int | None = None,
) -> tuple[int | None, str]:
    now = reference_now or datetime.now(timezone.utc)
    dt = parse_iso_datetime(published_at)
    if dt is not None:
        return max(0, int((now - dt).total_seconds() // 86400)), f"date structurée ({str(published_at)[:10]})"

    low = normalize_text(text[:header_chars] if header_chars else text)
    if m := _LABEL_WORD_RX.search(low):
        return (0, "aujourd'hui") if m.group("w").startswith(("aujourd", "today")) else (1, "hier")
    if m := _REL_RX.search(low):
        return _relative_to_days(int(m.group("n")), m.group("unit")), f"il y a {m.group('n')} {m.group('unit')}"
    if m := _AGO_RX.search(low):
        return _relative_to_days(int(m.group("n")), m.group("unit")), f"{m.group('n')} {m.group('unit')} ago"

    def _days_since(y: int, mo: int, d: int) -> int:
        return max(0, (now - datetime(y, mo, d, tzinfo=timezone.utc)).days)

    if m := _ISO_RX.search(low):
        y, mo, d = map(int, m.groups())
        with contextlib.suppress(ValueError):
            return _days_since(y, mo, d), f"date {y:04d}-{mo:02d}-{d:02d}"
    if m := _SLASH_RX.search(low):
        d, mo, y = map(int, m.groups())
        y += 2000 if y < 100 else 0
        with contextlib.suppress(ValueError):
            return _days_since(y, mo, d), f"date {d:02d}/{mo:02d}/{y:04d}"
    if m := _FR_DATE_RX.search(low):
        d, name, y = int(m.group(1)), m.group(2), int(m.group(3))
        with contextlib.suppress(ValueError):
            return _days_since(y, FRENCH_MONTHS[name], d), f"date {d} {name} {y}"
    return None, "date non détectée"


class SeenDB:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seen_jobs (
                job_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                url TEXT,
                match_score INTEGER,
                seen_at TEXT NOT NULL,
                status TEXT,
                filter_reason TEXT,
                rules_hash TEXT,
                fingerprint TEXT
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_seen_fingerprint ON seen_jobs(fingerprint)")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS location_cache (
                location_key TEXT PRIMARY KEY,
                country_code TEXT,
                dept_code TEXT,
                is_idf INTEGER,
                resolved_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def get_cached_location(self, location_key: str) -> dict | None:
        row = self.conn.execute(
            "SELECT country_code, dept_code, is_idf, resolved_at FROM location_cache WHERE location_key = ?",
            (location_key,),
        ).fetchone()
        if not row:
            return None
        country_code, dept_code, is_idf, resolved_at = row
        dt = parse_iso_datetime(resolved_at)
        if dt and datetime.now(timezone.utc) - dt > timedelta(days=90):
            return None
        return {
            "country_code": country_code,
            "dept_code": dept_code,
            "is_idf": None if is_idf is None else bool(is_idf),
        }

    def set_cached_location(
        self,
        location_key: str,
        country_code: str | None,
        dept_code: str | None,
        is_idf: bool | None,
    ) -> None:
        with critical_section():
            self.conn.execute(
                """
                INSERT OR REPLACE INTO location_cache
                    (location_key, country_code, dept_code, is_idf, resolved_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    location_key,
                    country_code,
                    dept_code,
                    None if is_idf is None else int(is_idf),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self.conn.commit()

    def should_skip(self, job_id: str, rules_hash: str, recheck_days: int = 0, now: datetime | None = None) -> bool:
        row = self.conn.execute(
            "SELECT status, rules_hash, seen_at FROM seen_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if not row:
            return False
        status, stored_hash, seen_at = row
        if status == "evaluated":
            return True
        if status == "filtered":
            if stored_hash != rules_hash:
                return False
            if recheck_days > 0:
                seen = parse_iso_datetime(seen_at)
                if seen and (now or datetime.now(timezone.utc)) - seen > timedelta(days=recheck_days):
                    return False
            return True
        return False

    def _upsert(
        self,
        job: dict,
        status: str,
        score: int | None,
        reason: str | None,
        rules_hash: str | None,
        fingerprint: str | None,
    ) -> None:
        with critical_section():
            self.conn.execute(
                """
                INSERT OR REPLACE INTO seen_jobs
                    (job_id, source, title, url, match_score, seen_at, status, filter_reason, rules_hash, fingerprint)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job["id"],
                    job.get("source", ""),
                    job.get("title") or "",
                    job.get("url", ""),
                    score,
                    datetime.now(timezone.utc).isoformat(),
                    status,
                    reason,
                    rules_hash,
                    fingerprint,
                ),
            )
            self.conn.commit()

    def mark_evaluated(self, job: dict, score: int, fingerprint: str | None = None) -> None:
        self._upsert(job, "evaluated", score, None, None, fingerprint)

    def mark_filtered(self, job: dict, reason: str, rules_hash: str) -> None:
        self._upsert(job, "filtered", 0, reason, rules_hash, None)

    def fingerprint_evaluated(self, fingerprint: str, exclude_id: str | None = None) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM seen_jobs WHERE fingerprint = ? AND status = 'evaluated' AND job_id != ? LIMIT 1",
            (fingerprint, exclude_id or ""),
        ).fetchone()
        return row is not None


def content_fingerprint(job: dict) -> str | None:
    company = job.get("company") or ""
    if is_unknown(company) or is_unknown(job.get("title")):
        return None
    basis = f"{normalize_text(job['title'])}|{normalize_text(company)}|{normalize_text(job.get('location') or '')}"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def prefilter_score(job: dict, _criteria: dict | None = None) -> float:
    age = job.get("age_days")
    return 0.0 if age is None else max(0.0, 30.0 - float(age))


def _format_salary_range(s_min: int | None, s_max: int | None) -> str:
    if s_min is not None and s_max is not None:
        return f"{s_min} – {s_max} k€"
    if s_min is not None:
        return f"dès {s_min} k€"
    if s_max is not None:
        return f"jusqu'à {s_max} k€"
    return "Non précisé"


def _age_str(item: dict) -> str:
    return f"{item['age_days']} j" if item.get("age_days") is not None else "non précisé"


def is_qualifying(item: dict, min_score: int) -> bool:
    return bool(item.get("meets_criteria")) and int(item.get("match_score", 0)) >= min_score


def format_markdown_report(evaluations: list[dict], partial: bool = False, min_score: int = 40) -> str:
    ordered = sorted(evaluations, key=lambda x: int(x.get("match_score", 0)), reverse=True)
    qualifying = [x for x in ordered if is_qualifying(x, min_score)]
    discarded = [x for x in ordered if not is_qualifying(x, min_score)]
    suffix = " _(résultats partiels)_" if partial else ""
    lines = [
        "# Rapport d'évaluation des offres d'emploi",
        f"_Mis à jour le {datetime.now().strftime('%Y-%m-%d %H:%M')}{suffix}_",
        "",
        f"**Retenues (score ≥ {min_score}) :** {len(qualifying)} | **Écartées :** {len(discarded)}",
        "",
        "## Offres retenues",
        "",
    ]
    if not qualifying:
        lines += [f"_Aucune offre ne valide pleinement les critères (score ≥ {min_score})._", ""]
    for item in qualifying:
        lines += [
            f"### {item.get('title', '?')} — {item.get('company', '?')} ({item.get('match_score', 0)}/100)",
            f"- **ID :** `{item.get('job_id', '')}` | **Remote :** `{item.get('remote_mode', '')}` | **Âge :** `{_age_str(item)}` | **Zone OK :** `{item.get('location_or_remote_ok')}`",
            f"- **Lieu :** {item.get('location', '')} | **Salaire :** {_format_salary_range(item.get('salary_min_k'), item.get('salary_max_k'))}",
            f"- **Lien :** {item.get('url', '')}",
            f"- **Résumé :** {item.get('summary', '')}",
            f"- **Atouts :** {', '.join(item.get('matching_skills') or []) or 'Aucun'}",
            f"- **Manques :** {', '.join(item.get('missing_skills') or []) or 'Aucun'}",
            f"- **Alertes :** {', '.join(item.get('red_flags') or []) or 'Aucune'}",
            "",
        ]
    lines += ["## Offres écartées ou hors cible", ""] + (
        ["_Aucune offre écartée enregistrée._", ""]
        if not discarded
        else [
            f"- **{item.get('title', '?')}** — {item.get('company', '?')} ({item.get('match_score', 0)}/100, "
            f"{_format_salary_range(item.get('salary_min_k'), item.get('salary_max_k'))}) : "
            f"{'; '.join(item.get('red_flags') or []) or 'Critères non validés'} — {item.get('url', '')}"
            for item in discarded
        ]
    )
    return "\n".join(lines + [""])


def format_filtered_report(filtered: list[dict], partial: bool = False) -> str:
    suffix = " _(résultats partiels)_" if partial else ""
    lines = ["# Rapport des offres filtrées", f"_Mis à jour le {datetime.now().strftime('%Y-%m-%d %H:%M')}{suffix}_", ""]
    for item in filtered:
        lines += [
            f"## {item.get('title') or 'Sans titre'} — {item.get('company') or item.get('source', '')}",
            f"- **ID :** `{item.get('job_id', '')}` | **Source :** `{item.get('source', '')}` | **Motif :** {item.get('filter_reason', '')}",
            f"- **Lieu :** {item.get('location') or 'Non précisé'} | **Lien :** {item.get('url') or ''}",
            "",
        ]
    return "\n".join(lines)


def write_scrape_outputs(out_dir: Path, jobs: list[dict]) -> None:
    lines = ["# Offres collectées", f"_Mis à jour le {datetime.now().strftime('%Y-%m-%d %H:%M')} — Total : {len(jobs)}_", ""]
    for item in jobs:
        lines += [
            f"## {item.get('title') or 'Sans titre'} — {item.get('source', '')}",
            f"- **ID :** `{item.get('id', '')}` | **Lieu :** {item.get('location', '')} | **Lien :** {item.get('url', '')}",
            "",
        ]
    with critical_section():
        atomic_write_text(out_dir / "offres_collectees.json", json.dumps(jobs, indent=2, ensure_ascii=False))
        atomic_write_text(out_dir / "rapport_collectees.md", "\n".join(lines))


@dataclass
class RunState:
    evaluations: dict[str, dict] = field(default_factory=dict)
    filtered: list[dict] = field(default_factory=list)
    evals_this_run: int = 0
    failures: int = 0
    quota_exceeded: bool = False
    circuit_open: bool = False
    interrupted: bool = False
    fatal_error: str | None = None


def persist_outputs(out_dir: Path, state: RunState, criteria: dict, partial: bool) -> None:
    pr = get_prompt_cfg(criteria)
    with critical_section():
        atomic_write_text(
            out_dir / "offres_evaluees.json",
            json.dumps(list(state.evaluations.values()), indent=2, ensure_ascii=False),
        )
        atomic_write_text(
            out_dir / "rapport_offres.md",
            format_markdown_report(list(state.evaluations.values()), partial, int(pr.get("report_min_score", 40))),
        )
        atomic_write_text(out_dir / "offres_filtrees.json", json.dumps(state.filtered, indent=2, ensure_ascii=False))
        atomic_write_text(out_dir / "rapport_filtrees.md", format_filtered_report(state.filtered, partial))


def load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [x for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []
    except (OSError, ValueError):
        log(f"[WARN] {path.name} illisible — ignoré.")
        return []


def passes_deterministic_filter(
    job: dict,
    criteria: dict,
    now: datetime | None = None,
    store: SeenDB | None = None,
    geo_resolver=None,
) -> tuple[bool, str]:
    from .geo import resolve_location_info

    resolver = geo_resolver or resolve_location_info
    now = now or datetime.now(timezone.utc)
    rt = get_runtime_cfg(criteria)
    title = (job.get("title") or "").strip()
    raw_location = (job.get("location") or "").strip()
    location_norm = normalize_text(raw_location)
    head = (job.get("description") or "")[: int(rt.get("header_chars", 800))]

    if hit := detect_expired_phrase(f"{title} {head}", rt.get("expired_offer_phrases", [])):
        return False, f"Offre marquée inactive/expirée ('{hit}')"
    if (valid_through := parse_iso_datetime(job.get("valid_through"))) and valid_through < now:
        return False, f"Offre expirée (validThrough {valid_through.date()})"
    for pattern in rt.get("blocked_title_patterns", []):
        if compiled_regex(pattern).search(title):
            return False, f"Titre exclu par motif '{pattern}'"

    if not is_unknown(raw_location) and not EXPLICIT_IDF_OR_REMOTE_RX.search(location_norm):
        geo = resolver(raw_location, store=store)
        dept = geo.get("dept_code")
        cc = geo.get("country_code")
        if cc == "fr" and dept and dept not in IDF_DEPTS:
            return False, f"Lieu hors IDF (département {dept})"
        if cc and cc != "fr":
            return False, f"Lieu non référencé comme commune FR ({raw_location})"

    return True, "OK"


def triage_jobs(
    jobs: list[dict],
    store: SeenDB,
    criteria: dict,
    rules_hash: str,
    state: RunState,
    force_refilter: bool = False,
) -> list[dict]:
    from .resilience import GracefulShutdown, should_stop

    rt = get_runtime_cfg(criteria)
    recheck_days = int(rt.get("filtered_recheck_days", 14))
    seen_fingerprints: set[str] = set()
    pending: list[dict] = []

    for job in jobs:
        if should_stop():
            raise GracefulShutdown("STOP")
        job_id = job["id"]
        if not force_refilter and store.should_skip(job_id, rules_hash, recheck_days):
            continue
        passed, reason = passes_deterministic_filter(job, criteria, store=store)
        if not passed:
            entry = {
                "job_id": job_id,
                "source": job.get("source", ""),
                "title": job.get("title", ""),
                "company": job.get("company", ""),
                "location": job.get("location", ""),
                "url": job.get("url", ""),
                "age_days": job.get("age_days"),
                "age_detail": job.get("age_detail"),
                "filter_reason": reason,
                "filtered_at": datetime.now(timezone.utc).isoformat(),
            }
            state.filtered = [x for x in state.filtered if x.get("job_id") != job_id] + [entry]
            store.mark_filtered(job, reason, rules_hash)
            continue
        fp = content_fingerprint(job)
        if fp and (fp in seen_fingerprints or store.fingerprint_evaluated(fp, job_id)):
            continue
        if fp:
            seen_fingerprints.add(fp)
        pending.append(job)

    pending.sort(key=lambda j: (str(j.get("source") or ""), -prefilter_score(j, criteria)))
    return pending
