from __future__ import annotations

import importlib
import json
import pkgutil
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .config import SAMPLE_JOB_ID, get_base_dir, get_runtime_cfg, get_sources_path
from .resilience import GracefulShutdown, interruptible_sleep, log, should_stop
from .storage import estimate_job_age_days, html_to_text, make_job_id, write_scrape_outputs

COLLECTOR_REGISTRY: dict[str, type[BaseCollector]] = {}


class BaseCollector(ABC):
    """Contrat commun à tous les collecteurs d'offres."""

    source_type: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.source_type:
            COLLECTOR_REGISTRY[cls.source_type] = cls

    def __init__(self, source_cfg: dict, criteria: dict, helpers: dict[str, Callable]) -> None:
        self.cfg = source_cfg
        self.criteria = criteria
        self.helpers = helpers

    @abstractmethod
    def collect(self, now: datetime | None = None) -> list[dict]:
        """Retourne une liste d'offres normalisées au format attendu par triage_jobs()."""
        raise NotImplementedError


def discover_collectors(package_dir: Path | None = None) -> dict[str, type[BaseCollector]]:
    """Découvre dynamiquement les collecteurs dans le dossier `collectors` sous `get_base_dir()`."""
    target_dir = (package_dir or (get_base_dir() / "collectors")).resolve()
    if not target_dir.is_dir():
        return dict(COLLECTOR_REGISTRY)
    parent = str(target_dir.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    pkg_name = target_dir.name
    for module_info in pkgutil.iter_modules([str(target_dir)]):
        if not module_info.name.startswith("_"):
            importlib.import_module(f"{pkg_name}.{module_info.name}")
    return dict(COLLECTOR_REGISTRY)


def default_collector_helpers() -> dict[str, Callable]:
    return {
        "log": log,
        "should_stop": should_stop,
        "sleep": interruptible_sleep,
        "make_job_id": make_job_id,
        "html_to_text": html_to_text,
        "estimate_job_age_days": estimate_job_age_days,
    }


def sample_job() -> dict:
    return {
        "id": SAMPLE_JOB_ID,
        "source": "sample",
        "category": "sample",
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


def collect_jobs(
    criteria: dict,
    out_dir: Path,
    *,
    sources_path: Path | str | None = None,
    no_france_travail: bool = False,
    write_outputs: bool = True,
) -> list[dict]:
    src_file = Path(sources_path).expanduser().resolve() if sources_path else get_sources_path()
    if not src_file.exists():
        return []
    registry = discover_collectors()
    sources = json.loads(src_file.read_text(encoding="utf-8")).get("sources", [])
    rt = get_runtime_cfg(criteria)
    helpers = default_collector_helpers()
    unique: dict[str, dict] = {}

    for src in sources:
        if should_stop():
            raise GracefulShutdown("STOP")
        if not src.get("enabled", True) or (no_france_travail and src.get("type") == "francetravail"):
            continue
        collector_cls = registry.get(src.get("type", ""))
        if collector_cls is None:
            continue
        src_name = src.get("name") or src.get("type") or "source"
        try:
            log(f"[COLLECT] {src_name} ({src.get('type')})...")
            jobs_src = collector_cls(src, rt, helpers).collect()
            log(f"[COLLECT] {src_name} -> {len(jobs_src)} offre(s)")
            for job in jobs_src:
                unique.setdefault(job["id"], job)
        except GracefulShutdown:
            raise
        except Exception as exc:
            log(f"[WARN] Collecteur '{src_name}' : {type(exc).__name__}: {exc}")

    jobs = list(unique.values())
    if write_outputs:
        write_scrape_outputs(out_dir, jobs)
    return jobs
