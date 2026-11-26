from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from core.collectors import (
    BaseCollector,
    COLLECTOR_REGISTRY,
    default_collector_helpers,
)

__all__ = [
    "BaseCollector",
    "COLLECTOR_REGISTRY",
    "default_collector_helpers",
    "discover_collectors",
]


def discover_collectors(package_dir: Path | None = None) -> dict[str, type[BaseCollector]]:
    """Importe dynamiquement tous les modules présents dans le package `collectors/`
    en alimentant le registre partagé de `core.collectors`.
    """
    target_dir = Path(package_dir).resolve() if package_dir is not None else Path(__file__).resolve().parent
    if not target_dir.is_dir():
        return dict(COLLECTOR_REGISTRY)
    for module_info in pkgutil.iter_modules([str(target_dir)]):
        if not module_info.name.startswith("_"):
            importlib.import_module(f"{__name__}.{module_info.name}")
    return dict(COLLECTOR_REGISTRY)
