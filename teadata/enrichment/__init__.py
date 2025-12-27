from typing import Dict
from .base import Enricher

_registry: Dict[str, Enricher] = {}


def enricher(name: str):
    def wrap(cls):
        if not issubclass(cls, Enricher):
            raise TypeError("enricher must extend Enricher")
        _registry[name] = cls()
        return cls

    return wrap


def run_enrichment(
    repo, cfg_path: str, year: int, *, datasets: list[str] | None = None
):
    # If datasets is None, run all registered
    selected = datasets or list(_registry.keys())
    stats = {}
    for name in selected:
        enr = _registry.get(name)
        if not enr:
            print(f"[enrich] '{name}' not registered; skipping")
            continue
        try:
            stats[name] = enr.apply(repo, cfg_path, year)
        except Exception as e:
            print(f"[enrich] {name} failed: {e}")
    return stats
