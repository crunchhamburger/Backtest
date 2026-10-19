"""Research configuration loading.

All experiment parameters live in ``config/research_config.json``. Modules read
them through :func:`load_config` so that a run is fully described by one file
(whose SHA-256 is stamped on every output).
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LAB_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = LAB_DIR / "config" / "research_config.json"
DEFAULT_CACHE_DIR = LAB_DIR / "data" / "cache"
DEFAULT_OUTPUT_DIR = LAB_DIR / "outputs"


@dataclass
class ResearchConfig:
    raw: dict
    path: Path
    sha256: str
    overrides: dict = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # Convenience accessors -------------------------------------------------
    @property
    def cost_bps(self) -> float:
        c = self.raw["costs"]
        return float(c["commission_bps"]) + float(c["slippage_bps"])

    @property
    def initial_capital(self) -> float:
        return float(self.raw["initial_capital"])

    @property
    def rebalance_band(self) -> float:
        return float(self.raw["rebalance_band"])

    def universe_tickers(self, name: str) -> list[str]:
        return list(self.raw["universes"][name]["tickers"])

    def enabled_universes(self) -> list[str]:
        return [k for k, v in self.raw["universes"].items() if v.get("enabled", True) and v["tickers"]]

    def all_tickers(self) -> list[str]:
        """Every ticker needed by any enabled universe or auxiliary signal."""
        out: list[str] = []
        for u in self.enabled_universes():
            out += self.universe_tickers(u)
        aux = self.raw["aux_tickers"]
        out += [aux["market"], aux["defensive"], aux["semis_leader"]]
        out += aux["risk_on_pair"] + aux["breadth"]
        for u in self.enabled_universes():
            out.append(self.raw["universes"][u]["benchmark"])
        seen, uniq = set(), []
        for t in out:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> ResearchConfig:
    """Load the JSON config, optionally applying nested ``overrides``.

    Example: ``load_config(overrides={"costs": {"slippage_bps": 20}})``.
    """
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    text = p.read_text()
    raw = json.loads(text)
    overrides = overrides or {}
    if overrides:
        raw = _deep_update(copy.deepcopy(raw), overrides)
    digest = hashlib.sha256((text + json.dumps(overrides, sort_keys=True)).encode()).hexdigest()
    return ResearchConfig(raw=raw, path=p, sha256=digest, overrides=overrides)
