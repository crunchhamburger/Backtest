import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from data_pipeline import assemble  # noqa: E402
from lab_config import load_config  # noqa: E402
from synthetic_data import synthetic_universe  # noqa: E402

TICKERS = ["SPY", "QQQ", "IEF", "XLY", "XLP", "XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLB", "SMH", "NVDA"]


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def raw_cash():
    return synthetic_universe(TICKERS, start="2001-01-01", end="2011-12-30", seed=11)


@pytest.fixture(scope="session")
def md(cfg, raw_cash):
    raw, cash = raw_cash
    return assemble(raw, cash, cfg, source="SYNTHETIC-TEST")


@pytest.fixture(scope="session")
def aux(cfg):
    a = cfg["aux_tickers"]
    return {"market": "SPY", "defensive": "IEF", "risk_on_num": "XLY", "risk_on_den": "XLP",
            "breadth_list": [t for t in a["breadth"]], "semis_leader": "SMH"}


def make_flat_md(cfg, n=40, price=100.0, opens=None, closes=None, cash_pct=0.0):
    """Tiny deterministic one-asset market for exact accounting tests."""
    idx = pd.bdate_range("2020-01-01", periods=n)
    o = np.full(n, price) if opens is None else np.asarray(opens, float)
    c = np.full(n, price) if closes is None else np.asarray(closes, float)
    raw = pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.001, "Low": np.minimum(o, c) * 0.999, "Close": c,
                        "Adj Close": c, "Volume": 1e6, "Dividends": 0.0, "Stock Splits": 0.0}, index=idx)
    cash = pd.Series(cash_pct, index=idx)
    return assemble({"SPY": raw}, cash, cfg, source="TEST")
