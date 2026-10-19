"""Perfect-knowledge long-only benchmarks (theoretical, impossible to implement).

Decision timing
---------------
Realistic strategies decide at the close of bar t and trade at the open of
bar t+1. The oracle is given the *same execution points* (market opens) but is
allowed to know the future: at the open of bar i it chooses long or cash for
the interval open[i] -> open[i+1], knowing that interval's return. It cannot
trade intraday, cannot short and cannot use leverage.

Benchmarks
----------
* Daily oracle (costless): long whenever the next open-to-open asset return
  beats the cash accrual. Also reported net of costs on the same path.
* Cost-aware oracle: dynamic programme maximising terminal log-wealth with a
  proportional cost per switch (unconstrained number of trades).
* Constrained oracle: the same dynamic programme with at most K entries
  (round trips) over the sample, for a grid of K.

Each oracle path is converted into target weights and run through the
standard engine, so its accounting is identical to the realistic strategies.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from backtest_engine import backtest
from data_pipeline import MarketData


def interval_returns(md: MarketData, ticker: str) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Open-to-next-open asset returns and matching cash returns, for listed bars."""
    b = md.bars[ticker]
    live = b["listed"].to_numpy()
    o = b["Open"].to_numpy(dtype=float)
    cg = md.cash_growth.to_numpy(dtype=float)
    idx = md.calendar
    # Start at the second listed bar so that the decision for every interval can be
    # expressed as a target on a listed bar (the engine executes target[t-1] at open[t]).
    pos = np.flatnonzero(live)[1:]
    a = o[pos[1:]] / o[pos[:-1]] - 1
    c = cg[pos[1:]]
    return a, c, idx[pos[:-1]]


def daily_oracle_path(a: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (a > c).astype(np.int8)


def dp_oracle_path(a: np.ndarray, c: np.ndarray, cost: float, max_entries: int | None = None) -> tuple[np.ndarray, float]:
    """Optimal long/cash path maximising log wealth with proportional switching costs.

    Starts in cash. ``max_entries`` caps the number of cash->long switches.
    Returns (path of 0/1 states per interval, terminal log wealth).
    """
    T = len(a)
    la = np.log1p(a)
    lc = np.log1p(c)
    lk = math.log(1 - cost) if cost > 0 else 0.0
    K = (T + 1) if max_entries is None else int(max_entries)
    K = min(K, T + 1)
    NEG = -1e18
    V0 = np.full(K + 1, NEG)  # in cash, k entries used
    V1 = np.full(K + 1, NEG)  # long
    V0[0] = 0.0
    bp0 = np.zeros((T, K + 1), dtype=np.int8)  # previous state for state 0
    bp1 = np.zeros((T, K + 1), dtype=np.int8)
    for i in range(T):
        # to cash: from cash (no cost) or from long (exit cost, same k)
        stay0 = V0
        from1 = V1 + lk
        n0 = np.maximum(stay0, from1)
        bp0[i] = (from1 > stay0).astype(np.int8)
        # to long: from long (no cost) or from cash with k-1 entries (entry cost)
        stay1 = V1
        from0 = np.full(K + 1, NEG)
        from0[1:] = V0[:-1] + lk
        n1 = np.maximum(stay1, from0)
        bp1[i] = np.where(from0 > stay1, 0, 1).astype(np.int8)
        V0 = n0 + lc[i]
        V1 = n1 + la[i]
    # terminal: liquidation cost if long at the end is not charged (mark-to-market, like strategies)
    k0, k1 = int(np.argmax(V0)), int(np.argmax(V1))
    if V0[k0] >= V1[k1]:
        s, k, best = 0, k0, V0[k0]
    else:
        s, k, best = 1, k1, V1[k1]
    path = np.zeros(T, dtype=np.int8)
    for i in range(T - 1, -1, -1):
        path[i] = s
        if s == 0:
            prev = int(bp0[i, k])
        else:
            prev = int(bp1[i, k])
            if prev == 0:
                k -= 1
        s = prev
    return path, float(best)


def path_to_target(md: MarketData, ticker: str, path: np.ndarray, dates: pd.DatetimeIndex) -> pd.Series:
    """Convert a per-interval path (state held from open[i]) into close-decided target weights.

    The engine executes target[t-1] at open[t]; therefore the state chosen for
    the interval starting at open[i] is written to the target of the previous bar.
    """
    tgt = pd.Series(0.0, index=md.calendar, name=ticker)
    pos = md.calendar.get_indexer(dates)
    prev = pos - 1
    ok = prev >= 0
    tgt.iloc[prev[ok]] = path[ok].astype(float)
    return tgt


def run_oracles(md: MarketData, ticker: str, cost_bps: float, trade_limits: list[int], capital: float = 100_000.0,
                start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> dict:
    """Compute all oracle variants for one asset over [start, end]."""
    a, c, dates = interval_returns(md, ticker)
    if start is not None or end is not None:
        m = np.ones(len(dates), dtype=bool)
        if start is not None:
            m &= dates >= start
        if end is not None:
            m &= dates <= end
        a, c, dates = a[m], c[m], dates[m]
    cost = cost_bps / 1e4
    out = {"dates": dates, "paths": {}, "results": {}, "summary": []}
    paths = {"Daily oracle (costless decision)": daily_oracle_path(a, c)}
    p, _ = dp_oracle_path(a, c, cost, None)
    paths["Cost-aware oracle (unlimited trades)"] = p
    for K in trade_limits:
        p, _ = dp_oracle_path(a, c, cost, K)
        paths[f"Constrained oracle (max {K} entries)"] = p
    for name, p in paths.items():
        tgt = path_to_target(md, ticker, p, dates)
        res_net = backtest(md, tgt, cost_bps, capital=capital, band=0.0)
        out["paths"][name] = p
        out["results"][name] = res_net
        if name.startswith("Daily"):
            out["results"][name + " [0 bps]"] = backtest(md, tgt, 0.0, capital=capital, band=0.0)
    return out


def missed_vs_avoided(md: MarketData, ticker: str, exposure: pd.Series, dates: pd.DatetimeIndex | None = None) -> dict:
    """Decompose a strategy's cash periods into missed rallies and avoided losses (open-to-open, log terms)."""
    a, c, d = interval_returns(md, ticker)
    w = exposure.reindex(d).fillna(0.0).to_numpy()
    if dates is not None:
        m = np.isin(d, dates)
        a, c, w = a[m], c[m], w[m]
    out_frac = 1 - np.clip(w, 0, 1)
    ex = np.log1p(a) - np.log1p(c)
    missed = float((out_frac * np.where(ex > 0, ex, 0)).sum())
    avoided = float((out_frac * np.where(ex < 0, -ex, 0)).sum())
    return {"missed_rallies_log": missed, "losses_avoided_log": avoided, "net_timing_log": avoided - missed}
