"""Synthetic market data for SOFTWARE TESTING ONLY.

Generates regime-switching, factor-correlated daily OHLCV with dividend and
split events, laid out exactly like a Yahoo download (``auto_adjust=False``).
Used by the unit tests and the ``--synthetic`` smoke run to exercise every code
path without network access. Outputs produced from this data are watermarked and
must never be interpreted as research results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline import RAW_COLUMNS

SYNTHETIC_BANNER = "SYNTHETIC DATA - SOFTWARE TEST ONLY - NOT RESEARCH RESULTS"


def business_calendar(start: str, end: str) -> pd.DatetimeIndex:
    days = pd.bdate_range(start, end)
    # Remove a few fixed-date holidays so calendars are not trivially regular.
    hol = days[((days.month == 1) & (days.day == 1)) | ((days.month == 7) & (days.day == 4)) |
               ((days.month == 12) & (days.day == 25))]
    return days.difference(hol)


def make_raw(cal: pd.DatetimeIndex, rng: np.random.Generator, market: np.ndarray, beta: float, vol: float,
             drift: float, div_yield: float, split_dates: list[pd.Timestamp] | None = None) -> pd.DataFrame:
    n = len(cal)
    idio = rng.standard_normal(n) * vol / np.sqrt(252)
    r = drift / 252 + beta * market + idio
    r = np.clip(r, -0.35, 0.35)
    gap = rng.standard_normal(n) * vol / np.sqrt(252) * 0.4
    tr_close = 50 * np.exp(np.cumsum(r))          # total-return path
    # Quarterly dividends: price drops by the dividend on ex-dates.
    q = pd.Series(cal.to_period("Q"), index=cal)
    ex = (q != q.shift(1)).to_numpy() & (np.arange(n) > 0)
    div = np.zeros(n)
    price = np.empty(n)
    price[0] = tr_close[0]
    for i in range(1, n):
        gross = price[i - 1] * tr_close[i] / tr_close[i - 1]
        if ex[i] and div_yield > 0:
            div[i] = round(price[i - 1] * div_yield / 4, 4)
            gross -= div[i]
        price[i] = max(gross, 0.01)
    close = price
    open_ = np.r_[close[0], close[:-1] * np.exp(gap[1:])]
    if div_yield > 0:
        open_ = np.where(div > 0, open_ - div, open_)
    hi = np.maximum(open_, close) * (1 + np.abs(rng.standard_normal(n)) * vol / np.sqrt(252) * 0.5)
    lo = np.minimum(open_, close) * (1 - np.abs(rng.standard_normal(n)) * vol / np.sqrt(252) * 0.5)
    vol_sh = np.round(1e6 * np.exp(rng.standard_normal(n) * 0.3))
    # Yahoo-style multiplicative dividend adjustment for Adj Close.
    factor = np.ones(n)
    f = 1.0
    for i in range(n - 1, 0, -1):
        factor[i] = f
        if div[i] > 0:
            f *= 1 - div[i] / close[i - 1]
    factor[0] = f
    splits = np.zeros(n)
    for d in split_dates or []:
        if d in cal:
            splits[cal.get_loc(d)] = 4.0
    df = pd.DataFrame({"Open": open_, "High": hi, "Low": lo, "Close": close, "Adj Close": close * factor,
                       "Volume": vol_sh, "Dividends": div, "Stock Splits": splits}, index=cal)
    df.index.name = "Date"
    return df[RAW_COLUMNS]


def synthetic_universe(tickers: list[str], start: str = "1995-01-01", end: str = "2026-10-16", seed: int = 7,
                       inception: dict | None = None) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Return raw bars for each ticker and a synthetic T-bill yield in percent."""
    rng = np.random.default_rng(seed)
    cal = business_calendar(start, end)
    n = len(cal)
    # Two-state Markov market factor (calm bull / turbulent bear).
    state = np.zeros(n, dtype=int)
    for i in range(1, n):
        p = 0.995 if state[i - 1] == 0 else 0.985
        state[i] = state[i - 1] if rng.random() < p else 1 - state[i - 1]
    mu = np.where(state == 0, 0.12, -0.15) / 252
    sd = np.where(state == 0, 0.13, 0.30) / np.sqrt(252)
    market = mu + sd * rng.standard_normal(n)
    inception = inception or {}
    out = {}
    for k, t in enumerate(tickers):
        trng = np.random.default_rng(seed * 1000 + k)
        is_stock = len(t) <= 4 and not t.startswith("X") and t not in {"SPY", "QQQ", "DIA", "IWM", "MDY", "VTI",
                                                                        "RSP", "SMH", "SOXX", "XSD", "PSI", "IEF"}
        beta = 1.0 if t == "SPY" else (0.2 if t == "IEF" else trng.uniform(0.6, 1.6))
        vol = 0.06 if t == "IEF" else (trng.uniform(0.25, 0.5) if is_stock else trng.uniform(0.08, 0.2))
        drift = trng.uniform(-0.02, 0.06)
        dy = 0.0 if is_stock else trng.uniform(0.005, 0.03)
        mkt = market if t != "IEF" else -0.2 * market
        splits = [cal[n // 2]] if is_stock else []
        df = make_raw(cal, trng, mkt, beta if t != "IEF" else 1.0, vol, drift, dy, splits)
        if t in inception:
            df = df[df.index >= pd.Timestamp(inception[t])]
        out[t] = df
    y = 2.5 + 2.0 * np.sin(np.arange(n) / 900.0)
    cash = pd.Series(np.clip(y, 0.05, None), index=cal, name="SYNTH:TBILL")
    return out, cash
