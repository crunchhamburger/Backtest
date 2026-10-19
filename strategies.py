"""Strategy signal library and the frozen 100-strategy registry.

Every strategy function receives a context object and returns *target weights
decided at the close* of each bar (Series for one asset, DataFrame when a
defensive/alternate asset or a universe is involved). The engine shifts them to
the next open; strategy code never handles execution timing.

Registry rules
--------------
* Exactly 100 strategies: 10 families x 10 variants, each with its own
  hypothesis, entry/exit rules, parameters, allocation rule, minimum lookback,
  benchmark and expected regime.
* ``grid`` lists a small neighbourhood of alternative parameter values used for
  parameter-sensitivity and walk-forward selection. The ``params`` value is the
  pre-registered default and is the only one used for headline results.
* :func:`freeze_registry` writes ``Strategy_Registry.csv`` and a SHA-256 hash
  of the canonical registry so later edits are detectable.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import indicators as ta
from data_pipeline import MarketData

# ============================================================================
# Context objects
# ============================================================================


class AssetContext:
    """Data access for a single-asset strategy."""

    def __init__(self, md: MarketData, ticker: str, aux: dict, cache: dict | None = None):
        self.md = md
        self.ticker = ticker
        self.df = md.bars[ticker]
        self.aux_names = aux
        self._cache = cache if cache is not None else {}
        self.ind = self._ind(ticker)
        self.close = self.df["Close"]
        self.index = md.calendar

    def _ind(self, t: str) -> ta.IndicatorCache:
        key = (id(self.md), t)
        if key not in self._cache:
            self._cache[key] = ta.IndicatorCache(self.md.bars[t])
        return self._cache[key]

    def aux(self, role_or_ticker: str) -> ta.IndicatorCache | None:
        t = self.aux_names.get(role_or_ticker, role_or_ticker)
        if t not in self.md.bars:
            return None
        return self._ind(t)

    def aux_ticker(self, role: str) -> str:
        return self.aux_names.get(role, role)

    def has(self, role: str) -> bool:
        return self.aux_ticker(role) in self.md.bars

    def listed(self, t: str | None = None) -> pd.Series:
        return self.md.bars[t or self.ticker]["listed"]

    def cash_cum(self, n: int) -> pd.Series:
        """Cumulative cash return over the trailing n bars (known at t)."""
        g = np.log1p(self.md.cash_growth)
        return np.expm1(g.rolling(n, min_periods=n).sum())


class UniverseContext:
    """Data access for a cross-sectional (rotation) strategy."""

    def __init__(self, md: MarketData, tickers: list[str], aux: dict, cache: dict | None = None):
        self.md = md
        self.tickers = [t for t in tickers if t in md.bars]
        self.aux_names = aux
        self._cache = cache if cache is not None else {}
        self.index = md.calendar

    def ind(self, t: str) -> ta.IndicatorCache:
        key = (id(self.md), t)
        if key not in self._cache:
            self._cache[key] = ta.IndicatorCache(self.md.bars[t])
        return self._cache[key]

    def close_panel(self) -> pd.DataFrame:
        return pd.DataFrame({t: self.md.bars[t]["Close"] for t in self.tickers})

    def cash_cum(self, n: int) -> pd.Series:
        g = np.log1p(self.md.cash_growth)
        return np.expm1(g.rolling(n, min_periods=n).sum())


# ============================================================================
# Generic helpers
# ============================================================================


def _arr(x) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.fillna(False).to_numpy(dtype=bool) if x.dtype == bool or x.dtype == object else \
            x.to_numpy(dtype=float)
    return np.asarray(x)


def _bool(x: pd.Series) -> np.ndarray:
    return x.astype("boolean").fillna(False).to_numpy(dtype=bool)


def cross_up(a: pd.Series, b: pd.Series | float) -> pd.Series:
    bb = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return (a > bb) & (a.shift(1) <= bb.shift(1))


def first_bar_of_month(index: pd.DatetimeIndex) -> np.ndarray:
    """True on the first trading day of each month (fully causal rebalance clock)."""
    p = pd.Series(index.to_period("M"), index=index)
    return (p != p.shift(1)).to_numpy()


def hold_between(w: pd.Series | pd.DataFrame, clock: np.ndarray):
    """Keep weights constant between rebalance bars."""
    if isinstance(w, pd.Series):
        return w.where(clock).ffill().fillna(0.0)
    return w.where(pd.Series(clock, index=w.index), axis=0).ffill().fillna(0.0)


def smooth_weights(raw: pd.Series, min_change: float = 0.10, step: float = 0.05) -> pd.Series:
    """Quantise continuous weights to ``step`` and change them only when the move is at least
    ``min_change`` (or when the quantised target is zero). Prevents vol-target style
    strategies from trading every day on tiny weight changes."""
    r = np.clip(np.nan_to_num(raw.to_numpy(dtype=float), nan=0.0), 0, 1)
    out = np.zeros(len(r))
    cur = 0.0
    for i, v in enumerate(r):
        q = round(v / step) * step
        if abs(q - cur) >= min_change - 1e-12 or (q == 0.0 and cur != 0.0):
            cur = q
        out[i] = cur
    return pd.Series(out, index=raw.index)


def latch(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Classic state machine: go long on ``entry`` when flat, go flat on ``exit_`` when long.

    Exit is evaluated first, so a bar satisfying both conditions while long exits
    and may re-enter on the next bar (faithful to the original notebook logic).
    """
    e, x = _bool(entry), _bool(exit_)
    s = np.zeros(len(e))
    st = 0.0
    for i in range(len(e)):
        if st == 1.0 and x[i]:
            st = 0.0
        elif st == 0.0 and e[i]:
            st = 1.0
        s[i] = st
    return pd.Series(s, index=entry.index)


def state_machine(ctx_df: pd.DataFrame, entry: pd.Series, exit_: pd.Series | None = None, *,
                  stop: str | None = None, stop_param: float = 0.0, atr: pd.Series | None = None,
                  peak_field: str = "Close", max_hold: int | None = None,
                  rearm_breakout: pd.Series | None = None, stop_param_series: pd.Series | None = None) -> pd.Series:
    """General long/flat state machine with optional trailing stop and time stop.

    * Entry when flat and ``entry`` is True and the system is *armed*.
    * Exit when ``exit_`` is True, or the trailing stop is hit at the close, or
      ``max_hold`` bars have elapsed since the fill.
    * After a stop or time exit the system is disarmed until ``entry`` turns
      False at least once (edge re-trigger) or ``rearm_breakout`` fires. This
      prevents immediate re-entry churn.
    * The trailing peak starts at the fill price (next open after the entry
      signal) and is updated with ``peak_field`` (Close or High) thereafter.
    Stops: ``pct`` (exit if close < peak*(1-p)), ``atr`` (close < peak - p*ATR).
    """
    e = _bool(entry)
    x = _bool(exit_) if exit_ is not None else np.zeros(len(e), dtype=bool)
    rb = _bool(rearm_breakout) if rearm_breakout is not None else np.zeros(len(e), dtype=bool)
    c = ctx_df["Close"].to_numpy(dtype=float)
    o = ctx_df["Open"].to_numpy(dtype=float)
    pk = ctx_df[peak_field].to_numpy(dtype=float)
    a = atr.to_numpy(dtype=float) if atr is not None else None
    sp = stop_param_series.to_numpy(dtype=float) if stop_param_series is not None else None
    n = len(e)
    s = np.zeros(n)
    st = 0.0
    armed = True
    peak = np.nan
    sig_bar = -1
    for i in range(n):
        if st == 1.0:
            if i == sig_bar + 1:
                peak = max(o[i], pk[i]) if not np.isnan(o[i]) else pk[i]
            elif not np.isnan(pk[i]):
                peak = max(peak, pk[i]) if not np.isnan(peak) else pk[i]
            out = False
            if x[i]:
                out = True
            elif stop is not None and not np.isnan(peak):
                p = sp[i] if sp is not None else stop_param
                if stop == "pct":
                    out = c[i] < peak * (1 - p)
                elif stop == "atr" and a is not None and not np.isnan(a[i]) and not np.isnan(p):
                    out = c[i] < peak - p * a[i]
                if out:
                    armed = False
            if not out and max_hold is not None and i - sig_bar >= max_hold:
                out = True
                armed = False
            if out:
                st = 0.0
                peak = np.nan
        else:
            if not armed and (not e[i] or rb[i]):
                armed = True if not e[i] else bool(rb[i])
                if rb[i] and e[i]:
                    st, sig_bar = 1.0, i
                    s[i] = st
                    continue
            if armed and e[i]:
                st, sig_bar = 1.0, i
        s[i] = st
    return pd.Series(s, index=entry.index)


def _finish(ctx: AssetContext, w, extra: str | None = None):
    """Zero weights before listing, clip to [0,1], and package output."""
    if isinstance(w, pd.Series):
        w = w.fillna(0.0).clip(0, 1) * ctx.listed().astype(float)
        return w.rename(ctx.ticker)
    return w


def _with_alt(ctx: AssetContext, w_asset: pd.Series, w_alt: pd.Series, alt_role: str) -> pd.DataFrame:
    """Combine asset weight with an alternate (defensive or benchmark) asset weight."""
    w_asset = w_asset.fillna(0.0).clip(0, 1) * ctx.listed().astype(float)
    alt = ctx.aux_ticker(alt_role)
    out = pd.DataFrame({ctx.ticker: w_asset})
    if alt in ctx.md.bars and alt != ctx.ticker:
        wa = (w_alt.fillna(0.0).clip(0, 1) * ctx.listed(alt).astype(float))
        wa = np.minimum(wa, 1 - out[ctx.ticker])
        out[alt] = wa
    return out


# ============================================================================
# Family 1: moving-average trend following
# ============================================================================

def ma_cross(ctx, fast, slow, ma="sma"):
    f = ctx.ind(ma, fast)
    s = ctx.ind(ma, slow)
    return _finish(ctx, (f > s).astype(float))


def price_above_ma(ctx, n, ma="sma"):
    return _finish(ctx, (ctx.close > ctx.ind(ma, n)).astype(float))


def ma_stack(ctx, n1, n2, n3):
    a, b, c = ctx.ind("sma", n1), ctx.ind("sma", n2), ctx.ind("sma", n3)
    return _finish(ctx, ((a > b) & (b > c)).astype(float))


def ma_vote(ctx, windows):
    votes = [(ctx.close > ctx.ind("sma", n)).astype(float).where(ctx.ind("sma", n).notna()) for n in windows]
    v = pd.concat(votes, axis=1)
    w = v.mean(axis=1).where(v.notna().all(axis=1), 0.0)
    return _finish(ctx, w)


def ma_slope(ctx, n, lookback):
    m = ctx.ind("sma", n)
    return _finish(ctx, (m > m.shift(lookback)).astype(float))


def monthly_price_above_ma(ctx, n):
    raw = (ctx.close > ctx.ind("sma", n)).astype(float).where(ctx.ind("sma", n).notna(), 0.0)
    return _finish(ctx, hold_between(raw, first_bar_of_month(ctx.index)))


# ============================================================================
# Family 2: asymmetric entry/exit
# ============================================================================

def asym_price_ma(ctx, exit_n, entry_n):
    se, sx = ctx.ind("sma", entry_n), ctx.ind("sma", exit_n)
    ready = se.notna() & sx.notna()
    return _finish(ctx, latch((ctx.close > se) & ready, (ctx.close < sx) & ready))


def permission_fast_exit(ctx, trig_n, slow_fast, slow_slow):
    t = ctx.ind("sma", trig_n)
    perm = ctx.ind("sma", slow_fast) > ctx.ind("sma", slow_slow)
    return _finish(ctx, latch((ctx.close > t) & perm, ctx.close < t))


def hysteresis_band(ctx, n, band):
    m = ctx.ind("sma", n)
    return _finish(ctx, latch(ctx.close > m * (1 + band), ctx.close < m * (1 - band)))


def atr_hysteresis(ctx, n, atr_n, k):
    m, a = ctx.ind("sma", n), ctx.ind("atr", atr_n)
    return _finish(ctx, latch(ctx.close > m + k * a, ctx.close < m - k * a))


def confirm_count(ctx, n, confirm):
    m = ctx.ind("sma", n)
    above = ctx.close > m
    return _finish(ctx, latch(ta.consecutive(above) >= confirm, ctx.close < m))


def two_stage_short(ctx, n1, n2):
    """Original notebook: 100% above both SMAs, 0% below both, else 50%."""
    a, b = ctx.ind("sma", n1), ctx.ind("sma", n2)
    hi, lo = np.maximum(a, b), np.minimum(a, b)
    w = pd.Series(np.where(ctx.close > hi, 1.0, np.where(ctx.close < lo, 0.0, 0.5)), index=ctx.index)
    return _finish(ctx, w.where(a.notna() & b.notna(), 0.0))


def two_stage_long(ctx, n_fast, n_slow):
    f, s = ctx.ind("sma", n_fast), ctx.ind("sma", n_slow)
    w = 0.5 * (ctx.close > f).astype(float) + 0.5 * (f > s).astype(float)
    return _finish(ctx, w.where(s.notna(), 0.0))


def shock_exit(ctx, vol_n, k, reentry_n, min_days):
    r = ctx.close.pct_change()
    sd = r.rolling(vol_n, min_periods=vol_n).std().shift(1)
    shock = r < -k * sd
    m = ctx.ind("sma", reentry_n)
    e = _bool(ctx.close > m)
    sh = _bool(shock)
    ready = _bool(sd.notna() & m.notna())
    s = np.zeros(len(e))
    st, since = 0.0, 0
    for i in range(len(e)):
        if not ready[i]:
            s[i] = 0.0
            continue
        if st == 1.0 and sh[i]:
            st, since = 0.0, 0
        elif st == 0.0:
            since += 1
            if e[i] and since >= min_days:
                st = 1.0
        s[i] = st
    return _finish(ctx, pd.Series(s, index=ctx.index))


def cross_asym(ctx, exit_fast, exit_slow, entry_fast, entry_slow):
    xf, xs = ctx.ind("sma", exit_fast), ctx.ind("sma", exit_slow)
    ef, es = ctx.ind("sma", entry_fast), ctx.ind("sma", entry_slow)
    return _finish(ctx, latch((ef > es) & (xf > xs), xf < xs))


# ============================================================================
# Family 3: momentum
# ============================================================================

def mom_abs(ctx, n, skip=0, threshold=0.0):
    m = ta.roc_skip(ctx.close, n, skip) if skip else ctx.ind("roc", n)
    return _finish(ctx, (m > threshold).astype(float))


def tsmom_excess(ctx, n):
    return _finish(ctx, (ctx.ind("roc", n) > ctx.cash_cum(n)).astype(float))


def mom_accel(ctx, n):
    m = ctx.ind("roc", n)
    return _finish(ctx, ((m > 0) & (m > m.shift(n))).astype(float))


def mom_all(ctx, horizons):
    conds = [ctx.ind("roc", h) > 0 for h in horizons]
    w = np.logical_and.reduce([_bool(c) for c in conds]).astype(float)
    return _finish(ctx, pd.Series(w, index=ctx.index))


def mom_vote(ctx, horizons):
    m = pd.concat([ctx.ind("roc", h) for h in horizons], axis=1)
    w = (m > 0).astype(float).mean(axis=1).where(m.notna().all(axis=1), 0.0)
    return _finish(ctx, w)


def mom_tstat(ctx, n, threshold):
    r = ctx.close.pct_change()
    t = r.rolling(n, min_periods=n).mean() / r.rolling(n, min_periods=n).std() * np.sqrt(n)
    return _finish(ctx, latch(t > threshold, t < 0))


def momentum_sma100_regime(ctx, mom_n, slope_n, exit_n, entry_n, regime_n):
    """Original notebook strategy 6."""
    m100 = ctx.ind("sma", regime_n)
    lag = m100.shift(slope_n)
    mom = ctx.ind("roc", mom_n)
    ready = m100.notna() & lag.notna() & mom.notna()
    ex = (ctx.close < ctx.ind("sma", exit_n)) & (m100 < lag)
    en = (ctx.close > ctx.ind("sma", entry_n)) & (mom > 0) & (m100 > lag)
    return _finish(ctx, latch(en & ready, ex & ready))


# ============================================================================
# Family 4: mean reversion
# ============================================================================

def rsi_reversion(ctx, n, entry, exit, trend_n=None, max_hold=10):
    r = ctx.ind("rsi", n)
    e = r < entry
    if trend_n:
        e = e & (ctx.close > ctx.ind("sma", trend_n))
    return _finish(ctx, state_machine(ctx.df, e, r > exit, max_hold=max_hold))


def bollinger_reversion(ctx, n, k, trend_n=None, max_hold=20):
    lo, mid, _ = ta.bollinger(ctx.close, n, k)
    e = ctx.close < lo
    if trend_n:
        e = e & (ctx.close > ctx.ind("sma", trend_n))
    return _finish(ctx, state_machine(ctx.df, e, ctx.close > mid, max_hold=max_hold))


def pct_deviation_reversion(ctx, n, dev, max_hold=30):
    m = ctx.ind("sma", n)
    return _finish(ctx, state_machine(ctx.df, ctx.close < m * (1 - dev), ctx.close > m, max_hold=max_hold))


def atr_pullback(ctx, n, k, max_hold=15):
    m, a = ctx.ind("sma", n), ctx.ind("atr", n)
    z = (ctx.close - m) / a
    return _finish(ctx, state_machine(ctx.df, z < -k, ctx.close > m, max_hold=max_hold))


def shock_reversal(ctx, vol_n, k, hold):
    r = ctx.close.pct_change()
    sd = r.rolling(vol_n, min_periods=vol_n).std().shift(1)
    e = r < -k * sd
    return _finish(ctx, state_machine(ctx.df, e, None, max_hold=hold))


def down_days(ctx, k, max_hold):
    d = ctx.close < ctx.close.shift(1)
    e = ta.consecutive(d) >= k
    up = ctx.close > ctx.close.shift(1)
    return _finish(ctx, state_machine(ctx.df, e, up, max_hold=max_hold))


def ibs_reversion(ctx, lo, hi, max_hold=5):
    v = ctx.ind("ibs")
    return _finish(ctx, state_machine(ctx.df, v < lo, v > hi, max_hold=max_hold))


def drawdown_buy(ctx, n, dd_entry, dd_exit):
    hi = ctx.close.rolling(n, min_periods=n).max()
    dd = ctx.close / hi - 1
    return _finish(ctx, latch(dd < -dd_entry, dd > -dd_exit))


# ============================================================================
# Family 5: breakouts
# ============================================================================

def donchian(ctx, n_entry, n_exit, trend_n=None):
    hh = ta.rolling_max(ctx.df["High"], n_entry)
    ll = ta.rolling_min(ctx.df["Low"], n_exit)
    e = ctx.close > hh
    if trend_n:
        e = e & (ctx.close > ctx.ind("sma", trend_n))
    return _finish(ctx, latch(e, ctx.close < ll))


def high_52w(ctx, n, exit_n):
    hh = ta.rolling_max(ctx.close, n)
    ll = ta.rolling_min(ctx.close, exit_n)
    return _finish(ctx, latch(ctx.close >= hh, ctx.close < ll))


def squeeze_breakout(ctx, n, k, squeeze_lookback, recent):
    lo, mid, up = ta.bollinger(ctx.close, n, k)
    bw = (up - lo) / mid
    tight = bw.rolling(recent, min_periods=recent).min() <= bw.rolling(squeeze_lookback, min_periods=squeeze_lookback).min() * 1.10
    return _finish(ctx, latch(tight & (ctx.close > up), ctx.close < mid))


def volume_breakout(ctx, n_entry, n_exit, vol_n, vol_mult):
    hh = ta.rolling_max(ctx.df["High"], n_entry)
    ll = ta.rolling_min(ctx.df["Low"], n_exit)
    v = ctx.df["Volume"]
    va = v.rolling(vol_n, min_periods=vol_n).mean().shift(1)
    return _finish(ctx, latch((ctx.close > hh) & (v > vol_mult * va), ctx.close < ll))


def breakout_atr_trail(ctx, n_entry, atr_n, mult):
    """Breakout entry with an ATR trailing stop from the highest close since the fill.
    With n_entry=10, atr_n=14, mult=2 this is the original notebook strategy 5."""
    hh = ta.rolling_max(ctx.df["High"], n_entry)
    a = ctx.ind("atr", atr_n)
    e = (ctx.close > hh) & a.notna()
    return _finish(ctx, state_machine(ctx.df, e, None, stop="atr", stop_param=mult, atr=a, peak_field="Close",
                                      rearm_breakout=e))


def keltner(ctx, n_ema, n_atr, k):
    m, a = ctx.ind("ema", n_ema), ctx.ind("atr", n_atr)
    return _finish(ctx, latch(ctx.close > m + k * a, ctx.close < m))


# ============================================================================
# Family 6: oscillators
# ============================================================================

def rsi_cross(ctx, n, lo, hi, max_hold=60):
    r = ctx.ind("rsi", n)
    return _finish(ctx, state_machine(ctx.df, cross_up(r, lo), r > hi, max_hold=max_hold))


def rsi_regime(ctx, n, level):
    return _finish(ctx, (ctx.ind("rsi", n) > level).astype(float))


def stoch_cross(ctx, n, smooth, lo, hi, max_hold=40):
    k, d = ctx.ind("stochastic", n, smooth)
    e = (k > d) & (k.shift(1) <= d.shift(1)) & (k.shift(1) < lo)
    return _finish(ctx, state_machine(ctx.df, e, k > hi, max_hold=max_hold))


def macd_signal(ctx, fast, slow, signal):
    line, sig, _ = ta.macd(ctx.close, fast, slow, signal)
    return _finish(ctx, (line > sig).astype(float))


def macd_positive(ctx, fast, slow, signal):
    line, sig, hist = ta.macd(ctx.close, fast, slow, signal)
    return _finish(ctx, latch((line > 0) & (hist > 0), line < 0))


def rsi_divergence(ctx, n, low_n, margin, exit_n, max_hold):
    r = ctx.ind("rsi", n)
    new_low = ctx.close <= ta.rolling_min(ctx.close, low_n)
    prior_rsi_low = ta.rolling_min(r, low_n)
    e = new_low & (r > prior_rsi_low + margin)
    return _finish(ctx, state_machine(ctx.df, e, ctx.close > ctx.ind("sma", exit_n), max_hold=max_hold))


def osc_combo(ctx, rsi_n, stoch_n):
    r = ctx.ind("rsi", rsi_n) > 50
    line, sig, _ = ta.macd(ctx.close)
    m = line > sig
    k, d = ctx.ind("stochastic", stoch_n, 3)
    s = k > d
    votes = r.astype(int) + m.astype(int) + s.astype(int)
    return _finish(ctx, latch(votes == 3, votes <= 1))


def williams_cross(ctx, n, lo, hi, max_hold=40):
    w = ctx.ind("williams_r", n)
    return _finish(ctx, state_machine(ctx.df, cross_up(w, lo), w > hi, max_hold=max_hold))


def cci_momentum(ctx, n, entry, exit):
    c = ctx.ind("cci", n)
    return _finish(ctx, latch(cross_up(c, entry), c < exit))


# ============================================================================
# Family 7: volatility and risk management
# ============================================================================

def trailing_stop_system(ctx, stop, param, reentry_n, peak_field="Close", atr_n=22, breakout_n=50):
    """Default-long trend participation with a trailing stop.

    Entry: close > SMA(reentry_n). After a stop-out, re-arm when the close drops
    below the SMA (then re-enter on a cross back above) or on a new
    ``breakout_n``-day closing high.
    """
    m = ctx.ind("sma", reentry_n)
    a = ctx.ind("atr", atr_n) if stop == "atr" else None
    rb = ctx.close > ta.rolling_max(ctx.close, breakout_n)
    e = (ctx.close > m) & m.notna()
    return _finish(ctx, state_machine(ctx.df, e, None, stop=stop, stop_param=param, atr=a,
                                      peak_field=peak_field, rearm_breakout=rb))


def adaptive_atr_stop(ctx, low_mult, high_mult, vol_n, rank_n, reentry_n, atr_n=14):
    v = ctx.ind("realized_vol", vol_n)
    pct = v.rolling(rank_n, min_periods=rank_n).rank(pct=True)
    mult = pd.Series(np.where(pct > 0.5, high_mult, low_mult), index=ctx.index).where(pct.notna())
    m = ctx.ind("sma", reentry_n)
    a = ctx.ind("atr", atr_n)
    rb = ctx.close > ta.rolling_max(ctx.close, 50)
    e = (ctx.close > m) & pct.notna()
    return _finish(ctx, state_machine(ctx.df, e, None, stop="atr", atr=a, stop_param_series=mult,
                                      rearm_breakout=rb))


def vol_target(ctx, target, n, estimator="boxcar"):
    if estimator == "ewma":
        v = ta.ewma_vol(ctx.close, 0.94, n)
    elif estimator == "downside":
        v = ctx.ind("downside_vol", n)
    else:
        v = ctx.ind("realized_vol", n)
    raw = (target / v).clip(upper=1.0).where(v.notna(), 0.0)
    return _finish(ctx, smooth_weights(raw))


def drawdown_exposure(ctx, n, dd1, dd2, w1, w2):
    hi = ctx.close.rolling(n, min_periods=n).max()
    dd = ctx.close / hi - 1
    w = pd.Series(np.where(dd > -dd1, 1.0, np.where(dd > -dd2, w1, w2)), index=ctx.index)
    return _finish(ctx, w.where(hi.notna(), 0.0))


def vol_regime_exposure(ctx, n, ref_n, high_weight):
    v = ctx.ind("realized_vol", n)
    med = v.rolling(ref_n, min_periods=ref_n).median()
    w = pd.Series(np.where(v < med, 1.0, high_weight), index=ctx.index).where(med.notna(), 0.0)
    return _finish(ctx, w)


def vol_spike_exit(ctx, short_n, long_n, exit_ratio, entry_ratio):
    vs, vl = ctx.ind("realized_vol", short_n), ctx.ind("realized_vol", long_n)
    ratio = vs / vl
    ready = ratio.notna()
    return _finish(ctx, latch((ratio < entry_ratio) & ready, ratio > exit_ratio))


# ============================================================================
# Family 8: regime detection
# ============================================================================

def market_sma_regime(ctx, n):
    mk = ctx.aux("market")
    if mk is None:
        return _finish(ctx, pd.Series(0.0, index=ctx.index))
    return _finish(ctx, (mk.df["Close"] > mk("sma", n)).astype(float))


def market_vol_regime(ctx, n, threshold):
    mk = ctx.aux("market")
    if mk is None:
        return _finish(ctx, pd.Series(0.0, index=ctx.index))
    v = mk("realized_vol", n)
    return _finish(ctx, ((v < threshold) & v.notna()).astype(float))


def adx_trend(ctx, n, threshold):
    a, p, m = ctx.ind("adx", n)
    return _finish(ctx, latch((a > threshold) & (p > m), p < m))


def momentum_scaled(ctx, n, scale):
    raw = (ctx.ind("roc", n) / scale).clip(0, 1)
    return _finish(ctx, smooth_weights(raw.where(raw.notna(), 0.0), min_change=0.25, step=0.25))


def dual_momentum(ctx, n):
    """Antonacci-style: asset if it beats both T-bills and the defensive bond fund over n bars,
    else the bond fund if it beats T-bills, else cash. Evaluated monthly."""
    ra = ctx.ind("roc", n)
    cc = ctx.cash_cum(n)
    d = ctx.aux("defensive")
    rd = d("roc", n) if d is not None else pd.Series(np.nan, index=ctx.index)
    rd_cmp = rd.fillna(-np.inf)
    asset = (ra > cc) & (ra > rd_cmp)
    alt = (~asset) & (rd > cc)
    clock = first_bar_of_month(ctx.index)
    wa = hold_between(asset.astype(float).where(ra.notna(), 0.0), clock)
    wd = hold_between(alt.astype(float).where(rd.notna(), 0.0), clock)
    return _with_alt(ctx, wa, wd, "defensive")


def defensive_switch(ctx, n):
    m = ctx.ind("sma", n)
    on = (ctx.close > m) & m.notna()
    off = (~on) & m.notna()
    return _with_alt(ctx, on.astype(float), off.astype(float), "defensive")


def variance_ratio_switch(ctx, q, n, fast, slow, rsi_n, rsi_entry, rsi_exit):
    vr = ta.variance_ratio(ctx.close, q, n)
    trend = (ctx.ind("sma", fast) > ctx.ind("sma", slow)).astype(float)
    mr = state_machine(ctx.df, ctx.ind("rsi", rsi_n) < rsi_entry, ctx.ind("rsi", rsi_n) > rsi_exit, max_hold=10)
    w = pd.Series(np.where(vr > 1.0, trend, mr), index=ctx.index).where(vr.notna(), 0.0)
    return _finish(ctx, w)


def risk_on_ratio(ctx, n):
    a, b = ctx.aux("risk_on_num"), ctx.aux("risk_on_den")
    if a is None or b is None:
        return _finish(ctx, pd.Series(0.0, index=ctx.index))
    ratio = a.df["Close"] / b.df["Close"]
    return _finish(ctx, (ratio > ta.sma(ratio, n)).astype(float))


def rate_regime(ctx, n, change, reduced_weight):
    y = ctx.md.cash_yield
    d = y - y.shift(n)
    w = pd.Series(np.where(d > change, reduced_weight, 1.0), index=ctx.index).where(d.notna(), 0.0)
    return _finish(ctx, w)


def breadth_regime(ctx, n, threshold):
    names = [t for t in ctx.aux_names.get("breadth_list", []) if t in ctx.md.bars]
    if not names:
        return _finish(ctx, pd.Series(0.0, index=ctx.index))
    above = pd.concat([(ctx.aux(t).df["Close"] > ctx.aux(t)("sma", n)).astype(float)
                       .where(ctx.aux(t)("sma", n).notna()) for t in names], axis=1)
    frac = above.mean(axis=1)
    valid = above.notna().sum(axis=1) >= max(3, len(names) // 2)
    return _finish(ctx, ((frac >= threshold) & valid).astype(float))


# ============================================================================
# Family 9: relative strength and rotation (universe strategies)
# ============================================================================

def rotation_topk(uctx: UniverseContext, score="mom", n=126, skip=0, k=3, abs_filter=False,
                  vol_filter=False, weighting="equal", reverse=False, cap=1.0):
    px = uctx.close_panel()
    if score == "mom":
        sc = px.shift(skip) / px.shift(n) - 1
    elif score == "riskadj":
        r = px.pct_change()
        sc = (px / px.shift(n) - 1) / (r.rolling(n, min_periods=n).std() * np.sqrt(252))
    else:
        raise ValueError(score)
    if reverse:
        sc = -sc
    vol63 = px.pct_change().rolling(63, min_periods=63).std()
    if vol_filter:
        thr = vol63.quantile(2 / 3, axis=1)
        sc = sc.where(vol63.le(thr, axis=0))
    clock = first_bar_of_month(uctx.index)
    cc = uctx.cash_cum(252)
    r12 = px / px.shift(252) - 1
    W = np.zeros(px.shape)
    scv = sc.to_numpy()
    vv = vol63.to_numpy()
    r12v = r12.to_numpy()
    ccv = cc.to_numpy()
    cur = np.zeros(px.shape[1])
    for i in range(len(px)):
        if clock[i]:
            row = scv[i]
            ok = ~np.isnan(row)
            cur = np.zeros(px.shape[1])
            if ok.sum() >= max(k, 2):
                order = np.argsort(-np.where(ok, row, -np.inf))[:k]
                picks = [j for j in order if ok[j]]
                if abs_filter:
                    picks_kept = [j for j in picks if not np.isnan(r12v[i, j]) and r12v[i, j] > ccv[i]]
                else:
                    picks_kept = picks
                if picks_kept:
                    if weighting == "invvol":
                        iv = np.array([1 / vv[i, j] if vv[i, j] > 0 else 0 for j in picks_kept])
                        ww = iv / iv.sum() if iv.sum() > 0 else np.ones(len(picks_kept)) / len(picks_kept)
                    else:
                        ww = np.ones(len(picks_kept)) / len(picks)
                    for j, x in zip(picks_kept, ww):
                        cur[j] = min(x, cap)
        W[i] = cur
    return pd.DataFrame(W, index=px.index, columns=px.columns)


def rs_vs_benchmark(uctx: UniverseContext, n=100, cap=0.25):
    bm = uctx.aux_names.get("market", "SPY")
    if bm not in uctx.md.bars:
        raise ValueError("benchmark missing")
    b = uctx.md.bars[bm]["Close"]
    px = uctx.close_panel()
    ratio = px.div(b, axis=0)
    strong = ratio > ratio.rolling(n, min_periods=n).mean()
    cnt = strong.sum(axis=1)
    w = strong.astype(float).div(cnt.replace(0, np.nan), axis=0).fillna(0.0).clip(upper=cap)
    return hold_between(w, first_bar_of_month(uctx.index))


def leadership_switch(ctx, n):
    """Hold the asset while its price ratio to the market is above its n-day average, else the market."""
    mk = ctx.aux_ticker("market")
    if mk == ctx.ticker or mk not in ctx.md.bars:
        raise NotApplicable("asset is the market proxy")
    ratio = ctx.close / ctx.md.bars[mk]["Close"]
    lead = (ratio > ta.sma(ratio, n))
    clock = first_bar_of_month(ctx.index)
    wa = hold_between(lead.astype(float).where(ta.sma(ratio, n).notna(), 0.0), clock)
    wm = hold_between((~lead).astype(float).where(ta.sma(ratio, n).notna(), 0.0), clock)
    return _with_alt(ctx, wa, wm, "market")


class NotApplicable(Exception):
    """Raised when a strategy is undefined for an asset (e.g. relative strength of SPY vs SPY)."""


# ============================================================================
# Family 10: hybrids
# ============================================================================

def trend_plus_mr(ctx, trend_n, rsi_n, entry, exit):
    up = (ctx.close > ctx.ind("sma", trend_n)) & ctx.ind("sma", trend_n).notna()
    r = ctx.ind("rsi", rsi_n)
    mr = state_machine(ctx.df, (r < entry) & ~up & ctx.ind("sma", trend_n).notna(), r > exit, max_hold=10)
    return _finish(ctx, np.maximum(up.astype(float), mr))


def momentum_atr_stop(ctx, n, atr_n, mult):
    m = ctx.ind("roc", n)
    a = ctx.ind("atr", atr_n)
    rb = ctx.close > ta.rolling_max(ctx.close, 50)
    return _finish(ctx, state_machine(ctx.df, (m > 0) & a.notna(), m <= 0, stop="atr", stop_param=mult, atr=a,
                                      rearm_breakout=rb))


def breakout_voltarget(ctx, n_entry, n_exit, target, vol_n):
    d = donchian(ctx, n_entry, n_exit)
    v = ctx.ind("realized_vol", vol_n)
    scale = (target / v).clip(upper=1.0).where(v.notna(), 0.0)
    return _finish(ctx, smooth_weights(d * scale))


def trend_two_stage(ctx, n_trig, n_mid, n_slow):
    f, m, s = ctx.ind("sma", n_trig), ctx.ind("sma", n_mid), ctx.ind("sma", n_slow)
    w = 0.5 * (m > s).astype(float) + 0.5 * ((ctx.close > f) & (f > m)).astype(float)
    return _finish(ctx, w.where(s.notna(), 0.0))


def kama_trend(ctx, n, fast, slow, slope_n):
    k = ta.kama(ctx.close, n, fast, slow)
    return _finish(ctx, latch((ctx.close > k) & (k > k.shift(slope_n)), ctx.close < k))


def risk_on_off(ctx, trend_n, vol_n, rank_n):
    m = ctx.ind("sma", trend_n)
    v = ctx.ind("realized_vol", vol_n)
    pct = v.rolling(rank_n, min_periods=rank_n).rank(pct=True)
    a = (ctx.close > m).astype(float) + (pct < 0.8).astype(float)
    ready = m.notna() & pct.notna()
    w_asset = (a / 2).where(ready, 0.0)
    w_def = (1 - a / 2).where(ready, 0.0)
    return _with_alt(ctx, w_asset, w_def, "defensive")


def ensemble_vote(ctx):
    c = ctx.close
    s1 = (ctx.ind("sma", 50) > ctx.ind("sma", 200)).astype(float).where(ctx.ind("sma", 200).notna())
    s2 = (ta.roc_skip(c, 252, 21) > 0).astype(float).where(ta.roc_skip(c, 252, 21).notna())
    s3 = donchian(ctx, 55, 20).where(ta.rolling_max(ctx.df["High"], 55).notna())
    line, sig, _ = ta.macd(c)
    s4 = (line > sig).astype(float).where(sig.notna())
    v = ctx.ind("realized_vol", 20)
    med = v.rolling(252, min_periods=252).median()
    s5 = (v < med).astype(float).where(med.notna())
    votes = pd.concat([s1, s2, s3, s4, s5], axis=1)
    w = votes.mean(axis=1).where(votes.notna().all(axis=1), 0.0)
    return _finish(ctx, w)


def tsmom_voltarget(ctx, n, target, vol_n):
    on = (ctx.ind("roc", n) > 0).astype(float)
    v = ctx.ind("realized_vol", vol_n)
    scale = (target / v).clip(upper=1.0).where(v.notna(), 0.0)
    return _finish(ctx, smooth_weights(on * scale))


def mr_in_momentum(ctx, mom_n, rsi_n, entry, exit, stop_pct, max_hold):
    mom = ctx.ind("roc", mom_n) > 0
    r = ctx.ind("rsi", rsi_n)
    return _finish(ctx, state_machine(ctx.df, mom & (r < entry), r > exit, stop="pct", stop_param=stop_pct,
                                      max_hold=max_hold))


def sma_cross_regime_scaled(ctx, fast, slow, regime_n, off_weight):
    base = (ctx.ind("sma", fast) > ctx.ind("sma", slow)).astype(float)
    reg = pd.Series(np.where(ctx.close > ctx.ind("sma", regime_n), 1.0, off_weight), index=ctx.index)
    return _finish(ctx, (base * reg).where(ctx.ind("sma", regime_n).notna(), 0.0))


# ============================================================================
# Registry
# ============================================================================

@dataclass(frozen=True)
class StrategySpec:
    id: str
    family: str
    name: str
    hypothesis: str
    entry: str
    exit: str
    params: dict
    allocation: str
    min_lookback: int
    benchmark: str
    regime: str
    func: str
    kind: str = "single"
    grid: dict = field(default_factory=dict)
    requires: tuple = ()
    origin: str = "new"

    def variants(self, max_variants: int = 6) -> list[dict]:
        """Default params first, then one-at-a-time neighbourhood perturbations."""
        out = [dict(self.params)]
        for k, vals in self.grid.items():
            for v in vals:
                if v == self.params.get(k):
                    continue
                p = dict(self.params)
                p[k] = v
                if p not in out:
                    out.append(p)
        return out[:max_variants]


FAMILIES = {
    "F01": "Moving-average trend following",
    "F02": "Asymmetric entry and exit",
    "F03": "Momentum",
    "F04": "Mean reversion",
    "F05": "Breakouts",
    "F06": "Oscillators",
    "F07": "Volatility and risk management",
    "F08": "Regime detection",
    "F09": "Relative strength and sector rotation",
    "F10": "Hybrid strategies",
}

BH = "Buy-and-hold same asset; secondary: universe benchmark (SPY, or SMH for semiconductor stocks)"
EWU = "Equal-weight universe (monthly rebalanced) and SPY"


def _S(**kw) -> StrategySpec:
    kw.setdefault("benchmark", BH)
    kw["requires"] = tuple(kw.get("requires", ()))
    return StrategySpec(**kw)


REGISTRY: list[StrategySpec] = [
    # ---------------- F01 Moving-average trend following ----------------
    _S(id="F01_01", family="F01", name="SMA 20/50 crossover (original)", origin="original_notebook",
       hypothesis="Intermediate-term trends persist because information diffuses gradually; a 20/50 crossover captures multi-month trends while stepping aside in sustained declines.",
       entry="Close-of-day SMA20 > SMA50", exit="SMA20 <= SMA50", params=dict(fast=20, slow=50, ma="sma"),
       allocation="100% asset when long, else cash", min_lookback=50, regime="Persistent multi-month trends",
       func="ma_cross", grid=dict(fast=[15, 25], slow=[40, 60])),
    _S(id="F01_02", family="F01", name="SMA 10/30 crossover (original)", origin="original_notebook",
       hypothesis="A faster crossover reacts sooner to trend changes, trading lower lag for more whipsaws; tests whether earlier exits improve downside protection.",
       entry="SMA10 > SMA30", exit="SMA10 <= SMA30", params=dict(fast=10, slow=30, ma="sma"),
       allocation="100% asset when long, else cash", min_lookback=30, regime="Trends with sharp reversals",
       func="ma_cross", grid=dict(fast=[8, 12], slow=[25, 35])),
    _S(id="F01_03", family="F01", name="SMA 50/200 golden cross",
       hypothesis="Long-horizon trend (golden/death cross) identifies primary bull and bear markets; few trades, so cost drag is minimal but lag is large.",
       entry="SMA50 > SMA200", exit="SMA50 <= SMA200", params=dict(fast=50, slow=200, ma="sma"),
       allocation="100% asset when long, else cash", min_lookback=200, regime="Long bull/bear cycles",
       func="ma_cross", grid=dict(fast=[40, 60], slow=[150, 250])),
    _S(id="F01_04", family="F01", name="EMA 12/26 crossover",
       hypothesis="Exponential weighting reduces lag versus equal-weighted SMAs at similar horizons, potentially improving entry/exit timing.",
       entry="EMA12 > EMA26", exit="EMA12 <= EMA26", params=dict(fast=12, slow=26, ma="ema"),
       allocation="100% asset when long, else cash", min_lookback=26, regime="Trending, moderate noise",
       func="ma_cross", grid=dict(fast=[9, 15], slow=[21, 32])),
    _S(id="F01_05", family="F01", name="Close above SMA200",
       hypothesis="Price relative to its 200-day average is a widely watched trend filter; staying out below it avoids the bulk of bear-market losses.",
       entry="Close > SMA200", exit="Close <= SMA200", params=dict(n=200, ma="sma"),
       allocation="100% asset when long, else cash", min_lookback=200, regime="Long bear markets",
       func="price_above_ma", grid=dict(n=[150, 250])),
    _S(id="F01_06", family="F01", name="Close above SMA100",
       hypothesis="A medium-term (≈5 month) price filter balances lag and whipsaw better than either 50- or 200-day filters.",
       entry="Close > SMA100", exit="Close <= SMA100", params=dict(n=100, ma="sma"),
       allocation="100% asset when long, else cash", min_lookback=100, regime="Medium-term trends",
       func="price_above_ma", grid=dict(n=[80, 120])),
    _S(id="F01_07", family="F01", name="Triple MA stack 10>50>200",
       hypothesis="Requiring short, medium and long trends to align filters out counter-trend rallies; only fully aligned trends are traded.",
       entry="SMA10 > SMA50 > SMA200", exit="Alignment breaks", params=dict(n1=10, n2=50, n3=200),
       allocation="100% asset when aligned, else cash", min_lookback=200, regime="Strong aligned uptrends",
       func="ma_stack", grid=dict(n1=[5, 20], n2=[40, 60])),
    _S(id="F01_08", family="F01", name="Multi-MA vote (20/50/100/200)",
       hypothesis="Averaging trend signals across horizons diversifies parameter risk; exposure scales with the breadth of trend confirmation.",
       entry="Weight = share of {SMA20,50,100,200} that Close exceeds", exit="Weight falls as filters fail",
       params=dict(windows=(20, 50, 100, 200)), allocation="0/25/50/75/100% asset by vote, rest cash",
       min_lookback=200, regime="Any trending regime", func="ma_vote",
       grid=dict(windows=[(10, 30, 60, 120), (30, 75, 150, 250)])),
    _S(id="F01_09", family="F01", name="SMA200 slope",
       hypothesis="The direction of the long-term average (rising vs falling) is a slower but less noisy trend measure than price crossing it.",
       entry="SMA200 today > SMA200 20 days ago", exit="SMA200 not rising over 20 days", params=dict(n=200, lookback=20),
       allocation="100% asset when slope positive, else cash", min_lookback=220, regime="Long trends",
       func="ma_slope", grid=dict(n=[150, 250], lookback=[10, 40])),
    _S(id="F01_10", family="F01", name="Monthly 10-month SMA (Faber)",
       hypothesis="Checking a long-term trend filter only monthly cuts whipsaw trades and costs relative to daily evaluation of the same rule.",
       entry="On first trading day of month: Close > SMA210", exit="On first trading day of month: Close <= SMA210",
       params=dict(n=210), allocation="100% asset or cash, held for the month", min_lookback=210,
       regime="Long bull/bear cycles", func="monthly_price_above_ma", grid=dict(n=[168, 252])),

    # ---------------- F02 Asymmetric entry/exit ----------------
    _S(id="F02_01", family="F02", name="Asymmetric price/SMA: exit<SMA20, enter>SMA10 (original)", origin="original_notebook",
       hypothesis="Exiting on a medium filter but re-entering on a faster one aims to cut losses while catching rebounds early (original notebook).",
       entry="Flat and Close > SMA10", exit="Long and Close < SMA20", params=dict(exit_n=20, entry_n=10),
       allocation="100% asset or cash", min_lookback=20, regime="Choppy markets with quick rebounds",
       func="asym_price_ma", grid=dict(exit_n=[15, 30], entry_n=[5, 15])),
    _S(id="F02_02", family="F02", name="Fast exit (<SMA20), slow re-entry (>SMA50)",
       hypothesis="Losses compound faster than gains; exiting quickly but demanding stronger evidence to re-enter limits drawdowns at the cost of missed rebounds (H3).",
       entry="Flat and Close > SMA50", exit="Long and Close < SMA20", params=dict(exit_n=20, entry_n=50),
       allocation="100% asset or cash", min_lookback=50, regime="Prolonged declines",
       func="asym_price_ma", grid=dict(exit_n=[10, 30], entry_n=[40, 100])),
    _S(id="F02_03", family="F02", name="Slow permission (SMA20>50), fast exit (<SMA10)",
       hypothesis="Only participate when the intermediate trend is up, but step aside on the first short-term weakness.",
       entry="SMA20 > SMA50 and Close > SMA10", exit="Close < SMA10", params=dict(trig_n=10, slow_fast=20, slow_slow=50),
       allocation="100% asset or cash", min_lookback=50, regime="Smooth uptrends",
       func="permission_fast_exit", grid=dict(trig_n=[5, 20], slow_slow=[40, 100])),
    _S(id="F02_04", family="F02", name="Hysteresis band ±2% around SMA100",
       hypothesis="A neutral zone around the trend line suppresses whipsaws caused by noise oscillating around the average.",
       entry="Close > 1.02 x SMA100", exit="Close < 0.98 x SMA100", params=dict(n=100, band=0.02),
       allocation="100% asset or cash", min_lookback=100, regime="Trends with noisy crossings",
       func="hysteresis_band", grid=dict(n=[50, 200], band=[0.01, 0.04])),
    _S(id="F02_05", family="F02", name="ATR hysteresis ±1 ATR around SMA50",
       hypothesis="Scaling the neutral band by recent volatility adapts whipsaw protection to each asset's noise level (important for volatile semiconductors).",
       entry="Close > SMA50 + 1 x ATR20", exit="Close < SMA50 - 1 x ATR20", params=dict(n=50, atr_n=20, k=1.0),
       allocation="100% asset or cash", min_lookback=50, regime="Volatile trends",
       func="atr_hysteresis", grid=dict(k=[0.5, 1.5], n=[30, 100])),
    _S(id="F02_06", family="F02", name="Time-confirmed re-entry (3 closes > SMA50)",
       hypothesis="Requiring several consecutive closes above the trend line before re-entering avoids false recoveries; exits remain immediate.",
       entry="3 consecutive closes > SMA50", exit="First close < SMA50", params=dict(n=50, confirm=3),
       allocation="100% asset or cash", min_lookback=50, regime="False-breakout prone markets",
       func="confirm_count", grid=dict(confirm=[2, 5], n=[30, 100])),
    _S(id="F02_07", family="F02", name="Two-stage 0/50/100 SMA10/20 (original)", origin="original_notebook",
       hypothesis="Partial exposure in ambiguous zones reduces the cost of being wrong in either direction (original notebook).",
       entry="100% when Close > both SMA10 and SMA20; 50% when between", exit="0% when Close < both",
       params=dict(n1=10, n2=20), allocation="0/50/100% asset, rest cash", min_lookback=20,
       regime="Mixed/choppy", func="two_stage_short", grid=dict(n1=[5, 15], n2=[30, 50])),
    _S(id="F02_08", family="F02", name="Two-stage long horizon (Close>SMA50, SMA50>SMA200)",
       hypothesis="Scaling exposure in two steps on medium and long trend signals smooths transitions between bull and bear regimes.",
       entry="+50% if Close > SMA50; +50% if SMA50 > SMA200", exit="Each half removed when its condition fails",
       params=dict(n_fast=50, n_slow=200), allocation="0/50/100% asset, rest cash", min_lookback=200,
       regime="Regime transitions", func="two_stage_long", grid=dict(n_fast=[30, 100], n_slow=[150, 250])),
    _S(id="F02_09", family="F02", name="Crash detector: exit on 2.5σ down day, re-enter >SMA20",
       hypothesis="Volatility clusters, so an extreme down day signals elevated near-term risk; exit immediately and re-enter only after recovery above SMA20.",
       entry="Close > SMA20 at least 5 bars after an exit", exit="Daily return < -2.5 x prior 60-day std",
       params=dict(vol_n=60, k=2.5, reentry_n=20, min_days=5), allocation="100% asset or cash", min_lookback=61,
       regime="Crash-prone, volatility-clustering markets", func="shock_exit",
       grid=dict(k=[2.0, 3.0], reentry_n=[10, 50])),
    _S(id="F02_10", family="F02", name="Exit on SMA10/30 cross, re-enter on SMA20/50 + SMA10/30",
       hypothesis="Direct test of H3 with the original systems: the faster 10/30 crossover exits, re-entry also requires the slower 20/50 confirmation.",
       entry="SMA20 > SMA50 and SMA10 > SMA30", exit="SMA10 < SMA30", params=dict(exit_fast=10, exit_slow=30, entry_fast=20, entry_slow=50),
       allocation="100% asset or cash", min_lookback=50, regime="Sharp declines followed by gradual recoveries",
       func="cross_asym", grid=dict(entry_slow=[40, 60], exit_slow=[25, 35])),

    # ---------------- F03 Momentum ----------------
    _S(id="F03_01", family="F03", name="3-month absolute momentum",
       hypothesis="Returns over the past quarter positively predict the next month's returns (time-series momentum at short-intermediate horizon).",
       entry="63-day return > 0", exit="63-day return <= 0", params=dict(n=63),
       allocation="100% asset or cash", min_lookback=63, regime="Persistent trends", func="mom_abs", grid=dict(n=[42, 84])),
    _S(id="F03_02", family="F03", name="6-month absolute momentum",
       hypothesis="Six-month price persistence reflects slow-moving fundamentals and investor under-reaction.",
       entry="126-day return > 0", exit="126-day return <= 0", params=dict(n=126),
       allocation="100% asset or cash", min_lookback=126, regime="Persistent trends", func="mom_abs", grid=dict(n=[105, 147])),
    _S(id="F03_03", family="F03", name="12-1 month momentum",
       hypothesis="Skipping the most recent month avoids short-term reversal contamination of the classic 12-month momentum signal.",
       entry="Return from t-252 to t-21 > 0", exit="That return <= 0", params=dict(n=252, skip=21),
       allocation="100% asset or cash", min_lookback=252, regime="Long trends", func="mom_abs", grid=dict(n=[210, 294], skip=[10, 42])),
    _S(id="F03_04", family="F03", name="Time-series momentum vs T-bills",
       hypothesis="Holding risk only when its trailing 12-month return beats cash (absolute momentum) avoids periods with negative risk premia.",
       entry="252-day return > cumulative 252-day T-bill return", exit="Otherwise", params=dict(n=252),
       allocation="100% asset or cash", min_lookback=252, regime="Bear markets with positive rates", func="tsmom_excess",
       grid=dict(n=[189, 315]), requires=("CASH",)),
    _S(id="F03_05", family="F03", name="Momentum acceleration",
       hypothesis="Improving momentum (current 3-month return above the prior 3-month return) signals strengthening trends with higher continuation probability.",
       entry="63-day return > 0 and > its value 63 days ago", exit="Either condition fails", params=dict(n=63),
       allocation="100% asset or cash", min_lookback=126, regime="Accelerating trends", func="mom_accel", grid=dict(n=[42, 84])),
    _S(id="F03_06", family="F03", name="Multi-horizon confirmation (3/6/12m all positive)",
       hypothesis="Requiring agreement across horizons filters out noisy single-horizon signals at the cost of later entries.",
       entry="63-, 126- and 252-day returns all > 0", exit="Any of them <= 0", params=dict(horizons=(63, 126, 252)),
       allocation="100% asset or cash", min_lookback=252, regime="Broad, persistent trends", func="mom_all",
       grid=dict(horizons=[(42, 126, 252), (63, 189, 252)])),
    _S(id="F03_07", family="F03", name="Graded multi-horizon momentum (1/3/6/12m)",
       hypothesis="Exposure proportional to the number of positive horizons diversifies horizon risk instead of betting on one lookback.",
       entry="Weight = share of {21,63,126,252}-day returns > 0", exit="Weight declines as horizons turn negative",
       params=dict(horizons=(21, 63, 126, 252)), allocation="0-100% in 25% steps, rest cash", min_lookback=252,
       regime="Any trending regime", func="mom_vote", grid=dict(horizons=[(10, 42, 126, 252), (21, 63, 189, 315)])),
    _S(id="F03_08", family="F03", name="Risk-adjusted momentum (t-stat > 1)",
       hypothesis="Trends that are statistically strong relative to noise (high t-statistic) are more likely to persist than raw positive returns.",
       entry="126-day mean daily return / std x sqrt(126) > 1", exit="t-stat < 0", params=dict(n=126, threshold=1.0),
       allocation="100% asset or cash", min_lookback=126, regime="Smooth trends", func="mom_tstat",
       grid=dict(threshold=[0.5, 1.5], n=[84, 189])),
    _S(id="F03_09", family="F03", name="1-month momentum",
       hypothesis="Very short-horizon momentum tests whether one-month persistence exists or whether short-term reversal dominates.",
       entry="21-day return > 0", exit="21-day return <= 0", params=dict(n=21),
       allocation="100% asset or cash", min_lookback=21, regime="Short-lived trends", func="mom_abs", grid=dict(n=[10, 30])),
    _S(id="F03_10", family="F03", name="5-day momentum + SMA100 regime (original)", origin="original_notebook",
       hypothesis="Short-term momentum aligned with a rising medium-term regime improves re-entry timing (original notebook strategy 6).",
       entry="Close > SMA10, 5-day return > 0 and SMA100 rising over 5 days", exit="Close < SMA20 and SMA100 falling over 5 days",
       params=dict(mom_n=5, slope_n=5, exit_n=20, entry_n=10, regime_n=100), allocation="100% asset or cash",
       min_lookback=105, regime="Medium-term trends", func="momentum_sma100_regime",
       grid=dict(regime_n=[50, 200], slope_n=[10, 20])),

    # ---------------- F04 Mean reversion ----------------
    _S(id="F04_01", family="F04", name="RSI(2) < 10 pullback",
       hypothesis="Short-term overreaction: extreme 2-day oversold readings in liquid assets tend to revert within days.",
       entry="RSI2 < 10", exit="RSI2 > 70 or 10 bars", params=dict(n=2, entry=10, exit=70, trend_n=None, max_hold=10),
       allocation="100% asset or cash", min_lookback=10, regime="Sideways / mean-reverting", func="rsi_reversion",
       grid=dict(entry=[5, 20], exit=[60, 80])),
    _S(id="F04_02", family="F04", name="RSI(2) < 10 above SMA200",
       hypothesis="Pullbacks inside long-term uptrends are buying opportunities; pullbacks in downtrends are not (trend-conditioned reversion).",
       entry="RSI2 < 10 and Close > SMA200", exit="RSI2 > 70 or 10 bars", params=dict(n=2, entry=10, exit=70, trend_n=200, max_hold=10),
       allocation="100% asset or cash", min_lookback=200, regime="Bull markets with pullbacks", func="rsi_reversion",
       grid=dict(entry=[5, 20], trend_n=[100, 250])),
    _S(id="F04_03", family="F04", name="Bollinger lower-band reversion",
       hypothesis="Closes beyond two standard deviations below the 20-day mean are statistically stretched and tend to revert to the mean.",
       entry="Close < SMA20 - 2 std", exit="Close > SMA20 or 20 bars", params=dict(n=20, k=2.0, trend_n=None, max_hold=20),
       allocation="100% asset or cash", min_lookback=20, regime="Range-bound", func="bollinger_reversion",
       grid=dict(k=[1.5, 2.5], n=[10, 30])),
    _S(id="F04_04", family="F04", name="Bollinger reversion with SMA200 trend filter",
       hypothesis="Band reversion works better when the long-term trend provides a tailwind.",
       entry="Close < lower band and Close > SMA200", exit="Close > SMA20 or 20 bars", params=dict(n=20, k=2.0, trend_n=200, max_hold=20),
       allocation="100% asset or cash", min_lookback=200, regime="Bull markets with dips", func="bollinger_reversion",
       grid=dict(k=[1.5, 2.5], trend_n=[100, 250])),
    _S(id="F04_05", family="F04", name="5% below SMA50 reversion",
       hypothesis="Fixed percentage deviations from the medium-term mean revert; deliberately not volatility-scaled to contrast with F04_06.",
       entry="Close < 0.95 x SMA50", exit="Close > SMA50 or 30 bars", params=dict(n=50, dev=0.05, max_hold=30),
       allocation="100% asset or cash", min_lookback=50, regime="Range-bound, low volatility", func="pct_deviation_reversion",
       grid=dict(dev=[0.03, 0.08], n=[20, 100])),
    _S(id="F04_06", family="F04", name="Volatility-normalised pullback (1.5 ATR below SMA10)",
       hypothesis="Measuring pullbacks in ATR units makes the oversold threshold comparable across assets of different volatility.",
       entry="(Close - SMA10)/ATR10 < -1.5", exit="Close > SMA10 or 15 bars", params=dict(n=10, k=1.5, max_hold=15),
       allocation="100% asset or cash", min_lookback=11, regime="Mean-reverting", func="atr_pullback",
       grid=dict(k=[1.0, 2.0], n=[5, 20])),
    _S(id="F04_07", family="F04", name="Reversal after 2σ daily loss (5-day hold)",
       hypothesis="Liquidity-driven large daily losses are partly reversed over the following week as liquidity providers are compensated.",
       entry="Daily return < -2 x prior 60-day std", exit="After 5 bars", params=dict(vol_n=60, k=2.0, hold=5),
       allocation="100% asset or cash", min_lookback=61, regime="High-volatility selloffs", func="shock_reversal",
       grid=dict(k=[1.5, 2.5], hold=[3, 10])),
    _S(id="F04_08", family="F04", name="Three consecutive down closes",
       hypothesis="Runs of down days reflect short-term selling pressure that tends to exhaust (behavioural overreaction).",
       entry="3 consecutive lower closes", exit="First up close or 5 bars", params=dict(k=3, max_hold=5),
       allocation="100% asset or cash", min_lookback=4, regime="Choppy markets", func="down_days",
       grid=dict(k=[2, 4], max_hold=[3, 10])),
    _S(id="F04_09", family="F04", name="Internal bar strength (IBS < 0.2)",
       hypothesis="Closing near the daily low reflects intraday liquidity pressure that reverses the next day(s).",
       entry="(Close-Low)/(High-Low) < 0.2", exit="IBS > 0.8 or 5 bars", params=dict(lo=0.2, hi=0.8, max_hold=5),
       allocation="100% asset or cash", min_lookback=1, regime="Liquid index ETFs, choppy", func="ibs_reversion",
       grid=dict(lo=[0.1, 0.3], hi=[0.7, 0.9])),
    _S(id="F04_10", family="F04", name="Buy 15% drawdown, sell near prior high",
       hypothesis="Long-horizon mean reversion: large drawdowns from 52-week highs eventually recover (buy-the-dip at the intermediate horizon).",
       entry="Close more than 15% below 252-day high", exit="Within 5% of 252-day high", params=dict(n=252, dd_entry=0.15, dd_exit=0.05),
       allocation="100% asset or cash", min_lookback=252, regime="Corrections in secular uptrends", func="drawdown_buy",
       grid=dict(dd_entry=[0.10, 0.20], dd_exit=[0.02, 0.10])),

    # ---------------- F05 Breakouts ----------------
    _S(id="F05_01", family="F05", name="Donchian 20/10 (Turtle S1)",
       hypothesis="Prices escaping a 20-day range reveal new information or demand imbalances that persist.",
       entry="Close > prior 20-day high", exit="Close < prior 10-day low", params=dict(n_entry=20, n_exit=10, trend_n=None),
       allocation="100% asset or cash", min_lookback=21, regime="Trending breakouts", func="donchian",
       grid=dict(n_entry=[15, 30], n_exit=[5, 15])),
    _S(id="F05_02", family="F05", name="Donchian 55/20 (Turtle S2)",
       hypothesis="Longer channel breakouts filter noise and capture larger trends with fewer trades.",
       entry="Close > prior 55-day high", exit="Close < prior 20-day low", params=dict(n_entry=55, n_exit=20, trend_n=None),
       allocation="100% asset or cash", min_lookback=56, regime="Large trends", func="donchian",
       grid=dict(n_entry=[40, 70], n_exit=[15, 30])),
    _S(id="F05_03", family="F05", name="52-week closing high breakout",
       hypothesis="Investors anchor to the 52-week high and under-react when it is broken (George-Hwang 52-week-high effect).",
       entry="Close >= prior 252-day closing high", exit="Close < prior 50-day closing low", params=dict(n=252, exit_n=50),
       allocation="100% asset or cash", min_lookback=253, regime="New highs in bull markets", func="high_52w",
       grid=dict(exit_n=[30, 100], n=[200, 300])),
    _S(id="F05_04", family="F05", name="50-day breakout, 25-day exit",
       hypothesis="Intermediate breakouts balance responsiveness and false-breakout risk.",
       entry="Close > prior 50-day high", exit="Close < prior 25-day low", params=dict(n_entry=50, n_exit=25, trend_n=None),
       allocation="100% asset or cash", min_lookback=51, regime="Intermediate trends", func="donchian",
       grid=dict(n_entry=[40, 60], n_exit=[20, 35])),
    _S(id="F05_05", family="F05", name="Volatility-squeeze breakout",
       hypothesis="Volatility compression precedes expansion; breakouts after a squeeze have more follow-through.",
       entry="Bollinger(20,2) bandwidth near its 126-day low within 10 bars and Close > upper band", exit="Close < SMA20",
       params=dict(n=20, k=2.0, squeeze_lookback=126, recent=10), allocation="100% asset or cash", min_lookback=146,
       regime="Low-vol consolidation then trend", func="squeeze_breakout", grid=dict(squeeze_lookback=[63, 252], k=[1.5, 2.5])),
    _S(id="F05_06", family="F05", name="Volume-confirmed 20-day breakout",
       hypothesis="Breakouts on heavy volume reflect genuine demand and are less likely to fail.",
       entry="Close > prior 20-day high and volume > 1.5 x 50-day avg", exit="Close < prior 10-day low",
       params=dict(n_entry=20, n_exit=10, vol_n=50, vol_mult=1.5), allocation="100% asset or cash", min_lookback=51,
       regime="News-driven breakouts", func="volume_breakout", grid=dict(vol_mult=[1.2, 2.0], n_entry=[15, 30])),
    _S(id="F05_07", family="F05", name="20-day breakout with SMA200 trend filter",
       hypothesis="Breakouts in the direction of the primary trend have higher success rates.",
       entry="Close > prior 20-day high and Close > SMA200", exit="Close < prior 10-day low",
       params=dict(n_entry=20, n_exit=10, trend_n=200), allocation="100% asset or cash", min_lookback=200,
       regime="Bull markets", func="donchian", grid=dict(trend_n=[100, 250], n_entry=[15, 30])),
    _S(id="F05_08", family="F05", name="20-day breakout + 3 ATR trailing stop",
       hypothesis="Letting winners run with a volatility-scaled trailing stop captures more of a trend than a fixed channel exit.",
       entry="Close > prior 20-day high", exit="Close < highest close since entry - 3 x ATR14",
       params=dict(n_entry=20, atr_n=14, mult=3.0), allocation="100% asset or cash", min_lookback=21,
       regime="Extended trends", func="breakout_atr_trail", grid=dict(mult=[2.5, 4.0], n_entry=[15, 30])),
    _S(id="F05_09", family="F05", name="10-day breakout + 2 ATR stop (original)", origin="original_notebook",
       hypothesis="Original notebook strategy 5: re-enter on a short breakout, protect with a tight ATR trailing stop.",
       entry="Close > prior 10-day high", exit="Close < highest close since entry - 2 x ATR14",
       params=dict(n_entry=10, atr_n=14, mult=2.0), allocation="100% asset or cash", min_lookback=15,
       regime="Short trends", func="breakout_atr_trail", grid=dict(mult=[1.5, 3.0], n_entry=[5, 20])),
    _S(id="F05_10", family="F05", name="Keltner channel breakout",
       hypothesis="Closes beyond an ATR-scaled envelope around the EMA indicate volatility-adjusted momentum bursts.",
       entry="Close > EMA20 + 2 x ATR10", exit="Close < EMA20", params=dict(n_ema=20, n_atr=10, k=2.0),
       allocation="100% asset or cash", min_lookback=20, regime="Momentum bursts", func="keltner",
       grid=dict(k=[1.5, 2.5], n_ema=[10, 50])),

    # ---------------- F06 Oscillators ----------------
    _S(id="F06_01", family="F06", name="RSI(14) cross above 30, exit > 70",
       hypothesis="Classic oversold/overbought: recovery from oversold marks exhaustion of selling.",
       entry="RSI14 crosses above 30", exit="RSI14 > 70 or 60 bars", params=dict(n=14, lo=30, hi=70, max_hold=60),
       allocation="100% asset or cash", min_lookback=15, regime="Oscillating ranges", func="rsi_cross",
       grid=dict(lo=[25, 35], hi=[65, 75])),
    _S(id="F06_02", family="F06", name="RSI(14) > 50 momentum regime",
       hypothesis="RSI above 50 is a smoothed momentum measure; treating it as a trend filter tests momentum rather than reversal use of oscillators.",
       entry="RSI14 > 50", exit="RSI14 <= 50", params=dict(n=14, level=50),
       allocation="100% asset or cash", min_lookback=15, regime="Trends", func="rsi_regime", grid=dict(n=[10, 21], level=[45, 55])),
    _S(id="F06_03", family="F06", name="RSI(14) < 35 in uptrend (SMA200)",
       hypothesis="Oscillator oversold signals are more reliable when confirmed by the primary trend.",
       entry="RSI14 < 35 and Close > SMA200", exit="RSI14 > 55 or 10 bars", params=dict(n=14, entry=35, exit=55, trend_n=200, max_hold=10),
       allocation="100% asset or cash", min_lookback=200, regime="Bull markets with pullbacks", func="rsi_reversion",
       grid=dict(entry=[30, 40], exit=[50, 65])),
    _S(id="F06_04", family="F06", name="Stochastic %K/%D bullish cross below 20",
       hypothesis="A %K/%D cross from oversold indicates turning short-term momentum.",
       entry="%K(14,3) crosses above %D with prior %K < 20", exit="%K > 80 or 40 bars", params=dict(n=14, smooth=3, lo=20, hi=80, max_hold=40),
       allocation="100% asset or cash", min_lookback=20, regime="Ranges", func="stoch_cross", grid=dict(lo=[10, 30], hi=[70, 90])),
    _S(id="F06_05", family="F06", name="MACD line above signal",
       hypothesis="MACD crossovers capture changes in trend momentum (second derivative of price).",
       entry="MACD(12,26) > signal(9)", exit="MACD <= signal", params=dict(fast=12, slow=26, signal=9),
       allocation="100% asset or cash", min_lookback=35, regime="Trends", func="macd_signal",
       grid=dict(fast=[8, 16], slow=[21, 34])),
    _S(id="F06_06", family="F06", name="MACD positive and above signal",
       hypothesis="Momentum transitions are most informative when MACD is already positive (trend and momentum aligned); exit only when MACD turns negative.",
       entry="MACD > 0 and histogram > 0", exit="MACD < 0", params=dict(fast=12, slow=26, signal=9),
       allocation="100% asset or cash", min_lookback=35, regime="Established trends", func="macd_positive",
       grid=dict(fast=[8, 16], slow=[21, 34])),
    _S(id="F06_07", family="F06", name="RSI bullish divergence proxy",
       hypothesis="A new 20-day price low without a new RSI low signals waning selling momentum (non-look-ahead divergence proxy).",
       entry="Close <= prior 20-day low close and RSI14 > prior 20-day RSI low + 5", exit="Close > SMA20 or 20 bars",
       params=dict(n=14, low_n=20, margin=5.0, exit_n=20, max_hold=20), allocation="100% asset or cash", min_lookback=35,
       regime="Bottoming", func="rsi_divergence", grid=dict(margin=[2.0, 10.0], low_n=[10, 40])),
    _S(id="F06_08", family="F06", name="Combined oscillator confirmation (RSI, MACD, Stochastic)",
       hypothesis="Requiring agreement among three oscillators reduces false signals relative to any single oscillator.",
       entry="RSI14 > 50, MACD > signal and %K > %D", exit="At most one of three remains true", params=dict(rsi_n=14, stoch_n=14),
       allocation="100% asset or cash", min_lookback=35, regime="Trends", func="osc_combo", grid=dict(rsi_n=[10, 21], stoch_n=[10, 21])),
    _S(id="F06_09", family="F06", name="Williams %R cross above -80",
       hypothesis="Williams %R recovering from deeply oversold marks short-term capitulation ending.",
       entry="%R(14) crosses above -80", exit="%R > -20 or 40 bars", params=dict(n=14, lo=-80, hi=-20, max_hold=40),
       allocation="100% asset or cash", min_lookback=15, regime="Ranges", func="williams_cross", grid=dict(lo=[-90, -70], hi=[-30, -10])),
    _S(id="F06_10", family="F06", name="CCI(20) momentum breakout",
       hypothesis="CCI above +100 signals unusually strong momentum relative to the typical deviation (momentum use of an oscillator).",
       entry="CCI20 crosses above +100", exit="CCI20 < 0", params=dict(n=20, entry=100, exit=0),
       allocation="100% asset or cash", min_lookback=40, regime="Momentum bursts", func="cci_momentum",
       grid=dict(entry=[50, 150], exit=[-50, 50])),

    # ---------------- F07 Volatility and risk management ----------------
    _S(id="F07_01", family="F07", name="Chandelier stop (3 ATR22 from high), SMA50 re-entry",
       hypothesis="A volatility-scaled trailing stop exits only when a decline is large relative to recent noise, protecting against crashes.",
       entry="Close > SMA50 (re-armed after stop by cross below/above SMA50 or new 50-day high)",
       exit="Close < highest high since entry - 3 x ATR22", params=dict(stop="atr", param=3.0, reentry_n=50, peak_field="High", atr_n=22),
       allocation="100% asset or cash", min_lookback=51, regime="Crashes after trends", func="trailing_stop_system",
       grid=dict(param=[2.0, 4.0], reentry_n=[20, 100])),
    _S(id="F07_02", family="F07", name="10% trailing stop, SMA50 re-entry",
       hypothesis="A fixed-percentage trailing stop caps drawdowns from peak; comparison with F07_01 tests fixed vs volatility-scaled stops.",
       entry="Close > SMA50 (re-armed after stop)", exit="Close < 0.90 x highest close since entry",
       params=dict(stop="pct", param=0.10, reentry_n=50, peak_field="Close"), allocation="100% asset or cash", min_lookback=51,
       regime="Large drawdowns", func="trailing_stop_system", grid=dict(param=[0.07, 0.15], reentry_n=[20, 100])),
    _S(id="F07_03", family="F07", name="Volatility targeting 15% (20-day)",
       hypothesis="Volatility is persistent and negatively related to subsequent risk-adjusted returns; scaling exposure inversely to recent volatility stabilises risk.",
       entry="Weight = min(1, 15% / 20-day realised vol), changed in ≥10% steps", exit="Weight shrinks as vol rises",
       params=dict(target=0.15, n=20, estimator="boxcar"), allocation="0-100% asset, rest cash", min_lookback=21,
       regime="Volatility clustering", func="vol_target", grid=dict(target=[0.10, 0.20], n=[10, 60])),
    _S(id="F07_04", family="F07", name="Drawdown-based exposure reduction",
       hypothesis="Large drawdowns from the 52-week high persist (momentum crashes cluster); cutting exposure in deep drawdowns lowers tail risk.",
       entry="100% when drawdown from 252-day high > -10%", exit="50% when -10% to -20%; 25% below -20%",
       params=dict(n=252, dd1=0.10, dd2=0.20, w1=0.5, w2=0.25), allocation="25/50/100% asset, rest cash", min_lookback=252,
       regime="Prolonged bear markets", func="drawdown_exposure", grid=dict(dd1=[0.07, 0.15], dd2=[0.15, 0.30])),
    _S(id="F07_05", family="F07", name="Volatility regime switch (vol below 1y median)",
       hypothesis="Low-volatility regimes carry better risk-adjusted equity returns than high-volatility regimes.",
       entry="100% when 20-day vol < its 252-day median", exit="50% otherwise", params=dict(n=20, ref_n=252, high_weight=0.5),
       allocation="50/100% asset, rest cash", min_lookback=272, regime="Calm bull markets", func="vol_regime_exposure",
       grid=dict(high_weight=[0.0, 0.75], n=[10, 60])),
    _S(id="F07_06", family="F07", name="Adaptive ATR stop (2 ATR calm / 4 ATR volatile)",
       hypothesis="Stops should be wider when volatility is high to avoid noise-driven exits, and tighter when calm.",
       entry="Close > SMA50 (re-armed after stop)", exit="Close < highest close since entry - k x ATR14; k=4 if 20d vol above its 1y median else 2",
       params=dict(low_mult=2.0, high_mult=4.0, vol_n=20, rank_n=252, reentry_n=50), allocation="100% asset or cash",
       min_lookback=272, regime="Shifting volatility", func="adaptive_atr_stop", grid=dict(low_mult=[1.5, 3.0], high_mult=[3.0, 5.0])),
    _S(id="F07_07", family="F07", name="Volatility-spike exit (5d vol > 2 x 60d vol)",
       hypothesis="Sudden increases in short-term volatility relative to its norm precede further turbulence.",
       entry="5-day vol / 60-day vol < 1.2", exit="5-day vol / 60-day vol > 2.0", params=dict(short_n=5, long_n=60, exit_ratio=2.0, entry_ratio=1.2),
       allocation="100% asset or cash", min_lookback=61, regime="Volatility shocks", func="vol_spike_exit",
       grid=dict(exit_ratio=[1.5, 2.5], entry_ratio=[1.0, 1.5])),
    _S(id="F07_08", family="F07", name="Downside-volatility targeting 10%",
       hypothesis="Only downside volatility matters for drawdown risk; targeting semideviation reduces exposure ahead of crashes without penalising upside volatility.",
       entry="Weight = min(1, 10% / 60-day downside semideviation)", exit="Weight shrinks as downside vol rises",
       params=dict(target=0.10, n=60, estimator="downside"), allocation="0-100% asset, rest cash", min_lookback=61,
       regime="Crash-prone markets", func="vol_target", grid=dict(target=[0.07, 0.14], n=[20, 120])),
    _S(id="F07_09", family="F07", name="Tight 5% trailing stop, SMA20 re-entry",
       hypothesis="Contrast hypothesis: tight stops are whipsawed by normal noise, destroying value through repeated exits and re-entries.",
       entry="Close > SMA20 (re-armed after stop)", exit="Close < 0.95 x highest close since entry",
       params=dict(stop="pct", param=0.05, reentry_n=20, peak_field="Close"), allocation="100% asset or cash", min_lookback=51,
       regime="Smooth low-volatility trends", func="trailing_stop_system", grid=dict(param=[0.03, 0.07], reentry_n=[10, 50])),
    _S(id="F07_10", family="F07", name="EWMA volatility targeting 12% (λ=0.94)",
       hypothesis="A faster exponentially weighted volatility estimate reacts sooner to volatility clustering than a boxcar window.",
       entry="Weight = min(1, 12% / EWMA vol)", exit="Weight shrinks as vol rises", params=dict(target=0.12, n=30, estimator="ewma"),
       allocation="0-100% asset, rest cash", min_lookback=30, regime="Volatility clustering", func="vol_target",
       grid=dict(target=[0.09, 0.16])),

    # ---------------- F08 Regime detection ----------------
    _S(id="F08_01", family="F08", name="Market SMA200 regime (SPY)",
       hypothesis="The broad-market primary trend governs risk appetite for every equity asset; hold the asset only in a market bull regime.",
       entry="SPY Close > SPY SMA200", exit="SPY Close <= SPY SMA200", params=dict(n=200),
       allocation="100% asset or cash", min_lookback=200, regime="Market-wide bear phases", func="market_sma_regime",
       grid=dict(n=[150, 250]), requires=("market",)),
    _S(id="F08_02", family="F08", name="Market realised-volatility regime (SPY vol < 20%)",
       hypothesis="High market volatility marks stress regimes with poor risk-adjusted returns for all equities.",
       entry="SPY 20-day realised vol < 20%", exit="SPY vol >= 20%", params=dict(n=20, threshold=0.20),
       allocation="100% asset or cash", min_lookback=21, regime="Calm markets", func="market_vol_regime",
       grid=dict(threshold=[0.15, 0.25], n=[10, 60]), requires=("market",)),
    _S(id="F08_03", family="F08", name="ADX trend-strength filter",
       hypothesis="Trend-following only pays when a trend exists; ADX > 20 with +DI > -DI identifies strong uptrends.",
       entry="ADX14 > 20 and +DI > -DI", exit="+DI < -DI", params=dict(n=14, threshold=20),
       allocation="100% asset or cash", min_lookback=42, regime="Strong trends", func="adx_trend",
       grid=dict(threshold=[15, 25], n=[10, 21])),
    _S(id="F08_04", family="F08", name="Momentum-scaled cash allocation",
       hypothesis="Exposure should rise with trend strength: 12-month return divided by 20%, capped at 100%.",
       entry="Weight = clip(252-day return / 20%, 0, 1) in 25% steps", exit="Weight falls with momentum",
       params=dict(n=252, scale=0.20), allocation="0-100% asset in 25% steps", min_lookback=252, regime="Strong trends",
       func="momentum_scaled", grid=dict(scale=[0.10, 0.30], n=[126, 315])),
    _S(id="F08_05", family="F08", name="Dual momentum with bond fallback (monthly)",
       hypothesis="Combining absolute momentum (vs T-bills) and relative momentum (vs bonds) rotates into the stronger of risk and safety assets.",
       entry="Monthly: asset if 12m return > T-bill and > IEF; else IEF if IEF 12m > T-bill", exit="Else cash",
       params=dict(n=252), allocation="100% asset, or 100% IEF, or cash", min_lookback=252, regime="Bear markets with falling rates",
       func="dual_momentum", grid=dict(n=[126, 189]), requires=("defensive", "CASH")),
    _S(id="F08_06", family="F08", name="Defensive switch to IEF below SMA200",
       hypothesis="Holding intermediate Treasuries instead of cash in equity downtrends captures flight-to-quality returns.",
       entry="Asset when Close > SMA200", exit="IEF when Close <= SMA200 (cash before IEF inception)", params=dict(n=200),
       allocation="100% asset or 100% IEF", min_lookback=200, regime="Equity bear markets with bond rallies",
       func="defensive_switch", grid=dict(n=[150, 250]), requires=("defensive",)),
    _S(id="F08_07", family="F08", name="Variance-ratio regime switch (trend vs reversion)",
       hypothesis="When returns are positively autocorrelated (variance ratio > 1) trend rules work; otherwise mean-reversion rules work.",
       entry="If VR(10,120) > 1: SMA20>SMA50; else RSI2<10 reversion", exit="Per active sub-strategy",
       params=dict(q=10, n=120, fast=20, slow=50, rsi_n=2, rsi_entry=10, rsi_exit=70), allocation="100% asset or cash",
       min_lookback=130, regime="Alternating autocorrelation regimes", func="variance_ratio_switch",
       grid=dict(n=[60, 250], q=[5, 20])),
    _S(id="F08_08", family="F08", name="Risk-on/off via XLY/XLP ratio",
       hypothesis="Consumer discretionary outperforming staples signals economic optimism and risk appetite that supports equities.",
       entry="XLY/XLP ratio > its 100-day average", exit="Ratio below its average", params=dict(n=100),
       allocation="100% asset or cash", min_lookback=100, regime="Business-cycle turns", func="risk_on_ratio",
       grid=dict(n=[50, 200]), requires=("risk_on_num", "risk_on_den")),
    _S(id="F08_09", family="F08", name="Interest-rate tightening regime",
       hypothesis="Rapid increases in short-term rates (tightening) depress equity valuations, particularly rate-sensitive sectors.",
       entry="100% unless 3m T-bill yield rose > 1pp over 252 days", exit="50% in tightening regimes",
       params=dict(n=252, change=0.01, reduced_weight=0.5), allocation="50/100% asset, rest cash", min_lookback=252,
       regime="Monetary tightening", func="rate_regime", grid=dict(change=[0.005, 0.02], reduced_weight=[0.0, 0.75]),
       requires=("CASH",)),
    _S(id="F08_10", family="F08", name="Sector breadth regime (≥50% sectors above SMA200)",
       hypothesis="Broad participation across sectors indicates a healthy market; narrow markets are fragile.",
       entry="At least half of 9 original sector ETFs above their SMA200", exit="Fewer than half", params=dict(n=200, threshold=0.5),
       allocation="100% asset or cash", min_lookback=200, regime="Breadth deterioration before declines", func="breadth_regime",
       grid=dict(threshold=[0.33, 0.67], n=[100, 250]), requires=("breadth_list",)),

    # ---------------- F09 Relative strength and rotation ----------------
    _S(id="F09_01", family="F09", name="Top-1 by 6-month momentum (monthly)", kind="rotation", benchmark=EWU,
       hypothesis="Cross-sectional momentum: the single strongest asset over 6 months continues to outperform (maximum concentration).",
       entry="Monthly: hold the asset with the highest 126-day return", exit="Replaced when no longer top-ranked",
       params=dict(score="mom", n=126, skip=0, k=1), allocation="100% in top asset", min_lookback=126,
       regime="Persistent leadership", func="rotation_topk", grid=dict(n=[63, 252])),
    _S(id="F09_02", family="F09", name="Top-2 by 6-month momentum (monthly)", kind="rotation", benchmark=EWU,
       hypothesis="Holding the two strongest assets retains momentum exposure with less idiosyncratic concentration.",
       entry="Monthly: top 2 by 126-day return, 50/50", exit="Dropped when out of top 2",
       params=dict(score="mom", n=126, skip=0, k=2), allocation="50% each in top 2", min_lookback=126,
       regime="Persistent leadership", func="rotation_topk", grid=dict(n=[63, 252])),
    _S(id="F09_03", family="F09", name="Top-3 by 12-1 momentum (monthly)", kind="rotation", benchmark=EWU,
       hypothesis="Classic Jegadeesh-Titman 12-1 cross-sectional momentum applied to industry/sector ETFs (Moskowitz-Grinblatt industry momentum).",
       entry="Monthly: top 3 by return t-252..t-21", exit="Dropped when out of top 3",
       params=dict(score="mom", n=252, skip=21, k=3), allocation="1/3 each in top 3", min_lookback=252,
       regime="Industry momentum", func="rotation_topk", grid=dict(k=[2, 4])),
    _S(id="F09_04", family="F09", name="Bottom-3 by 1-month return (short-term reversal)", kind="rotation", benchmark=EWU,
       hypothesis="Contrast hypothesis: one-month cross-sectional losers rebound (short-term reversal) rather than continue.",
       entry="Monthly: 3 worst 21-day performers", exit="Rebalanced monthly",
       params=dict(score="mom", n=21, skip=0, k=3, reverse=True), allocation="1/3 each", min_lookback=21,
       regime="Mean-reverting cross-section", func="rotation_topk", grid=dict(n=[10, 42])),
    _S(id="F09_05", family="F09", name="Top-3 by risk-adjusted 6m momentum", kind="rotation", benchmark=EWU,
       hypothesis="Ranking on return per unit of volatility avoids chasing the most volatile assets and improves risk-adjusted persistence.",
       entry="Monthly: top 3 by 126-day return / 126-day vol", exit="Dropped when out of top 3",
       params=dict(score="riskadj", n=126, skip=0, k=3), allocation="1/3 each", min_lookback=126,
       regime="Smooth leadership", func="rotation_topk", grid=dict(n=[63, 252])),
    _S(id="F09_06", family="F09", name="Top-3 6m momentum with absolute filter", kind="rotation", benchmark=EWU,
       hypothesis="Dual momentum: rank relatively, but hold cash in slots whose 12-month return trails T-bills, avoiding broad bear markets.",
       entry="Monthly: top 3 by 126-day return, each kept only if its 252-day return > T-bill", exit="Slot to cash otherwise",
       params=dict(score="mom", n=126, skip=0, k=3, abs_filter=True), allocation="Up to 1/3 each, rest cash", min_lookback=252,
       regime="Bear markets", func="rotation_topk", grid=dict(n=[63, 252]), requires=("CASH",)),
    _S(id="F09_07", family="F09", name="Top-3 6m momentum excluding high-vol tercile", kind="rotation", benchmark=EWU,
       hypothesis="Volatility filter: the highest-volatility assets have worse momentum persistence (lottery-like demand).",
       entry="Monthly: drop top-tercile 63-day vol, then top 3 by 126-day return", exit="Dropped when out of selection",
       params=dict(score="mom", n=126, skip=0, k=3, vol_filter=True), allocation="1/3 each", min_lookback=126,
       regime="Low-vol leadership", func="rotation_topk", grid=dict(n=[63, 252])),
    _S(id="F09_08", family="F09", name="Relative strength vs SPY, 25% cap", kind="rotation", benchmark=EWU,
       hypothesis="Assets outperforming the market (price ratio above its average) keep outperforming; a 25% cap controls concentration.",
       entry="Monthly: hold every asset whose ratio to SPY > its 100-day average", exit="Equal weight, capped at 25% each, remainder cash",
       params=dict(n=100, cap=0.25), allocation="Equal weight among qualifiers, max 25% each", min_lookback=100,
       regime="Persistent relative trends", func="rs_vs_benchmark", grid=dict(n=[50, 200], cap=[0.20, 0.34]), requires=("market",)),
    _S(id="F09_09", family="F09", name="Leadership switch: asset vs SPY (monthly)", kind="single",
       hypothesis="Hold the asset only while it leads the market (e.g. semiconductor leadership), otherwise hold the market itself.",
       entry="Monthly: asset if asset/SPY ratio > its 100-day average", exit="Else hold SPY",
       params=dict(n=100), allocation="100% asset or 100% SPY", min_lookback=100, regime="Sector leadership cycles",
       func="leadership_switch", grid=dict(n=[50, 200]), requires=("market",),
       benchmark="Buy-and-hold same asset and SPY"),
    _S(id="F09_10", family="F09", name="Top-4 by 6m momentum, inverse-vol weighted", kind="rotation", benchmark=EWU,
       hypothesis="Risk-parity weighting among momentum winners controls concentration in volatile leaders.",
       entry="Monthly: top 4 by 126-day return", exit="Weights proportional to 1/63-day vol",
       params=dict(score="mom", n=126, skip=0, k=4, weighting="invvol"), allocation="Inverse-vol weights across top 4",
       min_lookback=126, regime="Broad leadership", func="rotation_topk", grid=dict(k=[3, 5])),

    # ---------------- F10 Hybrid ----------------
    _S(id="F10_01", family="F10", name="Trend + mean-reversion overlay",
       hypothesis="Trend following misses V-shaped rebounds; adding oversold bounces when below SMA200 captures them.",
       entry="Long when Close > SMA200; also when below SMA200 and RSI2 < 5", exit="Below SMA200: RSI2 > 60 or 10 bars",
       params=dict(trend_n=200, rsi_n=2, entry=5, exit=60), allocation="100% asset or cash", min_lookback=200,
       regime="Bull trends plus sharp corrections", func="trend_plus_mr", grid=dict(trend_n=[100, 250], entry=[3, 10])),
    _S(id="F10_02", family="F10", name="6m momentum + 3 ATR trailing stop",
       hypothesis="Momentum entries capture trends; a volatility stop limits momentum-crash losses that pure momentum exits react to too slowly.",
       entry="126-day return > 0 (re-armed after a stop)", exit="126-day return <= 0 or Close < peak - 3 x ATR14",
       params=dict(n=126, atr_n=14, mult=3.0), allocation="100% asset or cash", min_lookback=126,
       regime="Trends ending in crashes", func="momentum_atr_stop", grid=dict(mult=[2.0, 4.0], n=[63, 252])),
    _S(id="F10_03", family="F10", name="Donchian 20/10 breakout + 15% vol target",
       hypothesis="Volatility-targeting breakout positions equalises risk across calm and turbulent breakouts.",
       entry="Donchian 20/10 long state x min(1, 15% / 20-day vol)", exit="Donchian exit",
       params=dict(n_entry=20, n_exit=10, target=0.15, vol_n=20), allocation="0-100% asset, rest cash", min_lookback=21,
       regime="Trending breakouts", func="breakout_voltarget", grid=dict(target=[0.10, 0.20], n_entry=[15, 55])),
    _S(id="F10_04", family="F10", name="Trend + two-stage exposure (50/200 base, 20/50 add-on)",
       hypothesis="Core exposure from the primary trend with tactical add-on from the intermediate trend balances participation and protection.",
       entry="+50% if SMA50 > SMA200; +50% if Close > SMA20 > SMA50", exit="Each half removed when its condition fails",
       params=dict(n_trig=20, n_mid=50, n_slow=200), allocation="0/50/100% asset", min_lookback=200,
       regime="Bull markets with corrections", func="trend_two_stage", grid=dict(n_trig=[10, 30], n_slow=[150, 250])),
    _S(id="F10_05", family="F10", name="Adaptive trend (KAMA)",
       hypothesis="Kaufman's adaptive average speeds up in trends and slows in noise, adapting trend sensitivity to market efficiency.",
       entry="Close > KAMA(10,2,30) and KAMA rising over 5 days", exit="Close < KAMA",
       params=dict(n=10, fast=2, slow=30, slope_n=5), allocation="100% asset or cash", min_lookback=40,
       regime="Alternating trend/noise", func="kama_trend", grid=dict(n=[20, 40], slow=[20, 50])),
    _S(id="F10_06", family="F10", name="Risk-on/risk-off asset/IEF allocation",
       hypothesis="Shift between equity and Treasuries based on trend and volatility state to keep diversified exposure instead of cash.",
       entry="Asset weight = 50% x [Close > SMA200] + 50% x [20d vol below its 1y 80th pct]", exit="Remainder in IEF",
       params=dict(trend_n=200, vol_n=20, rank_n=252), allocation="Asset/IEF at 0/50/100%", min_lookback=272,
       regime="Stress regimes with bond rallies", func="risk_on_off", grid=dict(trend_n=[100, 250], vol_n=[10, 60]),
       requires=("defensive",)),
    _S(id="F10_07", family="F10", name="Five-family ensemble vote",
       hypothesis="Diversifying across uncorrelated signal families (MA, momentum, breakout, MACD, volatility) is more robust than optimising one indicator (H10).",
       entry="Weight = share of {SMA50>200, 12-1 mom>0, Donchian 55/20 long, MACD>signal, vol<median}", exit="Weight falls with votes",
       params=dict(), allocation="0-100% in 20% steps", min_lookback=273, regime="Any", func="ensemble_vote", grid=dict()),
    _S(id="F10_08", family="F10", name="Time-series momentum + 12% vol target",
       hypothesis="Moskowitz-Ooi-Pedersen time-series momentum with volatility scaling improves risk-adjusted returns over raw momentum.",
       entry="252-day return > 0, weight min(1, 12% / 60-day vol)", exit="252-day return <= 0",
       params=dict(n=252, target=0.12, vol_n=60), allocation="0-100% asset, rest cash", min_lookback=252,
       regime="Persistent trends", func="tsmom_voltarget", grid=dict(target=[0.08, 0.16], n=[126, 189])),
    _S(id="F10_09", family="F10", name="Pullback in momentum regime with 5% stop",
       hypothesis="Buying short-term pullbacks only within positive 6-month momentum, with a protective stop, combines reversion entries and trend context.",
       entry="126-day return > 0 and RSI3 < 20", exit="RSI3 > 70, 5% trailing stop, or 10 bars",
       params=dict(mom_n=126, rsi_n=3, entry=20, exit=70, stop_pct=0.05, max_hold=10), allocation="100% asset or cash",
       min_lookback=126, regime="Uptrends with dips", func="mr_in_momentum", grid=dict(entry=[10, 30], stop_pct=[0.03, 0.08])),
    _S(id="F10_10", family="F10", name="SMA10/30 with SMA200 regime scaling",
       hypothesis="The original SMA10/30 signal is more reliable inside a long-term uptrend; take full size above SMA200 and half size below.",
       entry="SMA10 > SMA30: 100% if Close > SMA200 else 50%", exit="SMA10 <= SMA30",
       params=dict(fast=10, slow=30, regime_n=200, off_weight=0.5), allocation="0/50/100% asset", min_lookback=200,
       regime="Bull markets with corrections", func="sma_cross_regime_scaled", grid=dict(off_weight=[0.0, 0.25], regime_n=[150, 250])),
]

FUNCTIONS = {name: obj for name, obj in globals().items() if callable(obj)}


def get_spec(sid: str) -> StrategySpec:
    for s in REGISTRY:
        if s.id == sid:
            return s
    raise KeyError(sid)


def generate_weights(spec: StrategySpec, md: MarketData, target, aux: dict, params: dict | None = None,
                     cache: dict | None = None) -> pd.DataFrame:
    """Run a strategy and return a target-weight DataFrame indexed by ``md.calendar``.

    ``target`` is a ticker (single-asset strategies) or a list of tickers (rotation).
    """
    p = dict(spec.params if params is None else params)
    fn = FUNCTIONS[spec.func]
    if spec.kind == "rotation":
        ctx = UniverseContext(md, list(target), aux, cache)
        w = fn(ctx, **p)
    else:
        ctx = AssetContext(md, target, aux, cache)
        w = fn(ctx, **p)
    if isinstance(w, pd.Series):
        w = w.to_frame()
    w = w.reindex(md.calendar).fillna(0.0)
    return w


# ============================================================================
# Registry export / freeze
# ============================================================================

def registry_frame() -> pd.DataFrame:
    rows = []
    for s in REGISTRY:
        d = asdict(s)
        d["family_name"] = FAMILIES[s.family]
        d["params"] = json.dumps(s.params, sort_keys=True, default=list)
        d["grid"] = json.dumps(s.grid, sort_keys=True, default=list)
        d["requires"] = ",".join(s.requires)
        d["n_param_variants"] = len(s.variants())
        d["config_json"] = json.dumps({"id": s.id, "func": s.func, "kind": s.kind, "params": s.params},
                                      sort_keys=True, default=list)
        rows.append(d)
    cols = ["id", "family", "family_name", "name", "origin", "kind", "hypothesis", "entry", "exit", "params",
            "allocation", "min_lookback", "benchmark", "regime", "func", "grid", "n_param_variants", "requires",
            "config_json"]
    return pd.DataFrame(rows)[cols]


def registry_hash() -> str:
    canon = json.dumps([asdict(s) for s in REGISTRY], sort_keys=True, default=list)
    return hashlib.sha256(canon.encode()).hexdigest()


def validate_registry() -> None:
    ids = [s.id for s in REGISTRY]
    assert len(REGISTRY) == 100, f"expected 100 strategies, found {len(REGISTRY)}"
    assert len(set(ids)) == 100, "duplicate strategy ids"
    for fam in FAMILIES:
        n = sum(s.family == fam for s in REGISTRY)
        assert n == 10, f"family {fam} has {n} strategies"
    sigs = [(s.func, json.dumps(s.params, sort_keys=True, default=list)) for s in REGISTRY]
    assert len(set(sigs)) == 100, "two strategies share identical function and parameters"
    for s in REGISTRY:
        assert s.func in FUNCTIONS, f"{s.id}: unknown function {s.func}"
        for attr in ("hypothesis", "entry", "exit", "allocation", "benchmark", "regime"):
            assert getattr(s, attr), f"{s.id}: missing {attr}"
        assert s.min_lookback >= 1


def freeze_registry(out_dir: Path) -> dict:
    """Write Strategy_Registry.csv and registry_freeze.json (hash + timestamp)."""
    validate_registry()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = registry_frame()
    df.to_csv(out_dir / "Strategy_Registry.csv", index=False)
    info = {"n_strategies": len(REGISTRY), "registry_sha256": registry_hash(),
            "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "families": {k: v for k, v in FAMILIES.items()}}
    freeze_path = out_dir / "registry_freeze.json"
    if freeze_path.exists():
        prev = json.loads(freeze_path.read_text())
        if prev.get("registry_sha256") == info["registry_sha256"]:
            return prev  # unchanged registry keeps its original freeze timestamp
        info["previous_freeze"] = prev
    freeze_path.write_text(json.dumps(info, indent=2))
    return info
