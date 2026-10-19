"""Unified execution engine with share-level portfolio accounting.

Timing convention (enforced here, not in strategy code)
-------------------------------------------------------
* A strategy returns *target weights* ``w[t]`` computed from information
  available at the **close** of bar ``t``.
* The engine executes ``w[t]`` at the **open of bar t+1**. Signal price
  (close t) and execution price (open t+1) are therefore always different, and
  the overnight gap between them is borne by the strategy.
* The portfolio is marked to market at every close. Daily returns are
  close-to-close NAV changes; on an entry day the position earns open-to-close.
* Uninvested cash accrues the lagged 3-month T-bill rate between bars.
* Costs = (commission + slippage) bps x |traded notional|, charged per side.
* Long-only, unlevered: every target must satisfy 0 <= w <= 1 and sum(w) <= 1.
  Buys are scaled down if needed so that cash never goes negative after costs.
* The portfolio is rebalanced to target when any position's target changes or
  any drifted weight deviates from target by more than ``rebalance_band``.
* Fractional shares are allowed (documented simplification).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_pipeline import MarketData

EPS = 1e-12


@dataclass
class CostModel:
    commission_bps: float = 0.0
    slippage_bps: float = 10.0

    @property
    def rate(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 1e4

    @classmethod
    def from_total_bps(cls, bps: float) -> "CostModel":
        return cls(0.0, float(bps))


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    exposure: pd.Series
    costs: pd.Series
    turnover: pd.Series
    cash: pd.Series
    target: pd.DataFrame
    trades: pd.DataFrame | None = None
    equity_open: pd.Series | None = None
    meta: dict = field(default_factory=dict)


def _validate_targets(w: np.ndarray, names) -> None:
    if np.isnan(w).any():
        raise ValueError("target weights contain NaN; strategies must emit 0 when flat")
    if (w < -1e-9).any():
        raise ValueError(f"short position requested in {names}")
    if (w > 1 + 1e-9).any() or (w.sum(axis=1) > 1 + 1e-6).any():
        raise ValueError(f"leverage requested in {names}")


def simulate(open_: np.ndarray, close: np.ndarray, tradable: np.ndarray, target: np.ndarray,
             cash_growth: np.ndarray, cost_rate: float, capital: float, band: float,
             record_trades: bool = False):
    """Core loop. Arrays are (T, N) except ``cash_growth`` (T,).

    Returns a dict of numpy arrays (and a list of trade tuples if requested).
    """
    T, N = target.shape
    o = np.nan_to_num(open_, nan=0.0)
    c = np.nan_to_num(close, nan=0.0)
    shares = np.zeros(N)
    last_tgt = np.zeros(N)
    cash = float(capital)
    eq_close = np.empty(T)
    eq_open = np.empty(T)
    cash_hist = np.empty(T)
    costs = np.zeros(T)
    turn = np.zeros(T)
    w_close = np.zeros((T, N))
    trades = []
    for t in range(T):
        cash *= 1.0 + cash_growth[t]
        px = o[t]
        val = shares * px
        nav = cash + val.sum()
        if t > 0 and nav > 0:
            tgt = target[t - 1]
            cur_w = val / nav
            # Any target change or band breach triggers a rebalance of every tradable position,
            # so proceeds from trimming one asset are redeployed instead of drifting into cash.
            trig = tradable[t] & ((np.abs(tgt - last_tgt) > EPS) | (np.abs(cur_w - tgt) > band))
            mask = tradable[t] & (np.abs(tgt * nav - val) > 1e-9 * nav) if trig.any() else np.zeros(N, dtype=bool)
            if mask.any():
                delta = np.where(mask, tgt * nav - val, 0.0)
                pos = delta > 0
                d_b = delta[pos].sum()
                d_s = -delta[~pos].sum()
                max_b = max((cash + d_s * (1 - cost_rate)) / (1 + cost_rate), 0.0)
                if d_b > max_b:
                    delta[pos] *= max_b / d_b
                cost = cost_rate * np.abs(delta).sum()
                cash -= delta.sum() + cost
                if abs(cash) < 1e-9 * nav:
                    cash = 0.0
                new_sh = np.where(mask, shares + np.divide(delta, px, out=np.zeros(N), where=px > 0), shares)
                new_sh[mask & (tgt <= EPS)] = 0.0
                if record_trades:
                    for j in np.flatnonzero(mask):
                        trades.append((t, j, delta[j] / px[j], px[j], delta[j], cost_rate * abs(delta[j]),
                                       cur_w[j], tgt[j]))
                shares = new_sh
                costs[t] = cost
                turn[t] = np.abs(delta).sum() / nav
            last_tgt = np.where(tradable[t], tgt, last_tgt)
        eq_open[t] = cash + (shares * px).sum()
        hv = shares * c[t]
        nav_c = cash + hv.sum()
        eq_close[t] = nav_c
        cash_hist[t] = cash
        w_close[t] = hv / nav_c if nav_c > 0 else 0.0
    return dict(eq_close=eq_close, eq_open=eq_open, cash=cash_hist, costs=costs, turnover=turn,
                w_close=w_close, trades=trades)


def simulate_single(open_: np.ndarray, close: np.ndarray, tradable: np.ndarray, target: np.ndarray,
                    cash_growth: np.ndarray, cost_rate: float, capital: float, band: float):
    """Scalar fast path for one asset; numerically identical to :func:`simulate` (tested)."""
    T = len(target)
    o = np.nan_to_num(open_, nan=0.0).tolist()
    c = np.nan_to_num(close, nan=0.0).tolist()
    tg = target.tolist()
    tr = tradable.tolist()
    cg = cash_growth.tolist()
    sh = 0.0
    last = 0.0
    cash = float(capital)
    eq = [0.0] * T
    wc = [0.0] * T
    costs = [0.0] * T
    turn = [0.0] * T
    for t in range(T):
        cash *= 1.0 + cg[t]
        px = o[t]
        val = sh * px
        nav = cash + val
        if t > 0 and nav > 0:
            tgt = tg[t - 1]
            if tr[t] and (abs(tgt - last) > EPS or abs(val / nav - tgt) > band) and abs(tgt * nav - val) > 1e-9 * nav:
                delta = tgt * nav - val
                if delta > 0:
                    max_b = max(cash / (1 + cost_rate), 0.0)
                    if delta > max_b:
                        delta = max_b
                cost = cost_rate * abs(delta)
                cash -= delta + cost
                if abs(cash) < 1e-9 * nav:
                    cash = 0.0
                sh = 0.0 if tgt <= EPS else sh + delta / px
                costs[t] = cost
                turn[t] = abs(delta) / nav
            if tr[t]:
                last = tgt
        hv = sh * c[t]
        navc = cash + hv
        eq[t] = navc
        wc[t] = hv / navc if navc > 0 else 0.0
    return np.array(eq), np.array(wc), np.array(costs), np.array(turn)


def _panel(md: MarketData, cols: list[str], field_: str) -> np.ndarray:
    return np.column_stack([md.bars[c][field_].to_numpy(dtype=float) for c in cols])


def backtest(md: MarketData, target: pd.DataFrame | pd.Series, cost: CostModel | float = 10.0,
             capital: float = 100_000.0, band: float = 0.10, record_trades: bool = False,
             fast: bool = True) -> BacktestResult:
    """Run target weights through the engine.

    ``target`` columns are tickers present in ``md``; its index must equal
    ``md.calendar``. ``cost`` may be a :class:`CostModel` or total bps.
    """
    if isinstance(cost, (int, float)):
        cost = CostModel.from_total_bps(cost)
    if isinstance(target, pd.Series):
        target = target.to_frame(target.name or "asset")
    target = target.reindex(md.calendar).fillna(0.0)
    cols = list(target.columns)
    w = target.to_numpy(dtype=float)
    _validate_targets(w, cols)
    tradable = np.column_stack([md.bars[c]["tradable"].to_numpy(dtype=bool) for c in cols])
    # A target on a not-yet-listed asset cannot be executed; force it to zero.
    listed = np.column_stack([md.bars[c]["listed"].to_numpy(dtype=bool) for c in cols])
    w = np.where(listed, w, 0.0)
    o = _panel(md, cols, "Open")
    c = _panel(md, cols, "Close")
    cg = md.cash_growth.to_numpy(dtype=float)
    idx = md.calendar
    if fast and len(cols) == 1 and not record_trades:
        eq, wc, costs, turn = simulate_single(o[:, 0], c[:, 0], tradable[:, 0], w[:, 0], cg, cost.rate, capital, band)
        wc = wc[:, None]
        cash = eq * (1 - wc[:, 0])
        eq_open = None
        trades = None
    else:
        r = simulate(o, c, tradable, w, cg, cost.rate, capital, band, record_trades)
        eq, wc, costs, turn, cash = r["eq_close"], r["w_close"], r["costs"], r["turnover"], r["cash"]
        eq_open = pd.Series(r["eq_open"], index=idx)
        trades = None
        if record_trades:
            tr = pd.DataFrame(r["trades"], columns=["t", "j", "shares", "price", "notional", "cost",
                                                    "weight_before", "weight_target"])
            tr.insert(0, "exec_date", idx[tr["t"].to_numpy()] if len(tr) else pd.DatetimeIndex([]))
            tr.insert(1, "signal_date", idx[tr["t"].to_numpy() - 1] if len(tr) else pd.DatetimeIndex([]))
            tr.insert(2, "ticker", [cols[j] for j in tr["j"]])
            tr.insert(3, "side", np.where(tr["notional"] > 0, "BUY", "SELL"))
            trades = tr.drop(columns=["t", "j"])
    equity = pd.Series(eq, index=idx)
    prev = np.r_[capital, eq[:-1]]
    ret = pd.Series(eq / prev - 1, index=idx)
    gross = pd.Series((eq + costs) / prev - 1, index=idx)
    weights = pd.DataFrame(wc, index=idx, columns=cols)
    return BacktestResult(equity=equity, returns=ret, gross_returns=gross, weights=weights,
                          exposure=weights.sum(axis=1), costs=pd.Series(costs, index=idx),
                          turnover=pd.Series(turn, index=idx), cash=pd.Series(cash, index=idx),
                          target=target, trades=trades, equity_open=eq_open,
                          meta={"cost_bps": cost.rate * 1e4, "band": band, "capital": capital})


def buy_and_hold_target(md: MarketData, ticker: str) -> pd.Series:
    return md.bars[ticker]["listed"].astype(float).rename(ticker)


def static_mix_target(md: MarketData, ticker: str, weight: float) -> pd.Series:
    """Constant-weight asset/cash mix (exposure- or volatility-matched benchmark)."""
    return (md.bars[ticker]["listed"].astype(float) * float(np.clip(weight, 0, 1))).rename(ticker)


def equal_weight_target(md: MarketData, tickers: list[str], monthly: bool = True) -> pd.DataFrame:
    listed = pd.DataFrame({t: md.bars[t]["listed"] for t in tickers}).astype(float)
    w = listed.div(listed.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    if monthly:
        # Rebalance on the first trading day of each month (causal clock) and whenever
        # a new asset lists; hold weights (no drift rebalancing beyond the engine band) otherwise.
        per = pd.Series(md.calendar.to_period("M"), index=md.calendar)
        clock = (per != per.shift(1)) | (listed.diff().abs().sum(axis=1) > 0)
        w = w.where(clock, axis=0).ffill().fillna(0.0)
    return w


def ledger_reconcile(res: BacktestResult, md: MarketData, capital: float) -> pd.DataFrame:
    """Rebuild holdings and cash from the trade ledger and compare with engine equity.

    Returns a frame with the absolute reconciliation error per day.
    """
    if res.trades is None:
        raise ValueError("run backtest(record_trades=True) first")
    cols = list(res.target.columns)
    idx = md.calendar
    cg = md.cash_growth.to_numpy()
    shares = {c: 0.0 for c in cols}
    cash = capital
    tr = res.trades.copy()
    tr["pos"] = idx.get_indexer(tr["exec_date"])
    by_t = {k: g for k, g in tr.groupby("pos")}
    out = np.empty(len(idx))
    for t in range(len(idx)):
        cash *= 1 + cg[t]
        if t in by_t:
            for _, row in by_t[t].iterrows():
                cash -= row["notional"] + row["cost"]
                shares[row["ticker"]] += row["shares"]
                if row["weight_target"] <= EPS:
                    shares[row["ticker"]] = 0.0
        mv = sum(shares[c] * np.nan_to_num(md.bars[c]["Close"].iat[t]) for c in cols)
        out[t] = cash + mv
    return pd.DataFrame({"ledger_equity": out, "engine_equity": res.equity.to_numpy(),
                         "abs_error": np.abs(out - res.equity.to_numpy())}, index=idx)


# ----------------------------------------------------------------------------
# Legacy replication of the original Colab notebook (cells 5 and 7)
# ----------------------------------------------------------------------------
NOTEBOOK_REFERENCE = {
    # Values printed by SPY_First_Backtest_Starter.ipynb (Yahoo data 2010-01-04..2026-10-16,
    # 10 bps per position change, cash at 0%, Sharpe with rf = 0).
    "data_rows": 4223,
    "holdout_start": "2021-09-29",
    "holdout_trades": 24,
    "Development 70%": {"SMA 20/50 net": {"CAGR": 0.085, "Annual vol": 0.113, "Sharpe (rf=0)": 0.780, "Max drawdown": -0.243},
                        "Buy-and-hold": {"CAGR": 0.144, "Annual vol": 0.161, "Sharpe (rf=0)": 0.920, "Max drawdown": -0.320}},
    "Holdout 30%": {"SMA 20/50 net": {"CAGR": 0.038, "Annual vol": 0.116, "Sharpe (rf=0)": 0.381, "Max drawdown": -0.283},
                    "Buy-and-hold": {"CAGR": 0.136, "Annual vol": 0.173, "Sharpe (rf=0)": 0.826, "Max drawdown": -0.263}},
}


def legacy_notebook_sma2050(bars: pd.DataFrame, start: str = "2010-01-01", cost_bps: float = 10.0) -> dict:
    """Re-implement the original notebook's SMA20/50 computation exactly (open-to-open, 0% cash).

    Used only to confirm that the new engine's inputs reproduce the earlier
    numbers before the improved methodology is applied.
    """
    p = bars[bars["listed"]].loc[start:, ["Open", "High", "Low", "Close"]].copy()
    d = p.copy()
    d["sma20"] = d["Close"].rolling(20, min_periods=20).mean()
    d["sma50"] = d["Close"].rolling(50, min_periods=50).mean()
    d["signal_at_close"] = (d["sma20"] > d["sma50"]).astype(int)
    d["position_at_open"] = d["signal_at_close"].shift(1).fillna(0).astype(int)
    d["next_open_return"] = d["Open"].shift(-1) / d["Open"] - 1
    d["position_changes"] = d["position_at_open"].diff().abs()
    d.loc[d.index[0], "position_changes"] = d.loc[d.index[0], "position_at_open"]
    c = cost_bps / 10_000
    d["strategy_return"] = d["position_at_open"] * d["next_open_return"] - d["position_changes"] * c
    d["buy_hold_return"] = d["next_open_return"]
    d = d.dropna(subset=["next_open_return", "strategy_return"])
    cut = int(len(d) * 0.70)

    def met(r):
        eq = (1 + r).cumprod()
        vol = r.std(ddof=1) * np.sqrt(252)
        return {"CAGR": eq.iloc[-1] ** (252 / len(r)) - 1, "Annual vol": vol,
                "Sharpe (rf=0)": r.mean() * 252 / vol, "Max drawdown": (eq / eq.cummax() - 1).min()}

    out = {"data_rows": len(p), "holdout_start": str(d.index[cut].date()),
           "holdout_trades": int(d["position_changes"].iloc[cut:].sum())}
    for lab, fr in (("Development 70%", d.iloc[:cut]), ("Holdout 30%", d.iloc[cut:])):
        out[lab] = {"SMA 20/50 net": met(fr["strategy_return"]), "Buy-and-hold": met(fr["buy_hold_return"])}
    return out
