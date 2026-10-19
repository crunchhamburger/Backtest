"""Walk-forward (rolling-origin) evaluation.

Folds are calendar based: each test window is one calendar year, preceded by a
training window of ``train_years`` full calendar years (rolling) or all prior
history (expanding). Strategy signals are causal and computed once on the full
history; slicing their daily returns into windows therefore introduces no
leakage, while *selection* (of parameters or of strategies) uses only the
training window and is evaluated on the following test window.

Analyses supported
------------------
* In-sample vs out-of-sample performance of fixed rules per fold.
* Parameter selection within a strategy's pre-registered neighbourhood.
* Strategy selection across the registry (best-of-N), whose stitched
  out-of-sample record is the honest estimate of "picking the best backtest".
* Rank persistence: Spearman correlation of IS vs OOS ranks and top-decile
  retention.
* Cross-sectional selection (sector-specific vs common strategy, H6).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

ANN = 252


@dataclass(frozen=True)
class Fold:
    label: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def year_folds(index: pd.DatetimeIndex, first_date: pd.Timestamp, train_years: int = 5, test_years: int = 1,
               mode: str = "rolling", min_test_bars: int = 60) -> list[Fold]:
    """Calendar-year folds whose training window starts on or after ``first_date``."""
    first_date = pd.Timestamp(first_date)
    years = sorted(set(index.year))
    folds = []
    first_full_year = first_date.year + (0 if first_date <= pd.Timestamp(first_date.year, 1, 15) else 1)
    for y in years:
        tr0 = first_full_year if mode == "expanding" else y - train_years
        if y - train_years < first_full_year:
            continue
        test = index[(index.year >= y) & (index.year < y + test_years)]
        train = index[(index.year >= tr0) & (index.year < y)]
        if len(test) < min_test_bars or len(train) < ANN * min(train_years, 1):
            continue
        folds.append(Fold(f"{y}", train[0], train[-1], test[0], test[-1]))
    return folds


def _mask(index: pd.DatetimeIndex, a: pd.Timestamp, b: pd.Timestamp) -> np.ndarray:
    return (index >= a) & (index <= b)


def _sharpe_cols(X: np.ndarray) -> np.ndarray:
    sd = X.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 0, X.mean(axis=0) / sd * math.sqrt(ANN), np.nan)


def _cagr_cols(R: np.ndarray) -> np.ndarray:
    return np.exp(np.log1p(R).sum(axis=0) * ANN / max(len(R), 1)) - 1


def fold_table(R: pd.DataFrame, rf: pd.Series, folds: list[Fold]) -> dict[str, pd.DataFrame]:
    """IS/OOS Sharpe and CAGR for every column and fold."""
    idx = R.index
    X = R.to_numpy(dtype=float)
    f = rf.reindex(idx).to_numpy(dtype=float)
    out = {k: [] for k in ("is_sharpe", "oos_sharpe", "is_cagr", "oos_cagr")}
    for fd in folds:
        mi = _mask(idx, fd.train_start, fd.train_end)
        mo = _mask(idx, fd.test_start, fd.test_end)
        out["is_sharpe"].append(_sharpe_cols(X[mi] - f[mi, None]))
        out["oos_sharpe"].append(_sharpe_cols(X[mo] - f[mo, None]))
        out["is_cagr"].append(_cagr_cols(X[mi]))
        out["oos_cagr"].append(_cagr_cols(X[mo]))
    labels = [fd.label for fd in folds]
    return {k: pd.DataFrame(np.array(v), index=labels, columns=R.columns) for k, v in out.items()}


def select_and_stitch(R: pd.DataFrame, rf: pd.Series, folds: list[Fold], groups: dict[str, list],
                      metric: str = "sharpe") -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each group, pick the column with the best training-window metric in every fold
    and stitch the chosen columns' test-window returns.

    Returns (oos_returns: DataFrame indexed like R with NaN outside test windows,
    choices: DataFrame fold x group with the chosen column label).
    """
    tab = fold_table(R, rf, folds)
    key = "is_sharpe" if metric == "sharpe" else "is_cagr"
    oos = pd.DataFrame(np.nan, index=R.index, columns=list(groups))
    choices = pd.DataFrame(index=[fd.label for fd in folds], columns=list(groups), dtype=object)
    is_best = pd.DataFrame(index=choices.index, columns=list(groups), dtype=float)
    for fd in folds:
        mo = _mask(R.index, fd.test_start, fd.test_end)
        for g, cols in groups.items():
            vals = tab[key].loc[fd.label, cols]
            if vals.notna().sum() == 0:
                continue
            best = vals.idxmax()
            choices.loc[fd.label, g] = best
            is_best.loc[fd.label, g] = vals.max()
            oos.loc[mo, g] = R.loc[mo, best].to_numpy()
    choices.attrs["is_best_metric"] = is_best
    return oos, choices


def rank_persistence(is_tab: pd.DataFrame, oos_tab: pd.DataFrame, top_frac: float = 0.10) -> pd.DataFrame:
    """Per fold: Spearman rank correlation and retention of the IS top group."""
    rows = []
    for lab in is_tab.index:
        a, b = is_tab.loc[lab], oos_tab.loc[lab]
        ok = a.notna() & b.notna()
        n = int(ok.sum())
        if n < 5:
            continue
        rho = stats.spearmanr(a[ok], b[ok]).statistic
        k = max(1, int(round(top_frac * n)))
        top_is = set(a[ok].nlargest(k).index)
        top_oos = set(b[ok].nlargest(k).index)
        half_oos = set(b[ok].nlargest(n // 2).index)
        best = a[ok].idxmax()
        rows.append({"fold": lab, "n": n, "spearman": rho, "top_retention": len(top_is & top_oos) / k,
                     "top_in_upper_half": len(top_is & half_oos) / k,
                     "best_is_oos_rank_pct": float((b[ok] < b[ok][best]).mean()),
                     "chance_top_retention": k / n})
    return pd.DataFrame(rows)


def stitched_metrics(oos: pd.Series, rf: pd.Series) -> dict:
    s = oos.dropna()
    if len(s) < 20:
        return {"oos_bars": len(s), "oos_cagr": np.nan, "oos_sharpe": np.nan, "oos_max_dd": np.nan}
    f = rf.reindex(s.index).to_numpy()
    x = s.to_numpy() - f
    eq = np.r_[1.0, np.cumprod(1 + s.to_numpy())]
    return {"oos_bars": len(s), "oos_cagr": float(eq[-1] ** (ANN / len(s)) - 1),
            "oos_sharpe": float(x.mean() / x.std(ddof=1) * math.sqrt(ANN)) if x.std(ddof=1) > 0 else np.nan,
            "oos_max_dd": float((eq / np.maximum.accumulate(eq) - 1).min())}


def cross_sectional_selection(is_by_asset: dict[str, pd.DataFrame], oos_by_asset: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """H6: compare per-asset strategy selection with one common strategy for all assets.

    ``is_by_asset[a]`` / ``oos_by_asset[a]`` are fold x strategy Sharpe tables.
    For each fold: *specific* picks each asset's best IS strategy; *common* picks the
    strategy with the highest mean IS Sharpe across assets. Both are scored OOS.
    """
    assets = list(is_by_asset)
    folds = sorted(set().union(*[set(is_by_asset[a].index) for a in assets]))
    rows = []
    for f in folds:
        avail = [a for a in assets if f in is_by_asset[a].index and is_by_asset[a].loc[f].notna().any()]
        if len(avail) < 2:
            continue
        IS = pd.DataFrame({a: is_by_asset[a].loc[f] for a in avail}).T
        OOS = pd.DataFrame({a: oos_by_asset[a].loc[f] for a in avail}).T
        common = IS.mean(axis=0).idxmax()
        for a in avail:
            spec = IS.loc[a].idxmax()
            rows.append({"fold": f, "asset": a, "specific_strategy": spec, "common_strategy": common,
                         "specific_is": IS.loc[a, spec], "common_is": IS.loc[a, common],
                         "specific_oos": OOS.loc[a, spec], "common_oos": OOS.loc[a, common],
                         "registry_median_oos": OOS.loc[a].median()})
    return pd.DataFrame(rows)
