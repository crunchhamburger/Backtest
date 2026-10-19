"""Causal technical indicators.

Every function uses only observations up to and including bar ``t`` and
returns NaN until its full lookback is available (``min_periods == window``).
``tests/test_lookahead.py`` verifies this property by perturbing future data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def roc(x: pd.Series, n: int) -> pd.Series:
    """n-bar rate of change (simple return)."""
    return x / x.shift(n) - 1


def roc_skip(x: pd.Series, n: int, skip: int) -> pd.Series:
    """Return from t-n to t-skip (e.g. 12-1 momentum: n=252, skip=21)."""
    return x.shift(skip) / x.shift(n) - 1


def rolling_max(x: pd.Series, n: int, exclude_current: bool = True) -> pd.Series:
    r = x.rolling(n, min_periods=n).max()
    return r.shift(1) if exclude_current else r


def rolling_min(x: pd.Series, n: int, exclude_current: bool = True) -> pd.Series:
    r = x.rolling(n, min_periods=n).min()
    return r.shift(1) if exclude_current else r


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["Close"].shift(1)
    return pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    """Average true range as a simple n-bar mean of true range (as in the original notebook)."""
    return true_range(df).rolling(n, min_periods=n).mean()


def realized_vol(close: pd.Series, n: int) -> pd.Series:
    """Annualised close-to-close volatility over n bars."""
    return close.pct_change().rolling(n, min_periods=n).std(ddof=1) * np.sqrt(ANN)


def ewma_vol(close: pd.Series, lam: float = 0.94, min_periods: int = 30) -> pd.Series:
    r = close.pct_change()
    var = (r ** 2).ewm(alpha=1 - lam, adjust=False, min_periods=min_periods).mean()
    return np.sqrt(var * ANN)


def downside_vol(close: pd.Series, n: int) -> pd.Series:
    r = close.pct_change()
    d = np.minimum(r, 0.0)
    return np.sqrt((d ** 2).rolling(n, min_periods=n).mean() * ANN)


def rsi(close: pd.Series, n: int) -> pd.Series:
    """Wilder RSI."""
    d = close.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = au / ad
    out = 100 - 100 / (1 + rs)
    out = out.where(ad > 0, 100.0).where(au.notna())
    return out


def bollinger(close: pd.Series, n: int, k: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    m = sma(close, n)
    s = close.rolling(n, min_periods=n).std(ddof=0)
    return m - k * s, m, m + k * s


def stochastic(df: pd.DataFrame, n: int, smooth: int) -> tuple[pd.Series, pd.Series]:
    ll = df["Low"].rolling(n, min_periods=n).min()
    hh = df["High"].rolling(n, min_periods=n).max()
    raw = 100 * (df["Close"] - ll) / (hh - ll).replace(0, np.nan)
    k = raw.rolling(smooth, min_periods=smooth).mean()
    d = k.rolling(smooth, min_periods=smooth).mean()
    return k, d


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    line = line.where(close.notna().rolling(slow, min_periods=slow).count() >= slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def williams_r(df: pd.DataFrame, n: int) -> pd.Series:
    hh = df["High"].rolling(n, min_periods=n).max()
    ll = df["Low"].rolling(n, min_periods=n).min()
    return -100 * (hh - df["Close"]) / (hh - ll).replace(0, np.nan)


def cci(df: pd.DataFrame, n: int) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    m = sma(tp, n)
    mad = tp.rolling(n, min_periods=n).apply(lambda a: np.mean(np.abs(a - a.mean())), raw=True)
    return (tp - m) / (0.015 * mad.replace(0, np.nan))


def adx(df: pd.DataFrame, n: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder ADX with +DI and -DI."""
    up = df["High"].diff()
    dn = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = true_range(df)
    a = 1 / n
    atr_w = tr.ewm(alpha=a, adjust=False, min_periods=n).mean()
    pdi = 100 * plus_dm.ewm(alpha=a, adjust=False, min_periods=n).mean() / atr_w
    mdi = 100 * minus_dm.ewm(alpha=a, adjust=False, min_periods=n).mean() / atr_w
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=a, adjust=False, min_periods=n).mean(), pdi, mdi


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    change = (close - close.shift(n)).abs()
    vol = close.diff().abs().rolling(n, min_periods=n).sum()
    return change / vol.replace(0, np.nan)


def kama(close: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    er = efficiency_ratio(close, n).to_numpy()
    c = close.to_numpy(dtype=float)
    fsc, ssc = 2 / (fast + 1), 2 / (slow + 1)
    out = np.full(len(c), np.nan)
    for i in range(len(c)):
        if np.isnan(er[i]) or np.isnan(c[i]):
            continue
        sc = (er[i] * (fsc - ssc) + ssc) ** 2
        prev = out[i - 1] if i > 0 and not np.isnan(out[i - 1]) else c[i - 1] if i > 0 else c[i]
        out[i] = prev + sc * (c[i] - prev)
    return pd.Series(out, index=close.index)


def variance_ratio(close: pd.Series, q: int, n: int) -> pd.Series:
    """Lo-MacKinlay variance ratio of q-bar vs 1-bar log returns over a rolling n-bar window."""
    lr = np.log(close).diff()
    v1 = lr.rolling(n, min_periods=n).var()
    lq = np.log(close).diff(q)
    vq = lq.rolling(n, min_periods=n).var()
    return vq / (q * v1)


def ibs(df: pd.DataFrame) -> pd.Series:
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    return (df["Close"] - df["Low"]) / rng


def consecutive(cond: pd.Series) -> pd.Series:
    """Number of consecutive True values ending at each bar."""
    c = cond.fillna(False).astype(int)
    grp = (c == 0).cumsum()
    return c.groupby(grp).cumsum()


class IndicatorCache:
    """Memoises indicator computations for one OHLCV frame."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._c: dict = {}

    def __call__(self, name: str, *args):
        key = (name,) + tuple(args)
        if key not in self._c:
            fn = globals()[name]
            if name in ("atr", "true_range", "stochastic", "williams_r", "cci", "adx", "ibs"):
                self._c[key] = fn(self.df, *args)
            else:
                self._c[key] = fn(self.df["Close"], *args)
        return self._c[key]
