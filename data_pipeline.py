"""Historical data acquisition, caching, adjustment and validation.

Design
------
* Raw daily bars are downloaded once (Yahoo Finance via ``yfinance`` with
  ``auto_adjust=False`` so that both the split-adjusted ``Close`` and the
  split+dividend-adjusted ``Adj Close`` are kept, together with the dividend and
  split event columns) and cached as gzip CSV under ``data/cache/raw``.
* Total-return adjusted OHLC are built locally:
  ``factor = Adj Close / Close`` and ``Open/High/Low/Close *= factor``.
  Because Open and Close on the same day share one factor, the intraday
  relationship is preserved, and open-to-open returns across an ex-dividend date
  include the distribution.
* Every series is validated (ordering, duplicates, OHLC consistency, stale or
  extreme prints, dividend/split consistency, calendar gaps). Problems are
  reported in a table; ``error`` rows block the ticker.
* The short-term cash rate (3-month T-bill, FRED ``DGS3MO``; fallback Yahoo
  ``^IRX``) is lagged one observation to respect publication timing.

Nothing in this module fabricates data. If a source is unavailable and no cache
exists, :class:`DataUnavailableError` is raised.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lab_config import DEFAULT_CACHE_DIR, ResearchConfig

RAW_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]


class DataUnavailableError(RuntimeError):
    """Raised when neither a download nor a cached copy is available."""


# ----------------------------------------------------------------------------
# Containers
# ----------------------------------------------------------------------------
@dataclass
class MarketData:
    """Aligned, adjusted market data for a set of tickers.

    ``bars[ticker]`` is indexed by the master calendar and holds adjusted
    ``Open, High, Low, Close, Volume`` plus ``RawClose, Dividends, Splits,
    AdjFactor, listed, tradable``. Rows before listing are NaN and
    ``listed == False``.
    """

    bars: dict[str, pd.DataFrame]
    calendar: pd.DatetimeIndex
    cash_yield: pd.Series          # annualised decimal yield known at each date (already lagged)
    cash_growth: pd.Series         # growth factor - 1 accrued from previous bar to this bar
    validation: pd.DataFrame
    source: str
    manifest: dict = field(default_factory=dict)

    def tickers(self) -> list[str]:
        return list(self.bars)

    def first_valid(self, ticker: str) -> pd.Timestamp:
        b = self.bars[ticker]
        return b.index[b["listed"].to_numpy()].min()

    def subset(self, tickers: list[str]) -> "MarketData":
        return MarketData({t: self.bars[t] for t in tickers if t in self.bars}, self.calendar,
                          self.cash_yield, self.cash_growth, self.validation, self.source, self.manifest)


# ----------------------------------------------------------------------------
# Download / cache
# ----------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    safe = ticker.replace("^", "IDX_").replace("/", "_")
    return cache_dir / "raw" / f"{safe}.csv.gz"


def download_yahoo(ticker: str, start: str, end_exclusive: str, retries: int = 3) -> pd.DataFrame:
    """Download raw daily bars from Yahoo Finance (unadjusted for dividends)."""
    import yfinance as yf  # imported lazily so offline tests do not need it

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end_exclusive, auto_adjust=False, actions=True,
                             progress=False, multi_level_index=False, threads=False)
            if df is not None and not df.empty:
                df = df.copy()
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                for c in RAW_COLUMNS:
                    if c not in df.columns:
                        df[c] = 0.0 if c in ("Dividends", "Stock Splits") else np.nan
                return df[RAW_COLUMNS].sort_index()
            last_exc = RuntimeError("empty frame")
        except Exception as exc:  # network errors, rate limits
            last_exc = exc
        time.sleep(2 ** attempt)
    raise DataUnavailableError(f"Yahoo download failed for {ticker}: {last_exc}")


def download_fred(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        raw = pd.read_csv(url)
    except Exception as exc:
        raise DataUnavailableError(f"FRED download failed for {series_id}: {exc}") from exc
    s = pd.Series(pd.to_numeric(raw.iloc[:, 1], errors="coerce").to_numpy(),
                  index=pd.to_datetime(raw.iloc[:, 0], errors="coerce"), name=series_id)
    return s.dropna().sort_index()


def fetch_raw(ticker: str, cfg: ResearchConfig, cache_dir: Path = DEFAULT_CACHE_DIR,
              refresh: bool = False, offline: bool = False) -> pd.DataFrame:
    """Return raw bars for ``ticker`` from cache, downloading if needed."""
    path = _cache_path(cache_dir, ticker)
    if path.exists() and not refresh:
        return read_raw_csv(path)
    if offline:
        raise DataUnavailableError(f"{ticker}: no cached data at {path} and offline mode is on")
    df = download_yahoo(ticker, cfg["data_start"], cfg["data_end_exclusive"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="Date", compression="gzip")
    _update_manifest(cache_dir, ticker, path, source="yahoo:yfinance(auto_adjust=False,actions=True)")
    return df


def read_raw_csv(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "Date"
    for c in RAW_COLUMNS:
        if c not in df.columns:
            if c == "Adj Close":
                df[c] = df["Close"]
            elif c in ("Dividends", "Stock Splits"):
                df[c] = 0.0
            else:
                raise ValueError(f"{path}: missing column {c}")
    return df[RAW_COLUMNS].sort_index()


def import_csv(ticker: str, src: Path | str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Import a user-supplied daily CSV (Date, Open, High, Low, Close[, Adj Close, Volume,
    Dividends, Stock Splits]) into the cache so the lab can run on it."""
    df = read_raw_csv(src)
    path = _cache_path(cache_dir, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="Date", compression="gzip")
    _update_manifest(cache_dir, ticker, path, source=f"user_csv:{Path(src).name}")
    return path


def _update_manifest(cache_dir: Path, key: str, path: Path, source: str) -> None:
    mpath = cache_dir / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest[key] = {
        "file": str(path.relative_to(cache_dir)),
        "sha256": _sha256_file(path),
        "source": source,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def fetch_cash_rate(cfg: ResearchConfig, cache_dir: Path = DEFAULT_CACHE_DIR, refresh: bool = False,
                    offline: bool = False) -> tuple[pd.Series, str]:
    """Annualised 3-month T-bill yield in percent, from cache, FRED or Yahoo ^IRX."""
    path = cache_dir / "cash" / "cash_rate.csv.gz"
    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df.iloc[:, 0].dropna(), str(df.columns[0])
    if offline:
        raise DataUnavailableError(f"No cached cash rate at {path} and offline mode is on")
    errors = []
    for src in (cfg["cash"]["primary_source"], cfg["cash"]["fallback_source"]):
        kind, code = src.split(":", 1)
        try:
            if kind == "FRED":
                s = download_fred(code)
            else:
                s = download_yahoo(code, cfg["data_start"], cfg["data_end_exclusive"])["Close"].dropna()
            s.name = src
            path.parent.mkdir(parents=True, exist_ok=True)
            s.to_frame().to_csv(path, index_label="Date", compression="gzip")
            _update_manifest(cache_dir, "cash_rate", path, source=src)
            return s, src
        except DataUnavailableError as exc:
            errors.append(str(exc))
    raise DataUnavailableError("Cash rate unavailable: " + " | ".join(errors))


# ----------------------------------------------------------------------------
# Adjustment and validation
# ----------------------------------------------------------------------------
def adjust_bars(raw: pd.DataFrame) -> pd.DataFrame:
    """Build total-return adjusted OHLC from raw bars."""
    df = raw.copy()
    factor = (df["Adj Close"] / df["Close"]).astype(float)
    out = pd.DataFrame(index=df.index)
    for c in ("Open", "High", "Low", "Close"):
        out[c] = df[c].astype(float) * factor
    out["Volume"] = df["Volume"].astype(float)
    out["RawClose"] = df["Close"].astype(float)
    out["Dividends"] = df["Dividends"].fillna(0.0).astype(float)
    out["Splits"] = df["Stock Splits"].fillna(0.0).astype(float)
    out["AdjFactor"] = factor
    return out


def validate_bars(ticker: str, raw: pd.DataFrame, adj: pd.DataFrame, as_of: pd.Timestamp,
                  is_stock: bool) -> list[dict]:
    """Return a list of validation findings (severity, check, detail, count)."""
    rows: list[dict] = []

    def add(sev, check, detail, count=0):
        rows.append({"ticker": ticker, "severity": sev, "check": check, "detail": detail, "count": int(count)})

    idx = raw.index
    if not idx.is_monotonic_increasing:
        add("error", "monotonic_index", "dates not increasing")
    if idx.has_duplicates:
        add("error", "duplicate_dates", "duplicate dates", idx.duplicated().sum())
    if (idx > as_of).any():
        add("error", "future_dates", f"rows after as-of {as_of.date()}", (idx > as_of).sum())
    px = raw[["Open", "High", "Low", "Close", "Adj Close"]]
    nan_rows = px.isna().any(axis=1).sum()
    if nan_rows:
        add("warning", "missing_values", "rows with NaN OHLC (dropped)", nan_rows)
    nonpos = (px <= 0).any(axis=1).sum()
    if nonpos:
        add("error", "non_positive_price", "non-positive prices", nonpos)
    tol = 1e-6
    hi_bad = (raw["High"] < raw[["Open", "Close"]].max(axis=1) * (1 - 1e-4) - tol).sum()
    lo_bad = (raw["Low"] > raw[["Open", "Close"]].min(axis=1) * (1 + 1e-4) + tol).sum()
    if hi_bad or lo_bad:
        add("warning", "ohlc_consistency", f"High<max(O,C): {hi_bad}; Low>min(O,C): {lo_bad}", hi_bad + lo_bad)
    zero_vol = (raw["Volume"] <= 0).sum()
    if zero_vol:
        add("info", "zero_volume", "days with zero/negative volume", zero_vol)
    ret = adj["Close"].pct_change()
    limit = 0.40 if is_stock else 0.25
    extreme = (ret.abs() > limit).sum()
    if extreme:
        dates = ", ".join(str(d.date()) for d in ret.index[ret.abs() > limit][:5])
        add("warning", "extreme_return", f"|daily return|>{limit:.0%} on {dates}", extreme)
    stale = (adj["Close"].diff() == 0).astype(int)
    runs = stale.groupby((stale == 0).cumsum()).sum().max() if len(stale) else 0
    if runs >= 5:
        add("warning", "stale_prices", f"longest run of unchanged closes: {runs}", runs)
    # Gap consistency: open/prev-close ratio identical in raw and adjusted terms except on ex-dates.
    raw_gap = raw["Open"] / raw["Close"].shift(1)
    adj_gap = adj["Open"] / adj["Close"].shift(1)
    ex = raw["Dividends"].fillna(0) > 0
    mism = ((raw_gap - adj_gap).abs() > 1e-6) & ~ex & raw_gap.notna()
    if mism.sum():
        add("warning", "adjustment_gap_mismatch", "adjusted vs raw overnight gap differ on non-ex-dates", mism.sum())
    # Dividend consistency: adjusted close return should equal (Close+Div)/prevClose - 1 on ex-dates.
    if ex.any():
        r_adj = adj["Close"] / adj["Close"].shift(1) - 1
        r_tot = (raw["Close"] + raw["Dividends"]) / raw["Close"].shift(1) - 1
        r_alt = raw["Close"] / (raw["Close"].shift(1) - raw["Dividends"]) - 1
        bad = ex & ((r_adj - r_tot).abs() > 0.0025) & ((r_adj - r_alt).abs() > 0.0025)
        add("info", "dividends", f"{int(ex.sum())} ex-dividend dates", ex.sum())
        if bad.sum():
            add("warning", "dividend_adjustment", "adjusted return inconsistent with dividend on ex-date", bad.sum())
    # Adjustment factor must be non-decreasing through time (each distribution lowers past prices).
    f = adj["AdjFactor"]
    drops = (f.diff() < -1e-6).sum()
    if drops:
        add("warning", "adj_factor_non_monotone", "dividend adjustment factor decreases", drops)
    # Splits: split-adjusted Close should not jump by the split ratio.
    sp = raw["Stock Splits"].fillna(0) > 0
    if sp.any():
        rr = raw["Close"].pct_change()
        jumps = (sp & (rr.abs() > 0.3)).sum()
        add("info", "splits", f"{int(sp.sum())} split events", sp.sum())
        if jumps:
            add("error", "unadjusted_split", "price jump >30% on split date", jumps)
    return rows


# ----------------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------------
STOCK_TICKERS = {"NVDA", "AMD", "MU", "AVGO", "TSM", "ASML", "AMAT", "LRCX", "KLAC", "INTC"}


def _cash_series(rate_pct: pd.Series, calendar: pd.DatetimeIndex, lag_obs: int) -> tuple[pd.Series, pd.Series]:
    known = (rate_pct.sort_index().shift(lag_obs) / 100.0)
    y = known.reindex(known.index.union(calendar)).ffill().reindex(calendar)
    if y.isna().any():
        first_ok = y.first_valid_index()
        if first_ok is None:
            raise DataUnavailableError("Cash-rate history does not overlap the price calendar")
        y = y.bfill()  # only affects dates before the first published rate; reported in validation
    days = pd.Series(calendar, index=calendar).diff().dt.days.fillna(0).to_numpy()
    growth = np.zeros(len(calendar))
    growth[1:] = (1 + y.to_numpy()[:-1]) ** (days[1:] / 365.0) - 1
    return y.rename("cash_yield"), pd.Series(growth, index=calendar, name="cash_growth")


def assemble(raw_by_ticker: dict[str, pd.DataFrame], cash_rate_pct: pd.Series | None, cfg: ResearchConfig,
             source: str, calendar_ticker: str = "SPY", manifest: dict | None = None) -> MarketData:
    as_of = pd.Timestamp(cfg["as_of_date"])
    findings: list[dict] = []
    adjusted: dict[str, pd.DataFrame] = {}
    for t, raw in raw_by_ticker.items():
        raw = raw[raw.index <= as_of]
        raw = raw[~raw.index.duplicated(keep="last")].sort_index()
        adj = adjust_bars(raw)
        rows = validate_bars(t, raw, adj, as_of, t in STOCK_TICKERS)
        findings += rows
        if any(r["severity"] == "error" for r in rows):
            continue
        adjusted[t] = adj.dropna(subset=["Open", "High", "Low", "Close"])
    if calendar_ticker not in adjusted:
        raise DataUnavailableError(f"Calendar ticker {calendar_ticker} unavailable or failed validation")
    calendar = adjusted[calendar_ticker].index
    bars: dict[str, pd.DataFrame] = {}
    for t, adj in adjusted.items():
        first, last = adj.index.min(), adj.index.max()
        re = adj.reindex(calendar)
        listed = (calendar >= first) & (calendar <= last)
        missing = listed & re["Close"].isna().to_numpy()
        if missing.any():
            findings.append({"ticker": t, "severity": "warning", "check": "calendar_gaps",
                             "detail": "in-life master-calendar days missing; filled flat and marked non-tradable",
                             "count": int(missing.sum())})
            c = re["Close"].ffill()
            for col in ("Open", "High", "Low"):
                re[col] = re[col].where(~missing, c)
            re["Close"] = re["Close"].where(~missing, c)
            re["Volume"] = re["Volume"].where(~missing, 0.0)
            re[["Dividends", "Splits"]] = re[["Dividends", "Splits"]].fillna(0.0)
            re[["RawClose", "AdjFactor"]] = re[["RawClose", "AdjFactor"]].ffill()
        extra = adj.index.difference(calendar)
        if len(extra):
            findings.append({"ticker": t, "severity": "info", "check": "off_calendar_rows",
                             "detail": "rows on dates absent from master calendar dropped", "count": len(extra)})
        re["listed"] = listed
        re["tradable"] = listed & ~missing
        bars[t] = re
    if cash_rate_pct is None:
        if not cfg["cash"]["allow_zero_cash_if_unavailable"]:
            raise DataUnavailableError("Cash rate missing and zero-cash fallback disabled")
        cash_rate_pct = pd.Series(0.0, index=calendar)
        findings.append({"ticker": "CASH", "severity": "warning", "check": "cash_rate",
                         "detail": "cash rate unavailable; cash earns 0%", "count": 0})
    else:
        first_rate = cash_rate_pct.dropna().index.min()
        if first_rate > calendar[0]:
            findings.append({"ticker": "CASH", "severity": "warning", "check": "cash_rate_backfill",
                             "detail": f"rate starts {first_rate.date()}; earlier dates use first value",
                             "count": int((calendar < first_rate).sum())})
    y, g = _cash_series(cash_rate_pct, calendar, int(cfg["cash"]["publication_lag_days"]))
    val = pd.DataFrame(findings, columns=["ticker", "severity", "check", "detail", "count"])
    return MarketData(bars, calendar, y, g, val, source, manifest or {})


def load_market_data(cfg: ResearchConfig, tickers: list[str] | None = None,
                     cache_dir: Path = DEFAULT_CACHE_DIR, refresh: bool = False,
                     offline: bool = False, skip_missing: bool = True) -> MarketData:
    """Load (download or cache) every ticker and assemble validated MarketData."""
    tickers = tickers or cfg.all_tickers()
    raw: dict[str, pd.DataFrame] = {}
    failures: list[dict] = []
    for t in tickers:
        try:
            raw[t] = fetch_raw(t, cfg, cache_dir, refresh=refresh, offline=offline)
        except DataUnavailableError as exc:
            if not skip_missing:
                raise
            failures.append({"ticker": t, "severity": "error", "check": "unavailable", "detail": str(exc), "count": 0})
    if not raw:
        raise DataUnavailableError("No price data could be loaded. " + "; ".join(f["detail"] for f in failures[:3]))
    rate, src = fetch_cash_rate(cfg, cache_dir, refresh=refresh, offline=offline)
    mpath = cache_dir / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    md = assemble(raw, rate, cfg, source=f"cache:{cache_dir} cash:{src}", manifest=manifest)
    if failures:
        md.validation = pd.concat([md.validation, pd.DataFrame(failures)], ignore_index=True)
    return md


def data_inventory(md: MarketData) -> pd.DataFrame:
    rows = []
    for t, b in md.bars.items():
        live = b[b["listed"]]
        rows.append({"ticker": t, "first_date": live.index.min().date(), "last_date": live.index.max().date(),
                     "bars": len(live), "non_tradable_days": int((~b["tradable"] & b["listed"]).sum()),
                     "dividend_events": int((live["Dividends"] > 0).sum()),
                     "split_events": int((live["Splits"] > 0).sum()),
                     "first_close_adj": float(live["Close"].iloc[0]), "last_close_adj": float(live["Close"].iloc[-1])})
    return pd.DataFrame(rows)
