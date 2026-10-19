"""Data adjustment/validation, analytics, multiple-testing and oracle tests."""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import portfolio_analytics as pa
import strategies as S
from backtest_engine import backtest
from data_pipeline import adjust_bars, assemble, validate_bars
from oracle_benchmark import dp_oracle_path, interval_returns, path_to_target, daily_oracle_path


def test_dividend_adjustment_gives_total_return(raw_cash):
    raw = raw_cash[0]["SPY"]
    adj = adjust_bars(raw)
    ex = raw.index[raw["Dividends"] > 0]
    assert len(ex) > 10
    for d in ex[:20]:
        i = raw.index.get_loc(d)
        tr = (raw["Close"].iloc[i] + raw["Dividends"].iloc[i]) / raw["Close"].iloc[i - 1] - 1
        alt = raw["Close"].iloc[i] / (raw["Close"].iloc[i - 1] - raw["Dividends"].iloc[i]) - 1
        ar = adj["Close"].iloc[i] / adj["Close"].iloc[i - 1] - 1
        assert min(abs(ar - tr), abs(ar - alt)) < 1e-9
        # Open-to-open across the ex-date includes the distribution (no artificial loss).
        oo_adj = adj["Open"].iloc[i] / adj["Open"].iloc[i - 1]
        oo_raw = raw["Open"].iloc[i] / raw["Open"].iloc[i - 1]
        assert oo_adj > oo_raw


def test_validation_flags_errors(raw_cash):
    raw = raw_cash[0]["NVDA"].copy()
    sp = raw.index[raw["Stock Splits"] > 0][0]
    bad = raw.copy()
    bad.loc[bad.index >= sp, ["Open", "High", "Low", "Close", "Adj Close"]] /= 4.0   # unadjusted split
    rows = validate_bars("NVDA", bad, adjust_bars(bad), pd.Timestamp("2026-10-18"), True)
    assert any(r["check"] == "unadjusted_split" and r["severity"] == "error" for r in rows)
    fut = raw.copy()
    fut.index = fut.index + pd.Timedelta(days=365 * 20)
    rows = validate_bars("NVDA", fut, adjust_bars(fut), pd.Timestamp("2026-10-18"), True)
    assert any(r["check"] == "future_dates" for r in rows)


def test_calendar_gap_marked_non_tradable(cfg, raw_cash):
    raw, cash = raw_cash
    q = raw["QQQ"].drop(raw["QQQ"].index[500:503])
    md = assemble({"SPY": raw["SPY"], "QQQ": q}, cash, cfg, source="t")
    b = md.bars["QQQ"]
    assert (~b["tradable"]).sum() == 3 and b["listed"].iloc[501]
    assert (md.validation["check"] == "calendar_gaps").any()


def test_basic_metrics():
    r = np.array([0.1, -0.5, 0.2, 0.0])
    assert pa.max_drawdown(r) == pytest.approx(-0.5)
    assert pa.max_dd_duration(r) == 3
    x = np.random.default_rng(0).normal(0.0004, 0.01, 5000)
    assert pa.sharpe(x) == pytest.approx(x.mean() / x.std(ddof=1) * np.sqrt(252))
    assert pa.cagr(np.zeros(252)) == 0.0


def test_bh_fdr_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    rej, q = pa.bh_fdr(p, 0.05)
    assert np.allclose(q, [0.04, 0.0533333, 0.0533333, 0.2], atol=1e-6)
    assert rej.tolist() == [True, False, False, False]


def test_dsr_penalises_many_trials():
    rng = np.random.default_rng(1)
    R = rng.normal(0, 0.01, (2000, 50))
    R[:, 0] += 0.0006
    d = pa.deflated_sharpe(R, np.zeros(2000), [f"s{i}" for i in range(50)])
    assert d.loc[0, "dsr"] < d.loc[0, "psr_vs_0"]


def test_reality_check_null_not_rejected():
    rng = np.random.default_rng(2)
    D = rng.normal(0, 0.01, (1500, 30))
    idx = pa.stationary_bootstrap_indices(1500, 300, 10, 3)
    res = pa.reality_check(D, idx)
    assert res["rc_pvalue"] > 0.05 and res["spa_pvalue"] > 0.05
    D[:, 4] += 0.003
    res = pa.reality_check(D, idx)
    assert res["spa_pvalue"] < 0.05 and res["best_index"] == 4


def test_stationary_bootstrap_shape():
    idx = pa.stationary_bootstrap_indices(100, 7, 5, 0)
    assert idx.shape == (7, 100) and idx.min() >= 0 and idx.max() < 100


def test_oracle_dp_matches_bruteforce():
    rng = np.random.default_rng(4)
    for _ in range(100):
        T = int(rng.integers(2, 9))
        a, c = rng.normal(0, 0.02, T), np.full(T, 1e-4)
        cost = float(rng.choice([0.0, 0.001, 0.01]))
        K = rng.choice([None, 1, 2])
        best = -np.inf
        for p in itertools.product([0, 1], repeat=T):
            s, k, lw = 0, 0, 0.0
            for i, x in enumerate(p):
                if x != s:
                    lw += np.log(1 - cost)
                    k += x
                s = x
                lw += np.log1p(a[i] if x else c[i])
            if K is None or k <= K:
                best = max(best, lw)
        path, v = dp_oracle_path(a, c, cost, K)
        assert v == pytest.approx(best, abs=1e-10)
        # Reconstructed path achieves the reported value.
        s, lw = 0, 0.0
        for i, x in enumerate(path):
            if x != s:
                lw += np.log(1 - cost)
            s = x
            lw += np.log1p(a[i] if x else c[i])
        assert lw == pytest.approx(v, abs=1e-10)


def test_oracle_through_engine_dominates(md):
    a, c, d = interval_returns(md, "SPY")
    tgt = path_to_target(md, "SPY", daily_oracle_path(a, c), d)
    r = backtest(md, tgt, 0.0, band=0.0)
    bh = backtest(md, md.bars["SPY"]["listed"].astype(float).rename("SPY"), 0.0)
    assert r.equity.iloc[-1] > bh.equity.iloc[-1]
    # Costless daily oracle never loses money between consecutive opens.
    eo = r.equity_open.to_numpy()
    assert (np.diff(eo[2:]) >= -1e-6 * eo[2:-1]).all()


def test_registry_frozen_and_complete():
    S.validate_registry()
    lab = Path(__file__).resolve().parents[1]
    freeze = json.loads((lab / "registry_freeze.json").read_text())
    assert freeze["registry_sha256"] == S.registry_hash(), "registry changed after freeze - re-freeze deliberately and document why"
    df = pd.read_csv(lab / "Strategy_Registry.csv")
    assert len(df) == 100 and df["id"].is_unique
    assert df.groupby("family").size().eq(10).all()
    for col in ("hypothesis", "entry", "exit", "params", "allocation", "min_lookback", "benchmark", "regime", "config_json"):
        assert df[col].notna().all()


def test_original_six_present():
    origin = {s.id for s in S.REGISTRY if s.origin == "original_notebook"}
    assert origin == {"F01_01", "F01_02", "F02_01", "F02_07", "F05_09", "F03_10"}
