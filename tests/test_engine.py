"""Execution timing, transaction-cost and accounting tests for the backtest engine."""
import numpy as np
import pandas as pd
import pytest

import strategies as S
from backtest_engine import backtest, buy_and_hold_target, ledger_reconcile, simulate, simulate_single
from conftest import make_flat_md


def test_signal_executes_at_next_open(cfg):
    n = 30
    opens = 100 + np.arange(n) * 1.0
    closes = opens + 0.5
    md = make_flat_md(cfg, n, opens=opens, closes=closes)
    tgt = pd.Series(0.0, index=md.calendar, name="SPY")
    tgt.iloc[10:] = 1.0              # decided at the close of bar 10
    r = backtest(md, tgt, 10.0, record_trades=True)
    tr = r.trades
    assert len(tr) == 1
    assert tr["exec_date"].iloc[0] == md.calendar[11]
    assert tr["signal_date"].iloc[0] == md.calendar[10]
    assert tr["price"].iloc[0] == pytest.approx(opens[11])      # filled at the open, not the signal close
    assert r.exposure.iloc[10] == 0.0 and r.exposure.iloc[11] > 0.99


def test_entry_day_earns_open_to_close_including_gap(cfg):
    n = 20
    opens = np.full(n, 100.0)
    closes = np.full(n, 100.0)
    opens[11], closes[11] = 105.0, 110.0      # overnight gap up on the fill day
    md = make_flat_md(cfg, n, opens=opens, closes=closes)
    tgt = pd.Series(0.0, index=md.calendar, name="SPY")
    tgt.iloc[10:] = 1.0
    r = backtest(md, tgt, 0.0)
    # The gap (100 -> 105) is missed because the fill is at the open; only 105 -> 110 is earned.
    assert r.returns.iloc[11] == pytest.approx(110 / 105 - 1)


@pytest.mark.parametrize("bps", [0.0, 5.0, 10.0, 50.0])
def test_round_trip_cost_exact(cfg, bps):
    md = make_flat_md(cfg, 30)
    tgt = pd.Series(0.0, index=md.calendar, name="SPY")
    tgt.iloc[5:15] = 1.0
    r = backtest(md, tgt, bps, capital=100_000.0, record_trades=True)
    c = bps / 1e4
    notional = 100_000.0 / (1 + c)
    assert r.equity.iloc[-1] == pytest.approx(100_000.0 - 2 * c * notional, rel=1e-12)
    assert r.trades["cost"].sum() == pytest.approx(2 * c * notional, rel=1e-12)
    assert (r.equity_open.fillna(0) >= -1e-9).all() if r.equity_open is not None else True


def test_cash_accrues_lagged_rate(cfg):
    md = make_flat_md(cfg, 30, cash_pct=5.0)
    r = backtest(md, pd.Series(0.0, index=md.calendar, name="SPY"), 10.0)
    g = md.cash_growth.to_numpy()
    assert r.equity.iloc[-1] == pytest.approx(100_000.0 * np.prod(1 + g))
    assert g[0] == 0.0


def test_no_leverage_or_short(cfg):
    md = make_flat_md(cfg, 10)
    with pytest.raises(ValueError):
        backtest(md, pd.Series(1.5, index=md.calendar, name="SPY"))
    with pytest.raises(ValueError):
        backtest(md, pd.Series(-0.5, index=md.calendar, name="SPY"))


def test_cash_never_negative(md):
    rng = np.random.default_rng(3)
    cols = ["SPY", "QQQ", "XLK"]
    w = pd.DataFrame(rng.dirichlet(np.ones(4), len(md.calendar))[:, :3], index=md.calendar, columns=cols)
    keep = pd.Series(rng.random(len(w)) > 0.7, index=w.index)      # change whole rows so each row sums to <= 1
    w = w.where(keep, axis=0).ffill().fillna(0.0)
    r = backtest(md, w, 25.0, record_trades=True, band=0.05)
    assert (r.cash >= -1e-6).all()
    assert (r.exposure <= 1 + 1e-9).all()


def test_fast_path_matches_generic(md):
    rng = np.random.default_rng(5)
    b = md.bars["SPY"]
    w = pd.Series(np.round(rng.random(len(md.calendar)) * 4) / 4, index=md.calendar)
    w = w.where(rng.random(len(w)) > 0.9).ffill().fillna(0.0) * b["listed"]
    args = (b["Open"].to_numpy(), b["Close"].to_numpy(), b["tradable"].to_numpy(), w.to_numpy(), md.cash_growth.to_numpy(), 0.001, 1e5, 0.1)
    eq_fast, *_ = simulate_single(*args)
    gen = simulate(args[0][:, None], args[1][:, None], args[2][:, None], args[3][:, None], args[4], *args[5:])
    assert np.allclose(eq_fast, gen["eq_close"], rtol=1e-12)


@pytest.mark.parametrize("sid", ["F01_01", "F02_07", "F08_06", "F07_03"])
def test_ledger_reconciles_with_equity(md, aux, sid):
    w = S.generate_weights(S.get_spec(sid), md, "QQQ", aux)
    r = backtest(md, w, 10.0, record_trades=True)
    rc = ledger_reconcile(r, md, 100_000.0)
    assert (rc["abs_error"] / rc["engine_equity"]).max() < 1e-9


def test_rotation_ledger_reconciles(md, aux):
    tick = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]
    w = S.generate_weights(S.get_spec("F09_10"), md, tick, aux)
    r = backtest(md, w, 10.0, record_trades=True)
    rc = ledger_reconcile(r, md, 100_000.0)
    assert (rc["abs_error"] / rc["engine_equity"]).max() < 1e-9
    assert (r.exposure <= 1 + 1e-9).all()


def test_buy_and_hold_matches_price_path(md):
    r = backtest(md, buy_and_hold_target(md, "SPY"), 0.0)
    c = md.bars["SPY"]["Close"]
    o = md.bars["SPY"]["Open"]
    # After the entry at the second bar's open, equity tracks the adjusted close exactly.
    ratio = r.equity.iloc[2:] / c.iloc[2:]
    assert np.allclose(ratio, ratio.iloc[0])
    assert r.equity.iloc[1] == pytest.approx(100_000.0 * c.iloc[1] / o.iloc[1] * (1 + md.cash_growth.iloc[1]), rel=1e-9)


def test_costs_reduce_returns_monotonically(md, aux):
    w = S.generate_weights(S.get_spec("F01_02"), md, "QQQ", aux)
    finals = [backtest(md, w, c).equity.iloc[-1] for c in (0, 5, 10, 20, 50)]
    assert all(a > b for a, b in zip(finals, finals[1:]))
    r = backtest(md, w, 10.0)
    assert (r.gross_returns >= r.returns - 1e-15).all()
