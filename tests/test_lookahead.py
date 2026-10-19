"""Look-ahead tests: weights decided at bar t must not change when data after t changes."""
import numpy as np
import pandas as pd
import pytest

import indicators as ta
import strategies as S
from data_pipeline import MarketData


def perturb_after(md: MarketData, cutoff: pd.Timestamp, seed: int = 0) -> MarketData:
    rng = np.random.default_rng(seed)
    bars = {}
    for t, b in md.bars.items():
        b = b.copy()
        m = b.index > cutoff
        k = m.sum()
        shock = np.exp(np.cumsum(rng.normal(0, 0.03, k)))
        for c in ("Open", "High", "Low", "Close"):
            b.loc[m, c] = b.loc[m, c] * shock * (1 + rng.normal(0, 0.002, k))
        b.loc[m, "High"] = b.loc[m, ["Open", "High", "Low", "Close"]].max(axis=1)
        b.loc[m, "Low"] = b.loc[m, ["Open", "High", "Low", "Close"]].min(axis=1)
        b.loc[m, "Volume"] = b.loc[m, "Volume"] * rng.uniform(0.2, 5, k)
        bars[t] = b
    y = md.cash_yield.copy()
    y[y.index > cutoff] = rng.uniform(0, 0.08)
    g = md.cash_growth.copy()
    g[g.index > cutoff] = rng.uniform(0, 3e-4)
    return MarketData(bars, md.calendar, y, g, md.validation, md.source)


CUTOFFS = ["2005-06-15", "2008-10-09", "2010-03-01"]


@pytest.mark.parametrize("spec", [s for s in S.REGISTRY], ids=lambda s: s.id)
def test_strategy_has_no_lookahead(md, aux, spec):
    target = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"] if spec.kind == "rotation" else "QQQ"
    base = S.generate_weights(spec, md, target, aux)
    for i, c in enumerate(CUTOFFS):
        cut = pd.Timestamp(c)
        md2 = perturb_after(md, cut, seed=i)
        w2 = S.generate_weights(spec, md2, target, aux)
        a = base.loc[:cut]
        b = w2.loc[:cut].reindex(columns=a.columns).fillna(0.0)
        assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-12), f"{spec.id} weights before {c} depend on later data"


def test_detector_catches_leak(md, aux):
    """Meta-test: a deliberately leaky rule (uses tomorrow's close) must be flagged."""
    c = md.bars["QQQ"]["Close"]
    leaky = (c.shift(-1) > c).astype(float)
    cut = pd.Timestamp(CUTOFFS[0])
    c2 = perturb_after(md, cut)["QQQ"]["Close"] if False else perturb_after(md, cut).bars["QQQ"]["Close"]
    leaky2 = (c2.shift(-1) > c2).astype(float)
    assert not np.allclose(leaky.loc[:cut].to_numpy(), leaky2.loc[:cut].to_numpy())


@pytest.mark.parametrize("fn,args", [("sma", (20,)), ("ema", (12,)), ("rsi", (14,)), ("realized_vol", (20,)),
                                     ("roc", (63,)), ("downside_vol", (60,)), ("ewma_vol", (0.94, 30))])
def test_close_indicators_causal(md, fn, args):
    c = md.bars["SPY"]["Close"]
    cut = pd.Timestamp("2006-01-03")
    c2 = c.copy()
    c2[c2.index > cut] *= 1.7
    a, b = getattr(ta, fn)(c, *args), getattr(ta, fn)(c2, *args)
    assert np.allclose(a.loc[:cut].fillna(-9).to_numpy(), b.loc[:cut].fillna(-9).to_numpy())


def test_rolling_extremes_exclude_current_bar():
    s = pd.Series([1, 2, 3, 10, 4, 5], dtype=float)
    hh = ta.rolling_max(s, 3)
    assert np.isnan(hh.iloc[2]) and hh.iloc[3] == 3 and hh.iloc[4] == 10
