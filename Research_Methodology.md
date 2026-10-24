# Research Methodology

This document fixes how every number in the lab is produced. The decision rules
for hypotheses H1–H10 were written before any real-data result was generated
(registry freeze: `registry_freeze.json`, 2026-10-19 04:40 UTC).

## 1. Data

| Item | Choice |
|---|---|
| Prices | Yahoo Finance daily bars via `yfinance` with `auto_adjust=False, actions=True`, downloaded once and cached under `data/cache/raw/` |
| Cut-off | Last bar 2026-10-16 (data as of 2026-10-18); rows after the as-of date are rejected by validation |
| Adjustment | `factor = Adj Close / Close`; Open, High, Low, Close are all multiplied by the same daily factor. Close is already split-adjusted by Yahoo, so the result is a split- and distribution-adjusted (total-return) OHLC series. Open-to-open returns across an ex-date therefore include the dividend |
| Cash | 3-month Treasury constant-maturity yield (FRED `DGS3MO`, fallback Yahoo `^IRX`), lagged one observation for publication timing, accrued as `(1+y)^(calendar days/365) - 1` between bars |
| Calendar | SPY's trading days. Other instruments are re-indexed to it; missing in-life days are filled flat and marked non-tradable |

Validation (`data_pipeline.validate_bars`) checks ordering, duplicates, future
dates, non-positive prices, OHLC consistency, extreme moves (>25% ETFs, >40%
stocks), stale prices, dividend consistency on ex-dates, adjustment-factor
monotonicity, unadjusted splits and calendar gaps. Errors block a ticker;
warnings are written to `outputs/audit/data_validation.csv`.

Yahoo rounds `Adj Close`, which makes the factor jitter by about 1e-6 on
ordinary days. The adjustment-consistency checks therefore use a 1e-5 relative
tolerance. On the real cache the largest non-ex-date discrepancy is 2.4e-6, so
all earlier warnings at the 1e-6 tolerance were rounding noise.

## 2. Execution model (`backtest_engine.py`)

1. A strategy returns target weights decided with data up to and including the
   **close** of bar *t*.
2. The engine executes them at the **open of bar *t+1***. The signal price and
   the fill price are always different bars, and the overnight gap is borne by
   the strategy. Stops are evaluated at the close and filled at the next open
   (no intraday stops).
3. Costs = (commission + slippage) bps × |traded notional|, per side. Base case
   10 bps; sensitivity 0 / 5 / 10 / 20 / 50 bps.
4. Long-only, unlevered: weights in [0, 1], sum ≤ 1. Buys are scaled so that
   cash never goes negative after costs.
5. Partial positions are rebalanced when any target changes or any drifted
   weight leaves a ±10 percentage-point band; a trigger rebalances the whole
   portfolio.
6. Fractional shares; the portfolio is marked at every close. Daily returns are
   close-to-close NAV changes (an entry day earns open-to-close).
7. Every trade is written to a ledger; `ledger_reconcile` rebuilds equity from
   the ledger and must match the engine to 1e-9 relative.

## 3. Strategy registry (`strategies.py`, `Strategy_Registry.csv`)

100 strategies, 10 families × 10. Each has an id, a hypothesis, entry/exit
rules, parameters, an allocation rule, minimum lookback, benchmark, expected
regime, a machine-readable config, and a small parameter neighbourhood (`grid`)
used only for sensitivity and walk-forward parameter selection. The original
notebook's six strategies are reproduced unchanged (`origin =
original_notebook`): F01_01, F01_02, F02_01, F02_07, F05_09, F03_10.

The registry hash is stored in `registry_freeze.json`; a test fails if the
registry changes without a deliberate re-freeze.

## 4. Universes and periods

| Universe | Instruments | Secondary benchmark |
|---|---|---|
| A broad ETFs | SPY QQQ DIA IWM MDY VTI RSP | SPY |
| B sector SPDRs | XLK XLF XLE XLV XLI XLY XLP XLU XLB XLC XLRE | SPY |
| C semiconductor ETFs | SMH SOXX XSD PSI | SPY |
| D semiconductor stocks | NVDA AMD MU AVGO TSM ASML AMAT LRCX KLAC INTC | SMH |
| E historical S&P 500 | **not run** | – |

Universe E needs point-in-time index membership and delisted-company prices.
Neither is available, so no S&P 500 constituent study is presented; using
today's constituents would be survivorship-biased.

Periods:

* **full**: each instrument from its first bar + 273-bar warm-up (the longest
  strategy lookback) to 2026-10-16; all strategies on an instrument share the
  same window.
* **common**: 2011-01-03 onward, for instruments with enough history (XLC and
  XLRE excluded).
* **prior_dev / prior_holdout (SPY)**: 2010-01-04 to 2021-09-28 and 2021-09-29
  to 2026-10-16, the split used in the original notebook. Both were inspected
  before this project and are **exploratory, not untouched**.

The only genuinely untouched sample is prospective: data after 2026-10-18.

## 5. Metrics (`portfolio_analytics.py`)

CAGR, total return, annualised volatility, Sharpe (excess over the lagged
T-bill accrual), Sortino, maximum drawdown, longest drawdown duration, Calmar,
time invested, average exposure, entries/exits, win rate, profit factor,
average holding period, annual turnover, gross vs net CAGR and cost drag,
alpha/beta vs buy-and-hold (Newey-West t-stats), tracking error, information
ratio, and return concentration (share of log return from the top 1% of days,
CAGR excluding the best 10 days, best 12-month share).

Benchmarks for every strategy × instrument:

1. buy-and-hold of the same instrument (primary);
2. the universe benchmark (SPY, or SMH for semiconductor stocks);
3. an **exposure-matched** static mix: the instrument at the strategy's
   average exposure, rest in cash;
4. a **volatility-matched** static mix: instrument weight = strategy vol /
   instrument vol.

Rotation strategies are compared with an equal-weight universe (monthly
rebalanced) and SPY.

Regime labels (H2, H4, H5) are ex-ante: for bar *t* the trailing 126-bar
return divided by its trailing volatility × √(126/252) is computed through
*t-1*; ≥ 0.5 = up, ≤ −0.5 = down, otherwise sideways.

## 6. Walk-forward (`walk_forward.py`)

Calendar-year folds: train on the previous five full calendar years, test on
the next calendar year, roll forward one year. Signals are causal and computed
once on the full history, so slicing returns introduces no leakage; all
*selection* uses training windows only. Analyses: fixed-rule IS vs OOS per
fold; parameter selection within each strategy's neighbourhood; best-of-
registry selection (stitched OOS record of the strategy with the best training
Sharpe); rank persistence (Spearman, top-decile retention); and sector-specific
vs common selection (H6).

## 7. Statistical inference

* Stationary bootstrap (Politis-Romano, mean block 20 days, 1,000 reps,
  seed 20261018): 90% CIs for Sharpe, CAGR and maximum drawdown.
* One-sided bootstrap p-values for the daily return differential vs buy-and-
  hold; Benjamini-Hochberg FDR (q = 10%) and Holm (5%) across all
  strategy × instrument tests.
* Probabilistic and deflated Sharpe ratios (Bailey & López de Prado), with the
  number of effective trials estimated from the average pairwise correlation.
* White's Reality Check and Hansen's SPA test per instrument, on raw
  differentials and on volatility-scaled differentials.

Cross-asset tests treat instruments as independent observations. They are
correlated, so these p-values are optimistic and are read as descriptive.

## 8. Perfect-knowledge oracle (`oracle_benchmark.py`)

The oracle decides at each market open whether to hold the asset or cash for
the interval open[*i*] → open[*i+1*], knowing that interval's return. It uses
the same execution points as the strategies, cannot short or use leverage, and
is run through the same engine. Variants: daily oracle (costless decision rule,
reported net of costs), cost-aware dynamic programme (maximises terminal
log-wealth with 10 bps per switch), and the same programme with at most *K*
entries for K ∈ {1, 2, 5, 10, 20, 40, 80, 160}. The DP is verified against
brute-force enumeration in the tests. The oracle is mathematically informative
and impossible to implement.

## 9. Hypothesis decision rules (fixed before results)

Verdicts: **Supported** (predicted sign, test passes at 10%), **Contradicted**
(opposite sign, test passes), **Not resolved** (otherwise).

| H | Test |
|---|---|
| H1 | Mean Sharpe of fast MA systems (F01_02, F01_04, F01_01) minus slow ones (F01_03, F01_05, F01_10); semiconductors (C+D) vs broad ETFs (A), Mann-Whitney |
| H2 | SMA10/30 vs SMA20/50 across instruments: drawdown difference (sign test) and sideways-regime excess return (sign test, predicted negative) |
| H3 | Pairs differing only in re-entry speed (F02_02 vs F02_01, F02_10 vs F01_02, F02_06 vs F01_06); Sharpe and drawdown, sign tests |
| H4 | Momentum family Sharpe minus buy-and-hold, semiconductors vs broad, Mann-Whitney; plus excess return in up-trend regimes |
| H5 | Mean-reversion Sharpe in sideways regimes minus trend regimes, per instrument, sign test |
| H6 | Walk-forward: per-sector best IS strategy vs the strategy with the best mean IS Sharpe across sectors, OOS Sharpe, t-test on fold means |
| H7 | Material = median Sharpe drop ≥ 0.10 from 0 to 10 bps, or ≥ 25% relative fall in the share beating buy-and-hold |
| H8 | Supported if top-decile IS→OOS retention < 50% and < 75% of IS top-decile land in the OOS upper half |
| H9 | Among combos with shallower drawdowns than buy-and-hold, supported if fewer than half also beat a volatility-matched static mix on both CAGR and drawdown |
| H10 | Hybrids (F10) vs median single-idea family on four robustness measures; supported if better on ≥ 3 of 4 |

## 10. Quality control (`tests/`, 138 tests)

* Look-ahead: for all 100 strategies, weights up to a cut-off must be identical
  after randomly perturbing all prices, volumes and rates after it (three
  cut-offs); a deliberately leaky rule must be caught.
* Execution: signal at close t fills at open t+1 at the open price; the entry
  day earns open-to-close only; exact round-trip cost; lagged cash accrual; no
  leverage or shorting; cash never negative; the fast path equals the general
  path.
* Accounting: ledger reconciliation for single-asset and rotation portfolios;
  buy-and-hold tracks the adjusted close.
* Data: dividend adjustment gives total return across ex-dates; validation
  catches unadjusted splits and future dates; calendar gaps are non-tradable.
* Statistics: known-value tests for drawdown, Sharpe, BH-FDR; DSR penalises
  many trials; Reality Check / SPA do not reject under the null and do reject a
  planted edge; oracle DP equals brute force.
* Registry: exactly 100 strategies, 10 per family, complete metadata, hash
  equals the freeze.

## 11. Limitations

* Yahoo daily bars are research-grade, not exchange-grade; opening-auction
  slippage and market impact are not modelled beyond the bps assumption.
* Fractional shares, no taxes, no borrowing.
* SMH's pre-2011 history is the HOLDRS product; semiconductor ETFs changed
  construction over time.
* Strategy ideas come from public literature and the original notebook, so no
  historical period is truly unseen; walk-forward selection is the main
  defence, prospective tracking the only clean test.
