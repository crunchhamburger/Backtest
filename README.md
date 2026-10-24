# Quantitative Research Lab

A reproducible backtesting laboratory that runs a frozen registry of 100
long-only trading strategies across broad-market ETFs, S&P 500 sector ETFs,
semiconductor ETFs and semiconductor stocks, then tests whether any apparent
edge survives realistic costs, walk-forward validation and multiple-testing
corrections.

It grew out of a Colab notebook that tested SMA20/50 and five follow-up
strategies on SPY. Those six strategies are reproduced exactly (ids F01_01,
F01_02, F02_01, F02_07, F05_09, F03_10) inside a unified, tested engine.

**Not investment advice.** No strategy here is recommended for live trading.

## Quick start

### Google Colab (easiest)
Open `Quant_Research_Lab.ipynb` in Colab and choose **Runtime → Run all**. It
clones this repo, runs the tests, downloads and validates data, runs every
experiment (about 40 minutes) and lets you download the reports.

### Local
```bash
pip install -r requirements.txt
python -m pytest -q tests          # 138 tests, about 30 s
python run_experiments.py          # downloads data on first run, then uses data/cache
```

Useful options:

```bash
python run_experiments.py --offline                       # never touch the network
python run_experiments.py --universes A_broad,C_semi_etfs # subset of universes
python run_experiments.py --strategies F01_01,F01_02      # subset of strategies
python run_experiments.py --cost-bps 20                   # change the base cost
python run_experiments.py --start 2005-01-01 --end 2020-12-31
python run_experiments.py --quick                         # 200 bootstrap reps
python run_experiments.py --report-only                   # rebuild reports from outputs/audit/results.pkl
python run_experiments.py --synthetic                     # software smoke test on simulated data (watermarked)
```

## Layout

| File | Role |
|---|---|
| `config/research_config.json` | Every research setting: dates, costs, cash source, universes, walk-forward, statistics, oracle |
| `lab_config.py` | Loads the config (with overrides) and hashes it |
| `data_pipeline.py` | Download, cache, total-return adjustment, validation, cash rate |
| `indicators.py` | Causal technical indicators |
| `strategies.py` | Signal functions and the 100-strategy registry |
| `backtest_engine.py` | Next-open execution, costs, share/cash accounting, trade ledger |
| `portfolio_analytics.py` | Metrics, bootstrap, deflated Sharpe, Reality Check / SPA, FDR |
| `walk_forward.py` | Rolling train/test folds, selection, rank persistence |
| `oracle_benchmark.py` | Perfect-knowledge long/cash dynamic programme |
| `hypotheses.py` | Pre-registered tests for H1–H10 |
| `reporting.py` | Excel workbooks, figures, PDF reports |
| `run_experiments.py` | Orchestrator / command-line interface |
| `synthetic_data.py` | Simulated data for tests only |
| `tests/` | Look-ahead, timing, cost, accounting, data, statistics and registry tests |
| `Strategy_Registry.csv`, `registry_freeze.json` | The frozen registry and its hash |
| `Research_Methodology.md` | Full methodology and decision rules |

## Outputs (`outputs/`)

* `Strategy_Leaderboard.xlsx`: every strategy × instrument (full and common
  periods) with filters, plus family summary, cost sensitivity, parameter
  sensitivity, statistical tests, deflated Sharpe, bootstrap CIs, Reality
  Check / SPA, correlations, hypotheses, registry and data validation.
* `Sector_Performance.xlsx`, `Semiconductor_Strategy_Results.xlsx`: leaderboards
  and colour-coded strategy × instrument heatmaps.
* `Walk_Forward_Results.xlsx`: fold definitions, IS vs OOS per fold, rank
  persistence, best-of-registry and parameter-selection records.
* `Quant_Research_Report.pdf`, `SPY_Strategy_Comparison.pdf`.
* `figures/`: all charts.
* `audit/`: trade logs, SPY equity curves, ledger reconciliation, data
  inventory and validation, long-format metric tables.

## Extending

* **Add a ticker:** add it to a universe in `config/research_config.json`.
* **Add a strategy:** write a signal function in `strategies.py` that returns
  close-decided target weights, add a `StrategySpec` to `REGISTRY`, then
  deliberately re-freeze (`python -c "import strategies, pathlib; strategies.freeze_registry(pathlib.Path('.'))"`)
  and note why. The look-ahead test runs on it automatically.
* **Change costs / periods / universes:** edit the config or use the CLI
  options above. Every output records the config hash.

## Data

Price data are downloaded from Yahoo Finance by the user's own run and cached
locally in `data/cache/` (not committed). Cash rates come from FRED `DGS3MO`.
