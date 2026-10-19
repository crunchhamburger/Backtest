"""Experiment orchestrator: data -> 100 strategies x universes -> analytics -> reports.

Usage (from the ``quant_research_lab`` directory)::

    python run_experiments.py                    # full run with cached/downloaded data
    python run_experiments.py --offline          # cache only, never touch the network
    python run_experiments.py --universes A_broad --strategies F01_01,F01_02
    python run_experiments.py --cost-bps 20      # change the base cost assumption
    python run_experiments.py --start 2005-01-01 --end 2020-12-31
    python run_experiments.py --synthetic        # software smoke test (watermarked output)

Every stage writes its products to ``outputs/`` (or ``--out``). Intermediate
results are pickled to ``outputs/audit/results.pkl`` so that reports can be
rebuilt with ``python run_experiments.py --report-only``.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))

import portfolio_analytics as pa  # noqa: E402
import strategies as S  # noqa: E402
import walk_forward as wf  # noqa: E402
from backtest_engine import (CostModel, backtest, buy_and_hold_target, equal_weight_target,  # noqa: E402
                             ledger_reconcile, legacy_notebook_sma2050, NOTEBOOK_REFERENCE, static_mix_target)
from data_pipeline import MarketData, assemble, data_inventory, load_market_data  # noqa: E402
from lab_config import DEFAULT_CACHE_DIR, DEFAULT_OUTPUT_DIR, ResearchConfig, load_config  # noqa: E402
from oracle_benchmark import missed_vs_avoided, run_oracles  # noqa: E402

ORIGINAL_SIX = ["F01_01", "F01_02", "F02_01", "F02_07", "F05_09", "F03_10"]
_G: dict = {}  # globals shared with forked workers


# ============================================================================
# Helpers
# ============================================================================

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def aux_map(cfg: ResearchConfig) -> dict:
    a = cfg["aux_tickers"]
    return {"market": a["market"], "defensive": a["defensive"], "risk_on_num": a["risk_on_pair"][0],
            "risk_on_den": a["risk_on_pair"][1], "breadth_list": list(a["breadth"]), "semis_leader": a["semis_leader"]}


def universe_of(cfg: ResearchConfig) -> dict[str, str]:
    m = {}
    for u in cfg.enabled_universes():
        for t in cfg.universe_tickers(u):
            m.setdefault(t, u)
    return m


def eval_start_for(md: MarketData, ticker: str, warmup: int, user_start: pd.Timestamp | None) -> pd.Timestamp | None:
    b = md.bars[ticker]
    pos = np.flatnonzero(b["listed"].to_numpy())
    if len(pos) <= warmup + 252:
        return None
    s = md.calendar[pos[0] + warmup]
    if user_start is not None and user_start > s:
        s = md.calendar[md.calendar.searchsorted(user_start)]
    return s


def regime_labels(md: MarketData, ticker: str, n: int = 126, z: float = 0.5) -> pd.Series:
    """Ex-ante trend regime per bar: label for bar t uses data through t-1.

    z-score = trailing n-bar return / (trailing vol x sqrt(n/252)); 'up' if >= z,
    'down' if <= -z, else 'sideways'. Scale-free across assets of different volatility.
    """
    c = md.bars[ticker]["Close"]
    r = c / c.shift(n) - 1
    v = c.pct_change().rolling(n, min_periods=n).std() * math.sqrt(n)
    zz = (r / v).shift(1)
    lab = pd.Series(np.where(zz >= z, "up", np.where(zz <= -z, "down", "sideways")), index=c.index)
    return lab.where(zz.notna())


def window(s: pd.Series | pd.DataFrame, a, b):
    return s.loc[(s.index >= a) & (s.index <= b)]


def metric_row(res, md: MarketData, a, b, rf: pd.Series, bh_ret: pd.Series, ub_ret: pd.Series | None,
               single: bool, closes: pd.DataFrame | None = None) -> dict:
    r = window(res.returns, a, b)
    if len(r) < 60:
        return {}
    expo = window(res.exposure, a, b)
    if single:
        ep = pa.episodes_from_exposure(expo.to_numpy(), r.to_numpy())
    else:
        ep = pa.episodes_from_weights(window(res.weights, a, b), window(closes, a, b))
    m = pa.perf_metrics(r, rf, exposure=expo, turnover=window(res.turnover, a, b), gross=window(res.gross_returns, a, b),
                        costs=res.costs, equity=res.equity, bench=window(bh_ret, a, b), episodes=ep)
    if ub_ret is not None:
        ab = pa.alpha_beta(r.to_numpy(), window(ub_ret, a, b).to_numpy(), window(rf, a, b).to_numpy())
        m.update({f"ub_{k}": v for k, v in ab.items()})
        m["ub_cagr"] = pa.cagr(window(ub_ret, a, b).to_numpy())
    return m


# ============================================================================
# Per-asset worker
# ============================================================================

def process_asset(ticker: str) -> dict:
    """Run every single-asset strategy (defaults + variants + cost grid) on one asset."""
    md: MarketData = _G["md"]
    cfg: ResearchConfig = _G["cfg"]
    specs = _G["specs"]
    universe = _G["universe_of"][ticker]
    aux = aux_map(cfg)
    base = cfg.cost_bps
    cap = cfg.initial_capital
    band = cfg.rebalance_band
    rf = md.cash_growth
    warm = int(cfg["evaluation"]["warmup_bars"])
    a = eval_start_for(md, ticker, warm, _G.get("user_start"))
    end = _G.get("user_end") or md.calendar[-1]
    out = {"ticker": ticker, "universe": universe, "rows": [], "cost_rows": [], "variant_rows": [], "errors": [],
           "eval_start": a}
    if a is None:
        out["errors"].append({"ticker": ticker, "strategy": "*", "error": "insufficient history"})
        return out
    common_start = pd.Timestamp(cfg["evaluation"]["common_period_start"])
    periods = {"full": (a, end)}
    if a <= common_start:
        periods["common"] = (common_start, end)
    if ticker in cfg["evaluation"]["previously_inspected"]:
        s0, s1 = cfg["evaluation"]["previously_inspected"][ticker]
        periods["prior_dev_2010_2021"] = (pd.Timestamp(s0), pd.Timestamp("2021-09-28"))
        periods["prior_holdout_2021_2026"] = (pd.Timestamp("2021-09-29"), pd.Timestamp(s1))
    ub = cfg["universes"][universe]["benchmark"]
    bh = backtest(md, buy_and_hold_target(md, ticker), base, cap, band)
    ubh = bh if ub == ticker or ub not in md.bars else backtest(md, buy_and_hold_target(md, ub), base, cap, band)
    labels = regime_labels(md, ticker)
    asset_vol = window(bh.returns, a, end).std() * math.sqrt(252)
    cache: dict = {}
    R_def = {"BH": bh.returns.astype("float32")}
    R_var = {}
    bh_full = None
    for p_name, (pa_, pb_) in periods.items():
        row = metric_row(bh, md, pa_, pb_, rf, bh.returns, ubh.returns, True)
        if row:
            out["rows"].append({"ticker": ticker, "universe": universe, "strategy": "BH", "period": p_name, **row})
            if p_name == "full":
                bh_full = row
    for spec in specs:
        if spec.kind != "single":
            continue
        try:
            w = S.generate_weights(spec, md, ticker, aux, cache=cache)
        except S.NotApplicable as exc:
            out["errors"].append({"ticker": ticker, "strategy": spec.id, "error": f"not applicable: {exc}"})
            continue
        except Exception as exc:  # pragma: no cover - surfaced in the error log
            out["errors"].append({"ticker": ticker, "strategy": spec.id, "error": repr(exc),
                                  "trace": traceback.format_exc(limit=3)})
            continue
        res = backtest(md, w, base, cap, band)
        R_def[spec.id] = res.returns.astype("float32")
        R_var[f"{spec.id}@0"] = R_def[spec.id]
        for p_name, (pa_, pb_) in periods.items():
            row = metric_row(res, md, pa_, pb_, rf, bh.returns, ubh.returns, True)
            if not row:
                continue
            if p_name == "full":
                # Exposure- and volatility-matched static benchmarks (same asset, rest in cash).
                em = backtest(md, static_mix_target(md, ticker, row["avg_exposure"]), base, cap, band)
                vm_w = min(1.0, row["ann_vol"] / asset_vol) if asset_vol > 0 else 0.0
                vm = backtest(md, static_mix_target(md, ticker, vm_w), base, cap, band)
                emr, vmr = window(em.returns, pa_, pb_), window(vm.returns, pa_, pb_)
                f_ = window(rf, pa_, pb_).to_numpy()
                row.update({"expo_matched_cagr": pa.cagr(emr), "expo_matched_sharpe": pa.sharpe(emr, f_),
                            "expo_matched_max_dd": pa.max_drawdown(emr), "vol_matched_weight": vm_w,
                            "vol_matched_cagr": pa.cagr(vmr), "vol_matched_sharpe": pa.sharpe(vmr, f_),
                            "vol_matched_max_dd": pa.max_drawdown(vmr)})
                row["excess_cagr_vs_expo_matched"] = row["cagr"] - row["expo_matched_cagr"]
                row["excess_cagr_vs_vol_matched"] = row["cagr"] - row["vol_matched_cagr"]
                row["dd_improvement_vs_vol_matched"] = row["max_dd"] - row["vol_matched_max_dd"]
                mt = missed_vs_avoided(md, ticker, res.exposure, window(res.exposure, pa_, pb_).index)
                row.update(mt)
                # Regime-conditional performance (ex-ante labels).
                rr = window(res.returns, pa_, pb_)
                bb = window(bh.returns, pa_, pb_)
                ll = labels.reindex(rr.index)
                ff = window(rf, pa_, pb_)
                for lab in ("up", "sideways", "down"):
                    m_ = (ll == lab).to_numpy()
                    if m_.sum() > 60:
                        row[f"reg_{lab}_share"] = float(m_.mean())
                        row[f"reg_{lab}_sharpe"] = pa.sharpe(rr[m_], ff[m_])
                        row[f"reg_{lab}_excess_vs_bh_ann"] = float((rr[m_] - bb[m_]).mean() * 252)
                        row[f"reg_{lab}_bh_sharpe"] = pa.sharpe(bb[m_], ff[m_])
            out["rows"].append({"ticker": ticker, "universe": universe, "strategy": spec.id, "period": p_name, **row})
        # Transaction-cost sensitivity (full period).
        pa_, pb_ = periods["full"]
        for cb in cfg["costs"]["sensitivity_bps"]:
            rc = res if abs(cb - base) < 1e-12 else backtest(md, w, cb, cap, band)
            rr = window(rc.returns, pa_, pb_)
            out["cost_rows"].append({"ticker": ticker, "universe": universe, "strategy": spec.id, "cost_bps": cb,
                                     "cagr": pa.cagr(rr), "sharpe": pa.sharpe(rr, window(rf, pa_, pb_)),
                                     "max_dd": pa.max_drawdown(rr),
                                     "bh_sharpe": bh_full["sharpe"] if bh_full else np.nan,
                                     "bh_cagr": bh_full["cagr"] if bh_full else np.nan})
        # Parameter neighbourhood.
        if not _G.get("skip_variants"):
            for vi, params in enumerate(spec.variants()[1:], start=1):
                try:
                    wv = S.generate_weights(spec, md, ticker, aux, params=params, cache=cache)
                    rv = backtest(md, wv, base, cap, band)
                except Exception as exc:  # pragma: no cover
                    out["errors"].append({"ticker": ticker, "strategy": f"{spec.id}@{vi}", "error": repr(exc)})
                    continue
                R_var[f"{spec.id}@{vi}"] = rv.returns.astype("float32")
                rr = window(rv.returns, pa_, pb_)
                out["variant_rows"].append({"ticker": ticker, "strategy": spec.id, "variant": vi,
                                            "params": json.dumps(params, default=list, sort_keys=True),
                                            "cagr": pa.cagr(rr), "sharpe": pa.sharpe(rr, window(rf, pa_, pb_)),
                                            "max_dd": pa.max_drawdown(rr)})
    R_def = pd.DataFrame(R_def)
    out["returns"] = window(R_def, a, end)
    out["exposure_note"] = None
    # ---------------- Walk-forward on this asset ----------------
    wcfg = cfg["walk_forward"]
    folds = wf.year_folds(md.calendar[(md.calendar >= a) & (md.calendar <= end)], a, wcfg["train_years"],
                          wcfg["test_years"], wcfg["mode"])
    out["folds"] = folds
    if folds:
        Rw = window(R_def, a, end).astype(float)
        tab = wf.fold_table(Rw, rf, folds)
        out["wf_fixed"] = tab
        strat_cols = [c for c in Rw.columns if c != "BH"]
        oos_best, ch_best = wf.select_and_stitch(Rw, rf, folds, {"BEST_OF_REGISTRY": strat_cols})
        oos_mask = oos_best["BEST_OF_REGISTRY"].notna()
        out["wf_best"] = {"oos": oos_best["BEST_OF_REGISTRY"].astype("float32"), "choices": ch_best,
                          "bh_oos": Rw["BH"].where(oos_mask).astype("float32"),
                          "is_best": ch_best.attrs["is_best_metric"]}
        if R_var and not _G.get("skip_variants"):
            Rv = window(pd.DataFrame(R_var), a, end).astype(float)
            groups = {}
            for c in Rv.columns:
                groups.setdefault(c.split("@")[0], []).append(c)
            oos_sel, ch_sel = wf.select_and_stitch(Rv, rf, folds, groups)
            tabv = wf.fold_table(Rv, rf, folds)
            rows = []
            for sid, cols in groups.items():
                sel = wf.stitched_metrics(oos_sel[sid], rf)
                fixed = wf.stitched_metrics(Rv[f"{sid}@0"].where(oos_sel[sid].notna()), rf)
                is_best = ch_sel.attrs["is_best_metric"][sid]
                rows.append({"ticker": ticker, "strategy": sid, "n_variants": len(cols),
                             "is_sharpe_selected_mean": float(is_best.mean()),
                             "is_sharpe_default_mean": float(tabv["is_sharpe"][f"{sid}@0"].mean()),
                             "oos_sharpe_default_mean": float(tabv["oos_sharpe"][f"{sid}@0"].mean()),
                             **{f"sel_{k}": v for k, v in sel.items()}, **{f"fixed_{k}": v for k, v in fixed.items()},
                             "share_folds_default_chosen": float((ch_sel[sid] == f"{sid}@0").mean())})
            out["wf_param"] = pd.DataFrame(rows)
    return out


# ============================================================================
# Rotation strategies
# ============================================================================

def run_rotations(md: MarketData, cfg: ResearchConfig, specs, user_start=None, user_end=None) -> dict:
    aux = aux_map(cfg)
    base, cap, band = cfg.cost_bps, cfg.initial_capital, cfg.rebalance_band
    rf = md.cash_growth
    out = {"rows": [], "cost_rows": [], "variant_rows": [], "returns": {}, "wf": {}, "errors": [], "eval_start": {}}
    rot_specs = [s for s in specs if s.kind == "rotation"]
    if not rot_specs:
        return out
    spy = cfg["aux_tickers"]["market"]
    for u in cfg["rotation_universes"]:
        if u not in cfg.enabled_universes():
            continue
        tick = [t for t in cfg.universe_tickers(u) if t in md.bars]
        if len(tick) < 4:
            continue
        listed = pd.DataFrame({t: md.bars[t]["listed"] for t in tick})
        hist = listed.cumsum()
        ok = (hist >= int(cfg["evaluation"]["warmup_bars"])).sum(axis=1) >= 4
        if not ok.any():
            continue
        a = ok.idxmax()
        if user_start is not None and pd.Timestamp(user_start) > a:
            a = md.calendar[md.calendar.searchsorted(pd.Timestamp(user_start))]
        end = user_end or md.calendar[-1]
        out["eval_start"][u] = a
        ew = backtest(md, equal_weight_target(md, tick), base, cap, band)
        spy_bh = backtest(md, buy_and_hold_target(md, spy), base, cap, band) if spy in md.bars else ew
        closes = pd.DataFrame({t: md.bars[t]["Close"] for t in tick})
        common_start = pd.Timestamp(cfg["evaluation"]["common_period_start"])
        periods = {"full": (a, end)}
        if a <= common_start:
            periods["common"] = (common_start, end)
        R = {"EW": ew.returns.astype("float32")}
        for p_name, (pa_, pb_) in periods.items():
            row = metric_row(ew, md, pa_, pb_, rf, ew.returns, spy_bh.returns, False, closes)
            out["rows"].append({"ticker": f"{u}", "universe": u, "strategy": "EW", "period": p_name, **row})
        for spec in rot_specs:
            try:
                w = S.generate_weights(spec, md, tick, aux)
            except Exception as exc:
                out["errors"].append({"ticker": u, "strategy": spec.id, "error": repr(exc)})
                continue
            res = backtest(md, w, base, cap, band)
            R[spec.id] = res.returns.astype("float32")
            for p_name, (pa_, pb_) in periods.items():
                row = metric_row(res, md, pa_, pb_, rf, ew.returns, spy_bh.returns, False, closes)
                if row and p_name == "full":
                    em = backtest(md, equal_weight_target(md, tick) * row["avg_exposure"], base, cap, band)
                    emr = window(em.returns, pa_, pb_)
                    row.update({"expo_matched_cagr": pa.cagr(emr),
                                "expo_matched_sharpe": pa.sharpe(emr, window(rf, pa_, pb_)),
                                "excess_cagr_vs_expo_matched": row["cagr"] - pa.cagr(emr)})
                if row:
                    out["rows"].append({"ticker": u, "universe": u, "strategy": spec.id, "period": p_name, **row})
            pa_, pb_ = periods["full"]
            for cb in cfg["costs"]["sensitivity_bps"]:
                rc = res if abs(cb - base) < 1e-12 else backtest(md, w, cb, cap, band)
                rr = window(rc.returns, pa_, pb_)
                out["cost_rows"].append({"ticker": u, "universe": u, "strategy": spec.id, "cost_bps": cb,
                                         "cagr": pa.cagr(rr), "sharpe": pa.sharpe(rr, window(rf, pa_, pb_)),
                                         "max_dd": pa.max_drawdown(rr),
                                         "bh_sharpe": pa.sharpe(window(ew.returns, pa_, pb_), window(rf, pa_, pb_)),
                                         "bh_cagr": pa.cagr(window(ew.returns, pa_, pb_))})
            for vi, params in enumerate(spec.variants()[1:], start=1):
                try:
                    rv = backtest(md, S.generate_weights(spec, md, tick, aux, params=params), base, cap, band)
                except Exception as exc:
                    out["errors"].append({"ticker": u, "strategy": f"{spec.id}@{vi}", "error": repr(exc)})
                    continue
                rr = window(rv.returns, pa_, pb_)
                out["variant_rows"].append({"ticker": u, "strategy": spec.id, "variant": vi,
                                            "params": json.dumps(params, default=list, sort_keys=True),
                                            "cagr": pa.cagr(rr), "sharpe": pa.sharpe(rr, window(rf, pa_, pb_)),
                                            "max_dd": pa.max_drawdown(rr)})
        Rdf = window(pd.DataFrame(R), a, end)
        out["returns"][u] = Rdf
        folds = wf.year_folds(Rdf.index, a, cfg["walk_forward"]["train_years"], cfg["walk_forward"]["test_years"],
                              cfg["walk_forward"]["mode"])
        if folds:
            out["wf"][u] = wf.fold_table(Rdf.astype(float), rf, folds)
    return out


# ============================================================================
# Statistics across strategies
# ============================================================================

def statistical_tests(asset_out: dict, md: MarketData, cfg: ResearchConfig) -> dict:
    st = cfg["statistics"]
    reps, blk, seed = int(st["bootstrap_reps"]), float(st["bootstrap_mean_block"]), int(st["seed"])
    rf = md.cash_growth
    dsr_rows, ci_rows, p_rows, rc_rows = [], [], [], []
    corr_sum, corr_n = None, 0
    for t, o in asset_out.items():
        R = o.get("returns")
        if R is None or R.shape[1] < 3:
            continue
        R = R.astype(float)
        cols = [c for c in R.columns if c != "BH"]
        X = R[cols].to_numpy()
        f = rf.reindex(R.index).to_numpy()
        bh = R["BH"].to_numpy()
        idx = pa.stationary_bootstrap_indices(len(R), reps, blk, seed)
        d = pa.deflated_sharpe(X, f, cols)
        d.insert(0, "ticker", t)
        dsr_rows.append(d)
        ci = pa.bootstrap_metric_ci(np.column_stack([X, bh]), f, idx)
        ci.insert(0, "strategy", cols + ["BH"])
        ci.insert(0, "ticker", t)
        ci_rows.append(ci)
        D = X - bh[:, None]
        pv = pa.bootstrap_diff_pvalues(D, idx)
        p_rows.append(pd.DataFrame({"ticker": t, "strategy": cols, "mean_diff_vs_bh_ann": D.mean(axis=0) * 252,
                                    "p_boot_diff_gt0": pv}))
        rc = pa.reality_check(D, idx)
        rc["ticker"] = t
        rc["best_strategy"] = cols[rc["best_index"]]
        # Also test risk-adjusted outperformance: differential of returns scaled to B&H volatility.
        scale = np.where(X.std(axis=0) > 0, bh.std() / X.std(axis=0), 0.0)
        Dv = (X - f[:, None]) * scale - (bh - f)[:, None]
        rcv = pa.reality_check(Dv, idx)
        rc["spa_pvalue_vol_scaled"] = rcv["spa_pvalue"]
        rc["rc_pvalue_vol_scaled"] = rcv["rc_pvalue"]
        rc["best_strategy_vol_scaled"] = cols[rcv["best_index"]]
        rc_rows.append(rc)
        C = pd.DataFrame(X, columns=cols).corr()
        corr_sum = C if corr_sum is None else corr_sum.add(C, fill_value=0)
        corr_n += 1
    pvals = pd.concat(p_rows, ignore_index=True) if p_rows else pd.DataFrame()
    if not pvals.empty:
        alpha = float(st["fdr_alpha"])
        rej, q = pa.bh_fdr(pvals["p_boot_diff_gt0"].to_numpy(), alpha)
        pvals["q_bh_global"] = q
        pvals["reject_bh_global"] = rej
        pvals["reject_holm_global_5pct"] = pa.holm(pvals["p_boot_diff_gt0"].to_numpy(), 0.05)
    return {"dsr": pd.concat(dsr_rows, ignore_index=True) if dsr_rows else pd.DataFrame(),
            "ci": pd.concat(ci_rows, ignore_index=True) if ci_rows else pd.DataFrame(),
            "pvalues": pvals, "reality_check": pd.DataFrame(rc_rows),
            "avg_corr": (corr_sum / max(corr_n, 1)) if corr_sum is not None else pd.DataFrame()}


# ============================================================================
# SPY focused study
# ============================================================================

def spy_study(md: MarketData, cfg: ResearchConfig, out_dir: Path, asset_out: dict) -> dict:
    spy = cfg["aux_tickers"]["market"]
    if spy not in md.bars:
        return {}
    aux = aux_map(cfg)
    base, cap, band = cfg.cost_bps, cfg.initial_capital, cfg.rebalance_band
    res = {}
    trade_dir = out_dir / "audit" / "trade_logs"
    trade_dir.mkdir(parents=True, exist_ok=True)
    recon = []
    curves = {}
    for sid in ORIGINAL_SIX + ["F02_10", "F10_10", "F10_07"]:
        spec = S.get_spec(sid)
        w = S.generate_weights(spec, md, spy, aux)
        r = backtest(md, w, base, cap, band, record_trades=True)
        res[sid] = r
        r.trades.to_csv(trade_dir / f"{spy}_{sid}_trades.csv", index=False)
        rc = ledger_reconcile(r, md, cap)
        recon.append({"strategy": sid, "max_abs_error": float(rc["abs_error"].max()),
                      "max_rel_error": float((rc["abs_error"] / rc["engine_equity"]).max()), "n_trades": len(r.trades)})
        curves[sid] = r.equity
    bh = backtest(md, buy_and_hold_target(md, spy), base, cap, band, record_trades=True)
    res["BH"] = bh
    curves["BH"] = bh.equity
    pd.DataFrame(curves).to_csv(out_dir / "audit" / f"equity_curves_{spy}_core.csv", index_label="Date")
    pd.DataFrame(recon).to_csv(out_dir / "audit" / "ledger_reconciliation.csv", index=False)
    legacy = None
    try:
        legacy = legacy_notebook_sma2050(md.bars[spy])
    except Exception as exc:  # pragma: no cover
        legacy = {"error": repr(exc)}
    return {"results": res, "reconciliation": pd.DataFrame(recon), "legacy": legacy, "legacy_reference": NOTEBOOK_REFERENCE}


def oracle_study(md: MarketData, cfg: ResearchConfig, rows: pd.DataFrame, asset_out: dict) -> dict:
    spy = cfg["aux_tickers"]["market"]
    out = {}
    oc = cfg["oracle"]
    tickers = [t for t in [spy, "QQQ", "SMH", "NVDA", "XLU"] if t in md.bars and t in asset_out
               and asset_out[t].get("eval_start") is not None]
    for t in tickers:
        a = asset_out[t]["eval_start"]
        o = run_oracles(md, t, float(oc["cost_bps"]), list(oc["trade_limits"]), cfg.initial_capital, start=a)
        full = rows[(rows.ticker == t) & (rows.period == "full") & (rows.strategy != "BH")]
        best_is = full.sort_values("sharpe", ascending=False).iloc[0]["strategy"] if len(full) else None
        summ = []
        rf = md.cash_growth
        for name, r in o["results"].items():
            rr = window(r.returns, a, md.calendar[-1])
            summ.append({"ticker": t, "benchmark": name, "cagr": pa.cagr(rr), "sharpe": pa.sharpe(rr, window(rf, a, md.calendar[-1])),
                         "max_dd": pa.max_drawdown(rr), "time_invested": float((window(r.exposure, a, md.calendar[-1]) > 0.01).mean()),
                         "entries": int(((r.exposure > 0.01) & (r.exposure.shift(1) <= 0.01)).loc[a:].sum()),
                         "turnover_ann": float(window(r.turnover, a, md.calendar[-1]).sum() / (len(rr) / 252))})
        out[t] = {"oracle": o, "summary": pd.DataFrame(summ), "best_in_sample": best_is, "eval_start": a}
    return out


# ============================================================================
# Main pipeline
# ============================================================================

def load_data(cfg: ResearchConfig, args) -> MarketData:
    if args.synthetic:
        from synthetic_data import synthetic_universe
        tick = cfg.all_tickers()
        raw, cash = synthetic_universe(tick, inception={"XLC": "2018-06-19", "XLRE": "2015-10-08",
                                                         "AVGO": "2009-08-06", "XSD": "2006-02-07", "PSI": "2005-06-23"})
        return assemble(raw, cash, cfg, source="SYNTHETIC")
    return load_market_data(cfg, cache_dir=Path(args.cache) if args.cache else DEFAULT_CACHE_DIR,
                            refresh=args.refresh, offline=args.offline)


def run(args) -> dict:
    t0 = time.time()
    overrides = {}
    if args.cost_bps is not None:
        overrides = {"costs": {"commission_bps": 0.0, "slippage_bps": float(args.cost_bps)}}
    if args.quick:
        overrides.setdefault("statistics", {})["bootstrap_reps"] = 200
    cfg = load_config(args.config, overrides)
    out_dir = Path(args.out) if args.out else (LAB / "outputs_synthetic" if args.synthetic else DEFAULT_OUTPUT_DIR)
    (out_dir / "audit").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    log(f"Config {cfg.path.name} sha256={cfg.sha256[:12]} base cost={cfg.cost_bps} bps -> {out_dir}")

    freeze = json.loads((LAB / "registry_freeze.json").read_text())
    if freeze["registry_sha256"] != S.registry_hash():
        log("WARNING: strategy registry differs from the frozen version; results are flagged as post-freeze edits")
    S.validate_registry()

    md = load_data(cfg, args)
    if args.universes:
        keep = set()
        for u in args.universes.split(","):
            keep |= set(cfg.universe_tickers(u))
        cfg.raw["universes"] = {k: v for k, v in cfg.raw["universes"].items() if k in args.universes.split(",")}
        cfg.raw["rotation_universes"] = [u for u in cfg["rotation_universes"] if u in cfg.raw["universes"]]
    inv = data_inventory(md)
    inv.to_csv(out_dir / "audit" / "data_inventory.csv", index=False)
    md.validation.to_csv(out_dir / "audit" / "data_validation.csv", index=False)
    log(f"Data: {len(md.bars)} tickers, {md.calendar[0].date()}..{md.calendar[-1].date()} source={md.source}")

    specs = S.REGISTRY if not args.strategies else [S.get_spec(s) for s in args.strategies.split(",")]
    uo = universe_of(cfg)
    assets = [t for t in uo if t in md.bars]
    missing = [t for t in uo if t not in md.bars]
    _G.update(md=md, cfg=cfg, specs=specs, universe_of=uo, skip_variants=args.skip_variants,
              user_start=pd.Timestamp(args.start) if args.start else None,
              user_end=pd.Timestamp(args.end) if args.end else None)
    if args.end:
        _G["user_end"] = md.calendar[md.calendar.searchsorted(pd.Timestamp(args.end), side="right") - 1]
    log(f"Running {sum(s.kind == 'single' for s in specs)} single-asset strategies on {len(assets)} assets "
        f"(+{sum(s.kind == 'rotation' for s in specs)} rotation strategies); missing tickers: {missing}")
    asset_out = {}
    workers = max(1, min(args.workers or os.cpu_count() or 1, len(assets)))
    if workers > 1 and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(workers) as pool:
            for o in pool.imap_unordered(process_asset, assets):
                asset_out[o["ticker"]] = o
                log(f"  {o['ticker']}: {len(o['rows'])} metric rows, {len(o['errors'])} notes")
    else:
        for t in assets:
            o = process_asset(t)
            asset_out[t] = o
            log(f"  {t}: {len(o['rows'])} metric rows, {len(o['errors'])} notes")
    log("Rotation strategies ...")
    rot = run_rotations(md, cfg, specs, _G["user_start"], _G.get("user_end"))

    rows = pd.DataFrame([r for o in asset_out.values() for r in o["rows"]] + rot["rows"])
    cost_rows = pd.DataFrame([r for o in asset_out.values() for r in o["cost_rows"]] + rot["cost_rows"])
    var_rows = pd.DataFrame([r for o in asset_out.values() for r in o["variant_rows"]] + rot["variant_rows"])
    errors = pd.DataFrame([r for o in asset_out.values() for r in o["errors"]] + rot["errors"])
    reg = S.registry_frame().set_index("id")
    if not rows.empty:
        rows["family"] = rows["strategy"].map(lambda s: reg["family"].get(s, "BENCH"))
        rows["family_name"] = rows["family"].map(lambda f: S.FAMILIES.get(f, "Benchmark"))
        rows["strategy_name"] = rows["strategy"].map(lambda s: reg["name"].get(s, {"BH": "Buy-and-hold", "EW": "Equal-weight universe"}.get(s, s)))
    log("Statistical tests ...")
    stats_out = statistical_tests(asset_out, md, cfg)
    log("SPY study, oracle ...")
    spy = spy_study(md, cfg, out_dir, asset_out) if not args.strategies else {}
    oracle = oracle_study(md, cfg, rows, asset_out) if not rows.empty else {}
    results = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "synthetic": bool(args.synthetic), "config": cfg.raw, "config_sha256": cfg.sha256,
        "registry_sha256": S.registry_hash(), "registry_freeze": freeze, "data_source": md.source,
        "calendar": (md.calendar[0], md.calendar[-1]), "inventory": inv, "validation": md.validation,
        "rows": rows, "cost_rows": cost_rows, "variant_rows": var_rows, "errors": errors,
        "asset_out": asset_out, "rotation": rot, "stats": stats_out, "spy": spy, "oracle": oracle,
        "n_strategies_defined": len(S.REGISTRY), "n_strategies_run": len(specs),
        "runtime_sec": None, "python": platform.python_version(), "args": vars(args),
    }
    results["runtime_sec"] = time.time() - t0
    with open(out_dir / "audit" / "results.pkl", "wb") as fh:
        pickle.dump({k: v for k, v in results.items() if k != "spy"} | {"spy": {k: v for k, v in spy.items() if k != "results"}}, fh)
    rows.to_csv(out_dir / "audit" / "all_metrics_long.csv", index=False)
    cost_rows.to_csv(out_dir / "audit" / "cost_sensitivity_long.csv", index=False)
    var_rows.to_csv(out_dir / "audit" / "parameter_variants_long.csv", index=False)
    errors.to_csv(out_dir / "audit" / "run_notes_errors.csv", index=False)
    log(f"Core experiments done in {results['runtime_sec']:.0f}s")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--universes", help="comma list, e.g. A_broad,C_semi_etfs")
    ap.add_argument("--strategies", help="comma list of strategy ids")
    ap.add_argument("--cost-bps", type=float)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--quick", action="store_true", help="fewer bootstrap reps")
    ap.add_argument("--skip-variants", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args(argv)
    import reporting
    if args.report_only:
        out_dir = Path(args.out) if args.out else (LAB / "outputs_synthetic" if args.synthetic else DEFAULT_OUTPUT_DIR)
        with open(out_dir / "audit" / "results.pkl", "rb") as fh:
            results = pickle.load(fh)
        reporting.build_all(results, out_dir, md=None)
        return
    results = run(args)
    if not args.no_report:
        out_dir = Path(args.out) if args.out else (LAB / "outputs_synthetic" if args.synthetic else DEFAULT_OUTPUT_DIR)
        reporting.build_all(results, out_dir, md=_G.get("md"))


if __name__ == "__main__":
    main()
