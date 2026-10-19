"""Pre-specified tests for hypotheses H1-H10.

Each test states its decision rule *before* looking at results (documented in
Research_Methodology.md) and returns evidence, a verdict and caveats. Verdicts:

* ``Supported``      - the pre-specified effect has the predicted sign and passes its test;
* ``Contradicted``   - the effect has the opposite sign and passes its test;
* ``Not resolved``   - mixed signs or the test is not significant.

Cross-asset tests treat each asset as one observation (paired sign tests and
t-tests across assets). Assets are correlated, so these p-values are optimistic
and are reported as descriptive, not confirmatory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

SEMI_UNIVERSES = ("C_semi_etfs", "D_semi_stocks")
BROAD = ("A_broad",)


def _sign_test(x: np.ndarray) -> float:
    x = x[np.isfinite(x) & (x != 0)]
    if len(x) == 0:
        return np.nan
    return float(stats.binomtest(int((x > 0).sum()), len(x), 0.5).pvalue)


def _ttest(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 3 or np.std(x) == 0:
        return np.nan
    return float(stats.ttest_1samp(x, 0.0).pvalue)


def _verdict(effect: float, p: float, predicted_sign: int = 1, alpha: float = 0.10) -> str:
    if not np.isfinite(effect) or not np.isfinite(p):
        return "Not resolved"
    if p <= alpha and np.sign(effect) == predicted_sign:
        return "Supported"
    if p <= alpha and np.sign(effect) == -predicted_sign:
        return "Contradicted"
    return "Not resolved"


def _pivot(rows: pd.DataFrame, metric: str, period: str = "full") -> pd.DataFrame:
    r = rows[(rows.period == period)]
    return r.pivot_table(index="ticker", columns="strategy", values=metric)


def h1(rows: pd.DataFrame) -> dict:
    fast, slow = ["F01_02", "F01_04", "F01_01"], ["F01_03", "F01_05", "F01_10"]
    sh = _pivot(rows, "sharpe")
    uni = rows.drop_duplicates("ticker").set_index("ticker")["universe"]
    diff = (sh[fast].mean(axis=1) - sh[slow].mean(axis=1)).dropna()
    semi = diff[uni.reindex(diff.index).isin(SEMI_UNIVERSES)]
    broad = diff[uni.reindex(diff.index).isin(BROAD)]
    eff = semi.mean() - broad.mean() if len(semi) and len(broad) else np.nan
    p = float(stats.mannwhitneyu(semi, broad, alternative="two-sided").pvalue) if len(semi) > 2 and len(broad) > 2 else np.nan
    return {"id": "H1", "claim": "Faster SMA systems work better than slower ones on volatile semiconductor assets",
            "test": "Sharpe(fast: SMA10/30, EMA12/26, SMA20/50) - Sharpe(slow: SMA50/200, >SMA200, monthly SMA210); "
                    "compare the mean difference in semiconductors (C+D) with broad ETFs (A); Mann-Whitney p",
            "evidence": f"Fast-slow Sharpe: semis mean {semi.mean():+.3f} (n={len(semi)}, {np.mean(semi > 0):.0%} positive); "
                        f"broad mean {broad.mean():+.3f} (n={len(broad)}); difference {eff:+.3f}; p={p:.3f}",
            "effect": eff, "p": p,
            "verdict": "Supported" if (semi.mean() > 0 and _verdict(eff, p) == "Supported") else _verdict(eff, p),
            "detail": pd.DataFrame({"fast_minus_slow_sharpe": diff, "universe": uni.reindex(diff.index)})}


def h2(rows: pd.DataFrame) -> dict:
    full = rows[rows.period == "full"].set_index(["ticker", "strategy"])
    t = sorted(set(full.index.get_level_values(0)))
    rec = []
    for k in t:
        if (k, "F01_02") in full.index and (k, "F01_01") in full.index:
            a, b = full.loc[(k, "F01_02")], full.loc[(k, "F01_01")]
            rec.append({"ticker": k, "dd_improvement": a["max_dd"] - b["max_dd"],
                        "sharpe_diff": a["sharpe"] - b["sharpe"],
                        "down_regime_excess_diff": a.get("reg_down_excess_vs_bh_ann", np.nan) - b.get("reg_down_excess_vs_bh_ann", np.nan),
                        "sideways_excess_diff": a.get("reg_sideways_excess_vs_bh_ann", np.nan) - b.get("reg_sideways_excess_vs_bh_ann", np.nan)})
    d = pd.DataFrame(rec).set_index("ticker") if rec else pd.DataFrame()
    if d.empty:
        return {"id": "H2", "verdict": "Not resolved", "evidence": "no data", "claim": "", "test": "", "effect": np.nan, "p": np.nan}
    p_dd = _sign_test(d["dd_improvement"].to_numpy())
    p_side = _sign_test(d["sideways_excess_diff"].to_numpy())
    v1 = _verdict(d["dd_improvement"].median(), p_dd)
    v2 = _verdict(d["sideways_excess_diff"].median(), p_side, predicted_sign=-1)
    verdict = "Supported" if v1 == "Supported" and v2 == "Supported" else (
        "Contradicted" if v1 == "Contradicted" else "Not resolved")
    return {"id": "H2", "claim": "SMA10/30 offers better downside protection than SMA20/50 but lags in rapidly reversing (sideways) markets",
            "test": "Across assets: max-drawdown difference (10/30 minus 20/50, positive = shallower) and sideways-regime excess return difference; sign tests",
            "evidence": f"Drawdown shallower for 10/30 on {np.mean(d['dd_improvement'] > 0):.0%} of {len(d)} assets "
                        f"(median {d['dd_improvement'].median():+.3f}, p={p_dd:.3f}); sideways-regime excess return lower for 10/30 on "
                        f"{np.mean(d['sideways_excess_diff'] < 0):.0%} (median {d['sideways_excess_diff'].median():+.3%}/yr, p={p_side:.3f}); "
                        f"Sharpe higher for 10/30 on {np.mean(d['sharpe_diff'] > 0):.0%}",
            "effect": d["dd_improvement"].median(), "p": p_dd, "verdict": verdict, "detail": d}


def h3(rows: pd.DataFrame) -> dict:
    pairs = [("F02_02", "F02_01"), ("F02_10", "F01_02"), ("F02_06", "F01_06")]
    full = rows[rows.period == "full"]
    sh = full.pivot_table(index="ticker", columns="strategy", values="sharpe")
    dd = full.pivot_table(index="ticker", columns="strategy", values="max_dd")
    out, ev = [], []
    for a, b in pairs:
        if a in sh and b in sh:
            ds = (sh[a] - sh[b]).dropna()
            dd_ = (dd[a] - dd[b]).dropna()
            out.append({"pair": f"{a} vs {b}", "median_sharpe_diff": ds.median(), "share_sharpe_better": float((ds > 0).mean()),
                        "p_sharpe": _sign_test(ds.to_numpy()), "median_dd_improvement": dd_.median(),
                        "share_dd_better": float((dd_ > 0).mean()), "p_dd": _sign_test(dd_.to_numpy())})
            ev.append(f"{a} vs {b}: Sharpe better on {(ds > 0).mean():.0%} (median {ds.median():+.3f}), "
                      f"drawdown better on {(dd_ > 0).mean():.0%}")
    d = pd.DataFrame(out)
    if d.empty:
        return {"id": "H3", "verdict": "Not resolved", "evidence": "no data", "claim": "", "test": "", "effect": np.nan, "p": np.nan}
    verdicts = [_verdict(r.median_sharpe_diff, r.p_sharpe) for r in d.itertuples()]
    verdict = "Supported" if all(v == "Supported" for v in verdicts) else (
        "Contradicted" if all(v == "Contradicted" for v in verdicts) else "Not resolved")
    return {"id": "H3", "claim": "Faster exits and slower re-entries improve risk-adjusted returns by limiting drawdowns",
            "test": "Pairs differing only in re-entry speed: F02_02 (re-enter >SMA50) vs F02_01 (>SMA10); F02_10 (10/30 exit, 20/50 re-entry) vs F01_02 (10/30 both); "
                    "F02_06 (3-day confirmed re-entry above SMA50) vs F01_06-style symmetric filter. Sign tests across assets",
            "evidence": "; ".join(ev), "effect": d["median_sharpe_diff"].median(), "p": d["p_sharpe"].max(),
            "verdict": verdict, "detail": d}


def h4(rows: pd.DataFrame, regime_rows: pd.DataFrame | None = None) -> dict:
    full = rows[rows.period == "full"]
    mom = full[full.family == "F03"]
    uni = mom.groupby("ticker")["universe"].first()
    ex = mom.groupby("ticker")["sharpe_diff"].mean()
    up = mom.groupby("ticker")["reg_up_excess_vs_bh_ann"].mean() if "reg_up_excess_vs_bh_ann" in mom else pd.Series(dtype=float)
    semi = ex[uni.isin(SEMI_UNIVERSES)]
    broad = ex[uni.isin(BROAD)]
    eff = semi.mean() - broad.mean() if len(semi) and len(broad) else np.nan
    p = float(stats.mannwhitneyu(semi, broad, alternative="two-sided").pvalue) if len(semi) > 2 and len(broad) > 2 else np.nan
    su, bu = up[uni.isin(SEMI_UNIVERSES)], up[uni.isin(BROAD)]
    return {"id": "H4", "claim": "Semiconductor strategies benefit disproportionately from momentum during sustained industry expansion",
            "test": "Momentum family (F03) Sharpe minus buy-and-hold Sharpe, semis vs broad ETFs (Mann-Whitney); "
                    "and momentum excess return vs buy-and-hold inside ex-ante 'up' trend regimes",
            "evidence": f"Momentum Sharpe minus B&H: semis {semi.mean():+.3f} (n={len(semi)}), broad {broad.mean():+.3f} "
                        f"(n={len(broad)}), diff {eff:+.3f}, p={p:.3f}. In up-trend regimes momentum excess vs B&H: "
                        f"semis {su.mean():+.2%}/yr, broad {bu.mean():+.2%}/yr (negative = momentum lags holding during expansions)",
            "effect": eff, "p": p, "verdict": _verdict(eff, p),
            "detail": pd.DataFrame({"mom_sharpe_minus_bh": ex, "up_regime_excess": up, "universe": uni})}


def h5(rows: pd.DataFrame) -> dict:
    full = rows[(rows.period == "full") & (rows.family == "F04")]
    if "reg_sideways_sharpe" not in full:
        return {"id": "H5", "verdict": "Not resolved", "evidence": "no regime data", "claim": "", "test": "", "effect": np.nan, "p": np.nan}
    trend = full[["reg_up_sharpe", "reg_down_sharpe"]].mean(axis=1)
    diff = (full["reg_sideways_sharpe"] - trend)
    by_asset = diff.groupby(full["ticker"]).mean().dropna()
    exs = (full["reg_sideways_excess_vs_bh_ann"] - full[["reg_up_excess_vs_bh_ann", "reg_down_excess_vs_bh_ann"]].mean(axis=1))
    p = _sign_test(by_asset.to_numpy())
    return {"id": "H5", "claim": "Mean reversion performs better in sideways regimes than in strong directional trends",
            "test": "For each mean-reversion strategy x asset: Sharpe in ex-ante sideways regime minus average Sharpe in up/down regimes; sign test across assets",
            "evidence": f"Sideways minus trend-regime Sharpe: median {diff.median():+.3f}; positive for {np.mean(by_asset > 0):.0%} of "
                        f"{len(by_asset)} assets (p={p:.3f}); excess return vs B&H sideways minus trend: median {exs.median():+.2%}/yr",
            "effect": by_asset.median(), "p": p, "verdict": _verdict(by_asset.median(), p),
            "detail": full[["ticker", "strategy", "reg_up_sharpe", "reg_sideways_sharpe", "reg_down_sharpe"]]}


def h6(cs: pd.DataFrame) -> dict:
    if cs is None or cs.empty:
        return {"id": "H6", "verdict": "Not resolved", "evidence": "insufficient folds", "claim": "", "test": "", "effect": np.nan, "p": np.nan}
    d = (cs["specific_oos"] - cs["common_oos"]).dropna()
    by_fold = (cs.assign(d=cs["specific_oos"] - cs["common_oos"]).groupby("fold")["d"].mean()).dropna()
    p = _ttest(by_fold.to_numpy())
    opt = (cs["specific_is"] - cs["specific_oos"]).mean()
    return {"id": "H6", "claim": "Sector-specific strategy selection outperforms one common signal for all sectors, after model-selection bias",
            "test": "Walk-forward: each fold picks the best IS strategy per sector vs the strategy with the best mean IS Sharpe across sectors; "
                    "compare OOS Sharpe (t-test on fold means)",
            "evidence": f"OOS Sharpe specific minus common: mean {d.mean():+.3f} over {len(d)} sector-folds "
                        f"({np.mean(d > 0):.0%} positive); fold-mean t-test p={p:.3f}. Specific-selection IS->OOS Sharpe decay {opt:+.2f}",
            "effect": d.mean(), "p": p, "verdict": _verdict(by_fold.mean(), p), "detail": cs}


def h7(cost_rows: pd.DataFrame) -> dict:
    c = cost_rows.copy()
    c["beats_bh"] = c["sharpe"] > c["bh_sharpe"]
    g = c.groupby("cost_bps").agg(median_sharpe=("sharpe", "median"), share_beats_bh=("beats_bh", "mean"),
                                  median_cagr=("cagr", "median"))
    if 0.0 not in g.index or 10.0 not in g.index:
        return {"id": "H7", "verdict": "Not resolved", "evidence": "cost grid missing", "claim": "", "test": "", "effect": np.nan, "p": np.nan, "detail": g}
    p0 = c[c.cost_bps == 0].set_index(["ticker", "strategy"])["sharpe"]
    p10 = c[c.cost_bps == 10].set_index(["ticker", "strategy"])["sharpe"]
    drop = (p0 - p10).dropna()
    material = drop.median() >= 0.10 or (g.loc[0.0, "share_beats_bh"] - g.loc[10.0, "share_beats_bh"]) >= 0.25 * max(g.loc[0.0, "share_beats_bh"], 1e-9)
    lines = "; ".join(f"{int(k)} bps: median Sharpe {r.median_sharpe:.2f}, {r.share_beats_bh:.0%} beat B&H" for k, r in g.iterrows())
    return {"id": "H7", "claim": "Performance deteriorates materially once transaction costs and slippage are included",
            "test": "Re-run every strategy x asset at 0/5/10/20/50 bps. 'Material' (pre-specified) = median Sharpe drop >= 0.10 from 0 to 10 bps, "
                    "or >= 25% relative fall in the share of combinations beating buy-and-hold",
            "evidence": f"{lines}. Median Sharpe drop 0->10 bps {drop.median():.3f}; 90th pct {drop.quantile(0.9):.3f}",
            "effect": drop.median(), "p": np.nan,
            "verdict": "Supported" if material else "Contradicted (at 10 bps); see higher-cost rows", "detail": g}


def h8(rank: pd.DataFrame) -> dict:
    if rank is None or rank.empty:
        return {"id": "H8", "verdict": "Not resolved", "evidence": "no folds", "claim": "", "test": "", "effect": np.nan, "p": np.nan}
    rho = rank["spearman"].mean()
    ret = rank["top_retention"].mean()
    chance = rank["chance_top_retention"].mean()
    upper = rank["top_in_upper_half"].mean()
    p = _ttest(rank["top_retention"].to_numpy() - rank["chance_top_retention"].to_numpy())
    supported = ret < 0.5 and upper < 0.75
    return {"id": "H8", "claim": "Many top-ranked in-sample strategies fail to retain their ranking out of sample",
            "test": "Per asset and fold: Spearman(IS Sharpe rank, next-year OOS Sharpe rank); share of IS top-decile strategies still top-decile OOS. "
                    "Supported if retention < 50% and fewer than 75% stay in the upper half",
            "evidence": f"Mean Spearman {rho:+.3f} over {len(rank)} asset-folds; top-decile retention {ret:.0%} vs chance {chance:.0%} "
                        f"(t-test vs chance p={p:.3f}); {upper:.0%} of IS top-decile land in the OOS upper half; "
                        f"best-IS strategy's median OOS percentile {rank['best_is_oos_rank_pct'].median():.0%}",
            "effect": ret - chance, "p": p, "verdict": "Supported" if supported else "Contradicted", "detail": rank}


def h9(rows: pd.DataFrame) -> dict:
    full = rows[(rows.period == "full") & rows.strategy.str.match(r"F\d\d_") & rows["vol_matched_cagr"].notna()] \
        if "vol_matched_cagr" in rows else pd.DataFrame()
    if full.empty:
        return {"id": "H9", "verdict": "Not resolved", "evidence": "no matched benchmarks", "claim": "", "test": "", "effect": np.nan, "p": np.nan}
    appear = full[full["max_dd_improvement"] > 0]   # shallower drawdown than buy-and-hold
    keep_cagr = (appear["excess_cagr_vs_vol_matched"] > 0).mean()
    keep_dd = (appear["dd_improvement_vs_vol_matched"] > 0).mean()
    both = ((appear["excess_cagr_vs_vol_matched"] > 0) & (appear["dd_improvement_vs_vol_matched"] > 0)).mean()
    beat_bh_cagr = full[full["excess_cagr"] > 0]
    expo_keep = (beat_bh_cagr["excess_cagr_vs_expo_matched"] > 0).mean() if len(beat_bh_cagr) else np.nan
    return {"id": "H9", "claim": "Some apparent advantages disappear after matching invested exposure or volatility",
            "test": "Among strategy x asset combinations with a shallower drawdown than buy-and-hold, share that also beat a static asset/cash mix "
                    "with the same volatility (CAGR and drawdown). Supported if fewer than half retain both",
            "evidence": f"{len(appear)} of {len(full)} combos have shallower drawdowns than B&H; vs the volatility-matched mix "
                        f"{keep_cagr:.0%} have higher CAGR, {keep_dd:.0%} shallower drawdown, {both:.0%} both. "
                        f"Of {len(beat_bh_cagr)} combos beating B&H CAGR, {expo_keep:.0%} also beat the exposure-matched mix",
            "effect": both, "p": np.nan, "verdict": "Supported" if both < 0.5 else "Contradicted",
            "detail": appear[["ticker", "strategy", "max_dd_improvement", "excess_cagr_vs_vol_matched", "dd_improvement_vs_vol_matched"]]}


def h10(rows: pd.DataFrame, wf_param: pd.DataFrame, variant_rows: pd.DataFrame) -> dict:
    full = rows[(rows.period == "full") & rows.strategy.str.match(r"F\d\d_")]
    fam = full.groupby("family").agg(share_pos_sharpe_diff=("sharpe_diff", lambda x: float((x > 0).mean())),
                                     disp_sharpe_diff=("sharpe_diff", "std"), median_sharpe_diff=("sharpe_diff", "median"))
    if wf_param is not None and not wf_param.empty:
        wp = wf_param.assign(family=wf_param["strategy"].str[:3])
        wp["decay"] = wp["is_sharpe_default_mean"] - wp["oos_sharpe_default_mean"]
        fam = fam.join(wp.groupby("family")["decay"].median().rename("median_is_oos_decay"))
    if variant_rows is not None and not variant_rows.empty:
        vr = variant_rows.assign(family=variant_rows["strategy"].str[:3])
        sens = vr.groupby(["ticker", "strategy"])["sharpe"].std().reset_index()
        sens["family"] = sens["strategy"].str[:3]
        fam = fam.join(sens.groupby("family")["sharpe"].median().rename("median_param_sharpe_std"))
    ens = full[full.strategy == "F10_07"]["sharpe_diff"]
    others = fam.drop(index="F10", errors="ignore")
    hy = fam.loc["F10"] if "F10" in fam.index else None
    score = 0
    checks = []
    if hy is not None:
        for col, better in (("share_pos_sharpe_diff", "high"), ("disp_sharpe_diff", "low"),
                            ("median_is_oos_decay", "low"), ("median_param_sharpe_std", "low")):
            if col in fam:
                med = others[col].median()
                ok = hy[col] > med if better == "high" else hy[col] < med
                score += int(ok)
                checks.append(f"{col}: hybrids {hy[col]:.3f} vs median single-family {med:.3f} ({'better' if ok else 'worse'})")
    n = len(checks)
    verdict = "Supported" if n and score >= n - 0 and score >= 3 else ("Contradicted" if n and score <= 1 else "Not resolved")
    return {"id": "H10", "claim": "Combining complementary signals produces greater robustness than optimising a single indicator",
            "test": "Hybrid family (F10) vs the median of the nine single-idea families on four robustness measures: share of assets with "
                    "Sharpe above B&H, cross-asset dispersion, IS->OOS Sharpe decay, parameter sensitivity. Supported if better on >=3 of 4",
            "evidence": "; ".join(checks) + f". Ensemble F10_07 beats B&H Sharpe on {np.mean(ens > 0):.0%} of assets",
            "effect": score, "p": np.nan, "verdict": verdict, "detail": fam.reset_index()}


def evaluate_all(results: dict) -> list[dict]:
    rows = results["rows"]
    rows = rows[rows.ticker.isin([t for t in results["asset_out"]])] if "asset_out" in results else rows
    wf_param = pd.concat([o["wf_param"] for o in results["asset_out"].values() if "wf_param" in o], ignore_index=True) \
        if any("wf_param" in o for o in results["asset_out"].values()) else pd.DataFrame()
    return [h1(rows), h2(rows), h3(rows), h4(rows), h5(rows), h6(results.get("cross_sectional")),
            h7(results["cost_rows"][results["cost_rows"].ticker.isin(results["asset_out"].keys())]),
            h8(results.get("rank_persistence")), h9(rows), h10(rows, wf_param, results["variant_rows"])]
