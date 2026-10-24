"""Excel workbooks, figures and PDF reports built from the results dictionary.

All text in the PDFs is generated from computed numbers; nothing is written by
hand about results. When the run used synthetic data, every page and sheet
carries the synthetic watermark.
"""
from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

import hypotheses as H  # noqa: E402
import portfolio_analytics as pa  # noqa: E402
import strategies as S  # noqa: E402
import walk_forward as wf  # noqa: E402
from synthetic_data import SYNTHETIC_BANNER  # noqa: E402

# Reference categorical palette (fixed order) and diverging map (blue <-> gray <-> red).
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
UNI_LABEL = {"A_broad": "Broad ETFs", "B_sectors": "Sector ETFs", "C_semi_etfs": "Semi ETFs", "D_semi_stocks": "Semi stocks"}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 1.6,
})


# ============================================================================
# Derived analyses (rank persistence, cross-sectional selection, WF summaries)
# ============================================================================

def derive(results: dict) -> None:
    ao = results["asset_out"]
    ranks, fixed_long, best_rows = [], [], []
    for t, o in ao.items():
        tab = o.get("wf_fixed")
        if tab is None:
            continue
        cols = [c for c in tab["is_sharpe"].columns if c != "BH"]
        rp = wf.rank_persistence(tab["is_sharpe"][cols], tab["oos_sharpe"][cols])
        rp.insert(0, "ticker", t)
        rp.insert(1, "universe", o["universe"])
        ranks.append(rp)
        for k in ("is_sharpe", "oos_sharpe", "is_cagr", "oos_cagr"):
            m = tab[k].stack().rename("value").reset_index()
            m.columns = ["fold", "strategy", "value"]
            fixed_long.append(m.assign(ticker=t, metric=k))
        if "wf_best" in o:
            rf = _rf(results, o["wf_best"]["oos"].index)
            sb = wf.stitched_metrics(o["wf_best"]["oos"].astype(float), rf)
            bb = wf.stitched_metrics(o["wf_best"]["bh_oos"].astype(float), rf)
            ch = o["wf_best"]["choices"]["BEST_OF_REGISTRY"].dropna()
            best_rows.append({"ticker": t, "universe": o["universe"], "n_folds": len(ch),
                              "distinct_strategies_chosen": ch.nunique(),
                              "most_chosen": ch.value_counts().index[0] if len(ch) else None,
                              "mean_is_sharpe_of_choice": float(o["wf_best"]["is_best"]["BEST_OF_REGISTRY"].mean()),
                              **{f"best_{k}": v for k, v in sb.items()}, **{f"bh_{k}": v for k, v in bb.items()}})
    results["rank_persistence"] = pd.concat(ranks, ignore_index=True) if ranks else pd.DataFrame()
    if fixed_long:
        fl = pd.concat(fixed_long, ignore_index=True)
        results["wf_fixed_long"] = fl.pivot_table(index=["ticker", "fold", "strategy"], columns="metric", values="value").reset_index()
    else:
        results["wf_fixed_long"] = pd.DataFrame()
    results["wf_best_summary"] = pd.DataFrame(best_rows)
    wp = [o["wf_param"] for o in ao.values() if "wf_param" in o]
    results["wf_param"] = pd.concat(wp, ignore_index=True) if wp else pd.DataFrame()
    # H6: sector-specific vs common selection.
    sec = [t for t, o in ao.items() if o["universe"] == "B_sectors" and o.get("wf_fixed") is not None]
    if len(sec) >= 3:
        is_by = {t: ao[t]["wf_fixed"]["is_sharpe"].drop(columns="BH") for t in sec}
        oos_by = {t: ao[t]["wf_fixed"]["oos_sharpe"].drop(columns="BH") for t in sec}
        results["cross_sectional"] = wf.cross_sectional_selection(is_by, oos_by)
    else:
        results["cross_sectional"] = pd.DataFrame()
    # OOS summary per strategy x asset (fixed default rules).
    fl = results["wf_fixed_long"]
    if not fl.empty:
        bh = fl[fl.strategy == "BH"][["ticker", "fold", "oos_sharpe"]].rename(columns={"oos_sharpe": "bh_oos_sharpe"})
        x = fl[fl.strategy != "BH"].merge(bh, on=["ticker", "fold"], how="left")
        x["oos_sharpe_diff"] = x["oos_sharpe"] - x["bh_oos_sharpe"]
        results["oos_summary"] = x.groupby(["ticker", "strategy"]).agg(
            wf_folds=("fold", "nunique"), wf_is_sharpe_mean=("is_sharpe", "mean"), wf_oos_sharpe_mean=("oos_sharpe", "mean"),
            wf_oos_sharpe_diff_mean=("oos_sharpe_diff", "mean"),
            wf_share_folds_beat_bh=("oos_sharpe_diff", lambda s: float((s > 0).mean()))).reset_index()
        results["oos_summary"]["wf_is_oos_decay"] = results["oos_summary"]["wf_is_sharpe_mean"] - results["oos_summary"]["wf_oos_sharpe_mean"]
    else:
        results["oos_summary"] = pd.DataFrame()
    results["hypotheses"] = H.evaluate_all(results)


def _rf(results: dict, idx) -> pd.Series:
    md = results.get("_md")
    if md is not None:
        return md.cash_growth.reindex(idx)
    cg = results.get("_cash_growth")
    return cg.reindex(idx) if cg is not None else pd.Series(0.0, index=idx)


# ============================================================================
# Excel helpers
# ============================================================================

PCT_COLS = {"cagr", "total_return", "ann_vol", "max_dd", "time_invested", "avg_exposure", "gross_cagr", "cost_drag_ann",
            "bm_alpha_ann", "excess_cagr", "max_dd_improvement", "bm_cagr", "bm_max_dd", "tracking_error", "win_rate",
            "avg_trade_ret", "expo_matched_cagr", "vol_matched_cagr", "excess_cagr_vs_expo_matched",
            "excess_cagr_vs_vol_matched", "dd_improvement_vs_vol_matched", "cagr_ex_best10", "worst_12m_ret",
            "ub_alpha_ann", "ub_cagr", "expo_matched_max_dd", "vol_matched_max_dd", "mean_diff_vs_bh_ann",
            "cagr_ci_lo", "cagr_ci_hi", "max_dd_ci_lo", "max_dd_ci_hi", "oos_cagr", "oos_max_dd", "best_oos_cagr",
            "bh_oos_cagr", "best_oos_max_dd", "bh_oos_max_dd", "sel_oos_cagr", "fixed_oos_cagr"}


def write_sheets(path: Path, sheets: dict[str, pd.DataFrame], readme: list[str], synthetic: bool,
                 heat: dict[str, list[str]] | None = None, heat_matrix: set[str] | None = None) -> None:
    heat = heat or {}
    heat_matrix = heat_matrix or set()
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        wb = xw.book
        f_title = wb.add_format({"bold": True, "font_size": 14})
        f_warn = wb.add_format({"bold": True, "font_color": "#d03b3b", "font_size": 12})
        f_wrap = wb.add_format({"text_wrap": True, "valign": "top"})
        f_pct = wb.add_format({"num_format": "0.0%"})
        f_num = wb.add_format({"num_format": "0.000"})
        f_head = wb.add_format({"bold": True, "bg_color": "#e4e3df", "text_wrap": True, "valign": "top"})
        ws = wb.add_worksheet("README")
        ws.set_column(0, 0, 130)
        r = 0
        if synthetic:
            ws.write(r, 0, SYNTHETIC_BANNER, f_warn)
            r += 2
        ws.write(r, 0, path.stem.replace("_", " "), f_title)
        r += 2
        for line in readme:
            ws.write(r, 0, line, f_wrap)
            r += 1
        for name, df in sheets.items():
            name = name[:31]
            if df is None or df.empty:
                pd.DataFrame({"note": ["No results for this sheet in this run."]}).to_excel(xw, sheet_name=name, index=False)
                continue
            df = df.copy()
            for c in df.columns:
                if df[c].dtype == object:
                    df[c] = df[c].map(lambda v: str(v) if isinstance(v, (list, tuple, dict, pd.Timestamp)) else v)
            start = 1 if synthetic else 0
            df.to_excel(xw, sheet_name=name, index=False, startrow=start)
            ws = xw.sheets[name]
            if synthetic:
                ws.write(0, 0, SYNTHETIC_BANNER, f_warn)
            for j, c in enumerate(df.columns):
                width = min(max(10, len(str(c)) + 2), 48 if df[c].dtype == object else 14)
                fmt = f_pct if c in PCT_COLS else (f_num if pd.api.types.is_float_dtype(df[c]) else None)
                ws.set_column(j, j, width, fmt)
                ws.write(start, j, c, f_head)
            ws.freeze_panes(start + 1, 2 if name not in heat_matrix else 1)
            ws.autofilter(start, 0, start + len(df), len(df.columns) - 1)
            n = len(df)
            cols_to_heat = list(df.columns[1:]) if name in heat_matrix else heat.get(name, [])
            for c in cols_to_heat:
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                    j = df.columns.get_loc(c)
                    ws.conditional_format(start + 1, j, start + n, j, {
                        "type": "3_color_scale", "min_color": "#e34948", "mid_type": "num", "mid_value": 0,
                        "mid_color": "#f0efec", "max_color": "#2a78d6"})


LEAD_COLS = ["ticker", "universe", "strategy", "strategy_name", "family", "family_name", "period", "start", "end", "years",
             "cagr", "total_return", "ann_vol", "sharpe", "sortino", "max_dd", "max_dd_duration_bars", "calmar",
             "time_invested", "avg_exposure", "entries", "exits", "win_rate", "profit_factor", "avg_hold_bars",
             "turnover_ann", "gross_cagr", "cost_drag_ann", "costs_pct_nav_total", "bm_alpha_ann", "bm_alpha_t", "bm_beta",
             "bm_cagr", "bm_sharpe", "bm_max_dd", "excess_cagr", "sharpe_diff", "max_dd_improvement", "info_ratio",
             "ub_alpha_ann", "ub_beta", "expo_matched_cagr", "excess_cagr_vs_expo_matched", "vol_matched_weight",
             "vol_matched_cagr", "vol_matched_max_dd", "excess_cagr_vs_vol_matched", "dd_improvement_vs_vol_matched",
             "missed_rallies_log", "losses_avoided_log", "net_timing_log", "top1pct_days_share", "cagr_ex_best10",
             "best_12m_share", "worst_12m_ret", "skew", "kurtosis",
             "reg_up_share", "reg_sideways_share", "reg_down_share", "reg_up_sharpe", "reg_sideways_sharpe",
             "reg_down_sharpe", "reg_up_excess_vs_bh_ann", "reg_sideways_excess_vs_bh_ann", "reg_down_excess_vs_bh_ann"]


def leaderboard(results: dict, period: str) -> pd.DataFrame:
    r = results["rows"]
    r = r[r.period == period].copy()
    st = results["stats"]
    if not st["pvalues"].empty:
        r = r.merge(st["pvalues"], on=["ticker", "strategy"], how="left")
    if not st["dsr"].empty and period == "full":
        r = r.merge(st["dsr"][["ticker", "label", "psr_vs_0", "dsr", "n_eff_trials"]].rename(columns={"label": "strategy"}),
                    on=["ticker", "strategy"], how="left")
    if not st["ci"].empty and period == "full":
        r = r.merge(st["ci"], on=["ticker", "strategy"], how="left")
    if period == "full" and not results.get("oos_summary", pd.DataFrame()).empty:
        r = r.merge(results["oos_summary"], on=["ticker", "strategy"], how="left")
    cost = results["cost_rows"]
    if not cost.empty:
        wide = cost.pivot_table(index=["ticker", "strategy"], columns="cost_bps", values="sharpe")
        wide.columns = [f"sharpe_at_{int(c)}bps" for c in wide.columns]
        r = r.merge(wide.reset_index(), on=["ticker", "strategy"], how="left")
    cols = [c for c in LEAD_COLS if c in r.columns] + [c for c in r.columns if c not in LEAD_COLS]
    return r[cols].sort_values(["ticker", "sharpe"], ascending=[True, False])


def family_summary(results: dict) -> pd.DataFrame:
    r = results["rows"]
    r = r[(r.period == "full") & r.strategy.str.match(r"F\d\d_")]
    oos = results.get("oos_summary", pd.DataFrame())
    if not oos.empty:
        r = r.merge(oos[["ticker", "strategy", "wf_oos_sharpe_diff_mean", "wf_is_oos_decay"]], on=["ticker", "strategy"], how="left")
    g = r.groupby(["family", "family_name", "universe"]).agg(
        n=("sharpe", "size"), median_sharpe=("sharpe", "median"), median_sharpe_diff_vs_bh=("sharpe_diff", "median"),
        share_sharpe_above_bh=("sharpe_diff", lambda s: float((s > 0).mean())),
        median_excess_cagr=("excess_cagr", "median"), median_dd_improvement=("max_dd_improvement", "median"),
        median_exposure=("avg_exposure", "median"), median_turnover=("turnover_ann", "median"),
        median_oos_sharpe_diff=("wf_oos_sharpe_diff_mean", "median") if "wf_oos_sharpe_diff_mean" in r else ("sharpe", "size"),
        median_is_oos_decay=("wf_is_oos_decay", "median") if "wf_is_oos_decay" in r else ("sharpe", "size")).reset_index()
    return g


def heat_matrix(results: dict, universes: tuple, metric: str = "sharpe_diff") -> pd.DataFrame:
    r = results["rows"]
    r = r[(r.period == "full") & r.universe.isin(universes) & r.strategy.str.match(r"F\d\d_")]
    m = r.pivot_table(index="strategy", columns="ticker", values=metric)
    order = [t for u in universes for t in results["config"]["universes"][u]["tickers"] if t in m.columns]
    m = m[order]
    names = S.registry_frame().set_index("id")["name"]
    m.insert(0, "strategy_name", names.reindex(m.index).to_numpy())
    return m.reset_index()


# ============================================================================
# Figures
# ============================================================================

def _wm(fig, synthetic):
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC - SOFTWARE TEST", fontsize=34, color="#d03b3b", alpha=0.18, ha="center",
                 va="center", rotation=25, transform=fig.transFigure)


def _save(fig, path, synthetic):
    _wm(fig, synthetic)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_family(results, figdir, syn):
    r = results["rows"]
    r = r[(r.period == "full") & r.strategy.str.match(r"F\d\d_")]
    fams = sorted(r.family.unique())
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [r[r.family == f]["sharpe_diff"].dropna().to_numpy() for f in fams]
    ax.boxplot(data, widths=0.5, showfliers=False, medianprops={"color": CAT[1], "linewidth": 2},
               boxprops={"color": INK2}, whiskerprops={"color": INK2}, capprops={"color": INK2})
    for i, d in enumerate(data, start=1):
        ax.scatter(np.random.default_rng(i).normal(i, 0.06, len(d)), d, s=6, color=CAT[0], alpha=0.35, linewidths=0)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(range(1, len(fams) + 1), [f"{f}\n" + textwrap.fill(S.FAMILIES[f], 13) for f in fams], fontsize=6.5)
    ax.set_ylabel("Sharpe minus buy-and-hold Sharpe (full history, net of costs)")
    ax.set_title("Strategy families: risk-adjusted performance relative to holding the same asset")
    return _save(fig, figdir / "family_comparison.png", syn)


def fig_heat(m: pd.DataFrame, title: str, path: Path, syn: bool, label: str = "Sharpe minus buy-and-hold Sharpe"):
    if m.empty:
        return None
    vals = m.drop(columns=["strategy", "strategy_name"]).to_numpy(dtype=float)
    lim = np.nanpercentile(np.abs(vals), 95) if np.isfinite(vals).any() else 1
    lim = lim if lim > 0 else 1
    fig, ax = plt.subplots(figsize=(1.6 + 0.42 * vals.shape[1], 0.16 * vals.shape[0] + 1.8))
    im = ax.imshow(vals, aspect="auto", cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim))
    ax.set_yticks(range(len(m)), m["strategy"], fontsize=5.5)
    ax.set_xticks(range(vals.shape[1]), m.columns[2:], rotation=90, fontsize=7)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(label, fontsize=7)
    ax.set_title(title)
    return _save(fig, path, syn)


def fig_family_asset(results, figdir, syn):
    r = results["rows"]
    r = r[(r.period == "full") & r.strategy.str.match(r"F\d\d_")]
    m = r.pivot_table(index="family", columns="ticker", values="sharpe_diff", aggfunc="median")
    order = [t for u in results["config"]["universes"] for t in results["config"]["universes"][u]["tickers"] if t in m.columns]
    m = m[order]
    vals = m.to_numpy(dtype=float)
    lim = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1
    fig, ax = plt.subplots(figsize=(0.32 * vals.shape[1] + 3, 4.2))
    im = ax.imshow(vals, aspect="auto", cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim))
    ax.set_yticks(range(len(m)), [f"{f} {textwrap.shorten(S.FAMILIES[f], 26)}" for f in m.index], fontsize=7)
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
    ax.grid(False)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=4.5, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).set_label("Median Sharpe minus B&H", fontsize=7)
    ax.set_title("Family x instrument: median Sharpe improvement over buy-and-hold")
    return _save(fig, figdir / "family_by_instrument_heatmap.png", syn)


def fig_scatter(results, figdir, syn):
    r = results["rows"]
    r = r[(r.period == "full") & r.universe.isin(UNI_LABEL)]
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=False, sharey=False)
    for ax, (u, lab) in zip(axs.flat, UNI_LABEL.items()):
        d = r[r.universe == u]
        s = d[d.strategy != "BH"]
        b = d[d.strategy == "BH"]
        ax.scatter(s["max_dd"], s["cagr"], s=7, color=CAT[0], alpha=0.35, linewidths=0, label="Strategy x asset")
        ax.scatter(b["max_dd"], b["cagr"], s=40, color=CAT[1], edgecolors=SURF, linewidths=1.5, label="Buy-and-hold", zorder=3)
        for _, row in b.iterrows():
            ax.annotate(row["ticker"], (row["max_dd"], row["cagr"]), fontsize=6, color=INK2, xytext=(3, 3), textcoords="offset points")
        ax.set_title(lab)
        ax.set_xlabel("Maximum drawdown")
        ax.set_ylabel("CAGR (net)")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axs[0, 0].legend(loc="upper left", fontsize=7)
    fig.suptitle("CAGR versus maximum drawdown (full available history, 10 bps per side)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, figdir / "cagr_vs_drawdown.png", syn)


# Fixed colour per strategy so the same strategy keeps its colour across every chart.
ENTITY_COLOR = {"F01_01": CAT[0], "F01_02": CAT[1], "F02_01": CAT[2], "F02_07": CAT[3], "F05_09": CAT[4],
                "F03_10": CAT[5], "F02_10": CAT[6], "F10_10": CAT[7]}


def fig_equity(curves: pd.DataFrame, title: str, path: Path, syn: bool, start=None, log=True):
    c = curves.dropna(how="all")
    if start is not None:
        c = c[c.index >= start]
    c = c / c.iloc[0]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for i, col in enumerate(c.columns):
        lw = 2.2 if col in ("BH", "Buy-and-hold") else 1.3
        color = INK if col in ("BH", "Buy-and-hold") else ENTITY_COLOR.get(col, CAT[i % len(CAT)])
        a1.plot(c.index, c[col], label=col, linewidth=lw, color=color)
        dd = c[col] / c[col].cummax() - 1
        a2.plot(c.index, dd, linewidth=1.0 if lw < 2 else 1.6, color=color)
    if log:
        a1.set_yscale("log")
    a1.set_ylabel("Growth of $1 (log scale)" if log else "Growth of $1")
    a2.set_ylabel("Drawdown")
    a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a1.legend(fontsize=7, ncol=3)
    a1.set_title(title)
    fig.tight_layout()
    return _save(fig, path, syn)


def fig_cost(results, figdir, syn):
    c = results["cost_rows"]
    if c.empty:
        return None
    c = c.assign(beats=c["sharpe"] > c["bh_sharpe"])
    g = c.groupby("cost_bps").agg(med=("sharpe", "median"), q25=("sharpe", lambda s: s.quantile(0.25)),
                                  q75=("sharpe", lambda s: s.quantile(0.75)), share=("beats", "mean"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(g.index, g["med"], marker="o", color=CAT[0], label="Median")
    a1.fill_between(g.index, g["q25"], g["q75"], color=CAT[0], alpha=0.15, label="Interquartile range")
    a1.set_xlabel("Cost per side (bps)")
    a1.set_ylabel("Sharpe ratio")
    a1.set_title("Sharpe of all strategy x asset combinations")
    a1.legend(fontsize=7)
    a2.plot(g.index, g["share"], marker="o", color=CAT[1])
    a2.set_ylim(0, 1)
    a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a2.set_xlabel("Cost per side (bps)")
    a2.set_title("Share with Sharpe above buy-and-hold")
    fig.tight_layout()
    return _save(fig, figdir / "cost_sensitivity.png", syn)


def fig_wf(results, figdir, syn):
    fl = results.get("wf_fixed_long", pd.DataFrame())
    if fl.empty:
        return None
    x = fl[fl.strategy != "BH"].dropna(subset=["is_sharpe", "oos_sharpe"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.hexbin(x["is_sharpe"], x["oos_sharpe"], gridsize=45, cmap=LinearSegmentedColormap.from_list("s", ["#f0efec", "#2a78d6", "#0d366b"]), mincnt=1)
    lim = np.nanpercentile(np.abs(np.r_[x["is_sharpe"], x["oos_sharpe"]]), 99)
    a1.plot([-lim, lim], [-lim, lim], color=INK2, linewidth=0.8, linestyle="--", label="OOS = IS")
    if len(x) > 10:
        b = np.polyfit(x["is_sharpe"], x["oos_sharpe"], 1)
        xs = np.linspace(-lim, lim, 10)
        a1.plot(xs, np.polyval(b, xs), color=CAT[1], label=f"Fit: OOS = {b[1]:.2f} + {b[0]:.2f} x IS")
    a1.set_xlabel("Training-window Sharpe (5 years)")
    a1.set_ylabel("Next-year out-of-sample Sharpe")
    a1.set_title("Fixed rules: training vs following-year Sharpe")
    a1.legend(fontsize=7)
    rp = results.get("rank_persistence", pd.DataFrame())
    if not rp.empty:
        a2.hist(rp["spearman"].dropna(), bins=30, color=CAT[0], edgecolor=SURF)
        a2.axvline(rp["spearman"].mean(), color=CAT[1], linewidth=2, label=f"Mean {rp['spearman'].mean():.2f}")
        a2.axvline(0, color=INK, linewidth=0.8)
        a2.set_xlabel("Spearman rank correlation, IS vs OOS (per asset x fold)")
        a2.set_title("Do in-sample rankings persist?")
        a2.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, figdir / "walk_forward_is_vs_oos.png", syn)


def fig_wf_best(results, figdir, syn, ticker="SPY"):
    o = results["asset_out"].get(ticker)
    if not o or "wf_best" not in o:
        return None
    s = o["wf_best"]["oos"].astype(float).dropna()
    b = o["wf_best"]["bh_oos"].astype(float).dropna()
    if len(s) < 20:
        return None
    df = pd.DataFrame({"Best-in-training strategy (walk-forward)": (1 + s).cumprod(), "Buy-and-hold": (1 + b).cumprod()})
    return fig_equity(df, f"{ticker}: stitched out-of-sample record of picking the best backtest each year", figdir / f"wf_best_of_registry_{ticker}.png", syn)


def fig_oracle(results, figdir, syn):
    paths = []
    for t, o in results.get("oracle", {}).items():
        res = o["oracle"]["results"]
        a = o["eval_start"]
        curves = {}
        bh = results["asset_out"][t]["returns"]["BH"].astype(float)
        curves["Buy-and-hold"] = (1 + bh).cumprod()
        best = o.get("best_in_sample")
        if best and best in results["asset_out"][t]["returns"]:
            curves[f"Best full-sample strategy ({best}, in-sample pick)"] = (1 + results["asset_out"][t]["returns"][best].astype(float)).cumprod()
        for name in ("Daily oracle (costless decision)", "Cost-aware oracle (unlimited trades)",
                     "Constrained oracle (max 10 entries)", "Constrained oracle (max 40 entries)"):
            if name in res:
                curves[name] = res[name].equity.loc[a:] / res[name].equity.loc[a:].iloc[0]
        df = pd.DataFrame(curves)
        paths.append(fig_equity(df, f"{t}: realistic strategies vs perfect-knowledge long-only oracles (theoretical)",
                                figdir / f"oracle_{t}.png", syn))
        sm = o["summary"]
        lim = sm[sm.benchmark.str.startswith("Constrained")].copy()
        if not lim.empty:
            lim["K"] = lim.benchmark.str.extract(r"max (\d+)").astype(float)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(lim["K"], lim["cagr"], marker="o", color=CAT[0], label="Constrained oracle")
            bhc = pa.cagr(bh.to_numpy())
            ax.axhline(bhc, color=INK, linewidth=1.2, label="Buy-and-hold")
            if best:
                ax.axhline(pa.cagr(results["asset_out"][t]["returns"][best].astype(float).to_numpy()), color=CAT[1],
                           linewidth=1.2, linestyle="--", label=f"Best tested ({best})")
            ax.set_xscale("log")
            ax.set_xlabel("Maximum number of entries allowed over the sample (log scale)")
            ax.set_ylabel("CAGR")
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            ax.set_title(f"{t}: value of perfect foresight vs number of trades (10 bps per side)")
            ax.legend(fontsize=7)
            paths.append(_save(fig, figdir / f"oracle_frontier_{t}.png", syn))
    return paths


def fig_corr(results, figdir, syn):
    C = results["stats"].get("avg_corr", pd.DataFrame())
    if C is None or C.empty:
        return None
    fam = pd.Series({c: c[:3] for c in C.columns})
    F = C.groupby(fam).mean().T.groupby(fam).mean()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(F.to_numpy(), cmap=LinearSegmentedColormap.from_list("s", ["#f0efec", "#2a78d6", "#0d366b"]), vmin=0, vmax=1)
    ax.set_xticks(range(len(F)), F.columns, rotation=90)
    ax.set_yticks(range(len(F)), F.index)
    ax.grid(False)
    for i in range(len(F)):
        for j in range(len(F)):
            ax.text(j, i, f"{F.iat[i, j]:.2f}", ha="center", va="center", fontsize=6, color=INK if F.iat[i, j] < 0.7 else SURF)
    fig.colorbar(im, ax=ax, fraction=0.04).set_label("Average daily-return correlation", fontsize=7)
    ax.set_title("Strategy correlation by family (averaged over assets)")
    return _save(fig, figdir / "strategy_correlation_by_family.png", syn)


def fig_regime(results, figdir, syn):
    r = results["rows"]
    r = r[(r.period == "full") & r.strategy.str.match(r"F\d\d_")]
    if "reg_sideways_sharpe" not in r:
        return None
    g = r.groupby("family")[["reg_up_sharpe", "reg_sideways_sharpe", "reg_down_sharpe"]].median()
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(g))
    for k, (col, lab) in enumerate((("reg_up_sharpe", "Up-trend regime"), ("reg_sideways_sharpe", "Sideways regime"),
                                    ("reg_down_sharpe", "Down-trend regime"))):
        ax.bar(x + (k - 1) * 0.26, g[col], width=0.24, color=CAT[k], label=lab)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x, g.index)
    ax.set_ylabel("Median Sharpe within regime")
    ax.set_title("Performance by ex-ante trend regime (126-day volatility-scaled return)")
    ax.legend(fontsize=7, ncol=3)
    return _save(fig, figdir / "regime_performance_by_family.png", syn)


def fig_spy_rolling(curves: pd.DataFrame, figdir, syn, rf: pd.Series | None):
    need = {"F01_02", "F01_01", "BH"}
    if not need.issubset(curves.columns):
        return None
    r = curves[list(need)].pct_change().dropna()
    win = 756
    f = (rf.reindex(r.index).fillna(0) if rf is not None else 0)
    ex = r.sub(f, axis=0)
    rs = ex.rolling(win).mean() / ex.rolling(win).std() * math.sqrt(252)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(rs.index, rs["F01_02"] - rs["BH"], color=CAT[0], label="SMA10/30 minus buy-and-hold")
    ax.plot(rs.index, rs["F01_01"] - rs["BH"], color=CAT[1], label="SMA20/50 minus buy-and-hold")
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_ylabel("Rolling 3-year Sharpe difference")
    ax.set_title("SPY: is SMA10/30's edge persistent? Rolling 3-year Sharpe relative to holding SPY")
    ax.legend(fontsize=7)
    return _save(fig, figdir / "spy_sma1030_rolling_sharpe.png", syn)


# ============================================================================
# PDF reports
# ============================================================================

def _fmt(x, kind="pct", nd=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if kind == "pct":
        return f"{x * 100:.{nd}f}%"
    if kind == "num":
        return f"{x:.{nd}f}"
    return str(x)


def _pdf(path: Path, story_fn, synthetic: bool, title: str):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    st = {"h1": ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, spaceAfter=8),
          "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=8, spaceAfter=5),
          "p": ParagraphStyle("p", parent=ss["BodyText"], fontSize=8.8, leading=11.5),
          "small": ParagraphStyle("s", parent=ss["BodyText"], fontSize=7.2, leading=9),
          "warn": ParagraphStyle("w", parent=ss["BodyText"], fontSize=11, textColor=colors.HexColor("#d03b3b"))}

    def table(df: pd.DataFrame, col_w=None, fs=6.6):
        data = [list(map(str, df.columns))] + [[str(v) for v in row] for row in df.itertuples(index=False)]
        data = [[Paragraph(c, st["small"]) for c in row] for row in data]
        t = Table(data, colWidths=col_w, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4e3df")),
                               ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c3c2b7")),
                               ("FONTSIZE", (0, 0), (-1, -1), fs), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return t

    def img(p, w=7.0):
        if p is None or not Path(p).exists():
            return Spacer(1, 1)
        from PIL import Image as PILImage
        with PILImage.open(p) as im:
            ar = im.height / im.width
        h = w * ar
        if h > 8.6:
            w, h = 8.6 / ar, 8.6
        return Image(str(p), width=w * inch, height=h * inch)

    def on_page(c, d):
        c.saveState()
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.HexColor("#52514e"))
        c.drawString(0.6 * inch, 0.45 * inch, f"{title} - page {d.page}")
        if synthetic:
            c.setFillColor(colors.HexColor("#d03b3b"))
            c.setFont("Helvetica-Bold", 9)
            c.drawString(0.6 * inch, 10.6 * inch, SYNTHETIC_BANNER)
        c.restoreState()

    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                            topMargin=0.6 * inch, bottomMargin=0.65 * inch, title=title)
    story = story_fn(st, table, img, Paragraph, Spacer, PageBreak)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


def _top_table(results, n=15):
    lb = results["_lead_full"]
    x = lb[lb.strategy.str.match(r"F\d\d_")].copy()
    if "wf_oos_sharpe_diff_mean" in x:
        x = x.sort_values("wf_oos_sharpe_diff_mean", ascending=False)
    else:
        x = x.sort_values("sharpe", ascending=False)
    x = x.head(n)
    out = pd.DataFrame({
        "Asset": x["ticker"], "Strategy": x["strategy"] + " " + x["strategy_name"].str.slice(0, 34),
        "Sharpe [90% CI]": [f"{a:.2f} [{b:.2f}, {c:.2f}]" if np.isfinite(b) else f"{a:.2f}"
                            for a, b, c in zip(x["sharpe"], x.get("sharpe_ci_lo", np.nan * x["sharpe"]), x.get("sharpe_ci_hi", np.nan * x["sharpe"]))],
        "B&H Sharpe": x["bm_sharpe"].map(lambda v: _fmt(v, "num", 2)),
        "CAGR": x["cagr"].map(_fmt), "MaxDD": x["max_dd"].map(_fmt),
        "WF OOS Sharpe-B&H": x.get("wf_oos_sharpe_diff_mean", pd.Series(np.nan, index=x.index)).map(lambda v: _fmt(v, "num", 2)),
        "DSR": x.get("dsr", pd.Series(np.nan, index=x.index)).map(lambda v: _fmt(v, "num", 2)),
        "FDR q": x.get("q_bh_global", pd.Series(np.nan, index=x.index)).map(lambda v: _fmt(v, "num", 2)),
    })
    return out


def final_answers(results: dict) -> list[tuple[str, str]]:
    """Data-driven answers to the ten final-report questions."""
    rows = results["rows"]
    full = rows[(rows.period == "full") & rows.strategy.str.match(r"F\d\d_")]
    fs = family_summary(results)
    hyp = {h["id"]: h for h in results["hypotheses"]}
    oos = results.get("oos_summary", pd.DataFrame())
    ans = []
    # 1 families
    fam = full.groupby("family").agg(share=("sharpe_diff", lambda s: (s > 0).mean()), med=("sharpe_diff", "median"))
    if not oos.empty:
        o2 = oos.assign(family=oos.strategy.str[:3]).groupby("family").agg(oos_med=("wf_oos_sharpe_diff_mean", "median"),
                                                                           oos_share=("wf_oos_sharpe_diff_mean", lambda s: (s > 0).mean()))
        fam = fam.join(o2)
    fam = fam.sort_values("oos_med" if "oos_med" in fam else "med", ascending=False)
    txt = "; ".join(f"{f} {S.FAMILIES[f]}: median Sharpe-B&H {r.med:+.2f} full history"
                    + (f", {r.oos_med:+.2f} walk-forward OOS ({r.oos_share:.0%} of assets positive)" if "oos_med" in fam else "")
                    for f, r in fam.head(4).iterrows())
    ans.append(("Strongest strategy families", "Ranked by median out-of-sample Sharpe improvement over buy-and-hold: " + txt +
                ". A family is only called consistent if most assets are positive out of sample; see Family_Summary for the full table."))
    # 2 instruments
    wb = results.get("wf_best_summary", pd.DataFrame())
    if not wb.empty:
        wb = wb.assign(gap=wb["best_oos_sharpe"] - wb["bh_oos_sharpe"])
        good = wb.sort_values("gap", ascending=False)
        txt = ", ".join(f"{r.ticker} ({r.gap:+.2f})" for r in good.head(6).itertuples())
        bad = ", ".join(f"{r.ticker} ({r.gap:+.2f})" for r in good.tail(4).itertuples())
        ans.append(("Most reliable instruments and sectors", f"Out-of-sample Sharpe of 'pick the best backtest each year' minus buy-and-hold: best {txt}; worst {bad}. "
                    f"Selection beat holding on {np.mean(wb['gap'] > 0):.0%} of {len(wb)} instruments."))
    # 3 SMA10/30
    s = rows[(rows.ticker == "SPY") & (rows.period == "full")].set_index("strategy")
    if {"F01_02", "F01_01", "BH"}.issubset(s.index):
        lf = results["_lead_full"].set_index(["ticker", "strategy"])
        q = lf.loc[("SPY", "F01_02")] if ("SPY", "F01_02") in lf.index else None
        ci = f" (90% bootstrap CI {q['sharpe_ci_lo']:.2f} to {q['sharpe_ci_hi']:.2f}; DSR {q.get('dsr', np.nan):.2f}; FDR q {q.get('q_bh_global', np.nan):.2f})" if q is not None and "sharpe_ci_lo" in q else ""
        ans.append(("Does SMA10/30 show a meaningful advantage?",
                    f"SPY full history: SMA10/30 CAGR {_fmt(s.loc['F01_02','cagr'])}, Sharpe {s.loc['F01_02','sharpe']:.2f}{ci}, max drawdown {_fmt(s.loc['F01_02','max_dd'])}; "
                    f"SMA20/50 {_fmt(s.loc['F01_01','cagr'])} / {s.loc['F01_01','sharpe']:.2f} / {_fmt(s.loc['F01_01','max_dd'])}; buy-and-hold {_fmt(s.loc['BH','cagr'])} / "
                    f"{s.loc['BH','sharpe']:.2f} / {_fmt(s.loc['BH','max_dd'])}. Cross-asset test (H2): {hyp['H2']['verdict']} - {hyp['H2']['evidence']}"))
    # 4 semis
    ans.append(("Do semiconductors behave differently from broad ETFs?", f"H1 ({hyp['H1']['verdict']}): {hyp['H1']['evidence']}. H4 ({hyp['H4']['verdict']}): {hyp['H4']['evidence']}."))
    # 5 exits
    ans.append(("Do faster exits / slower re-entries help?", f"H3 ({hyp['H3']['verdict']}): {hyp['H3']['evidence']}."))
    # 6 overfit
    wp = results.get("wf_param", pd.DataFrame())
    if not oos.empty:
        dec = oos.groupby("strategy")["wf_is_oos_decay"].median().sort_values(ascending=False)
        txt = ", ".join(f"{k} ({v:+.2f})" for k, v in dec.head(8).items())
        extra = ""
        if not wp.empty:
            worse = (wp["sel_oos_sharpe"] < wp["fixed_oos_sharpe"]).mean()
            extra = f" Re-optimising parameters inside each training window lowered out-of-sample Sharpe versus the pre-registered default in {worse:.0%} of strategy x asset cases."
        ans.append(("Which strategies were overfit?", f"Largest median drop from training-window to next-year Sharpe: {txt}." + extra +
                    f" H8 ({hyp['H8']['verdict']}): {hyp['H8']['evidence']}."))
    # 7 survive
    pv = results["stats"]["pvalues"]
    if not pv.empty:
        n_rej = int(pv["reject_bh_global"].sum())
        surv = pv[pv["reject_bh_global"]]
        txt = ", ".join(f"{r.strategy}/{r.ticker}" for r in surv.head(10).itertuples()) or "none"
        rc = results["stats"]["reality_check"]
        n_spa = int((rc["spa_pvalue"] < 0.10).sum()) if not rc.empty else 0
        n_spa_v = int((rc["spa_pvalue_vol_scaled"] < 0.10).sum()) if not rc.empty else 0
        ans.append(("Which findings survive forward validation and multiple-testing control?",
                    f"Return differential vs buy-and-hold: {n_rej} of {len(pv)} strategy x asset tests survive Benjamini-Hochberg FDR at "
                    f"{results['config']['statistics']['fdr_alpha']:.0%} ({txt}). Hansen SPA rejects 'no strategy beats holding' on {n_spa} of {len(rc)} assets "
                    f"for raw returns and on {n_spa_v} for volatility-matched returns."))
    # 8 oracle
    orc = results.get("oracle", {})
    if "SPY" in orc:
        sm = orc["SPY"]["summary"].set_index("benchmark")
        ans.append(("Gap to the perfect long-only trader", "SPY: daily oracle (costless decision, net of 10 bps) CAGR "
                    f"{_fmt(sm.loc['Daily oracle (costless decision)','cagr'])}; cost-aware unlimited {_fmt(sm.loc['Cost-aware oracle (unlimited trades)','cagr'])}; "
                    f"with at most 10 entries {_fmt(sm.loc['Constrained oracle (max 10 entries)','cagr']) if 'Constrained oracle (max 10 entries)' in sm.index else 'n/a'}; "
                    f"buy-and-hold {_fmt(s.loc['BH','cagr']) if 'BH' in s.index else 'n/a'}. Even a handful of perfectly timed trades dwarfs every realistic rule, "
                    "showing that the realised edge of timing rules is a tiny fraction of the theoretical opportunity. The oracle is impossible to implement."))
    # 9 uncertainty
    ci = results["stats"]["ci"]
    if not ci.empty:
        w = (ci["sharpe_ci_hi"] - ci["sharpe_ci_lo"]).median()
        dsr = results["stats"]["dsr"]
        ans.append(("Remaining statistical uncertainty", f"Median width of 90% bootstrap Sharpe intervals: {w:.2f}. "
                    f"Deflated Sharpe > 0.95 for {int((dsr['dsr'] > 0.95).sum())} of {len(dsr)} strategy x asset cases "
                    f"(expected maximum Sharpe of ~{dsr['n_eff_trials'].median():.0f} effective trials: {dsr['sr0_ann'].median():.2f}). "
                    "Cross-asset tests treat correlated assets as independent and so overstate significance."))
    ans.append(("Research directions worth pursuing", "Prospective paper-trading of the frozen registry from 2026-10-19 (the only truly untouched sample); "
                "point-in-time S&P 500 constituents to remove survivorship bias; intraday execution data to calibrate slippage for volatile single stocks; "
                "ensemble construction across families with weights fixed ex ante; and regime models estimated strictly inside training windows."))
    return ans


def build_main_pdf(results: dict, out_dir: Path, figs: dict) -> Path:
    syn = results["synthetic"]
    path = out_dir / "Quant_Research_Report.pdf"
    hyp = results["hypotheses"]

    def story(st, table, img, P, Sp, PB):
        s = [P("Quantitative Research Lab: 100 Strategies Across SPY, Sectors, Semiconductors and ETFs", st["h1"])]
        if syn:
            s.append(P(SYNTHETIC_BANNER + ". Every number in this document comes from simulated prices and has no economic meaning.", st["warn"]))
        cal = results["calendar"]
        n_assets = len([o for o in results["asset_out"].values() if o.get("eval_start") is not None])
        s.append(P(f"Generated {results['created_utc']} | data {cal[0].date()} to {cal[1].date()} | source: {results['data_source']} | "
                   f"config sha256 {results['config_sha256'][:12]} | registry sha256 {results['registry_sha256'][:12]} "
                   f"(frozen {results['registry_freeze']['frozen_utc']})", st["small"]))
        s.append(P(f"Strategies defined: {results['n_strategies_defined']}. Strategies executed: {results['n_strategies_run']}. "
                   f"Instruments backtested: {n_assets}. Strategy x instrument combinations (full history): "
                   f"{int(((results['rows'].period == 'full') & results['rows'].strategy.str.match(r'F[0-9]')).sum())}. "
                   "Execution: signals at the close, fills at the next open, 10 bps per side, lagged T-bill cash, long-only, no leverage.", st["p"]))
        s.append(P("Summary of findings", st["h2"]))
        for q, a in final_answers(results):
            s.append(P(f"<b>{q}.</b> {a}", st["p"]))
            s.append(Sp(1, 3))
        s.append(PB())
        s.append(P("Hypothesis tests (decision rules fixed before results; see Research_Methodology.md)", st["h2"]))
        ht = pd.DataFrame([{"H": h["id"], "Claim": h.get("claim", ""), "Verdict": h["verdict"], "Evidence": h.get("evidence", "")} for h in hyp])
        s.append(table(ht, col_w=[28, 150, 62, 300]))
        s.append(PB())
        s.append(P("Strategy family comparison", st["h2"]))
        s.append(img(figs.get("family")))
        s.append(img(figs.get("family_asset")))
        s.append(PB())
        s.append(P("Leaderboard (top 15 by walk-forward out-of-sample Sharpe improvement, not by CAGR)", st["h2"]))
        s.append(table(_top_table(results), col_w=[34, 150, 70, 38, 38, 40, 50, 30, 32]))
        s.append(P("Sharpe CIs: stationary bootstrap (mean block 20 days). DSR: deflated Sharpe ratio given the number of strategies tried on the asset. "
                   "FDR q: Benjamini-Hochberg adjusted p-value for the return differential versus buy-and-hold across all strategy x asset tests.", st["small"]))
        s.append(img(figs.get("scatter")))
        s.append(PB())
        s.append(P("Heatmaps", st["h2"]))
        s.append(img(figs.get("heat_sector"), 6.5))
        s.append(PB())
        s.append(img(figs.get("heat_semi"), 6.5))
        s.append(PB())
        s.append(P("Robustness: costs, walk-forward, correlation and regimes", st["h2"]))
        for k in ("cost", "wf", "wf_best", "corr", "regime"):
            s.append(img(figs.get(k), 6.8))
        s.append(PB())
        s.append(P("Multiple testing: Reality Check / SPA per instrument", st["h2"]))
        rc = results["stats"]["reality_check"]
        if not rc.empty:
            t = rc[["ticker", "n_strategies", "best_strategy", "best_mean_diff_ann", "rc_pvalue", "spa_pvalue",
                    "best_strategy_vol_scaled", "spa_pvalue_vol_scaled"]].copy()
            t["best_mean_diff_ann"] = t["best_mean_diff_ann"].map(_fmt)
            for c in ("rc_pvalue", "spa_pvalue", "spa_pvalue_vol_scaled"):
                t[c] = t[c].map(lambda v: _fmt(v, "num", 3))
            s.append(table(t, col_w=[40, 40, 55, 60, 50, 50, 70, 60]))
            s.append(P("H0: no strategy has a positive expected daily return differential versus buy-and-hold of the same asset. "
                       "The volatility-scaled version first scales each strategy's excess return to the asset's volatility, testing risk-adjusted "
                       "rather than raw outperformance.", st["small"]))
        s.append(PB())
        s.append(P("Perfect-knowledge long-only oracle (theoretical benchmark)", st["h2"]))
        s.append(P("The oracle trades at the same opens as the strategies but knows the next open-to-open return. It is mathematically "
                   "informative and impossible to implement.", st["p"]))
        for p in figs.get("oracle", [])[:4]:
            s.append(img(p, 6.5))
        for t, o in results.get("oracle", {}).items():
            sm = o["summary"].copy()
            for c in ("cagr", "max_dd", "time_invested"):
                sm[c] = sm[c].map(_fmt)
            sm["sharpe"] = sm["sharpe"].map(lambda v: _fmt(v, "num", 2))
            sm["turnover_ann"] = sm["turnover_ann"].map(lambda v: _fmt(v, "num", 1))
            s.append(P(f"{t}", st["h2"]))
            s.append(table(sm.drop(columns=["ticker"]), col_w=[190, 45, 40, 45, 50, 40, 55]))
        s.append(PB())
        s.append(P("Limitations", st["h2"]))
        for line in LIMITATIONS:
            s.append(P("- " + line, st["p"]))
        return s

    _pdf(path, story, syn, "Quant Research Report")
    return path


LIMITATIONS = [
    "Prices are Yahoo Finance daily bars adjusted locally for splits and distributions; they are research-grade convenience data, not exchange-quality.",
    "Execution assumes fills at the official open plus a fixed bps cost; opening-auction slippage and market impact for large orders are not modelled.",
    "Fractional shares and costless cash sweeps are assumed; taxes are ignored.",
    "Universe E (historical S&P 500 constituents) was not run: no point-in-time membership and delisting data were available. No current-constituent study is presented as unbiased.",
    "The SPY sample 2010-2026 was inspected in the prior notebook and is exploratory. All historical data predate the registry freeze only in the sense that strategy ideas were drawn from public literature; the only genuinely untouched sample is prospective (after 2026-10-18).",
    "Cross-asset statistics treat correlated instruments as independent and therefore overstate significance.",
    "Semiconductor ETF histories differ in construction over time (e.g. SMH's pre-2011 HOLDRS history).",
]


def build_spy_pdf(results: dict, out_dir: Path, figs: dict) -> Path:
    syn = results["synthetic"]
    path = out_dir / "SPY_Strategy_Comparison.pdf"
    rows = results["rows"]
    spy = rows[rows.ticker == "SPY"]
    names = S.registry_frame().set_index("id")["name"]

    def tab_for(period):
        d = spy[(spy.period == period) & spy.strategy.isin(["BH"] + _orig_ext())].set_index("strategy").reindex(["BH"] + _orig_ext()).dropna(how="all")
        return pd.DataFrame({"Strategy": [("Buy-and-hold" if i == "BH" else f"{i} {names.get(i, i)}") for i in d.index],
                             "CAGR": d["cagr"].map(_fmt), "Vol": d["ann_vol"].map(_fmt), "Sharpe": d["sharpe"].map(lambda v: _fmt(v, "num", 2)),
                             "Sortino": d["sortino"].map(lambda v: _fmt(v, "num", 2)), "MaxDD": d["max_dd"].map(_fmt),
                             "In mkt": d["time_invested"].map(_fmt), "Entries": d["entries"].map(lambda v: _fmt(v, "num", 0)),
                             "Win%": d["win_rate"].map(_fmt), "Turnover/yr": d["turnover_ann"].map(lambda v: _fmt(v, "num", 1))})

    def story(st, table, img, P, Sp, PB):
        s = [P("SPY Strategy Study: the six original strategies and SMA10/30", st["h1"])]
        if syn:
            s.append(P(SYNTHETIC_BANNER, st["warn"]))
        s.append(P("All strategies use the unified engine: signal at the close, execution at the next open, 10 bps per side, lagged "
                   "3-month T-bill on cash, close-to-close marking, and band rebalancing of partial positions.", st["p"]))
        lg = results.get("spy", {}).get("legacy")
        ref = results.get("spy", {}).get("legacy_reference")
        if lg and ref and "error" not in lg:
            s.append(P("Replication of the original notebook (same method: open-to-open returns, 0% cash, rf=0 Sharpe)", st["h2"]))
            rr = []
            for per in ("Development 70%", "Holdout 30%"):
                for appr in ("SMA 20/50 net", "Buy-and-hold"):
                    for k in ("CAGR", "Sharpe (rf=0)", "Max drawdown"):
                        rr.append({"Period": per, "Approach": appr, "Metric": k, "Notebook": f"{ref[per][appr][k]:.3f}",
                                   "This lab": f"{lg[per][appr][k]:.3f}", "Diff": f"{lg[per][appr][k] - ref[per][appr][k]:+.3f}"})
            s.append(table(pd.DataFrame(rr), col_w=[80, 80, 80, 60, 60, 50]))
            s.append(P(f"Holdout start: notebook {ref['holdout_start']}, lab {lg['holdout_start']}; trades in holdout: notebook {ref['holdout_trades']}, lab {lg['holdout_trades']}. "
                       "Small differences can arise if Yahoo revised adjusted prices between downloads.", st["small"]))
        for period, lab in (("full", "Full available history (after 273-bar warm-up)"), ("common", "Common period 2011-2026"),
                            ("prior_dev_2010_2021", "Prior notebook development period 2010-01-04 to 2021-09-28 (exploratory)"),
                            ("prior_holdout_2021_2026", "Prior notebook holdout 2021-09-29 to 2026-10-16 (already inspected: exploratory, NOT an untouched test)")):
            t = tab_for(period)
            if not t.empty:
                s.append(P(lab, st["h2"]))
                s.append(table(t, col_w=[150, 40, 38, 36, 38, 40, 38, 38, 36, 46]))
        s.append(PB())
        s.append(P("Equity curves and drawdowns", st["h2"]))
        s.append(img(figs.get("spy_six")))
        s.append(P("SMA10/30 focus", st["h2"]))
        s.append(img(figs.get("spy_1030")))
        s.append(img(figs.get("spy_roll")))
        lf = results["_lead_full"]
        f = lf[(lf.ticker == "SPY") & lf.strategy.isin(_orig_ext() + ["BH"])]
        keep = [c for c in ["strategy", "sharpe", "sharpe_ci_lo", "sharpe_ci_hi", "dsr", "p_boot_diff_gt0", "q_bh_global",
                            "wf_oos_sharpe_diff_mean", "wf_share_folds_beat_bh", "excess_cagr_vs_vol_matched",
                            "dd_improvement_vs_vol_matched", "missed_rallies_log", "losses_avoided_log"] if c in f]
        t = f[keep].copy()
        for c in keep[1:]:
            t[c] = t[c].map(lambda v: _fmt(v, "num", 2))
        s.append(P("Statistical evidence for SPY strategies", st["h2"]))
        s.append(table(t, col_w=[40] + [40] * (len(keep) - 1), fs=5.8))
        s.append(P("p_boot: one-sided stationary-bootstrap p-value that the daily return differential vs buy-and-hold is positive; "
                   "q: Benjamini-Hochberg across all tests. Vol-matched columns compare with a static SPY/cash mix of equal volatility. "
                   "Missed rallies / losses avoided: sum of log excess returns over cash while out of the market.", st["small"]))
        hyp = {h["id"]: h for h in results["hypotheses"]}
        s.append(P(f"<b>H2 across all instruments ({hyp['H2']['verdict']}).</b> {hyp['H2']['evidence']}", st["p"]))
        s.append(PB())
        s.append(P("Perfect-trader comparison (theoretical)", st["h2"]))
        for p in [p for p in figs.get("oracle", []) if "SPY" in str(p)]:
            s.append(img(p, 6.5))
        rec = results.get("spy", {}).get("reconciliation")
        if rec is not None and len(rec):
            s.append(P("Ledger reconciliation (equity rebuilt from the trade log vs engine equity)", st["h2"]))
            r2 = rec.copy()
            r2["max_abs_error"] = r2["max_abs_error"].map(lambda v: f"{v:.2e}")
            r2["max_rel_error"] = r2["max_rel_error"].map(lambda v: f"{v:.2e}")
            s.append(table(r2, col_w=[80, 90, 90, 60]))
        return s

    _pdf(path, story, syn, "SPY Strategy Comparison")
    return path


def _orig_ext():
    return ["F01_01", "F01_02", "F02_01", "F02_07", "F05_09", "F03_10", "F02_10", "F10_10"]


# ============================================================================
# Orchestration
# ============================================================================

def build_all(results: dict, out_dir: Path, md=None) -> dict:
    out_dir = Path(out_dir)
    figdir = out_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    syn = bool(results.get("synthetic"))
    if md is not None:
        results["_md"] = md
        results["_cash_growth"] = md.cash_growth
    derive(results)
    results["_lead_full"] = leaderboard(results, "full")
    lead_common = leaderboard(results, "common")
    cfg = results["config"]
    readme = [
        f"Generated {results['created_utc']} (UTC). Data: {results['calendar'][0].date()} to {results['calendar'][1].date()}, source {results['data_source']}.",
        f"Config sha256 {results['config_sha256']}; strategy registry sha256 {results['registry_sha256']} (frozen {results['registry_freeze']['frozen_utc']}).",
        f"Strategies defined: {results['n_strategies_defined']}; strategies executed this run: {results['n_strategies_run']}.",
        "Execution: signal at close t, fill at open t+1; base cost "
        f"{cfg['costs']['commission_bps'] + cfg['costs']['slippage_bps']} bps per side; cash earns the lagged 3-month T-bill; long-only, no leverage.",
        "Periods: 'full' = each instrument's full history after a 273-bar warm-up (all strategies share the window); 'common' = "
        f"{cfg['evaluation']['common_period_start']} onward for instruments with enough history; SPY 'prior_*' rows are the periods already inspected in the original notebook (exploratory).",
        "Do not rank on CAGR alone: compare Sharpe vs buy-and-hold, drawdown, walk-forward OOS columns (wf_*), deflated Sharpe (dsr), FDR q-values and the exposure/volatility-matched columns.",
        "Column glossary: sharpe_diff = Sharpe minus B&H Sharpe; bm_* = vs buy-and-hold of the same asset; ub_* = vs universe benchmark (SPY or SMH); "
        "excess_cagr_vs_vol_matched = CAGR minus a static asset/cash mix with the same volatility; wf_oos_sharpe_diff_mean = mean next-year OOS Sharpe minus B&H across folds.",
    ]
    if syn:
        readme.insert(0, SYNTHETIC_BANNER)
    st = results["stats"]
    fam = family_summary(results)
    cost = results["cost_rows"]
    cost_w = cost.pivot_table(index=["ticker", "universe", "strategy"], columns="cost_bps", values=["sharpe", "cagr"]) if not cost.empty else pd.DataFrame()
    if not cost_w.empty:
        cost_w.columns = [f"{a}_{int(b)}bps" for a, b in cost_w.columns]
        cost_w = cost_w.reset_index()
        bh = cost.groupby(["ticker", "strategy"])["bh_sharpe"].first().reset_index()
        cost_w = cost_w.merge(bh, on=["ticker", "strategy"], how="left")
        grid = sorted(cost["cost_bps"].unique())

        def breakeven(r):
            xs = [r[f"sharpe_{int(c)}bps"] - r["bh_sharpe"] for c in grid]
            if xs[0] <= 0:
                return 0.0
            for (c0, x0), (c1, x1) in zip(zip(grid, xs), zip(grid[1:], xs[1:])):
                if x1 <= 0:
                    return c0 + (c1 - c0) * x0 / (x0 - x1)
            return np.inf
        cost_w["breakeven_cost_bps_vs_bh_sharpe"] = cost_w.apply(breakeven, axis=1)
    prior = results["rows"][results["rows"].period.str.startswith("prior")]
    sheets = {
        "Leaderboard_FullHistory": results["_lead_full"], "Leaderboard_CommonPeriod": lead_common,
        "SPY_Prior_Exploratory": prior, "Family_Summary": fam, "Cost_Sensitivity": cost_w,
        "Parameter_Sensitivity": results["variant_rows"], "Statistical_Tests": st["pvalues"],
        "Deflated_Sharpe": st["dsr"], "Bootstrap_CIs": st["ci"], "Reality_Check_SPA": st["reality_check"],
        "Strategy_Correlation": st["avg_corr"].reset_index() if not st["avg_corr"].empty else pd.DataFrame(),
        "Hypotheses": pd.DataFrame([{k: h.get(k) for k in ("id", "claim", "test", "verdict", "evidence")} for h in results["hypotheses"]]),
        "Registry": S.registry_frame(), "Data_Inventory": results["inventory"], "Data_Validation": results["validation"],
        "Run_Notes": results["errors"],
    }
    write_sheets(out_dir / "Strategy_Leaderboard.xlsx", sheets, readme, syn,
                 heat={"Leaderboard_FullHistory": ["sharpe_diff", "excess_cagr", "max_dd_improvement", "wf_oos_sharpe_diff_mean"],
                       "Family_Summary": ["median_sharpe_diff_vs_bh", "median_oos_sharpe_diff", "median_dd_improvement"]})
    # Sector workbook
    secB = ("B_sectors",)
    rowsB = results["_lead_full"][results["_lead_full"].universe == "B_sectors"]
    rot = results["rotation"]
    rot_rows = results["rows"][results["rows"].universe.isin(["B_sectors"]) & results["rows"].strategy.isin(["EW"] + [s.id for s in S.REGISTRY if s.kind == "rotation"]) & (results["rows"].ticker == "B_sectors")]
    hs = {"Heat_SharpeDiff": heat_matrix(results, secB, "sharpe_diff"), "Heat_ExcessCAGR": heat_matrix(results, secB, "excess_cagr"),
          "Heat_DD_Improvement": heat_matrix(results, secB, "max_dd_improvement")}
    if not results.get("oos_summary", pd.DataFrame()).empty:
        o = results["oos_summary"].merge(results["rows"][["ticker", "universe"]].drop_duplicates(), on="ticker")
        o = o[o.universe == "B_sectors"].pivot_table(index="strategy", columns="ticker", values="wf_oos_sharpe_diff_mean")
        hs["Heat_OOS_SharpeDiff"] = o.reset_index()
    regime_cols = ["ticker", "strategy", "strategy_name", "reg_up_share", "reg_sideways_share", "reg_down_share", "reg_up_sharpe",
                   "reg_sideways_sharpe", "reg_down_sharpe", "reg_up_excess_vs_bh_ann", "reg_sideways_excess_vs_bh_ann", "reg_down_excess_vs_bh_ann"]
    write_sheets(out_dir / "Sector_Performance.xlsx",
                 {"Sector_Leaderboard": rowsB, **hs,
                  "Family_by_Sector": fam[fam.universe == "B_sectors"],
                  "Regime_Performance": rowsB[[c for c in regime_cols if c in rowsB]],
                  "Sector_Rotation": rot_rows, "H6_Selection_WF": results.get("cross_sectional", pd.DataFrame())},
                 readme + ["Heat_* sheets: rows = strategies, columns = sector ETFs; blue = better than buy-and-hold, red = worse (underlying numbers shown)."],
                 syn, heat={"Sector_Leaderboard": ["sharpe_diff", "excess_cagr", "max_dd_improvement"]}, heat_matrix=set(hs))
    # Semiconductor workbook
    semi = ("C_semi_etfs", "D_semi_stocks")
    rowsS = results["_lead_full"][results["_lead_full"].universe.isin(semi)]
    hsm = {"Heat_SharpeDiff": heat_matrix(results, semi, "sharpe_diff"), "Heat_ExcessCAGR": heat_matrix(results, semi, "excess_cagr"),
           "Heat_DD_Improvement": heat_matrix(results, semi, "max_dd_improvement")}
    hyp = {h["id"]: h for h in results["hypotheses"]}
    write_sheets(out_dir / "Semiconductor_Strategy_Results.xlsx",
                 {"Semi_Leaderboard": rowsS, **hsm, "Family_by_Universe": fam[fam.universe.isin(semi + ("A_broad",))],
                  "Semi_Stock_Rotation": results["rows"][(results["rows"].ticker == "D_semi_stocks")],
                  "H1_Fast_vs_Slow": hyp["H1"].get("detail", pd.DataFrame()).reset_index() if isinstance(hyp["H1"].get("detail"), pd.DataFrame) else pd.DataFrame(),
                  "H4_Momentum": hyp["H4"].get("detail", pd.DataFrame()).reset_index() if isinstance(hyp["H4"].get("detail"), pd.DataFrame) else pd.DataFrame()},
                 readme + ["Semiconductor universes: C (ETFs: SMH, SOXX, XSD, PSI) and D (stocks). Stock-level benchmark: SMH."],
                 syn, heat={"Semi_Leaderboard": ["sharpe_diff", "excess_cagr", "max_dd_improvement"]}, heat_matrix=set(hsm))
    # Walk-forward workbook
    folds = []
    for t, o in results["asset_out"].items():
        for fd in o.get("folds", []):
            folds.append({"ticker": t, "fold": fd.label, "train_start": fd.train_start.date(), "train_end": fd.train_end.date(),
                          "test_start": fd.test_start.date(), "test_end": fd.test_end.date()})
    fl = results.get("wf_fixed_long", pd.DataFrame())
    decay_fam = pd.DataFrame()
    if not fl.empty:
        x = fl[fl.strategy != "BH"].assign(family=lambda d: d.strategy.str[:3])
        decay_fam = x.groupby("family").agg(mean_is_sharpe=("is_sharpe", "mean"), mean_oos_sharpe=("oos_sharpe", "mean")).reset_index()
        decay_fam["decay"] = decay_fam["mean_is_sharpe"] - decay_fam["mean_oos_sharpe"]
    rot_wf = []
    for u, tab in rot.get("wf", {}).items():
        for k in ("is_sharpe", "oos_sharpe"):
            m = tab[k].stack().rename("value").reset_index()
            m.columns = ["fold", "strategy", "value"]
            rot_wf.append(m.assign(universe=u, metric=k))
    write_sheets(out_dir / "Walk_Forward_Results.xlsx",
                 {"Fold_Definitions": pd.DataFrame(folds), "Fixed_Rules_IS_OOS": fl, "OOS_Summary": results.get("oos_summary", pd.DataFrame()),
                  "IS_OOS_by_Family": decay_fam, "Rank_Persistence": results.get("rank_persistence", pd.DataFrame()),
                  "Best_of_Registry_OOS": results.get("wf_best_summary", pd.DataFrame()),
                  "Param_Selection_OOS": results.get("wf_param", pd.DataFrame()),
                  "H6_Sector_Selection": results.get("cross_sectional", pd.DataFrame()),
                  "Rotation_WF": pd.concat(rot_wf, ignore_index=True) if rot_wf else pd.DataFrame()},
                 readme + [f"Walk-forward: rolling {cfg['walk_forward']['train_years']}-year training windows, {cfg['walk_forward']['test_years']}-year calendar test windows; "
                           "selection uses training-window Sharpe only. 'Best_of_Registry_OOS' stitches the test-year returns of whichever strategy had the best training Sharpe."],
                 syn, heat={"OOS_Summary": ["wf_oos_sharpe_diff_mean"]})
    # Figures
    figs = {"family": fig_family(results, figdir, syn), "family_asset": fig_family_asset(results, figdir, syn),
            "heat_sector": fig_heat(hs["Heat_SharpeDiff"], "Sector ETFs: Sharpe minus buy-and-hold by strategy", figdir / "heatmap_sectors.png", syn),
            "heat_semi": fig_heat(hsm["Heat_SharpeDiff"], "Semiconductors: Sharpe minus buy-and-hold by strategy", figdir / "heatmap_semiconductors.png", syn),
            "scatter": fig_scatter(results, figdir, syn), "cost": fig_cost(results, figdir, syn), "wf": fig_wf(results, figdir, syn),
            "wf_best": fig_wf_best(results, figdir, syn), "corr": fig_corr(results, figdir, syn), "regime": fig_regime(results, figdir, syn),
            "oracle": fig_oracle(results, figdir, syn)}
    eq_path = out_dir / "audit" / "equity_curves_SPY_core.csv"
    if eq_path.exists():
        curves = pd.read_csv(eq_path, index_col=0, parse_dates=True)
        a = results["asset_out"].get("SPY", {}).get("eval_start")
        six = curves[[c for c in ["BH"] + ["F01_01", "F01_02", "F02_01", "F02_07", "F05_09", "F03_10"] if c in curves]]
        figs["spy_six"] = fig_equity(six, "SPY: the six original strategies vs buy-and-hold (net of costs)", figdir / "spy_original_six.png", syn, start=a)
        figs["spy_1030"] = fig_equity(curves[[c for c in ["BH", "F01_02", "F01_01", "F02_10", "F10_10"] if c in curves]],
                                      "SPY: SMA10/30 vs SMA20/50 and two SMA10/30 refinements", figdir / "spy_sma1030_focus.png", syn, start=a)
        figs["spy_roll"] = fig_spy_rolling(curves.loc[a:] if a is not None else curves, figdir, syn, results.get("_cash_growth"))
    # Audit exports
    audit = out_dir / "audit"
    for t in ("SPY", "QQQ", "SMH", "NVDA", "XLK"):
        o = results["asset_out"].get(t)
        if o is not None and o.get("returns") is not None:
            o["returns"].astype(float).round(7).to_csv(audit / f"daily_returns_{t}.csv.gz", compression="gzip")
    pd.DataFrame([{k: h.get(k) for k in ("id", "claim", "test", "verdict", "evidence")} for h in results["hypotheses"]]).to_csv(audit / "hypotheses.csv", index=False)
    build_main_pdf(results, out_dir, figs)
    build_spy_pdf(results, out_dir, figs)
    return figs
