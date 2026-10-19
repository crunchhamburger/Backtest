"""Performance analytics and statistical inference.

Metrics follow standard definitions (all annualised with 252 bars/year):

* CAGR = (final/initial)^(252/T) - 1 ; total return = final/initial - 1
* Sharpe = mean(r - rf) / std(r - rf) * sqrt(252)   (rf = lagged T-bill accrual)
* Sortino = mean(r - rf) / sqrt(mean(min(r - rf, 0)^2)) * sqrt(252)
* Max drawdown from the running peak of the equity curve (including the start)
* Drawdown duration = longest run of bars spent below a prior peak
* Calmar = CAGR / |max drawdown|
* Alpha/beta: OLS of strategy excess returns on benchmark excess returns,
  Newey-West (HAC, 10 lags) t-statistics; alpha annualised x252.

Inference tools: stationary bootstrap (Politis-Romano), probabilistic and
deflated Sharpe ratios (Bailey & Lopez de Prado), White's Reality Check and
Hansen's SPA test, Benjamini-Hochberg FDR and Holm corrections.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

ANN = 252
EULER_GAMMA = 0.5772156649015329


# ============================================================================
# Basic metrics
# ============================================================================

def equity_from_returns(r: pd.Series | np.ndarray) -> np.ndarray:
    return np.cumprod(1 + np.asarray(r, dtype=float))


def drawdown(r) -> np.ndarray:
    eq = np.r_[1.0, equity_from_returns(r)]
    return (eq / np.maximum.accumulate(eq) - 1)[1:]


def max_drawdown(r) -> float:
    dd = drawdown(r)
    return float(dd.min()) if len(dd) else np.nan


def max_dd_duration(r) -> int:
    dd = drawdown(r)
    under = dd < -1e-12
    best = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        best = max(best, cur)
    return int(best)


def cagr(r) -> float:
    r = np.asarray(r, dtype=float)
    if len(r) == 0:
        return np.nan
    g = np.prod(1 + r)
    return float(g ** (ANN / len(r)) - 1) if g > 0 else -1.0


def sharpe(r, rf=None) -> float:
    r = np.asarray(r, dtype=float)
    x = r - (np.asarray(rf, dtype=float) if rf is not None else 0.0)
    s = x.std(ddof=1)
    return float(x.mean() / s * math.sqrt(ANN)) if s > 0 else np.nan


def sortino(r, rf=None) -> float:
    r = np.asarray(r, dtype=float)
    x = r - (np.asarray(rf, dtype=float) if rf is not None else 0.0)
    dd = np.sqrt(np.mean(np.minimum(x, 0) ** 2))
    return float(x.mean() / dd * math.sqrt(ANN)) if dd > 0 else np.nan


def alpha_beta(r, b, rf, lags: int = 10) -> dict:
    """HAC-robust regression of excess strategy returns on excess benchmark returns."""
    import statsmodels.api as sm

    y = np.asarray(r) - np.asarray(rf)
    x = np.asarray(b) - np.asarray(rf)
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 30 or np.std(x[ok]) == 0:
        return {"alpha_ann": np.nan, "alpha_t": np.nan, "alpha_p": np.nan, "beta": np.nan}
    X = sm.add_constant(x[ok])
    m = sm.OLS(y[ok], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"alpha_ann": float(m.params[0] * ANN), "alpha_t": float(m.tvalues[0]), "alpha_p": float(m.pvalues[0]),
            "beta": float(m.params[1])}


def episodes_from_exposure(exposure: np.ndarray, ret: np.ndarray) -> pd.DataFrame:
    """Position episodes for single-asset strategies.

    An episode starts on the first bar with exposure > 0 (entry fill day) and
    ends on the first bar where exposure returns to 0 (exit fill day, which
    carries the overnight gap and exit cost). Episode return compounds the
    strategy's net daily returns over those bars, so it includes costs and the
    cash yield on any uninvested fraction.
    """
    on = exposure > 1e-9
    rows = []
    i, n = 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            end = min(j, n - 1)
            closed = j < n
            rr = np.prod(1 + ret[i:end + 1]) - 1
            rows.append({"start": i, "end": end, "bars": j - i, "ret": rr, "closed": closed})
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(rows, columns=["start", "end", "bars", "ret", "closed"])


def episodes_from_weights(weights: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """Approximate per-holding episodes for multi-asset strategies (contribution = w[t-1] * r[t])."""
    rows = []
    r = closes.pct_change().fillna(0.0).to_numpy()
    W = weights.to_numpy()
    for j in range(W.shape[1]):
        on = W[:, j] > 1e-9
        contrib = np.r_[0.0, W[:-1, j] * r[1:, j]]
        i, n = 0, len(on)
        while i < n:
            if on[i]:
                k = i
                while k < n and on[k]:
                    k += 1
                end = min(k, n - 1)
                rows.append({"start": i, "end": end, "bars": k - i, "ret": float(np.prod(1 + contrib[i + 1:end + 1]) - 1),
                             "closed": k < n, "ticker": weights.columns[j]})
                i = k + 1
            else:
                i += 1
    return pd.DataFrame(rows, columns=["start", "end", "bars", "ret", "closed", "ticker"])


def trade_stats(ep: pd.DataFrame) -> dict:
    if ep.empty:
        return {"entries": 0, "exits": 0, "win_rate": np.nan, "profit_factor": np.nan, "avg_hold_bars": np.nan,
                "avg_trade_ret": np.nan}
    wins = ep["ret"][ep["ret"] > 0].sum()
    losses = -ep["ret"][ep["ret"] < 0].sum()
    return {"entries": int(len(ep)), "exits": int(ep["closed"].sum()), "win_rate": float((ep["ret"] > 0).mean()),
            "profit_factor": float(wins / losses) if losses > 0 else np.inf,
            "avg_hold_bars": float(ep["bars"].mean()), "avg_trade_ret": float(ep["ret"].mean())}


def return_concentration(r) -> dict:
    """How much of the compounded log return comes from a handful of days/windows."""
    r = np.asarray(r, dtype=float)
    lr = np.log1p(r)
    total = lr.sum()
    srt = np.sort(lr)[::-1]
    k1 = max(1, int(round(0.01 * len(r))))
    best10 = srt[:10].sum()
    out = {"top1pct_days_share": float(srt[:k1].sum() / total) if total > 0 else np.nan,
           "cagr_ex_best10": float(math.exp((total - best10) * ANN / len(r)) - 1) if len(r) else np.nan}
    if len(r) > ANN:
        roll = pd.Series(lr).rolling(ANN).sum()
        out["best_12m_share"] = float(roll.max() / total) if total > 0 else np.nan
        out["worst_12m_ret"] = float(math.exp(roll.min()) - 1)
    return out


def perf_metrics(ret: pd.Series, rf: pd.Series, exposure: pd.Series | None = None, turnover: pd.Series | None = None,
                 gross: pd.Series | None = None, costs: pd.Series | None = None, equity: pd.Series | None = None,
                 bench: pd.Series | None = None, episodes: pd.DataFrame | None = None) -> dict:
    r = ret.to_numpy(dtype=float)
    f = rf.reindex(ret.index).to_numpy(dtype=float)
    T = len(r)
    years = T / ANN
    m = {
        "start": ret.index[0].date() if T else None, "end": ret.index[-1].date() if T else None,
        "years": years, "cagr": cagr(r), "total_return": float(np.prod(1 + r) - 1),
        "ann_vol": float(r.std(ddof=1) * math.sqrt(ANN)), "sharpe": sharpe(r, f), "sortino": sortino(r, f),
        "max_dd": max_drawdown(r), "max_dd_duration_bars": max_dd_duration(r),
        "skew": float(stats.skew(r)), "kurtosis": float(stats.kurtosis(r, fisher=False)),
    }
    m["calmar"] = m["cagr"] / abs(m["max_dd"]) if m["max_dd"] < 0 else np.nan
    if exposure is not None:
        e = exposure.reindex(ret.index).to_numpy(dtype=float)
        m["time_invested"] = float((e > 0.01).mean())
        m["avg_exposure"] = float(e.mean())
    if turnover is not None:
        m["turnover_ann"] = float(turnover.reindex(ret.index).sum() / years) if years > 0 else np.nan
    if gross is not None:
        g = gross.reindex(ret.index).to_numpy(dtype=float)
        m["gross_cagr"] = cagr(g)
        m["gross_total_return"] = float(np.prod(1 + g) - 1)
        m["cost_drag_ann"] = m["gross_cagr"] - m["cagr"]
    if costs is not None and equity is not None:
        eq_prev = equity.shift(1).reindex(ret.index)
        m["costs_pct_nav_total"] = float((costs.reindex(ret.index) / eq_prev).fillna(0).sum())
    if bench is not None:
        b = bench.reindex(ret.index).to_numpy(dtype=float)
        m.update({f"bm_{k}": v for k, v in alpha_beta(r, b, f).items()})
        d = r - b
        te = d.std(ddof=1) * math.sqrt(ANN)
        m["tracking_error"] = float(te)
        m["info_ratio"] = float(d.mean() * ANN / te) if te > 0 else np.nan
        m["bm_cagr"] = cagr(b)
        m["bm_sharpe"] = sharpe(b, f)
        m["bm_max_dd"] = max_drawdown(b)
        m["excess_cagr"] = m["cagr"] - m["bm_cagr"]
        m["sharpe_diff"] = m["sharpe"] - m["bm_sharpe"]
        m["max_dd_improvement"] = m["max_dd"] - m["bm_max_dd"]
    if episodes is not None:
        m.update(trade_stats(episodes))
    m.update(return_concentration(r))
    return m


# ============================================================================
# Bootstrap
# ============================================================================

def stationary_bootstrap_indices(T: int, reps: int, mean_block: float, seed: int) -> np.ndarray:
    """Politis-Romano stationary bootstrap index matrix of shape (reps, T)."""
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    idx = np.empty((reps, T), dtype=np.int32)
    starts = rng.integers(0, T, size=(reps, T))
    new_block = rng.random((reps, T)) < p
    new_block[:, 0] = True
    cur = starts[:, 0].copy()
    for t in range(T):
        cur = np.where(new_block[:, t], starts[:, t], (cur + 1) % T)
        idx[:, t] = cur
    return idx


def bootstrap_metric_ci(R: np.ndarray, RF: np.ndarray, idx: np.ndarray, level: float = 0.90) -> pd.DataFrame:
    """Bootstrap CIs for Sharpe, CAGR and max drawdown for each column of R (T x K)."""
    reps = idx.shape[0]
    K = R.shape[1]
    sh = np.empty((reps, K))
    cg = np.empty((reps, K))
    md = np.empty((reps, K))
    X = R - RF[:, None]
    for b in range(reps):
        ii = idx[b]
        xb = X[ii]
        rb = R[ii]
        s = xb.std(axis=0, ddof=1)
        sh[b] = np.where(s > 0, xb.mean(axis=0) / s * math.sqrt(ANN), np.nan)
        lg = np.log1p(rb).sum(axis=0)
        cg[b] = np.exp(lg * ANN / len(ii)) - 1
        eq = np.vstack([np.ones(K), np.cumprod(1 + rb, axis=0)])
        md[b] = (eq / np.maximum.accumulate(eq, axis=0) - 1).min(axis=0)
    a = (1 - level) / 2
    q = lambda M: (np.nanquantile(M, a, axis=0), np.nanquantile(M, 1 - a, axis=0))
    out = {}
    for name, M in (("sharpe", sh), ("cagr", cg), ("max_dd", md)):
        lo, hi = q(M)
        out[f"{name}_ci_lo"], out[f"{name}_ci_hi"] = lo, hi
    out["p_sharpe_le_0"] = (sh <= 0).mean(axis=0)
    return pd.DataFrame(out)


def bootstrap_diff_pvalues(D: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """One-sided bootstrap p-values for H0: mean(D_k) <= 0, using the centred bootstrap."""
    T = D.shape[0]
    mu = D.mean(axis=0)
    cnt = np.zeros(D.shape[1])
    for b in range(idx.shape[0]):
        mb = D[idx[b]].mean(axis=0)
        cnt += (mb - mu) >= mu
    return (cnt + 1) / (idx.shape[0] + 1)


def reality_check(D: np.ndarray, idx: np.ndarray) -> dict:
    """White (2000) Reality Check and Hansen (2005) consistent SPA test.

    ``D`` is T x K of daily performance differentials (strategy minus benchmark).
    H0: no strategy has positive expected differential.
    """
    T, K = D.shape
    mu = D.mean(axis=0)
    reps = idx.shape[0]
    boot_mu = np.empty((reps, K))
    for b in range(reps):
        boot_mu[b] = D[idx[b]].mean(axis=0)
    # White RC (non-studentised)
    V = math.sqrt(T) * mu.max()
    Vb = math.sqrt(T) * (boot_mu - mu).max(axis=1)
    p_rc = float(((Vb >= V).sum() + 1) / (reps + 1))
    # Hansen SPA (studentised, consistent recentring)
    omega = np.sqrt(T) * boot_mu.std(axis=0, ddof=1)
    omega = np.where(omega > 0, omega, np.nan)
    t_obs = np.nanmax(np.maximum(math.sqrt(T) * mu / omega, 0.0))
    # Hansen's consistent recentring: only clearly inferior strategies (mean below the
    # -sqrt(2 log log T) * omega / sqrt(T) threshold) keep their own mean; all others sit on the null boundary.
    thr = -np.sqrt((omega ** 2 / T) * 2 * math.log(math.log(T)))
    mu_c = np.where(mu <= thr, mu, 0.0)
    Z = math.sqrt(T) * (boot_mu - mu + mu_c) / omega
    t_b = np.nanmax(np.maximum(Z, 0.0), axis=1)
    p_spa = float(((t_b >= t_obs).sum() + 1) / (reps + 1))
    return {"n_strategies": K, "best_index": int(np.argmax(mu)), "best_mean_diff_ann": float(mu.max() * ANN),
            "rc_pvalue": p_rc, "spa_pvalue": p_spa}


# ============================================================================
# Sharpe-ratio inference
# ============================================================================

def probabilistic_sharpe(sr: float, T: int, skew: float, kurt: float, sr_ref: float = 0.0) -> float:
    """PSR with per-period (non-annualised) Sharpe ratios; ``kurt`` is raw (non-excess) kurtosis."""
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denom <= 0 or T < 3:
        return np.nan
    return float(stats.norm.cdf((sr - sr_ref) * math.sqrt(T - 1) / math.sqrt(denom)))


def expected_max_sharpe(var_sr: float, n_trials: float) -> float:
    """Expected maximum per-period Sharpe among ``n_trials`` independent zero-skill trials."""
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * math.e))
    return float(math.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def effective_trials(R: np.ndarray) -> float:
    """Effective number of independent trials from the average pairwise correlation."""
    K = R.shape[1]
    if K < 2:
        return 1.0
    C = np.corrcoef(R, rowvar=False)
    rho = np.nanmean(C[np.triu_indices(K, 1)])
    rho = min(max(rho, 0.0), 1.0)
    return float(rho + (1 - rho) * K)


def deflated_sharpe(R: np.ndarray, RF: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Deflated Sharpe ratio for every column of a T x K return matrix (all trials on one asset)."""
    X = R - RF[:, None]
    T = X.shape[0]
    sd = X.std(axis=0, ddof=1)
    sr = np.where(sd > 0, X.mean(axis=0) / sd, np.nan)
    var_sr = float(np.nanvar(sr, ddof=1))
    n_eff = effective_trials(R)
    sr0 = expected_max_sharpe(var_sr, n_eff)
    sk = stats.skew(X, axis=0)
    ku = stats.kurtosis(X, axis=0, fisher=False)
    rows = []
    for k, lab in enumerate(labels):
        rows.append({"label": lab, "sharpe_ann": sr[k] * math.sqrt(ANN),
                     "psr_vs_0": probabilistic_sharpe(sr[k], T, sk[k], ku[k], 0.0),
                     "dsr": probabilistic_sharpe(sr[k], T, sk[k], ku[k], sr0),
                     "sr0_ann": sr0 * math.sqrt(ANN), "n_trials": R.shape[1], "n_eff_trials": n_eff})
    return pd.DataFrame(rows)


# ============================================================================
# Multiple-testing corrections
# ============================================================================

def bh_fdr(p: np.ndarray, alpha: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg: returns (reject mask, adjusted q-values)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    qv = np.empty(n)
    qv[order] = np.minimum(q, 1.0)
    return qv <= alpha, qv


def holm(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    rej = np.zeros(n, dtype=bool)
    for i, k in enumerate(order):
        if p[k] <= alpha / (n - i):
            rej[k] = True
        else:
            break
    return rej
