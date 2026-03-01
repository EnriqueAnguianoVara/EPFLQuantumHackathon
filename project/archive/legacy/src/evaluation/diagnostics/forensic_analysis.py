from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import anderson, chi2, jarque_bera, kstest, kurtosis, norm, skew
from sklearn.decomposition import PCA

from src.data.loader import load_train


def autocorr_at_lag(series: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= len(series):
        return float("nan")
    x0 = series[:-lag] - series[:-lag].mean()
    x1 = series[lag:] - series[lag:].mean()
    denom = float(np.sqrt((x0 @ x0) * (x1 @ x1)))
    return float((x0 @ x1) / denom) if denom > 0.0 else float("nan")


def variance_ratio(series: np.ndarray, horizon: int) -> float:
    diff_1 = np.diff(series)
    diff_h = series[horizon:] - series[:-horizon]
    var_1 = float(np.var(diff_1, ddof=1))
    var_h = float(np.var(diff_h, ddof=1))
    return float(var_h / (horizon * var_1)) if var_1 > 0.0 else float("nan")


def adf_style_tstat(series: np.ndarray, max_lag: int = 10) -> float:
    """Lightweight Dickey-Fuller proxy using OLS with lagged differences."""
    dy = np.diff(series)
    y_lag = series[:-1]
    n_lags = min(max_lag, max(1, len(dy) // 8))
    n_obs = len(dy) - n_lags
    if n_obs <= 5:
        return float("nan")

    design = np.ones((n_obs, 2 + n_lags))
    design[:, 1] = y_lag[n_lags:]
    for lag in range(1, n_lags + 1):
        design[:, 1 + lag] = dy[n_lags - lag : len(dy) - lag]
    target = dy[n_lags:]
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ coef
    sigma2 = float((resid @ resid) / max(1, n_obs - design.shape[1]))
    cov = sigma2 * np.linalg.pinv(design.T @ design)
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    return float(coef[1] / se) if se > 0.0 else float("nan")


def ljung_box_pass_rate(diffs: np.ndarray, max_lag: int = 10) -> float:
    n_rows, n_cols = diffs.shape
    q_stats = np.zeros(n_cols, dtype=float)
    for col in range(n_cols):
        q_value = 0.0
        for lag in range(1, max_lag + 1):
            ac = autocorr_at_lag(diffs[:, col], lag)
            if np.isnan(ac):
                continue
            q_value += (ac**2) / max(1, n_rows - lag)
        q_stats[col] = n_rows * (n_rows + 2) * q_value
    pvals = 1.0 - chi2.cdf(q_stats, df=max_lag)
    return float((pvals > 0.05).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run forensic tests for a synthetic Gaussian random-walk hypothesis.")
    parser.add_argument("--max-cols", type=int, default=224, help="Number of columns to test (for speed control).")
    args = parser.parse_args()

    _, prices, _ = load_train()
    prices = prices[:, : args.max_cols]
    diffs = np.diff(prices, axis=0)
    n_rows, n_cols = prices.shape

    jb_pass = []
    ks_pass = []
    ad_pass = 0
    adf_non_reject = 0
    vr2 = []
    vr5 = []
    beta_vals = []

    for col in range(n_cols):
        series = prices[:, col]
        delta = diffs[:, col]
        if np.std(delta) == 0.0:
            continue
        z = (delta - delta.mean()) / delta.std(ddof=0)
        jb_pass.append(float(jarque_bera(delta).pvalue > 0.05))
        ks_pass.append(float(kstest(z, "norm").pvalue > 0.05))
        ad = anderson(z, "norm")
        if ad.statistic < ad.critical_values[2]:
            ad_pass += 1
        adf_non_reject += int(adf_style_tstat(series) > -2.87)
        vr2.append(variance_ratio(series, 2))
        vr5.append(variance_ratio(series, 5))
        x = series[:-1]
        y = series[1:]
        design = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        beta_vals.append(float(coef[1]))

    z_prices = (prices - prices.mean(axis=0, keepdims=True)) / np.maximum(prices.std(axis=0, keepdims=True), 1e-8)
    pca = PCA(n_components=min(6, n_cols, n_rows - 1), random_state=0).fit(z_prices)

    print("== Forensic Analysis ==")
    print(f"rows={n_rows}")
    print(f"cols={n_cols}")
    print(f"jb_pass_rate={np.mean(jb_pass):.4f}")
    print(f"ks_pass_rate={np.mean(ks_pass):.4f}")
    print(f"anderson_pass_rate={ad_pass / n_cols:.4f}")
    print(f"excess_kurtosis_mean={float(np.mean([kurtosis(diffs[:, i]) for i in range(n_cols)])):.6f}")
    print(f"skew_mean={float(np.mean([skew(diffs[:, i]) for i in range(n_cols)])):.6f}")
    print(f"ljung_box_pass_rate={ljung_box_pass_rate(diffs):.4f}")
    print(f"unit_root_non_reject_rate={adf_non_reject / n_cols:.4f}")
    print(f"vr2_mean={float(np.nanmean(vr2)):.6f}")
    print(f"vr5_mean={float(np.nanmean(vr5)):.6f}")
    print(f"ols_beta_mean={float(np.nanmean(beta_vals)):.6f}")
    print(f"ols_beta_gap_from_1={float(np.nanmean(np.abs(np.array(beta_vals) - 1.0))):.6f}")
    print(f"pca_k90={int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.90) + 1)}")
    print(f"pca_k99={int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.99) + 1)}")

    evidence = []
    if np.mean(jb_pass) >= 0.80 and np.mean(ks_pass) >= 0.80:
        evidence.append("increments are broadly Gaussian")
    if ljung_box_pass_rate(diffs) >= 0.70:
        evidence.append("increments are close to temporally independent")
    if adf_non_reject / n_cols >= 0.75:
        evidence.append("levels are consistent with a unit-root process")
    if abs(np.nanmean(vr2) - 1.0) <= 0.10 and abs(np.nanmean(vr5) - 1.0) <= 0.20:
        evidence.append("variance scaling is close to a random walk")
    if np.nanmean(np.abs(np.array(beta_vals) - 1.0)) <= 0.05:
        evidence.append("one-step OLS slope is close to 1")
    if np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.99) + 1 <= 4:
        evidence.append("the surface is strongly low-rank")

    verdict = (
        "strongly consistent with a synthetic low-rank Gaussian random walk"
        if len(evidence) >= 5
        else "partially consistent with a synthetic low-rank Gaussian random walk"
    )

    print("== Verdict ==")
    print(f"verdict={verdict}")
    for idx, item in enumerate(evidence, start=1):
        print(f"{idx}. {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
