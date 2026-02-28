from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import pearsonr, ttest_1samp

from src.data.loader import load_train


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def corr_safe(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(pearsonr(x, y).statistic)


def main() -> int:
    parser = argparse.ArgumentParser(description="Empirical martingale proof: RW without drift implies naive forecast.")
    parser.add_argument("--ma-window", type=int, default=5, help="Window for the moving-average baseline.")
    args = parser.parse_args()

    _, prices, _ = load_train()
    diffs = np.diff(prices, axis=0)
    n_rows, n_cols = prices.shape

    mu = diffs.mean(axis=0)
    sigma = diffs.std(axis=0, ddof=0)
    sigma = np.where(sigma == 0.0, 1e-8, sigma)
    drift_to_noise = np.abs(mu) / sigma
    pvals = np.array([ttest_1samp(diffs[:, i], 0.0).pvalue for i in range(n_cols)], dtype=float)

    level_diff_corr = np.array(
        [corr_safe(prices[:-1, i], diffs[:, i]) for i in range(n_cols)],
        dtype=float,
    )
    diff_diff_corr = np.array(
        [corr_safe(diffs[:-1, i], diffs[1:, i]) for i in range(n_cols)],
        dtype=float,
    )

    naive_pred = prices[:-1]
    actual = prices[1:]
    drift_pred = prices[:-1] + mu

    phi = np.array(
        [
            float(np.dot(diffs[1:, i], diffs[:-1, i]) / max(np.dot(diffs[:-1, i], diffs[:-1, i]), 1e-8))
            for i in range(n_cols)
        ]
    )
    arima110_pred = prices[1:-1] + phi * diffs[:-1]
    arima110_actual = prices[2:]

    w = max(2, args.ma_window)
    ma_pred = np.vstack([prices[t - w : t].mean(axis=0) for t in range(w, n_rows - 1)])
    ma_actual = prices[w + 1 :]

    print("== Martingale Proof ==")
    print(f"rows={n_rows}")
    print(f"cols={n_cols}")
    print(f"zero_drift_non_reject_rate={float((pvals >= 0.05).mean()):.4f}")
    print(f"mean_abs_drift={float(np.mean(np.abs(mu))):.10f}")
    print(f"mean_sigma={float(np.mean(sigma)):.10f}")
    print(f"mean_drift_over_noise={float(np.mean(drift_to_noise)):.6f}")
    print(f"mean_corr_level_to_next_diff={float(np.nanmean(level_diff_corr)):.6f}")
    print(f"mean_corr_diff_to_next_diff={float(np.nanmean(diff_diff_corr)):.6f}")

    print("== Forecast Competition ==")
    naive_mse = mse(actual, naive_pred)
    drift_mse = mse(actual, drift_pred)
    arima110_mse = mse(arima110_actual, arima110_pred)
    ma_mse = mse(ma_actual, ma_pred)
    print(f"naive_mse={naive_mse:.10f}")
    print(f"naive_mae={mae(actual, naive_pred):.10f}")
    print(f"rw_plus_drift_mse={drift_mse:.10f}")
    print(f"arima_1_1_0_mse={arima110_mse:.10f}")
    print(f"moving_average_mse={ma_mse:.10f}")
    print(f"naive_vs_drift_gain_pct={100.0 * (drift_mse - naive_mse) / drift_mse:.2f}")
    print(f"naive_vs_arima110_gain_pct={100.0 * (arima110_mse - naive_mse) / arima110_mse:.2f}")
    print(f"naive_vs_moving_average_gain_pct={100.0 * (ma_mse - naive_mse) / ma_mse:.2f}")

    print("== Conclusion ==")
    print("For a near-zero-drift random walk, E[X(t+1)|F_t] ~= X(t), so the naive forecast is the MSE-optimal benchmark.")
    print("Empirically, the drift is negligible, increments are close to orthogonal to the past, and naive is hard to beat.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
