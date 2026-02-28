from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import norm, pearsonr, ttest_rel

from src.data.loader import load_train

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Calibration:
    mu: np.ndarray
    sigma: np.ndarray
    last: np.ndarray
    chol: np.ndarray


class QuantumRandomWalk:
    """Simple discrete coined quantum walk used as a synthetic noise source."""

    def __init__(self, n_steps: int = 40, n_positions: int = 201, coin_angle: float = np.pi / 4) -> None:
        self.n_steps = n_steps
        self.n_positions = n_positions
        self.center = n_positions // 2
        self.coin_angle = coin_angle

    def evolve(self) -> np.ndarray:
        state = np.zeros((self.n_positions, 2), dtype=complex)
        state[self.center] = np.array([1.0, 1j]) / np.sqrt(2.0)
        coin = np.array(
            [
                [np.cos(self.coin_angle), np.sin(self.coin_angle)],
                [np.sin(self.coin_angle), -np.cos(self.coin_angle)],
            ]
        )
        for _ in range(self.n_steps):
            for pos in range(self.n_positions):
                state[pos] = coin @ state[pos]
            shifted = np.zeros_like(state)
            for pos in range(self.n_positions):
                if pos + 1 < self.n_positions:
                    shifted[pos + 1, 0] += state[pos, 0]
                if pos - 1 >= 0:
                    shifted[pos - 1, 1] += state[pos, 1]
            state = shifted
        probs = np.sum(np.abs(state) ** 2, axis=1)
        return probs / probs.sum()

    def sample(self, size: int, probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        support = np.arange(self.n_positions) - self.center
        return rng.choice(support, size=size, p=probs)


def calibrate(prices: np.ndarray) -> Calibration:
    diffs = np.diff(prices, axis=0)
    mu = diffs.mean(axis=0)
    sigma = diffs.std(axis=0, ddof=0)
    sigma = np.where(sigma == 0.0, 1e-8, sigma)
    corr = np.corrcoef(diffs.T)
    corr = np.nan_to_num(corr, nan=0.0)
    corr_reg = 0.9 * corr + 0.1 * np.eye(corr.shape[0])
    chol = np.linalg.cholesky(corr_reg)
    return Calibration(mu=mu, sigma=sigma, last=prices[-1].copy(), chol=chol)


def generate_classical(cal: Calibration, n_days: int, n_sims: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_cols = len(cal.mu)
    paths = np.zeros((n_sims, n_days, n_cols), dtype=float)
    for sim in range(n_sims):
        current = cal.last.copy()
        for day in range(n_days):
            z = rng.standard_normal(n_cols)
            current = np.maximum(current + cal.mu + cal.sigma * (cal.chol @ z), 1e-6)
            paths[sim, day] = current
    return paths


def generate_quantum(
    cal: Calibration,
    n_days: int,
    n_sims: int,
    seed: int,
    qrw: QuantumRandomWalk,
    probs_main: np.ndarray,
    probs_alt: np.ndarray,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    qrw_std = float(np.sqrt(qrw.n_steps))
    n_cols = len(cal.mu)
    paths = np.zeros((n_sims, n_days, n_cols), dtype=float)
    for sim in range(n_sims):
        current = cal.last.copy()
        for day in range(n_days):
            probs = probs_main if rng.random() < 0.7 else probs_alt
            raw = qrw.sample(n_cols, probs, rng)
            scaled = np.clip(raw / (3.0 * qrw_std), -0.999, 0.999)
            z = norm.ppf((scaled + 1.0) / 2.0)
            current = np.maximum(current + cal.mu + cal.sigma * (cal.chol @ z), 1e-6)
            paths[sim, day] = current
    return paths


def summarize_paths(classical_paths: np.ndarray, quantum_paths: np.ndarray, cal: Calibration) -> dict[str, float]:
    c_mean = classical_paths.mean(axis=0)
    q_mean = quantum_paths.mean(axis=0)
    c_std = classical_paths.std(axis=0)
    q_std = quantum_paths.std(axis=0)

    horizon = classical_paths.shape[1]
    expected = np.vstack([cal.last + (day + 1) * cal.mu for day in range(horizon)])

    c_bias = float(np.mean(c_mean - expected))
    q_bias = float(np.mean(q_mean - expected))
    mean_abs_gap = float(np.mean(np.abs(c_mean - q_mean)))
    combined_sigma = np.sqrt(c_std**2 + q_std**2)
    gap_over_sigma = float(np.mean(np.abs(c_mean - q_mean) / np.maximum(combined_sigma, 1e-8)))

    overlap = []
    corr_values = []
    ttest_keep = 0
    for day in range(horizon):
        c_lo, c_hi = c_mean[day] - 2.0 * c_std[day], c_mean[day] + 2.0 * c_std[day]
        q_lo, q_hi = q_mean[day] - 2.0 * q_std[day], q_mean[day] + 2.0 * q_std[day]
        overlap.append(float((np.maximum(c_lo, q_lo) < np.minimum(c_hi, q_hi)).mean()))
        corr_values.append(float(pearsonr(c_mean[day], q_mean[day]).statistic))
        if float(ttest_rel(c_mean[day], q_mean[day]).pvalue) >= 0.05:
            ttest_keep += 1

    return {
        "classical_bias_mean": c_bias,
        "quantum_bias_mean": q_bias,
        "mean_abs_gap": mean_abs_gap,
        "gap_over_sigma_mean": gap_over_sigma,
        "band_overlap_mean": float(np.mean(overlap)),
        "band_overlap_last_day": float(overlap[-1]),
        "path_corr_mean": float(np.mean(corr_values)),
        "path_corr_last_day": float(corr_values[-1]),
        "ttest_non_reject_ratio": ttest_keep / horizon,
    }


def plot_mean_paths(classical_paths: np.ndarray, quantum_paths: np.ndarray, output_path: Path) -> None:
    c_mean = classical_paths.mean(axis=0).mean(axis=1)
    q_mean = quantum_paths.mean(axis=0).mean(axis=1)
    c_std = classical_paths.std(axis=0).mean(axis=1)
    q_std = quantum_paths.std(axis=0).mean(axis=1)

    days = np.arange(1, len(c_mean) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, c_mean, label="Classical RW", linewidth=2.0)
    ax.plot(days, q_mean, label="Quantum RW", linewidth=2.0)
    ax.fill_between(days, c_mean - 2.0 * c_std, c_mean + 2.0 * c_std, alpha=0.15)
    ax.fill_between(days, q_mean - 2.0 * q_std, q_mean + 2.0 * q_std, alpha=0.15)
    ax.set_title("Classical vs Quantum Random-Walk Mean Paths")
    ax.set_xlabel("Forecast day")
    ax.set_ylabel("Average surface value")
    ax.grid(True, alpha=0.25)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare classical and quantum random-walk generators.")
    parser.add_argument("--days", type=int, default=60, help="Forecast horizon for synthetic paths.")
    parser.add_argument("--sims", type=int, default=300, help="Number of Monte Carlo paths per generator.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for both generators.")
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        help="Optional path to save a convergence plot.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _, prices, _ = load_train()
    cal = calibrate(prices)

    qrw = QuantumRandomWalk()
    probs = qrw.evolve()
    probs_alt = QuantumRandomWalk(coin_angle=np.pi / 4 + 0.05).evolve()

    classical_paths = generate_classical(cal, args.days, args.sims, args.seed)
    quantum_paths = generate_quantum(cal, args.days, args.sims, args.seed, qrw, probs, probs_alt)
    summary = summarize_paths(classical_paths, quantum_paths, cal)

    print("== Convergence Analysis ==")
    print(f"classical_bias_mean={summary['classical_bias_mean']:.10f}")
    print(f"quantum_bias_mean={summary['quantum_bias_mean']:.10f}")
    print(f"mean_abs_gap={summary['mean_abs_gap']:.10f}")
    print(f"gap_over_sigma_mean={summary['gap_over_sigma_mean']:.6f}")
    print(f"band_overlap_mean={summary['band_overlap_mean']:.4f}")
    print(f"band_overlap_last_day={summary['band_overlap_last_day']:.4f}")
    print(f"path_corr_mean={summary['path_corr_mean']:.6f}")
    print(f"path_corr_last_day={summary['path_corr_last_day']:.6f}")
    print(f"ttest_non_reject_ratio={summary['ttest_non_reject_ratio']:.4f}")

    if args.save_plot is not None:
        plot_mean_paths(classical_paths, quantum_paths, args.save_plot.resolve())
        print(f"saved_plot={args.save_plot.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
