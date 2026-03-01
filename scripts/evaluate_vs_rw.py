#!/usr/bin/env python3
"""
Auxiliary (non-official) validation against synthetic RW 6-day surfaces.

Usage:
    python scripts/evaluate_vs_rw.py
    python scripts/evaluate_vs_rw.py --rw-file "C:\\Users\\jorge\\Downloads\\rw_6days.xlsx"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import get_price_column_names
from src.evaluation.metrics import all_metrics


TRAINED_DIR = ROOT / "trained_models"


def extract_r2(metrics: Dict[str, float]) -> float:
    if "R2" in metrics:
        return float(metrics["R2"])
    if "R²" in metrics:
        return float(metrics["R²"])
    if "RÂ²" in metrics:
        return float(metrics["RÂ²"])
    for k, v in metrics.items():
        if str(k).startswith("R"):
            return float(v)
    return float("nan")


def load_rw_future_surfaces(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"RW file not found: {path}")

    df = pd.read_excel(path, engine="openpyxl")
    cols = [c for c in get_price_column_names() if c in df.columns]
    if len(cols) != 224:
        raise ValueError(f"RW file must contain 224 price columns; found {len(cols)}")

    if "Type" in df.columns:
        type_col = df["Type"].astype(str).str.lower()
        rw_rows = df[type_col.str.contains("future")]
        if rw_rows.empty:
            rw_rows = df
    else:
        rw_rows = df

    rw_vals = rw_rows[cols].astype(float).to_numpy()
    if rw_vals.shape[0] < 6:
        raise ValueError(f"RW file has {rw_vals.shape[0]} rows, expected at least 6 future rows")

    return rw_vals[:6]


def main() -> int:
    parser = argparse.ArgumentParser(description="Auxiliary RW validation (non-official)")
    parser.add_argument(
        "--rw-file",
        default=str(Path.home() / "Downloads" / "rw_6days.xlsx"),
        help="Path to rw_6days.xlsx (default: ~/Downloads/rw_6days.xlsx)",
    )
    args = parser.parse_args()

    rw_file = Path(args.rw_file)
    rw_truth = load_rw_future_surfaces(rw_file)

    model_files = {
        "Ridge": "ridge_future_prices.npy",
        "Naive Persistence": "naive_future_prices.npy",
        "XGBoost": None,  # no direct 6-day artifact currently
        "LSTM": None,     # no direct 6-day artifact currently
        "QRC": "qrc_future_prices.npy",
        "QAE": "qae_future_prices.npy",
        "QKGP": "qkgp_future_prices.npy",
        "QRLSTM": "qrlstm_future_prices.npy",
    }

    results = {}
    per_horizon_mae = {}
    for name, fname in model_files.items():
        if fname is None:
            continue
        p = TRAINED_DIR / fname
        if not p.exists():
            continue
        pred = np.load(p)
        if pred.shape != rw_truth.shape:
            continue

        # Horizon-wise MAE: one value per forecast day (h=1..6)
        horizon_mae = np.mean(np.abs(pred - rw_truth), axis=1)
        per_horizon_mae[name] = {
            f"h{h+1}": float(v) for h, v in enumerate(horizon_mae)
        }

        m = all_metrics(rw_truth, pred)
        results[name] = {
            "MAE": float(m["MAE"]),
            "RMSE": float(m["RMSE"]),
            "R2": extract_r2(m),
        }

    if not results:
        raise RuntimeError("No compatible model predictions found to compare against RW truth.")

    sorted_rows = sorted(results.items(), key=lambda kv: kv[1]["MAE"])
    best_model, best_metrics = sorted_rows[0]
    best_model_by_avg_horizon = min(
        per_horizon_mae.items(),
        key=lambda kv: float(np.mean(list(kv[1].values()))),
    )[0]

    out = {
        "note": "Auxiliary RW validation only; not part of official challenge benchmark.",
        "rw_file": str(rw_file),
        "n_days": int(rw_truth.shape[0]),
        "n_points": int(rw_truth.shape[1]),
        "best_model_by_mae": best_model,
        "best_model_metrics": best_metrics,
        "best_model_by_avg_horizon_mae": best_model_by_avg_horizon,
        "per_horizon_mae": per_horizon_mae,
        "results": {k: v for k, v in sorted_rows},
    }

    out_path = TRAINED_DIR / "rw_validation_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Saved {out_path}")
    print(f"Best model vs RW (MAE): {best_model} -> {best_metrics['MAE']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
