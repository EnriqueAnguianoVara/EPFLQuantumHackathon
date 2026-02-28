#!/usr/bin/env python3
"""
Notebook 02 — Classical Baselines Training

Trains Ridge, XGBoost, and LSTM models on the swaption dataset,
evaluates with walk-forward validation, and saves everything to trained_models/.

Run from the project root:
    python notebooks/02_classical_baselines.py
"""

import sys
import os
import time
import json
import pickle
from pathlib import Path

import numpy as np

# Project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all
from src.data.preprocessing import full_pipeline, normalize, pca_reduce, denormalize
from src.evaluation.metrics import all_metrics, surface_error_grid
from src.models.classical.ridge import RidgeBaseline
from src.models.classical.xgboost_model import GBTBaseline
from src.models.classical.lstm import LSTMBaseline

TRAINED_DIR = ROOT / "trained_models"
TRAINED_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# 1. Load and preprocess data
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  1. LOADING DATA")
print("=" * 60)

data = load_all()
prices = data["train_prices"]  # (494, 224)
dates = data["train_dates"]

# Config
N_PCA = 6
WINDOW = 20
HORIZON = 1
VAL_RATIO = 0.15

pipe = full_pipeline(prices, n_pca_components=N_PCA, window_size=WINDOW, horizon=HORIZON, val_ratio=VAL_RATIO)

X_train = pipe["X_train"]  # (399, 120)
Y_train = pipe["Y_train"]  # (399, 6)
X_val = pipe["X_val"]      # (55, 120)
Y_val = pipe["Y_val"]      # (55, 6)

scaler = pipe["scaler"]
pca_reducer = pipe["pca_reducer"]
train_pca = pipe["train_pca"]
val_pca = pipe["val_pca"]
all_pca = pipe["all_pca"]

print(f"  Train: {X_train.shape[0]} samples")
print(f"  Val:   {X_val.shape[0]} samples")
print(f"  PCA components: {N_PCA} (cumvar: {pca_reducer.cumulative_variance[N_PCA-1]:.4f})")
print()

# Save preprocessing objects
with open(TRAINED_DIR / "scalers.pkl", "wb") as f:
    pickle.dump({
        "scaler": scaler,
        "pca_reducer": pca_reducer,
        "config": {
            "n_pca": N_PCA,
            "window_size": WINDOW,
            "horizon": HORIZON,
            "val_ratio": VAL_RATIO,
        }
    }, f)
print("  Saved scalers.pkl ✓")

# ══════════════════════════════════════════════════════════════════════════
# 2. Ridge Regression
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  2. RIDGE REGRESSION")
print("=" * 60)

best_ridge = None
best_ridge_mae = float("inf")

for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    ridge = RidgeBaseline(alpha=alpha)
    t0 = time.time()
    ridge.fit(X_train, Y_train)
    dt = time.time() - t0

    Y_pred_val = ridge.predict(X_val)
    m = all_metrics(Y_val, Y_pred_val)

    print(f"  alpha={alpha:6.2f} | MAE={m['MAE']:.6f} | RMSE={m['RMSE']:.6f} | R²={m['R²']:.4f} | {dt:.2f}s")

    if m["MAE"] < best_ridge_mae:
        best_ridge_mae = m["MAE"]
        best_ridge = ridge
        best_ridge_alpha = alpha

print(f"\n  ★ Best alpha: {best_ridge_alpha}")

# Save
with open(TRAINED_DIR / "ridge_model.pkl", "wb") as f:
    pickle.dump(best_ridge, f)
print("  Saved ridge_model.pkl ✓")

# Evaluate on val in original space
Y_pred_pca = best_ridge.predict(X_val)
Y_pred_norm = pca_reducer.inverse_transform(Y_pred_pca)
Y_pred_original = denormalize(Y_pred_norm, scaler)
Y_val_norm = pca_reducer.inverse_transform(Y_val)
Y_val_original = denormalize(Y_val_norm, scaler)

ridge_metrics_original = all_metrics(Y_val_original, Y_pred_original)
print(f"\n  Metrics in ORIGINAL price space:")
for k, v in ridge_metrics_original.items():
    print(f"    {k}: {v:.6f}")

# ══════════════════════════════════════════════════════════════════════════
# 3. XGBoost
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  3. XGBOOST")
print("=" * 60)

try:
    best_xgb = None
    best_xgb_mae = float("inf")

    configs = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1},
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05},
        {"n_estimators": 300, "max_depth": 7, "learning_rate": 0.05},
    ]

    for cfg in configs:
        gbt = GBTBaseline(backend="xgboost", **cfg)
        t0 = time.time()
        gbt.fit(X_train, Y_train)
        dt = time.time() - t0

        Y_pred_val = gbt.predict(X_val)
        m = all_metrics(Y_val, Y_pred_val)

        label = f"n={cfg['n_estimators']},d={cfg['max_depth']},lr={cfg['learning_rate']}"
        print(f"  {label:25s} | MAE={m['MAE']:.6f} | RMSE={m['RMSE']:.6f} | R²={m['R²']:.4f} | {dt:.1f}s")

        if m["MAE"] < best_xgb_mae:
            best_xgb_mae = m["MAE"]
            best_xgb = gbt

    with open(TRAINED_DIR / "xgboost_model.pkl", "wb") as f:
        pickle.dump(best_xgb, f)
    print(f"\n  ★ Best XGBoost: {best_xgb.get_params()}")
    print("  Saved xgboost_model.pkl ✓")

    # Original space metrics
    Y_pred_pca = best_xgb.predict(X_val)
    Y_pred_original = denormalize(pca_reducer.inverse_transform(Y_pred_pca), scaler)
    xgb_metrics_original = all_metrics(Y_val_original, Y_pred_original)
    print(f"\n  Metrics in ORIGINAL price space:")
    for k, v in xgb_metrics_original.items():
        print(f"    {k}: {v:.6f}")

except ImportError as e:
    print(f"  ⚠ XGBoost not available: {e}")
    print("  Skipping. Install with: pip install xgboost")
    xgb_metrics_original = None

# ══════════════════════════════════════════════════════════════════════════
# 4. LSTM
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  4. LSTM")
print("=" * 60)

try:
    import torch

    best_lstm = None
    best_lstm_mae = float("inf")

    configs_lstm = [
        {"hidden_dim": 16, "num_layers": 1, "epochs": 150, "lr": 0.005},
        {"hidden_dim": 32, "num_layers": 1, "epochs": 150, "lr": 0.003},
        {"hidden_dim": 32, "num_layers": 2, "epochs": 200, "lr": 0.003, "dropout": 0.1},
    ]

    for cfg in configs_lstm:
        lstm = LSTMBaseline(n_pca=N_PCA, window_size=WINDOW, **cfg)
        t0 = time.time()
        lstm.fit(X_train, Y_train, X_val, Y_val)
        dt = time.time() - t0

        Y_pred_val = lstm.predict(X_val)
        m = all_metrics(Y_val, Y_pred_val)

        label = f"h={cfg['hidden_dim']},L={cfg['num_layers']}"
        ep_trained = len(lstm.train_losses)
        print(f"  {label:12s} | MAE={m['MAE']:.6f} | RMSE={m['RMSE']:.6f} | R²={m['R²']:.4f} | {ep_trained} ep | {dt:.1f}s")

        if m["MAE"] < best_lstm_mae:
            best_lstm_mae = m["MAE"]
            best_lstm = lstm

    # Save
    best_lstm.save(str(TRAINED_DIR / "lstm_weights.pt"))
    with open(TRAINED_DIR / "lstm_config.pkl", "wb") as f:
        pickle.dump(best_lstm.get_params(), f)
    print(f"\n  ★ Best LSTM: {best_lstm.get_params()}")
    print("  Saved lstm_weights.pt + lstm_config.pkl ✓")

    # Original space metrics
    Y_pred_pca = best_lstm.predict(X_val)
    Y_pred_original = denormalize(pca_reducer.inverse_transform(Y_pred_pca), scaler)
    lstm_metrics_original = all_metrics(Y_val_original, Y_pred_original)
    print(f"\n  Metrics in ORIGINAL price space:")
    for k, v in lstm_metrics_original.items():
        print(f"    {k}: {v:.6f}")

except ImportError as e:
    print(f"  ⚠ PyTorch not available: {e}")
    print("  Skipping. Install with: pip install torch")
    lstm_metrics_original = None

# ══════════════════════════════════════════════════════════════════════════
# 5. Rolling predictions (6 future days)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  5. ROLLING PREDICTIONS (6 future days)")
print("=" * 60)

# Use all data to predict 6 days ahead
all_norm, _ = normalize(prices, scaler)
all_pca_full, _ = pca_reduce(all_norm, reducer=pca_reducer)

# Ridge rolling
ridge_future_pca = best_ridge.predict_rolling(all_pca_full, n_steps=6, window_size=WINDOW, n_pca=N_PCA)
ridge_future_norm = pca_reducer.inverse_transform(ridge_future_pca)
ridge_future_prices = denormalize(ridge_future_norm, scaler)
print(f"  Ridge future predictions shape: {ridge_future_prices.shape}")
print(f"  Ridge future price range: [{ridge_future_prices.min():.4f}, {ridge_future_prices.max():.4f}]")

# Save rolling predictions
np.save(TRAINED_DIR / "ridge_future_pca.npy", ridge_future_pca)
np.save(TRAINED_DIR / "ridge_future_prices.npy", ridge_future_prices)

# ══════════════════════════════════════════════════════════════════════════
# 6. Missing data imputation (2 dates)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  6. MISSING DATA IMPUTATION")
print("=" * 60)

# Missing dates: 04/08/2051 and 20/09/2051
# Find their approximate position in the timeline
# These dates ARE in the training period but were removed
# Strategy: interpolate in PCA space from neighboring days

test_df = data["test_df"]
test_info = data["test_info"]

# Parse training dates to find neighbors
import pandas as pd
train_dates_parsed = pd.to_datetime(data["train_dates"], dayfirst=True)

for idx in test_info["missing_indices"]:
    missing_date = test_df.iloc[idx]["date"]
    mask = test_info["missing_masks"][idx]  # True = known value

    # Find nearest neighbors in training data
    diffs = np.abs((train_dates_parsed - missing_date).total_seconds().values)
    sorted_idx = np.argsort(diffs)
    prev_idx = sorted_idx[0]
    next_idx = sorted_idx[1]

    # Ensure prev < missing < next chronologically
    if train_dates_parsed[prev_idx] > train_dates_parsed[next_idx]:
        prev_idx, next_idx = next_idx, prev_idx

    # Interpolate in PCA space
    z_prev = all_pca_full[prev_idx]
    z_next = all_pca_full[next_idx]
    z_interp = 0.5 * (z_prev + z_next)

    # Decode
    interp_norm = pca_reducer.inverse_transform(z_interp.reshape(1, -1))
    interp_prices = denormalize(interp_norm, scaler)[0]

    n_missing = int((~mask).sum())
    missing_col_indices = np.where(~mask)[0]

    print(f"  Date {missing_date.strftime('%Y-%m-%d')}: {n_missing} missing values")
    print(f"    Interpolated between {train_dates_parsed[prev_idx].strftime('%Y-%m-%d')} "
          f"and {train_dates_parsed[next_idx].strftime('%Y-%m-%d')}")
    print(f"    Predicted values: {interp_prices[missing_col_indices]}")

# ══════════════════════════════════════════════════════════════════════════
# 7. Summary
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  7. SUMMARY — Validation Metrics (original price space)")
print("=" * 60)

summary = {"ridge": ridge_metrics_original}
if xgb_metrics_original:
    summary["xgboost"] = xgb_metrics_original
if lstm_metrics_original:
    summary["lstm"] = lstm_metrics_original

print(f"\n  {'Model':<12} {'MAE':>10} {'RMSE':>10} {'R²':>10} {'Max Err':>10}")
print(f"  {'-'*54}")
for name, metrics in summary.items():
    print(f"  {name:<12} {metrics['MAE']:>10.6f} {metrics['RMSE']:>10.6f} "
          f"{metrics['R²']:>10.6f} {metrics['Max Error']:>10.6f}")

# Save summary
with open(TRAINED_DIR / "baselines_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n  Saved baselines_summary.json ✓")
print("\n  ✅ Phase 2 complete!")
