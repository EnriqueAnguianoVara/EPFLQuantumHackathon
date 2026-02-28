#!/usr/bin/env python3
"""
Notebook 05 — Final Predictions

Loads all trained models, generates ensemble predictions for:
- 6 future days (24/12/2051 → 01/01/2052)
- 2 missing dates (04/08/2051, 20/09/2051)

Fills test_template.xlsx and saves as results.xlsx.

Run from project root:
    python notebooks/05_final_predictions.py
"""

import sys
import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all
from src.data.preprocessing import normalize, pca_reduce, denormalize
from src.evaluation.metrics import all_metrics
from src.models.ensemble import Ensemble
from src.utils.excel_writer import fill_test_template

TRAINED_DIR = ROOT / "trained_models"
DATA_DIR = ROOT / "data"

# ══════════════════════════════════════════════════════════════════════════
# 1. Load data and preprocessing
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  1. LOADING DATA & MODELS")
print("=" * 60)

data = load_all()
prices = data["train_prices"]
dates = data["train_dates"]
test_df = data["test_df"]
test_info = data["test_info"]

with open(TRAINED_DIR / "scalers.pkl", "rb") as f:
    prep = pickle.load(f)
scaler = prep["scaler"]
pca_reducer = prep["pca_reducer"]
config = prep["config"]

all_norm, _ = normalize(prices, scaler)
all_pca, _ = pca_reduce(all_norm, reducer=pca_reducer)

N_PCA = config["n_pca"]
WINDOW = config["window_size"]

print(f"  Train: {prices.shape}")
print(f"  PCA: {all_pca.shape}")

# ══════════════════════════════════════════════════════════════════════════
# 2. Load future predictions from all models
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  2. LOADING MODEL PREDICTIONS")
print("=" * 60)

future_predictions = {}

# Ridge
if (TRAINED_DIR / "ridge_future_prices.npy").exists():
    future_predictions["Ridge"] = np.load(TRAINED_DIR / "ridge_future_prices.npy")
    print(f"  Ridge: {future_predictions['Ridge'].shape}")

# QRC
if (TRAINED_DIR / "qrc_future_prices.npy").exists():
    future_predictions["QRC"] = np.load(TRAINED_DIR / "qrc_future_prices.npy")
    print(f"  QRC: {future_predictions['QRC'].shape}")

# QAE
if (TRAINED_DIR / "qae_future_prices.npy").exists():
    future_predictions["QAE"] = np.load(TRAINED_DIR / "qae_future_prices.npy")
    print(f"  QAE: {future_predictions['QAE'].shape}")

if not future_predictions:
    print("  ⚠ No predictions found! Run notebooks 02-04 first.")
    sys.exit(1)

print(f"\n  Available models: {list(future_predictions.keys())}")

# ══════════════════════════════════════════════════════════════════════════
# 3. Ensemble future predictions
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  3. ENSEMBLE FUTURE PREDICTIONS")
print("=" * 60)

# Use validation data to find optimal weights
split_idx = int(len(prices) * (1 - config["val_ratio"]))
val_prices = prices[split_idx:]

# Build validation predictions from each model (1-step on val set)
val_predictions = {}

# Ridge
if (TRAINED_DIR / "ridge_model.pkl").exists():
    with open(TRAINED_DIR / "ridge_model.pkl", "rb") as f:
        ridge = pickle.load(f)
    from src.data.preprocessing import create_flat_windows
    val_norm, _ = normalize(val_prices, scaler)
    val_pca, _ = pca_reduce(val_norm, reducer=pca_reducer)
    X_val, Y_val = create_flat_windows(val_pca, window_size=WINDOW, horizon=1)
    ridge_val_pca = ridge.predict(X_val)
    ridge_val_norm = pca_reducer.inverse_transform(ridge_val_pca)
    val_predictions["Ridge"] = denormalize(ridge_val_norm, scaler)

# For QRC and QAE, use their future predictions directly (no val split available easily)
# We'll use simple mean for the ensemble

ensemble = Ensemble(method="mean")
if val_predictions:
    Y_val_norm = pca_reducer.inverse_transform(Y_val)
    Y_val_prices = denormalize(Y_val_norm, scaler)
    # Only optimize if we have >1 model with val predictions
    if len(val_predictions) > 1:
        ensemble = Ensemble(method="optimal")
        ensemble.fit(val_predictions, Y_val_prices)
    else:
        ensemble.weights = {name: 1.0 / len(future_predictions) for name in future_predictions}
        ensemble.model_names = list(future_predictions.keys())
        ensemble.is_fitted = True
else:
    ensemble.weights = {name: 1.0 / len(future_predictions) for name in future_predictions}
    ensemble.model_names = list(future_predictions.keys())
    ensemble.is_fitted = True

# Generate ensemble future predictions
ensemble_future = ensemble.predict(future_predictions)

print(f"  Ensemble weights: {ensemble.weights}")
print(f"  Ensemble future shape: {ensemble_future.shape}")
print(f"  Price range: [{ensemble_future.min():.4f}, {ensemble_future.max():.4f}]")

# Compare individual models
future_dates = ["24/12/2051", "26/12/2051", "27/12/2051",
                "29/12/2051", "30/12/2051", "01/01/2052"]

print(f"\n  {'Model':<10} {'Mean Price':>12} {'Min':>10} {'Max':>10}")
print(f"  {'-'*45}")
for name, pred in future_predictions.items():
    print(f"  {name:<10} {pred.mean():>12.6f} {pred.min():>10.4f} {pred.max():>10.4f}")
print(f"  {'Ensemble':<10} {ensemble_future.mean():>12.6f} {ensemble_future.min():>10.4f} {ensemble_future.max():>10.4f}")
print(f"  {'Last known':<10} {prices[-1].mean():>12.6f} {prices[-1].min():>10.4f} {prices[-1].max():>10.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 4. Missing data imputation
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  4. MISSING DATA IMPUTATION")
print("=" * 60)

train_dates_parsed = pd.to_datetime(dates, dayfirst=True)
imputed_missing = {}

for idx in test_info["missing_indices"]:
    missing_date = test_df.iloc[idx]["date"]
    mask = test_info["missing_masks"][idx]

    # Find nearest neighbors in training data
    diffs = np.abs((train_dates_parsed - missing_date).total_seconds().values)
    sorted_indices = np.argsort(diffs)
    i_before = sorted_indices[0]
    i_after = sorted_indices[1]
    if train_dates_parsed[i_before] > train_dates_parsed[i_after]:
        i_before, i_after = i_after, i_before

    # Method 1: PCA interpolation
    z_before = all_pca[i_before]
    z_after = all_pca[i_after]
    z_interp = 0.5 * (z_before + z_after)
    pca_interp_norm = pca_reducer.inverse_transform(z_interp.reshape(1, -1))
    pca_interp = denormalize(pca_interp_norm, scaler)[0]

    # Method 2: Direct price interpolation
    direct_interp = 0.5 * (prices[i_before] + prices[i_after])

    # Method 3: QAE latent interpolation (if available)
    if (TRAINED_DIR / "latent_vectors.npy").exists():
        Z_all = np.load(TRAINED_DIR / "latent_vectors.npy")
        import torch
        from src.models.quantum.autoencoder import QuantumAutoencoder
        q_ae = QuantumAutoencoder(
            input_dim=224, pre_quantum_dim=6,
            n_modes=6, n_photons=3, latent_dim=8,
        )
        q_ae.load_state_dict(
            torch.load(str(TRAINED_DIR / "q_autoencoder_weights.pt"),
                       map_location="cpu", weights_only=True)
        )
        q_ae.eval()
        z_qae = 0.5 * (Z_all[i_before] + Z_all[i_after])
        with torch.no_grad():
            qae_interp_norm = q_ae.decode(
                torch.tensor(z_qae, dtype=torch.float32).unsqueeze(0)
            ).numpy()
        qae_interp = denormalize(qae_interp_norm, scaler)[0]
        # Ensemble: average PCA + direct + QAE
        imputed_full = (pca_interp + direct_interp + qae_interp) / 3.0
    else:
        # Average PCA + direct
        imputed_full = (pca_interp + direct_interp) / 2.0

    # For known values, keep the originals from test template
    # For missing values, use our imputation
    known_values = test_df.iloc[idx, 2:].values  # skip 'type' and 'date'
    final_surface = np.zeros(224)
    for j in range(224):
        if mask[j]:
            # Known value — keep it
            final_surface[j] = float(known_values[j])
        else:
            # Missing — use imputation
            final_surface[j] = imputed_full[j]

    imputed_missing[idx] = final_surface

    n_missing = int((~mask).sum())
    missing_cols = np.where(~mask)[0]
    print(f"  Date {missing_date.strftime('%Y-%m-%d')}: {n_missing} values imputed")
    print(f"    Neighbors: {train_dates_parsed[i_before].strftime('%Y-%m-%d')} "
          f"& {train_dates_parsed[i_after].strftime('%Y-%m-%d')}")
    for col in missing_cols:
        print(f"    Col {col}: imputed = {final_surface[col]:.6f}")

# ══════════════════════════════════════════════════════════════════════════
# 5. Fill Excel template
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  5. FILLING EXCEL TEMPLATE")
print("=" * 60)

output_path = fill_test_template(
    predictions_future=ensemble_future,
    predictions_missing=imputed_missing,
    template_path=DATA_DIR / "test_template.xlsx",
    output_path=DATA_DIR / "results.xlsx",
)

print(f"  Saved: {output_path}")

# Verify
import openpyxl
wb = openpyxl.load_workbook(output_path, data_only=True)
ws = wb.active
na_count = 0
for row in ws.iter_rows(min_row=2, values_only=True):
    for v in row:
        if v == "NA" or v is None:
            na_count += 1
print(f"  Remaining NAs: {na_count}")
if na_count == 0:
    print("  ✅ All values filled!")
else:
    print(f"  ⚠ {na_count} NAs remaining")

# ══════════════════════════════════════════════════════════════════════════
# 6. Save final summary
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  6. FINAL SUMMARY")
print("=" * 60)

def to_python(obj):
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

final_summary = {
    "ensemble_weights": ensemble.weights,
    "ensemble_method": ensemble.method,
    "models_used": list(future_predictions.keys()),
    "future_dates": future_dates,
    "future_mean_prices": ensemble_future.mean(axis=1).tolist(),
    "missing_dates_imputed": [
        test_df.iloc[idx]["date"].strftime("%Y-%m-%d")
        for idx in test_info["missing_indices"]
    ],
}

with open(TRAINED_DIR / "final_summary.json", "w") as f:
    json.dump(to_python(final_summary), f, indent=2)
print("  Saved final_summary.json ✓")

# Print nice table
print(f"\n  Future Predictions:")
print(f"  {'Date':<15} {'Mean Price':>12}")
print(f"  {'-'*28}")
for i, date in enumerate(future_dates):
    print(f"  {date:<15} {ensemble_future[i].mean():>12.6f}")

print(f"\n  ✅ Phase 4 complete! results.xlsx is ready for submission.")
