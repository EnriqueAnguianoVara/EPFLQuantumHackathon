#!/usr/bin/env python3
"""
Notebook 04 — Quantum Autoencoder

Trains the quantum autoencoder (Phase 1: reconstruction, Phase 2: temporal),
compares with classical bottleneck ablation, and generates predictions.

Run from project root:
    python notebooks/04_quantum_autoencoder.py
"""

import sys
import os
import time
import json
import pickle
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all
from src.data.preprocessing import normalize, pca_reduce, denormalize
from src.evaluation.metrics import all_metrics
from src.models.quantum.autoencoder import (
    QuantumAutoencoder,
    ClassicalAutoencoder,
    AutoencoderTrainer,
    TemporalPredictor,
    SwaptionPredictor,
)

TRAINED_DIR = ROOT / "trained_models"
TRAINED_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  1. LOADING DATA")
print("=" * 60)

data = load_all()
prices = data["train_prices"]

VAL_RATIO = 0.15
WINDOW = 20
LATENT_DIM = 8

with open(TRAINED_DIR / "scalers.pkl", "rb") as f:
    prep = pickle.load(f)
scaler = prep["scaler"]
pca_reducer = prep["pca_reducer"]

# Normalize (we train the autoencoder on NORMALIZED 224-dim surfaces)
all_norm, _ = normalize(prices, scaler)
split_idx = int(len(prices) * (1 - VAL_RATIO))
train_norm = all_norm[:split_idx]
val_norm = all_norm[split_idx:]

print(f"  Train: {train_norm.shape}")
print(f"  Val:   {val_norm.shape}")
print(f"  All:   {all_norm.shape}")

# ══════════════════════════════════════════════════════════════════════════
# 2. Train QUANTUM Autoencoder
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  2. QUANTUM AUTOENCODER")
print("=" * 60)

q_autoencoder = QuantumAutoencoder(
    input_dim=224,
    pre_quantum_dim=6,
    n_modes=6,
    n_photons=3,
    latent_dim=LATENT_DIM,
    circuit_depth=1,
)

n_params = sum(p.numel() for p in q_autoencoder.parameters())
n_trainable = sum(p.numel() for p in q_autoencoder.parameters() if p.requires_grad)
print(f"  Total params: {n_params}")
print(f"  Trainable params: {n_trainable}")

q_trainer = AutoencoderTrainer(
    model=q_autoencoder,
    lr=0.01,
    epochs=200,
    batch_size=32,
    patience=25,
)

print("\n  Training quantum autoencoder...")
q_history = q_trainer.fit(train_norm, val_norm, verbose=True)

# Evaluate reconstruction
q_recon_train = q_trainer.reconstruct(train_norm)
q_recon_val = q_trainer.reconstruct(val_norm)

q_train_metrics = all_metrics(train_norm, q_recon_train)
q_val_metrics = all_metrics(val_norm, q_recon_val)

print(f"\n  Reconstruction metrics (normalized space):")
print(f"    Train: MAE={q_train_metrics['MAE']:.6f}, R²={q_train_metrics['R²']:.6f}")
print(f"    Val:   MAE={q_val_metrics['MAE']:.6f}, R²={q_val_metrics['R²']:.6f}")

# Reconstruction in original price space
q_recon_val_prices = denormalize(q_recon_val, scaler)
val_prices = denormalize(val_norm, scaler)
q_val_orig_metrics = all_metrics(val_prices, q_recon_val_prices)
print(f"\n  Reconstruction metrics (original price space):")
print(f"    Val: MAE={q_val_orig_metrics['MAE']:.6f}, R²={q_val_orig_metrics['R²']:.6f}")

# ══════════════════════════════════════════════════════════════════════════
# 3. Train CLASSICAL Autoencoder (ablation)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  3. CLASSICAL AUTOENCODER (ablation)")
print("=" * 60)

c_autoencoder = ClassicalAutoencoder(
    input_dim=224,
    pre_quantum_dim=6,
    latent_dim=LATENT_DIM,
)

c_params = sum(p.numel() for p in c_autoencoder.parameters())
print(f"  Classical params: {c_params}")

c_trainer = AutoencoderTrainer(
    model=c_autoencoder,
    lr=0.01,
    epochs=200,
    batch_size=32,
    patience=25,
)

print("\n  Training classical autoencoder...")
c_history = c_trainer.fit(train_norm, val_norm, verbose=True)

c_recon_val = c_trainer.reconstruct(val_norm)
c_val_metrics = all_metrics(val_norm, c_recon_val)

c_recon_val_prices = denormalize(c_recon_val, scaler)
c_val_orig_metrics = all_metrics(val_prices, c_recon_val_prices)

print(f"\n  Classical reconstruction (normalized):")
print(f"    Val: MAE={c_val_metrics['MAE']:.6f}, R²={c_val_metrics['R²']:.6f}")
print(f"  Classical reconstruction (original):")
print(f"    Val: MAE={c_val_orig_metrics['MAE']:.6f}, R²={c_val_orig_metrics['R²']:.6f}")

# ══════════════════════════════════════════════════════════════════════════
# 4. Comparison: Quantum vs Classical bottleneck
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  4. QUANTUM vs CLASSICAL BOTTLENECK")
print("=" * 60)

print(f"\n  {'Metric':<15} {'Quantum':>12} {'Classical':>12} {'Winner':>10}")
print(f"  {'-'*50}")

for metric in ["MAE", "RMSE", "R²"]:
    q_val = q_val_orig_metrics[metric]
    c_val = c_val_orig_metrics[metric]

    if metric == "R²":
        winner = "Quantum" if q_val > c_val else "Classical"
    else:
        winner = "Quantum" if q_val < c_val else "Classical"

    print(f"  {metric:<15} {q_val:>12.6f} {c_val:>12.6f} {winner:>10}")

# ══════════════════════════════════════════════════════════════════════════
# 5. Extract latent vectors and train temporal predictor
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  5. TEMPORAL MODEL IN LATENT SPACE")
print("=" * 60)

# Extract latents for ALL data using quantum autoencoder
Z_all = q_trainer.extract_latents(all_norm)  # (494, LATENT_DIM)
Z_train = Z_all[:split_idx]
Z_val = Z_all[split_idx:]

print(f"  Latent vectors: {Z_all.shape}")
print(f"  Latent range: [{Z_all.min():.4f}, {Z_all.max():.4f}]")
print(f"  Latent std: {Z_all.std(axis=0).mean():.4f}")

# Train temporal predictor
temporal = TemporalPredictor(
    latent_dim=LATENT_DIM,
    window_size=WINDOW,
    alpha=1.0,
)
temporal.fit(Z_train)
print(f"  Temporal model fitted ✓")

# Evaluate temporal prediction
Z_val_pred = []
for t in range(WINDOW, len(Z_val)):
    window = Z_val[t - WINDOW : t]
    z_pred = temporal.predict(window)
    Z_val_pred.append(z_pred)
Z_val_pred = np.array(Z_val_pred)
Z_val_actual = Z_val[WINDOW:]

temporal_metrics = all_metrics(Z_val_actual, Z_val_pred)
print(f"\n  Temporal prediction in latent space:")
print(f"    MAE={temporal_metrics['MAE']:.6f}, R²={temporal_metrics['R²']:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 6. End-to-end: predict future surfaces
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  6. FUTURE PREDICTIONS (6 days)")
print("=" * 60)

predictor = SwaptionPredictor(
    autoencoder=q_autoencoder,
    temporal_predictor=temporal,
)

# Predict 6 future normalized surfaces
future_norm = predictor.predict_future(Z_all, n_steps=6)
future_prices = denormalize(future_norm, scaler)

print(f"  Future surfaces shape: {future_prices.shape}")
print(f"  Price range: [{future_prices.min():.4f}, {future_prices.max():.4f}]")

# ══════════════════════════════════════════════════════════════════════════
# 7. Impute missing data
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  7. MISSING DATA IMPUTATION")
print("=" * 60)

import pandas as pd

test_df = data["test_df"]
test_info = data["test_info"]
train_dates_parsed = pd.to_datetime(data["train_dates"], dayfirst=True)

imputed_surfaces = {}

for idx in test_info["missing_indices"]:
    missing_date = test_df.iloc[idx]["date"]
    mask = test_info["missing_masks"][idx]

    # Find nearest neighbors
    diffs = np.abs((train_dates_parsed - missing_date).total_seconds().values)
    sorted_idx = np.argsort(diffs)
    i_before = sorted_idx[0]
    i_after = sorted_idx[1]
    if train_dates_parsed[i_before] > train_dates_parsed[i_after]:
        i_before, i_after = i_after, i_before

    # Method 1: Latent space interpolation
    surface_interp_norm = predictor.impute_interpolation(Z_all, i_before, i_after)
    surface_interp = denormalize(surface_interp_norm.reshape(1, -1), scaler)[0]

    # Method 2: Masked optimization (uses known values)
    known_norm = all_norm[i_before].copy()  # Use neighbor as approximate known
    # Get actual known values from test template
    test_prices = test_df.iloc[idx, 2:].values  # skip type and date cols... adjust as needed
    # We use the interpolated z as initialization
    z_init = 0.5 * (Z_all[i_before] + Z_all[i_after])

    imputed_surfaces[idx] = surface_interp

    n_missing = int((~mask).sum())
    print(f"  Date {missing_date}: {n_missing} missing values imputed")
    print(f"    Interpolated between days {i_before} and {i_after}")

# ══════════════════════════════════════════════════════════════════════════
# 8. Save everything
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  8. SAVING")
print("=" * 60)

# Save quantum autoencoder
torch.save(q_autoencoder.state_dict(), str(TRAINED_DIR / "q_autoencoder_weights.pt"))
print("  Saved q_autoencoder_weights.pt ✓")

# Save classical autoencoder
torch.save(c_autoencoder.state_dict(), str(TRAINED_DIR / "c_autoencoder_weights.pt"))
print("  Saved c_autoencoder_weights.pt ✓")

# Save temporal predictor
with open(TRAINED_DIR / "temporal_model.pkl", "wb") as f:
    pickle.dump(temporal, f)
print("  Saved temporal_model.pkl ✓")

# Save latent vectors
np.save(TRAINED_DIR / "latent_vectors.npy", Z_all)
print("  Saved latent_vectors.npy ✓")

# Save predictions
np.save(TRAINED_DIR / "qae_future_prices.npy", future_prices)
np.save(TRAINED_DIR / "qae_future_norm.npy", future_norm)
print("  Saved future predictions ✓")

# Save training histories and comparison
comparison = {
    "quantum": {
        "reconstruction_val_MAE": q_val_orig_metrics["MAE"],
        "reconstruction_val_RMSE": q_val_orig_metrics["RMSE"],
        "reconstruction_val_R2": q_val_orig_metrics["R²"],
        "n_params": n_trainable,
        "epochs_trained": q_history["epochs_trained"],
        "best_val_loss": q_history["best_val_loss"],
    },
    "classical": {
        "reconstruction_val_MAE": c_val_orig_metrics["MAE"],
        "reconstruction_val_RMSE": c_val_orig_metrics["RMSE"],
        "reconstruction_val_R2": c_val_orig_metrics["R²"],
        "n_params": c_params,
        "epochs_trained": c_history["epochs_trained"],
        "best_val_loss": c_history["best_val_loss"],
    },
    "temporal_latent_MAE": temporal_metrics["MAE"],
    "temporal_latent_R2": temporal_metrics["R²"],
    "latent_dim": LATENT_DIM,
}

def to_python(obj):
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    return obj

with open(TRAINED_DIR / "autoencoder_comparison.json", "w") as f:
    json.dump(to_python(comparison), f, indent=2)

# Save training curves
np.save(TRAINED_DIR / "q_train_losses.npy", np.array(q_history["train_losses"]))
np.save(TRAINED_DIR / "q_val_losses.npy", np.array(q_history["val_losses"]))
np.save(TRAINED_DIR / "c_train_losses.npy", np.array(c_history["train_losses"]))
np.save(TRAINED_DIR / "c_val_losses.npy", np.array(c_history["val_losses"]))
print("  Saved training curves ✓")

print("\n  ✅ Phase 3b (Quantum Autoencoder) complete!")
