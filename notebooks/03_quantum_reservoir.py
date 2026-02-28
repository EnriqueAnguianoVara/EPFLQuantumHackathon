#!/usr/bin/env python3
"""
Notebook 03 — Quantum Reservoir Computing

Trains QRC models with ablation over circuit parameters,
computes memory capacity, and saves best model.

Run from project root:
    python notebooks/03_quantum_reservoir.py
"""

import sys
import os
import time
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all
from src.data.preprocessing import normalize, pca_reduce, denormalize, full_pipeline
from src.evaluation.metrics import all_metrics
from src.models.quantum.reservoir import QuantumReservoir

TRAINED_DIR = ROOT / "trained_models"
TRAINED_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  1. LOADING DATA")
print("=" * 60)

data = load_all()
prices = data["train_prices"]  # (494, 224)

N_PCA = 6
WINDOW = 20
VAL_RATIO = 0.15

# Load saved preprocessing from Phase 2
with open(TRAINED_DIR / "scalers.pkl", "rb") as f:
    prep = pickle.load(f)
scaler = prep["scaler"]
pca_reducer = prep["pca_reducer"]

# Prepare data
all_norm, _ = normalize(prices, scaler)
all_pca, _ = pca_reduce(all_norm, reducer=pca_reducer)

split_idx = int(len(prices) * (1 - VAL_RATIO))
train_pca = all_pca[:split_idx]
val_pca = all_pca[split_idx:]

print(f"  Train PCA: {train_pca.shape}")
print(f"  Val PCA:   {val_pca.shape}")
print(f"  Total PCA: {all_pca.shape}")

# ══════════════════════════════════════════════════════════════════════════
# 2. Ablation study: vary n_modes and n_photons
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  2. ABLATION STUDY")
print("=" * 60)

ablation_results = []

configs = [
    {"n_modes": 6,  "n_photons": 2, "encoding": "last"},
    {"n_modes": 6,  "n_photons": 2, "encoding": "flatten"},
    {"n_modes": 6,  "n_photons": 3, "encoding": "last"},
    {"n_modes": 6,  "n_photons": 3, "encoding": "flatten"},
    {"n_modes": 8,  "n_photons": 3, "encoding": "last"},
    {"n_modes": 8,  "n_photons": 3, "encoding": "flatten"},
    {"n_modes": 10, "n_photons": 4, "encoding": "last"},
    {"n_modes": 10, "n_photons": 4, "encoding": "flatten"},
]

best_model = None
best_mae = float("inf")

for cfg in configs:
    n_modes = cfg["n_modes"]
    n_photons = cfg["n_photons"]
    encoding = cfg["encoding"]

    print(f"\n  --- n_modes={n_modes}, n_photons={n_photons}, encoding={encoding} ---")

    try:
        t0 = time.time()

        qrc = QuantumReservoir(
            n_modes=n_modes,
            n_photons=n_photons,
            n_pca=N_PCA,
            window_size=WINDOW,
            encoding=encoding,
            readout_alpha=1.0,
            seed=42,
        )

        print(f"  Reservoir output dim: {qrc.q_output_size}")

        # Extract features for train
        train_features = qrc.extract_all_features(train_pca, verbose=True)

        # Targets: next step
        train_targets = train_pca[WINDOW:]
        assert train_features.shape[0] == train_targets.shape[0]

        # Fit readout
        qrc.fit(train_features, train_targets)

        # Validate
        val_features = qrc.extract_all_features(val_pca, verbose=False)
        val_targets = val_pca[WINDOW:]
        val_pred_pca = qrc.predict(val_features)

        # Metrics in PCA space
        m_pca = all_metrics(val_targets, val_pred_pca)

        # Metrics in original price space
        val_pred_norm = pca_reducer.inverse_transform(val_pred_pca)
        val_pred_prices = denormalize(val_pred_norm, scaler)
        val_actual_norm = pca_reducer.inverse_transform(val_targets)
        val_actual_prices = denormalize(val_actual_norm, scaler)
        m_orig = all_metrics(val_actual_prices, val_pred_prices)

        dt = time.time() - t0

        result = {
            "n_modes": n_modes,
            "n_photons": n_photons,
            "encoding": encoding,
            "q_output_size": qrc.q_output_size,
            "pca_MAE": m_pca["MAE"],
            "pca_R2": m_pca["R²"],
            "orig_MAE": m_orig["MAE"],
            "orig_RMSE": m_orig["RMSE"],
            "orig_R2": m_orig["R²"],
            "time_seconds": dt,
        }
        ablation_results.append(result)

        print(f"  PCA  → MAE={m_pca['MAE']:.6f}, R²={m_pca['R²']:.4f}")
        print(f"  Orig → MAE={m_orig['MAE']:.6f}, R²={m_orig['R²']:.4f}")
        print(f"  Time: {dt:.1f}s")

        if m_orig["MAE"] < best_mae:
            best_mae = m_orig["MAE"]
            best_model = qrc
            best_config = cfg

    except Exception as e:
        print(f"  ⚠ Error: {e}")
        ablation_results.append({
            "n_modes": n_modes,
            "n_photons": n_photons,
            "error": str(e),
        })

# Print summary table
print("\n  " + "-" * 70)
print(f"  {'Modes':>5} {'Photons':>7} {'Enc':>8} {'Fock dim':>8} {'MAE':>10} {'R2':>8} {'Time':>8}")
print("  " + "-" * 70)
for r in ablation_results:
    if "error" in r:
        print(f"  {r['n_modes']:>5} {r['n_photons']:>7} {r.get('encoding', '-'):>8} {'ERROR':>8}")
    else:
        print(f"  {r['n_modes']:>5} {r['n_photons']:>7} {r['encoding']:>8} {r['q_output_size']:>8} "
              f"{r['orig_MAE']:>10.6f} {r['orig_R2']:>8.4f} {r['time_seconds']:>7.1f}s")

print(f"\n  Best: n_modes={best_config['n_modes']}, n_photons={best_config['n_photons']}, encoding={best_config['encoding']}")

# ══════════════════════════════════════════════════════════════════════════
# 3. Memory capacity of best reservoir
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  3. MEMORY CAPACITY")
print("=" * 60)

mc = best_model.get_memory_capacity(train_pca, max_lag=30)
total_mc = mc.sum()
print(f"  Total memory capacity: {total_mc:.2f}")
print(f"  Per-lag MC (first 10): {[f'{v:.3f}' for v in mc[:10]]}")

# ══════════════════════════════════════════════════════════════════════════
# 4. Rolling prediction (6 future days)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  4. ROLLING PREDICTIONS")
print("=" * 60)

qrc_future_pca = best_model.predict_rolling(all_pca, n_steps=6)
qrc_future_norm = pca_reducer.inverse_transform(qrc_future_pca)
qrc_future_prices = denormalize(qrc_future_norm, scaler)

print(f"  Future predictions shape: {qrc_future_prices.shape}")
print(f"  Price range: [{qrc_future_prices.min():.4f}, {qrc_future_prices.max():.4f}]")

# ══════════════════════════════════════════════════════════════════════════
# 5. Save everything
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  5. SAVING")
print("=" * 60)

# Cannot pickle QuantumLayer (Perceval objects), so save the readout + config
qrc_save = {
    "readout": best_model.readout,
    "config": best_model.get_params(),
}
with open(TRAINED_DIR / "qrc_model.pkl", "wb") as f:
    pickle.dump(qrc_save, f)
print("  Saved qrc_model.pkl (readout + config) ✓")

np.save(TRAINED_DIR / "qrc_future_pca.npy", qrc_future_pca)
np.save(TRAINED_DIR / "qrc_future_prices.npy", qrc_future_prices)
np.save(TRAINED_DIR / "qrc_memory_capacity.npy", mc)
print("  Saved future predictions + memory capacity ✓")

with open(TRAINED_DIR / "qrc_ablation.json", "w") as f:
    json.dump(ablation_results, f, indent=2)
print("  Saved qrc_ablation.json ✓")

print("\n  ✅ Phase 3a (Quantum Reservoir) complete!")
