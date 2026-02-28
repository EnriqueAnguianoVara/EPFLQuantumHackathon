#!/usr/bin/env python3
"""
Phase 1 — Full Test Suite

Run from the project root:
    python test_phase1.py

Tests all data loading, preprocessing, surface utilities, metrics,
and walk-forward validation WITHOUT requiring plotly or streamlit.
"""

import sys
import os
import time

# Ensure we can import src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════
section("1. DATA LOADER")
# ══════════════════════════════════════════════════════════════

from src.data.loader import (
    load_train, load_test_template, load_all,
    parse_header, parse_all_headers,
    TENORS, MATURITIES, MATURITY_LABELS, N_TENORS, N_MATURITIES, N_PRICES,
)

# Constants
check("TENORS count", len(TENORS) == 14, f"got {len(TENORS)}")
check("MATURITIES count", len(MATURITIES) == 16, f"got {len(MATURITIES)}")
check("N_PRICES = 224", N_PRICES == 224, f"got {N_PRICES}")
check("MATURITY_LABELS count", len(MATURITY_LABELS) == 16)

# Header parsing
parsed = parse_header("Tenor : 5; Maturity : 0.0833333333333333")
check("parse_header tenor", parsed["tenor"] == 5.0)
check("parse_header maturity", abs(parsed["maturity"] - 0.0833333) < 1e-5)
check("parse_header invalid", parse_header("Date") is None)

# Load train
t0 = time.time()
train_df, train_prices, train_dates = load_train()
dt = time.time() - t0
check("train_prices shape", train_prices.shape == (494, 224), f"got {train_prices.shape}")
check("train no NaNs", not np.any(np.isnan(train_prices)))
check("train_dates length", len(train_dates) == 494)
check("train first date", train_dates[0] == "01/01/2050", f"got {train_dates[0]}")
check("train last date", train_dates[-1] == "23/12/2051", f"got {train_dates[-1]}")
check("train_df has 'date' col", "date" in train_df.columns)
check("train price range", train_prices.min() > 0 and train_prices.max() < 1)
print(f"  ⏱  Train loaded in {dt:.2f}s")

# Load test template
test_df, test_info = load_test_template()
check("test_df shape", test_df.shape[0] == 8, f"got {test_df.shape[0]} rows")
check("future rows", test_info["future_indices"] == [0, 1, 2, 3, 4, 5])
check("missing rows", test_info["missing_indices"] == [6, 7])
check("missing row 6 has 4 NAs",
      int((~test_info["missing_masks"][6]).sum()) == 4,
      f"got {int((~test_info['missing_masks'][6]).sum())}")
check("missing row 7 has 6 NAs",
      int((~test_info["missing_masks"][7]).sum()) == 6,
      f"got {int((~test_info['missing_masks'][7]).sum())}")

# Load all
data = load_all()
check("load_all has all keys",
      all(k in data for k in ["train_prices", "test_df", "test_info", "tenors", "maturities"]))


# ══════════════════════════════════════════════════════════════
section("2. PREPROCESSING")
# ══════════════════════════════════════════════════════════════

from src.data.preprocessing import (
    normalize, denormalize, pca_reduce,
    create_windows, create_flat_windows,
    train_val_split, full_pipeline,
    Scaler, PCAReducer,
)

# Normalization
prices_norm, scaler = normalize(train_prices)
check("normalize shape", prices_norm.shape == (494, 224))
check("normalize mean ≈ 0", np.abs(prices_norm.mean(axis=0)).max() < 1e-10)
check("normalize std ≈ 1", np.abs(prices_norm.std(axis=0) - 1).max() < 1e-10)

# Denormalize roundtrip
prices_back = denormalize(prices_norm, scaler)
check("denormalize roundtrip", np.allclose(train_prices, prices_back, atol=1e-10))

# PCA
reduced, reducer = pca_reduce(prices_norm, n_components=10)
check("PCA shape", reduced.shape == (494, 10))
check("PCA variance sums ≤ 1", reducer.cumulative_variance[-1] <= 1.0 + 1e-10)
check("PCA 3 components > 99%", reducer.cumulative_variance[2] > 0.99,
      f"got {reducer.cumulative_variance[2]:.4f}")
print(f"  📊 PCA cumulative variance: {[f'{v:.4f}' for v in reducer.cumulative_variance[:6]]}")

# PCA inverse roundtrip
reconstructed = reducer.inverse_transform(reduced)
recon_error = np.mean((prices_norm - reconstructed) ** 2)
check("PCA 10-comp reconstruction error < 1e-4", recon_error < 1e-4, f"got {recon_error:.2e}")

# Windows
X_3d, Y_3d = create_windows(reduced, window_size=20, horizon=1)
check("windows 3D X shape", X_3d.shape == (474, 20, 10), f"got {X_3d.shape}")
check("windows 3D Y shape", Y_3d.shape == (474, 1, 10), f"got {Y_3d.shape}")

X_flat, Y_flat = create_flat_windows(reduced, window_size=20, horizon=1)
check("windows flat X shape", X_flat.shape == (474, 200), f"got {X_flat.shape}")
check("windows flat Y shape", Y_flat.shape == (474, 10), f"got {Y_flat.shape}")

# Multi-horizon windows
X_mh, Y_mh = create_windows(reduced, window_size=20, horizon=5)
check("multi-horizon X shape", X_mh.shape == (470, 20, 10), f"got {X_mh.shape}")
check("multi-horizon Y shape", Y_mh.shape == (470, 5, 10), f"got {Y_mh.shape}")

# Train/val split
train_split, val_split, split_idx = train_val_split(train_prices, val_ratio=0.15)
check("split preserves total", train_split.shape[0] + val_split.shape[0] == 494)
check("split ratio ~85/15",
      abs(train_split.shape[0] / 494 - 0.85) < 0.02,
      f"got {train_split.shape[0]/494:.2f}")

# Full pipeline
pipe = full_pipeline(train_prices, n_pca_components=6, window_size=20, horizon=1)
check("pipeline X_train cols = window*pca",
      pipe["X_train"].shape[1] == 20 * 6,
      f"got {pipe['X_train'].shape[1]}")
check("pipeline Y_train cols = pca",
      pipe["Y_train"].shape[1] == 6,
      f"got {pipe['Y_train'].shape[1]}")
check("pipeline has scaler", pipe["scaler"].is_fitted)
check("pipeline has pca_reducer", pipe["pca_reducer"].is_fitted)


# ══════════════════════════════════════════════════════════════
section("3. SURFACE UTILITIES")
# ══════════════════════════════════════════════════════════════

from src.utils.surface import flat_to_grid, grid_to_flat, batch_flat_to_grid

# Single surface roundtrip
flat = train_prices[0]
grid = flat_to_grid(flat)
check("flat_to_grid shape", grid.shape == (14, 16), f"got {grid.shape}")

flat_back = grid_to_flat(grid)
check("grid_to_flat roundtrip", np.allclose(flat, flat_back))

# Verify the mapping is correct:
# flat[0] should be Tenor:1, Maturity:0.083 (first tenor, first maturity)
# flat[14] should be Tenor:1, Maturity:0.25 (first tenor, second maturity)
# grid[0,0] = Tenor:1, Maturity:0.083
# grid[0,1] = Tenor:1, Maturity:0.25
check("grid[0,0] = flat[0] (T1,M1)", abs(grid[0, 0] - flat[0]) < 1e-15)
check("grid[0,1] = flat[14] (T1,M2)", abs(grid[0, 1] - flat[14]) < 1e-15)
check("grid[1,0] = flat[1] (T2,M1)", abs(grid[1, 0] - flat[1]) < 1e-15)

# Batch
batch = train_prices[:5]
grids = batch_flat_to_grid(batch)
check("batch_flat_to_grid shape", grids.shape == (5, 14, 16), f"got {grids.shape}")
check("batch consistency", np.allclose(grids[0], grid))


# ══════════════════════════════════════════════════════════════
section("4. EVALUATION METRICS")
# ══════════════════════════════════════════════════════════════

from src.evaluation.metrics import mae, rmse, mape, r_squared, max_error, all_metrics, surface_error_grid

np.random.seed(42)
y_true = np.random.randn(50, 224)
y_pred = y_true + 0.1 * np.random.randn(50, 224)

check("MAE > 0", mae(y_true, y_pred) > 0)
check("RMSE > MAE", rmse(y_true, y_pred) >= mae(y_true, y_pred))
check("R² close to 1", r_squared(y_true, y_pred) > 0.95)
check("max_error > 0", max_error(y_true, y_pred) > 0)

# Perfect prediction
check("MAE perfect = 0", mae(y_true, y_true) == 0.0)
check("R² perfect = 1", r_squared(y_true, y_true) == 1.0)

# all_metrics
m = all_metrics(y_true, y_pred)
check("all_metrics keys", set(m.keys()) == {"MAE", "RMSE", "MAPE (%)", "R²", "Max Error"})

# surface_error_grid
err_grid = surface_error_grid(y_true, y_pred)
check("surface_error_grid shape", err_grid.shape == (14, 16), f"got {err_grid.shape}")


# ══════════════════════════════════════════════════════════════
section("5. EXCEL WRITER (dry run)")
# ══════════════════════════════════════════════════════════════

from src.utils.excel_writer import fill_test_template
from pathlib import Path

# Don't actually write, just verify the function signature works
check("fill_test_template is callable", callable(fill_test_template))


# ══════════════════════════════════════════════════════════════
section("6. WALK-FORWARD VALIDATION (quick)")
# ══════════════════════════════════════════════════════════════

from src.evaluation.walk_forward import walk_forward_validation
from sklearn.linear_model import Ridge

# Quick test with PCA data
pca_data = pipe["all_pca"]  # (494, 6)

def train_fn(X, Y):
    model = Ridge(alpha=1.0)
    model.fit(X, Y)
    return model

def predict_fn(model, X):
    return model.predict(X)

t0 = time.time()
wf = walk_forward_validation(
    pca_data,
    train_fn=train_fn,
    predict_fn=predict_fn,
    initial_train_size=400,
    step_size=10,  # large step for quick test
    window_size=20,
    horizon=1,
)
dt = time.time() - t0

check("walk-forward ran", wf["n_steps"] > 0, f"got {wf['n_steps']} steps")
check("walk-forward has metrics", "MAE" in wf["metrics"])
print(f"  ⏱  Walk-forward ({wf['n_steps']} steps) in {dt:.2f}s")
print(f"  📊 Metrics: MAE={wf['metrics']['MAE']:.6f}, R²={wf['metrics']['R²']:.4f}")


# ══════════════════════════════════════════════════════════════
section("SUMMARY")
# ══════════════════════════════════════════════════════════════

total = PASS + FAIL
print(f"\n  Results: {PASS}/{total} passed, {FAIL} failed\n")

if FAIL == 0:
    print("  🎉 ALL TESTS PASSED — Phase 1 is fully functional!")
    print()
    print("  Next steps:")
    print("    1. pip install plotly streamlit  (if not done)")
    print("    2. streamlit run app.py")
    print("    3. Navigate to 📊 Market Explorer")
    print()
else:
    print(f"  ⚠️  {FAIL} test(s) failed. Check the output above.")
    sys.exit(1)
