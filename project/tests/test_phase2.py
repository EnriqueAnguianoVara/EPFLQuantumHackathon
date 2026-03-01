#!/usr/bin/env python3
"""
Phase 2 — Test Suite for Classical Baselines

Tests model instantiation, fit, predict, and rolling prediction
for Ridge, XGBoost, and LSTM.

Run from project root:
    python tests/test_phase2.py
"""

import sys
import os
import time
from pathlib import Path

# Avoid UnicodeEncodeError on cp1252 terminals (Windows default).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


# ── Load data ────────────────────────────────────────────────────────────
section("0. SETUP")

from src.data.loader import load_all
from src.data.preprocessing import full_pipeline

data = load_all()
prices = data["train_prices"]
pipe = full_pipeline(prices, n_pca_components=6, window_size=20, horizon=1)

X_train = pipe["X_train"]
Y_train = pipe["Y_train"]
X_val = pipe["X_val"]
Y_val = pipe["Y_val"]
all_pca = pipe["all_pca"]

print(f"  X_train: {X_train.shape}, Y_train: {Y_train.shape}")
print(f"  X_val:   {X_val.shape}, Y_val:   {Y_val.shape}")

# ══════════════════════════════════════════════════════════════════════════
section("1. RIDGE REGRESSION")
# ══════════════════════════════════════════════════════════════════════════

from src.models.classical.ridge import RidgeBaseline
from src.evaluation.metrics import all_metrics

ridge = RidgeBaseline(alpha=1.0)
check("Ridge instantiation", not ridge.is_fitted)

t0 = time.time()
ridge.fit(X_train, Y_train)
dt = time.time() - t0
check("Ridge fit", ridge.is_fitted)
print(f"  ⏱  Fit time: {dt:.3f}s")

# Predict batch
Y_pred = ridge.predict(X_val)
check("Ridge predict shape", Y_pred.shape == Y_val.shape, f"got {Y_pred.shape}")

# Predict single
Y_single = ridge.predict(X_val[0])
check("Ridge predict single", Y_single.shape == (1, 6), f"got {Y_single.shape}")

# Metrics
m = all_metrics(Y_val, Y_pred)
check("Ridge R² > 0.5", m["R²"] > 0.5, f"R²={m['R²']:.4f}")
print(f"  📊 MAE={m['MAE']:.6f}, RMSE={m['RMSE']:.6f}, R²={m['R²']:.4f}")

# Rolling prediction
future = ridge.predict_rolling(all_pca, n_steps=6, window_size=20, n_pca=6)
check("Ridge rolling shape", future.shape == (6, 6), f"got {future.shape}")
check("Ridge rolling no NaN", not np.any(np.isnan(future)))

# Get params
params = ridge.get_params()
check("Ridge get_params", "alpha" in params)

# ══════════════════════════════════════════════════════════════════════════
section("2. XGBOOST")
# ══════════════════════════════════════════════════════════════════════════

try:
    from src.models.classical.xgboost_model import GBTBaseline

    gbt = GBTBaseline(backend="xgboost", n_estimators=50, max_depth=3)
    check("XGBoost instantiation", not gbt.is_fitted)

    t0 = time.time()
    gbt.fit(X_train, Y_train)
    dt = time.time() - t0
    check("XGBoost fit", gbt.is_fitted)
    print(f"  ⏱  Fit time: {dt:.2f}s")

    Y_pred = gbt.predict(X_val)
    check("XGBoost predict shape", Y_pred.shape == Y_val.shape, f"got {Y_pred.shape}")

    m = all_metrics(Y_val, Y_pred)
    check("XGBoost R² > 0.3", m["R²"] > 0.3, f"R²={m['R²']:.4f}")
    print(f"  📊 MAE={m['MAE']:.6f}, RMSE={m['RMSE']:.6f}, R²={m['R²']:.4f}")

    future = gbt.predict_rolling(all_pca, n_steps=6, window_size=20, n_pca=6)
    check("XGBoost rolling shape", future.shape == (6, 6), f"got {future.shape}")

    importances = gbt.feature_importance()
    check("XGBoost feature importance shape", importances.shape[0] == 6)

    params = gbt.get_params()
    check("XGBoost get_params", "n_estimators" in params)

except ImportError:
    print("  ⚠ xgboost not installed, skipping (pip install xgboost)")

# Try LightGBM backend
try:
    gbt_lgb = GBTBaseline(backend="lightgbm", n_estimators=50, max_depth=3)
    gbt_lgb.fit(X_train, Y_train)
    Y_pred_lgb = gbt_lgb.predict(X_val)
    check("LightGBM backend works", Y_pred_lgb.shape == Y_val.shape)
except ImportError:
    print("  ⚠ lightgbm not installed, skipping (pip install lightgbm)")

# ══════════════════════════════════════════════════════════════════════════
section("3. LSTM")
# ══════════════════════════════════════════════════════════════════════════

try:
    import torch
    from src.models.classical.lstm import LSTMBaseline, LSTMModel

    # Test raw model
    model = LSTMModel(input_dim=6, hidden_dim=16, num_layers=1)
    x = torch.randn(4, 20, 6)  # (batch, seq, features)
    y = model(x)
    check("LSTMModel forward", y.shape == (4, 6), f"got {y.shape}")

    # Test wrapper
    lstm = LSTMBaseline(
        n_pca=6, window_size=20,
        hidden_dim=16, num_layers=1,
        epochs=10, batch_size=32, lr=0.01, patience=5,
    )
    check("LSTM instantiation", not lstm.is_fitted)

    t0 = time.time()
    lstm.fit(X_train, Y_train, X_val, Y_val)
    dt = time.time() - t0
    check("LSTM fit", lstm.is_fitted)
    check("LSTM train losses recorded", len(lstm.train_losses) > 0)
    check("LSTM val losses recorded", len(lstm.val_losses) > 0)
    print(f"  ⏱  Fit time: {dt:.1f}s ({len(lstm.train_losses)} epochs)")

    Y_pred = lstm.predict(X_val)
    check("LSTM predict shape", Y_pred.shape == Y_val.shape, f"got {Y_pred.shape}")

    m = all_metrics(Y_val, Y_pred)
    print(f"  📊 MAE={m['MAE']:.6f}, RMSE={m['RMSE']:.6f}, R²={m['R²']:.4f}")
    # R² might be low with only 10 epochs, just check it doesn't crash
    check("LSTM metrics computed", "R²" in m)

    # Single prediction
    Y_single = lstm.predict(X_val[0])
    check("LSTM predict single", Y_single.shape == (1, 6), f"got {Y_single.shape}")

    # Rolling
    future = lstm.predict_rolling(all_pca, n_steps=6, window_size=20, n_pca=6)
    check("LSTM rolling shape", future.shape == (6, 6), f"got {future.shape}")
    check("LSTM rolling no NaN", not np.any(np.isnan(future)))

    # Save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmppath = f.name
    lstm.save(tmppath)
    check("LSTM save", os.path.exists(tmppath))

    lstm2 = LSTMBaseline(n_pca=6, window_size=20, hidden_dim=16, num_layers=1)
    lstm2.load(tmppath)
    check("LSTM load", lstm2.is_fitted)

    Y_pred2 = lstm2.predict(X_val[:5])
    Y_pred1 = lstm.predict(X_val[:5])
    check("LSTM save/load consistency", np.allclose(Y_pred1, Y_pred2, atol=1e-5))

    os.unlink(tmppath)

    params = lstm.get_params()
    check("LSTM get_params", "hidden_dim" in params and "trainable_params" in params)

except ImportError:
    print("  ⚠ torch not installed, skipping (pip install torch)")

# ══════════════════════════════════════════════════════════════════════════
section("4. TRAINING NOTEBOOK (dry run check)")
# ══════════════════════════════════════════════════════════════════════════

notebook_path = str(ROOT / "notebooks" / "02_classical_baselines.py")
check("Training notebook exists", os.path.exists(notebook_path), notebook_path)

# ══════════════════════════════════════════════════════════════════════════
section("5. STREAMLIT PAGE (import check)")
# ══════════════════════════════════════════════════════════════════════════

page_path = str(ROOT / "pages" / "2_Classical_Baselines.py")
check("Streamlit page exists", os.path.exists(page_path), page_path)

# ══════════════════════════════════════════════════════════════════════════
section("SUMMARY")
# ══════════════════════════════════════════════════════════════════════════

total = PASS + FAIL
print(f"\n  Results: {PASS}/{total} passed, {FAIL} failed\n")

if FAIL == 0:
    print("  🎉 ALL PHASE 2 TESTS PASSED!")
    print()
    print("  Next steps:")
    print("    python notebooks/02_classical_baselines.py   # Train & save models")
    print("    streamlit run app.py                         # View results")
else:
    print(f"  ⚠️  {FAIL} test(s) failed. Check the output above.")
    sys.exit(1)
