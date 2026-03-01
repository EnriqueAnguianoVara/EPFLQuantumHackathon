#!/usr/bin/env python3
"""
Phase 3 — Test Suite for Quantum Models

Tests circuit builders, reservoir, and autoencoder.
Requires: merlinquantum, perceval-quandela, torch

Run from project root:
    python tests/test_phase3.py
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


# ── Setup ────────────────────────────────────────────────────────────────
section("0. SETUP")

from src.data.loader import load_all
from src.data.preprocessing import normalize, pca_reduce, full_pipeline
from src.evaluation.metrics import all_metrics

data = load_all()
prices = data["train_prices"]
pipe = full_pipeline(prices, n_pca_components=6, window_size=20, horizon=1)
all_pca = pipe["all_pca"]
train_pca = pipe["train_pca"]
all_norm = pipe["all_norm"]
train_norm = pipe["train_norm"]

print(f"  all_pca: {all_pca.shape}, train_norm: {train_norm.shape}")

try:
    import torch
    import merlin
    import perceval
    HAS_QUANTUM = True
    print(f"  PyTorch: {torch.__version__}")
    print(f"  MerLin: available ✓")
    print(f"  Perceval: available ✓")
except ImportError as e:
    HAS_QUANTUM = False
    print(f"  ⚠ Missing quantum deps: {e}")
    print(f"  Only structural tests will run.")


# ══════════════════════════════════════════════════════════════════════════
section("1. CIRCUIT BUILDERS")
# ══════════════════════════════════════════════════════════════════════════

if HAS_QUANTUM:
    from src.models.quantum.circuits import (
        build_reservoir_circuit,
        build_autoencoder_circuit,
        build_variational_circuit,
    )

    # Reservoir circuit
    res = build_reservoir_circuit(n_modes=6, n_photons=2, input_size=6, seed=42)
    check("Reservoir circuit created", res["layer"] is not None)
    check("Reservoir output_size > 0", res["output_size"] > 0, f"got {res['output_size']}")
    check("Reservoir n_modes", res["n_modes"] == 6)

    # Test forward pass
    x = torch.randn(1, 6)
    with torch.no_grad():
        out = res["layer"](x)
    check("Reservoir forward pass", out.shape == (1, res["output_size"]),
          f"got {out.shape}")

    # Autoencoder circuit
    ae = build_autoencoder_circuit(n_modes=6, n_photons=3, input_size=6)
    check("Autoencoder circuit created", ae["layer"] is not None)
    check("Autoencoder output_size > 0", ae["raw_output_size"] > 0)

    x = torch.randn(1, 6)
    with torch.no_grad():
        out = ae["layer"](x)
    check("Autoencoder forward pass", out.shape[0] == 1)

    # Variational circuit
    var = build_variational_circuit(n_modes=6, n_photons=3, input_size=6, n_layers=2)
    check("Variational circuit created", var["layer"] is not None)

    x = torch.randn(1, 6)
    with torch.no_grad():
        out = var["layer"](x)
    check("Variational forward pass", out.shape[0] == 1)

    # Trainable parameters exist in autoencoder circuit
    trainable = sum(p.numel() for p in ae["layer"].parameters() if p.requires_grad)
    check("Autoencoder has trainable params", trainable > 0, f"got {trainable}")

else:
    print("  Skipping circuit tests (no MerLin)")


# ══════════════════════════════════════════════════════════════════════════
section("2. QUANTUM RESERVOIR")
# ══════════════════════════════════════════════════════════════════════════

if HAS_QUANTUM:
    from src.models.quantum.reservoir import QuantumReservoir

    qrc = QuantumReservoir(
        n_modes=6, n_photons=2, n_pca=6,
        window_size=20, encoding="last", seed=42,
    )
    check("QRC instantiation", not qrc.is_fitted)

    # Extract features for a single window
    window = train_pca[:20]
    feat = qrc.extract_features(window)
    check("QRC extract features shape", feat.shape[1] == qrc.q_output_size,
          f"got {feat.shape}")

    # Extract all features (small subset for speed)
    subset = train_pca[:50]
    t0 = time.time()
    all_feat = qrc.extract_all_features(subset, verbose=False)
    dt = time.time() - t0
    check("QRC extract all features", all_feat.shape == (30, qrc.q_output_size),
          f"got {all_feat.shape}")
    print(f"  ⏱  50 samples feature extraction: {dt:.2f}s")

    # Fit
    targets = subset[20:]
    qrc.fit(all_feat, targets)
    check("QRC fit", qrc.is_fitted)

    # Predict
    pred = qrc.predict(all_feat[:5])
    check("QRC predict shape", pred.shape == (5, 6), f"got {pred.shape}")

    # Rolling
    rolling = qrc.predict_rolling(train_pca[:50], n_steps=3)
    check("QRC rolling shape", rolling.shape == (3, 6), f"got {rolling.shape}")
    check("QRC rolling no NaN", not np.any(np.isnan(rolling)))

    # Memory capacity
    mc = qrc.get_memory_capacity(subset, max_lag=10)
    check("QRC memory capacity shape", mc.shape == (10,))
    check("QRC memory capacity values ≥ 0", np.all(mc >= 0))

    # Params
    params = qrc.get_params()
    check("QRC get_params", "n_modes" in params and "q_output_size" in params)

else:
    print("  Skipping QRC tests (no MerLin)")


# ══════════════════════════════════════════════════════════════════════════
section("3. QUANTUM AUTOENCODER")
# ══════════════════════════════════════════════════════════════════════════

if HAS_QUANTUM:
    from src.models.quantum.autoencoder import (
        QuantumAutoencoder, ClassicalAutoencoder,
        AutoencoderTrainer, TemporalPredictor, SwaptionPredictor,
    )

    # Quantum model
    q_ae = QuantumAutoencoder(
        input_dim=224, pre_quantum_dim=6,
        n_modes=6, n_photons=3, latent_dim=8,
    )
    n_params = sum(p.numel() for p in q_ae.parameters())
    check("QAE instantiation", n_params > 0, f"{n_params} params")

    # Forward pass
    x = torch.randn(2, 224)
    with torch.no_grad():
        x_hat = q_ae(x)
    check("QAE forward shape", x_hat.shape == (2, 224), f"got {x_hat.shape}")

    # Encode
    with torch.no_grad():
        z = q_ae.encode(x)
    check("QAE encode shape", z.shape == (2, 8), f"got {z.shape}")

    # Decode
    with torch.no_grad():
        x_dec = q_ae.decode(z)
    check("QAE decode shape", x_dec.shape == (2, 224), f"got {x_dec.shape}")

    # Classical model (ablation)
    c_ae = ClassicalAutoencoder(input_dim=224, pre_quantum_dim=6, latent_dim=8)
    with torch.no_grad():
        c_hat = c_ae(x)
    check("Classical AE forward", c_hat.shape == (2, 224))

    # Trainer (quick — 3 epochs)
    trainer = AutoencoderTrainer(q_ae, lr=0.01, epochs=3, batch_size=16, patience=5)
    small_data = train_norm[:30]
    history = trainer.fit(small_data, verbose=False)
    check("QAE training ran", history["epochs_trained"] > 0)

    # Extract latents
    Z = trainer.extract_latents(small_data)
    check("QAE latents shape", Z.shape == (30, 8), f"got {Z.shape}")

    # Reconstruct
    recon = trainer.reconstruct(small_data)
    check("QAE reconstruction shape", recon.shape == (30, 224))

    # Temporal predictor
    temporal = TemporalPredictor(latent_dim=8, window_size=5, alpha=1.0)
    Z_fake = np.random.randn(30, 8)
    temporal.fit(Z_fake)
    check("Temporal fit", temporal.is_fitted)

    z_next = temporal.predict(Z_fake[:5])
    check("Temporal predict shape", z_next.shape == (8,), f"got {z_next.shape}")

    z_future = temporal.predict_rolling(Z_fake, n_steps=3)
    check("Temporal rolling shape", z_future.shape == (3, 8), f"got {z_future.shape}")

    # Swaption predictor
    predictor = SwaptionPredictor(q_ae, temporal)

    future_surf = predictor.predict_future(Z_fake, n_steps=3)
    check("SwaptionPredictor future shape", future_surf.shape == (3, 224),
          f"got {future_surf.shape}")

    # Interpolation imputation
    imputed = predictor.impute_interpolation(Z_fake, idx_before=10, idx_after=12)
    check("SwaptionPredictor impute shape", imputed.shape == (224,),
          f"got {imputed.shape}")

    # Masked imputation
    known = np.random.randn(224).astype(np.float32)
    mask = np.ones(224, dtype=bool)
    mask[0:5] = False
    imputed_masked = predictor.impute_masked(known, mask, n_steps=10, lr=0.1)
    check("SwaptionPredictor masked impute shape", imputed_masked.shape == (224,))

else:
    print("  Skipping Autoencoder tests (no MerLin)")


# ══════════════════════════════════════════════════════════════════════════
section("4. FILE CHECKS")
# ══════════════════════════════════════════════════════════════════════════

base = str(ROOT)
files = [
    "src/models/quantum/circuits.py",
    "src/models/quantum/reservoir.py",
    "src/models/quantum/autoencoder.py",
    "src/models/quantum/__init__.py",
    "notebooks/03_quantum_reservoir.py",
    "notebooks/04_quantum_autoencoder.py",
    "pages/3_Quantum_Approaches.py",
    "pages/5_Comparison.py",
]
for f in files:
    check(f"File exists: {f}", os.path.exists(os.path.join(base, f)))


# ══════════════════════════════════════════════════════════════════════════
section("SUMMARY")
# ══════════════════════════════════════════════════════════════════════════

total = PASS + FAIL
print(f"\n  Results: {PASS}/{total} passed, {FAIL} failed")

if not HAS_QUANTUM:
    print(f"\n  ⚠ Quantum tests skipped — install merlinquantum + perceval-quandela + torch")

if FAIL == 0:
    print("\n  🎉 ALL PHASE 3 TESTS PASSED!")
    print()
    print("  Next steps:")
    print("    python notebooks/03_quantum_reservoir.py   # Train QRC")
    print("    python notebooks/04_quantum_autoencoder.py  # Train QAE")
    print("    streamlit run app.py                        # View results")
else:
    print(f"\n  ⚠️  {FAIL} test(s) failed.")
    sys.exit(1)
