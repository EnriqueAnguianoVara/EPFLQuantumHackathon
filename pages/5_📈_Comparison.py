"""
📈 Comparison — Head-to-head evaluation of all models.
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_all
from src.data.preprocessing import normalize, pca_reduce, denormalize, create_flat_windows
from src.evaluation.metrics import all_metrics, surface_error_grid
from src.utils.surface import plot_surface_heatmap

st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")
TRAINED_DIR = ROOT / "trained_models"

data = load_all(ROOT / "data")
prices = data["train_prices"]

st.title("📈 Model Comparison")
st.markdown("Head-to-head evaluation of classical baselines vs quantum models.")

# ── Load all available metrics ───────────────────────────────────────────
results = {}

if (TRAINED_DIR / "baselines_summary.json").exists():
    with open(TRAINED_DIR / "baselines_summary.json") as f:
        baselines = json.load(f)
    for name, metrics in baselines.items():
        results[name.capitalize()] = metrics

if (TRAINED_DIR / "qrc_ablation.json").exists():
    with open(TRAINED_DIR / "qrc_ablation.json") as f:
        ablation = json.load(f)
    best_qrc = min([r for r in ablation if "error" not in r], key=lambda r: r["orig_MAE"])
    results["QRC"] = {
        "MAE": best_qrc["orig_MAE"],
        "RMSE": best_qrc["orig_RMSE"],
        "R²": best_qrc["orig_R2"],
    }

if (TRAINED_DIR / "quantum_extra_summary.json").exists():
    with open(TRAINED_DIR / "quantum_extra_summary.json") as f:
        extra_q = json.load(f)
    for name in ("QKGP", "QRLSTM"):
        item = extra_q.get(name, {})
        if item.get("status") == "ok":
            results[name] = {
                "MAE": item.get("val_MAE", np.nan),
                "RMSE": item.get("val_RMSE", np.nan),
                "R2": item.get("val_R2", np.nan),
            }

if (TRAINED_DIR / "autoencoder_comparison.json").exists():
    with open(TRAINED_DIR / "autoencoder_comparison.json") as f:
        ae_comp = json.load(f)
    if "quantum" in ae_comp:
        results["QAE"] = {
            "MAE": ae_comp["quantum"]["reconstruction_val_MAE"],
            "RMSE": ae_comp["quantum"]["reconstruction_val_RMSE"],
            "R²": ae_comp["quantum"]["reconstruction_val_R2"],
        }
    if "classical" in ae_comp:
        results["Classical AE"] = {
            "MAE": ae_comp["classical"]["reconstruction_val_MAE"],
            "RMSE": ae_comp["classical"]["reconstruction_val_RMSE"],
            "R²": ae_comp["classical"]["reconstruction_val_R2"],
        }

for _m in results.values():
    if "R2" not in _m:
        for _k, _v in _m.items():
            if str(_k).startswith("R"):
                _m["R2"] = _v
                break

if not results:
    st.warning("No model results found. Run the training notebooks first.")
    st.stop()

# ── Section 1: Metrics table ─────────────────────────────────────────────
st.header("1. Validation Metrics (Original Price Space)")

df_results = pd.DataFrame(results).T
df_results.index.name = "Model"

# Ensure consistent columns
for col in ["MAE", "RMSE", "R2"]:
    if col not in df_results.columns:
        df_results[col] = np.nan

st.dataframe(
    df_results[["MAE", "RMSE", "R2"]].style.format({
        "MAE": "{:.6f}",
        "RMSE": "{:.6f}",
        "R2": "{:.6f}",
    }).highlight_min(axis=0, subset=["MAE", "RMSE"], color="#2d5a27")
    .highlight_max(axis=0, subset=["R2"], color="#2d5a27"),
    use_container_width=True,
)

# Bar chart
import plotly.graph_objects as go

model_names = list(results.keys())
mae_vals = [results[m].get("MAE", 0) for m in model_names]
r2_vals = [results[m].get("R2", 0) for m in model_names]

col1, col2 = st.columns(2)

with col1:
    fig_mae = go.Figure(go.Bar(
        x=model_names, y=mae_vals,
        marker_color=["#FF6B35" if "Q" not in m and "quantum" not in m.lower()
                       else "#7030A0" for m in model_names],
    ))
    fig_mae.update_layout(title="MAE by Model", yaxis_title="MAE", height=400)
    st.plotly_chart(fig_mae, use_container_width=True)

with col2:
    fig_r2 = go.Figure(go.Bar(
        x=model_names, y=r2_vals,
        marker_color=["#FF6B35" if "Q" not in m and "quantum" not in m.lower()
                       else "#7030A0" for m in model_names],
    ))
    fig_r2.update_layout(title="R2 by Model", yaxis_title="R2", height=400)
    st.plotly_chart(fig_r2, use_container_width=True)

st.markdown(
    "🟣 **Purple** = Quantum models &nbsp;&nbsp; 🟠 **Orange** = Classical models"
)

# ── Section 2: Quantum vs Classical bottleneck ────────────────────────────
st.header("2. Quantum vs Classical Bottleneck (Autoencoder)")

if (TRAINED_DIR / "autoencoder_comparison.json").exists():
    q_mae = ae_comp["quantum"]["reconstruction_val_MAE"]
    c_mae = ae_comp["classical"]["reconstruction_val_MAE"]

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Quantum MAE", f"{q_mae:.6f}")
    col_b.metric("Classical MAE", f"{c_mae:.6f}")

    if q_mae < c_mae:
        diff = (c_mae - q_mae) / c_mae * 100
        col_c.metric("Quantum Advantage", f"{diff:.1f}%", delta=f"-{diff:.1f}%")
    else:
        diff = (q_mae - c_mae) / c_mae * 100
        col_c.metric("Classical Advantage", f"{diff:.1f}%", delta=f"+{diff:.1f}%")

    st.markdown(
        "Both autoencoders share identical architecture (encoder: 224→64→16→6, "
        "decoder: 8→16→64→224). The only difference is the bottleneck: "
        "QuantumLayer + LexGrouping vs Linear + Tanh."
    )

# ── Section 3: Future prediction comparison ───────────────────────────────
st.header("3. Future Predictions Comparison")

future_files = {
    "Ridge": "ridge_future_prices.npy",
    "QRC": "qrc_future_prices.npy",
    "QAE": "qae_future_prices.npy",
    "QKGP": "qkgp_future_prices.npy",
    "QRLSTM": "qrlstm_future_prices.npy",
}

future_preds = {}
for name, fname in future_files.items():
    path = TRAINED_DIR / fname
    if path.exists():
        future_preds[name] = np.load(path)

if future_preds:
    future_dates = ["24/12/2051", "26/12/2051", "27/12/2051",
                    "29/12/2051", "30/12/2051", "01/01/2052"]

    day = st.selectbox("Select day", range(6),
                       format_func=lambda i: future_dates[i])

    n_models = len(future_preds)
    cols = st.columns(min(n_models + 1, 4))

    # Last known
    with cols[0]:
        st.markdown("**Last Known**")
        fig = plot_surface_heatmap(prices[-1], title="23/12/2051",
                                   zmin=prices.min(), zmax=prices.max())
        st.plotly_chart(fig, use_container_width=True)

    for i, (name, pred) in enumerate(future_preds.items()):
        col_idx = (i + 1) % len(cols)
        with cols[col_idx]:
            st.markdown(f"**{name}**")
            fig = plot_surface_heatmap(pred[day], title=name,
                                       zmin=prices.min(), zmax=prices.max())
            st.plotly_chart(fig, use_container_width=True)

    # Divergence between models
    if len(future_preds) >= 2:
        st.subheader("Model Agreement")
        all_preds = np.stack(list(future_preds.values()))  # (n_models, 6, 224)
        std_across = all_preds[:, day, :].std(axis=0)  # (224,) std per cell

        from src.utils.surface import flat_to_grid
        from src.data.loader import TENORS, MATURITY_LABELS

        std_grid = flat_to_grid(std_across)
        fig_std = go.Figure(go.Heatmap(
            z=std_grid,
            x=MATURITY_LABELS,
            y=[str(t) for t in TENORS],
            colorscale="Reds",
            colorbar=dict(title="Std Dev"),
        ))
        fig_std.update_layout(
            title=f"Prediction Disagreement — {future_dates[day]}",
            xaxis_title="Maturity", yaxis_title="Tenor",
            yaxis=dict(autorange="reversed"),
            height=400,
        )
        st.plotly_chart(fig_std, use_container_width=True)

        st.markdown(
            "Higher values (red) indicate cells where models disagree the most. "
            "Low disagreement suggests higher confidence in the prediction."
        )

# ── Section 4: Ensemble info ─────────────────────────────────────────────
st.header("4. Ensemble Configuration")

if (TRAINED_DIR / "final_summary.json").exists():
    with open(TRAINED_DIR / "final_summary.json") as f:
        summary = json.load(f)
    st.json(summary)
else:
    st.info("Run `python notebooks/05_final_predictions.py` to generate ensemble results.")

