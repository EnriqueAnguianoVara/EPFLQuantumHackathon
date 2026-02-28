"""
Model comparison page.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_all, TENORS, MATURITY_LABELS
from src.utils.surface import plot_surface_heatmap, flat_to_grid

st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")
TRAINED_DIR = ROOT / "trained_models"

data = load_all(ROOT / "data")
prices = data["train_prices"]

st.title("📈 Model Comparison")
st.markdown("Head-to-head evaluation of classical and quantum approaches.")


def _extract_r2(metrics: dict) -> float:
    if "R2" in metrics:
        return metrics["R2"]
    if "R²" in metrics:
        return metrics["R²"]
    if "RÂ²" in metrics:
        return metrics["RÂ²"]
    for k, v in metrics.items():
        if str(k).startswith("R"):
            return v
    return np.nan


results = {}

if (TRAINED_DIR / "baselines_summary.json").exists():
    with open(TRAINED_DIR / "baselines_summary.json") as f:
        baselines = json.load(f)
    for name, m in baselines.items():
        results[name.replace("_", " ").title()] = {
            "MAE": m.get("MAE", np.nan),
            "RMSE": m.get("RMSE", np.nan),
            "R2": _extract_r2(m),
        }

if (TRAINED_DIR / "qrc_ablation.json").exists():
    with open(TRAINED_DIR / "qrc_ablation.json") as f:
        ablation = json.load(f)
    valid = [r for r in ablation if "error" not in r]
    if valid:
        best_qrc = min(valid, key=lambda r: r["orig_MAE"])
        results["QRC"] = {
            "MAE": best_qrc.get("orig_MAE", np.nan),
            "RMSE": best_qrc.get("orig_RMSE", np.nan),
            "R2": best_qrc.get("orig_R2", np.nan),
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

ae_comp = {}
if (TRAINED_DIR / "autoencoder_comparison.json").exists():
    with open(TRAINED_DIR / "autoencoder_comparison.json") as f:
        ae_comp = json.load(f)
    if "quantum" in ae_comp:
        results["QAE"] = {
            "MAE": ae_comp["quantum"].get("reconstruction_val_MAE", np.nan),
            "RMSE": ae_comp["quantum"].get("reconstruction_val_RMSE", np.nan),
            "R2": ae_comp["quantum"].get("reconstruction_val_R2", np.nan),
        }
    if "classical" in ae_comp:
        results["Classical AE"] = {
            "MAE": ae_comp["classical"].get("reconstruction_val_MAE", np.nan),
            "RMSE": ae_comp["classical"].get("reconstruction_val_RMSE", np.nan),
            "R2": ae_comp["classical"].get("reconstruction_val_R2", np.nan),
        }

if not results:
    st.warning("No model results found. Run the training notebooks first.")
    st.stop()

st.header("1. Validation Metrics")
st.caption("R2 shown here is the standard coefficient of determination: R2 = 1 - SS_res/SS_tot")

df_results = pd.DataFrame(results).T
df_results.index.name = "Model"
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

import plotly.graph_objects as go

model_names = list(df_results.index)
mae_vals = [df_results.loc[m, "MAE"] for m in model_names]
r2_vals = [df_results.loc[m, "R2"] for m in model_names]

col1, col2 = st.columns(2)
with col1:
    fig_mae = go.Figure(go.Bar(
        x=model_names, y=mae_vals,
        marker_color=["#FF6B35" if "Q" not in m and "Quantum" not in m else "#7030A0" for m in model_names],
    ))
    fig_mae.update_layout(title="MAE by Model", yaxis_title="MAE", height=380)
    st.plotly_chart(fig_mae, use_container_width=True)

with col2:
    fig_r2 = go.Figure(go.Bar(
        x=model_names, y=r2_vals,
        marker_color=["#FF6B35" if "Q" not in m and "Quantum" not in m else "#7030A0" for m in model_names],
    ))
    fig_r2.update_layout(title="R2 by Model", yaxis_title="R2", height=380)
    st.plotly_chart(fig_r2, use_container_width=True)

st.header("2. Autoencoder Bottleneck (Quantum vs Classical)")
if ae_comp:
    q_mae = ae_comp.get("quantum", {}).get("reconstruction_val_MAE", np.nan)
    c_mae = ae_comp.get("classical", {}).get("reconstruction_val_MAE", np.nan)
    q_r2 = ae_comp.get("quantum", {}).get("reconstruction_val_R2", np.nan)
    c_r2 = ae_comp.get("classical", {}).get("reconstruction_val_R2", np.nan)
    c1, c2 = st.columns(2)
    c1.metric("Quantum AE MAE / R2", f"{q_mae:.6f} / {q_r2:.6f}")
    c2.metric("Classical AE MAE / R2", f"{c_mae:.6f} / {c_r2:.6f}")
else:
    st.info("Autoencoder comparison artifact not found.")

st.header("3. Future Prediction Surfaces")
future_files = {
    "Ridge": "ridge_future_prices.npy",
    "QRC": "qrc_future_prices.npy",
    "QAE": "qae_future_prices.npy",
    "QKGP": "qkgp_future_prices.npy",
    "QRLSTM": "qrlstm_future_prices.npy",
    "Naive": "naive_future_prices.npy",
}
future_preds = {}
for name, fname in future_files.items():
    p = TRAINED_DIR / fname
    if p.exists():
        future_preds[name] = np.load(p)

if future_preds:
    future_dates = ["24/12/2051", "26/12/2051", "27/12/2051", "29/12/2051", "30/12/2051", "01/01/2052"]
    day = st.selectbox("Select day", range(6), format_func=lambda i: future_dates[i])

    cols = st.columns(min(len(future_preds) + 1, 4))
    with cols[0]:
        fig = plot_surface_heatmap(prices[-1], title="Last Known", zmin=prices.min(), zmax=prices.max())
        st.plotly_chart(fig, use_container_width=True)

    for i, (name, pred) in enumerate(future_preds.items()):
        col_idx = (i + 1) % len(cols)
        with cols[col_idx]:
            fig = plot_surface_heatmap(pred[day], title=name, zmin=prices.min(), zmax=prices.max())
            st.plotly_chart(fig, use_container_width=True)

    if len(future_preds) >= 2:
        all_preds = np.stack(list(future_preds.values()))
        std_across = all_preds[:, day, :].std(axis=0)
        std_grid = flat_to_grid(std_across)
        fig_std = go.Figure(go.Heatmap(
            z=std_grid,
            x=MATURITY_LABELS,
            y=[str(t) for t in TENORS],
            colorscale="Reds",
            colorbar=dict(title="Std Dev"),
        ))
        fig_std.update_layout(
            title=f"Prediction Disagreement - {future_dates[day]}",
            xaxis_title="Maturity",
            yaxis_title="Tenor",
            yaxis=dict(autorange="reversed"),
            height=380,
        )
        st.plotly_chart(fig_std, use_container_width=True)
else:
    st.info("No future prediction artifacts found.")
