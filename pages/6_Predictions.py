"""
Final predictions and submission page.
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

from src.data.loader import load_all, TENORS, MATURITY_LABELS, N_TENORS
from src.utils.surface import plot_surface_heatmap

st.set_page_config(page_title="Final Predictions", layout="wide")
TRAINED_DIR = ROOT / "trained_models"
DATA_DIR = ROOT / "data"

data = load_all(ROOT / "data")
prices = data["train_prices"]
test_df = data["test_df"]
test_info = data["test_info"]

st.title("Final Predictions")
st.markdown("Forecast uses the best available quantum model (not the ensemble).")


def load_best_quantum_forecast():
    candidates = []

    if (TRAINED_DIR / "qrc_ablation.json").exists() and (TRAINED_DIR / "qrc_future_prices.npy").exists():
        with open(TRAINED_DIR / "qrc_ablation.json") as f:
            ablation = json.load(f)
        valid = [r for r in ablation if "error" not in r]
        if valid:
            best_qrc = min(valid, key=lambda r: r["orig_MAE"])
            candidates.append(("QRC", float(best_qrc["orig_MAE"]), np.load(TRAINED_DIR / "qrc_future_prices.npy")))

    if (TRAINED_DIR / "quantum_extra_summary.json").exists():
        with open(TRAINED_DIR / "quantum_extra_summary.json") as f:
            extra = json.load(f)
        for name, fname in [("QKGP", "qkgp_future_prices.npy"), ("QRLSTM", "qrlstm_future_prices.npy")]:
            item = extra.get(name, {})
            path = TRAINED_DIR / fname
            if item.get("status") == "ok" and path.exists():
                candidates.append((name, float(item.get("val_MAE", np.inf)), np.load(path)))

    if not candidates and (TRAINED_DIR / "qae_future_prices.npy").exists():
        candidates.append(("QAE", np.inf, np.load(TRAINED_DIR / "qae_future_prices.npy")))

    if not candidates:
        return None

    best = min(candidates, key=lambda x: x[1])
    return {"model_name": best[0], "val_mae": best[1], "future_prices": best[2]}


best_quantum = load_best_quantum_forecast()
if best_quantum is None:
    st.error("No quantum forecast artifacts found. Run `python run_pipeline.py` first.")
    st.stop()

st.success(
    f"Using best quantum model: {best_quantum['model_name']}"
    + ("" if np.isinf(best_quantum["val_mae"]) else f" (val MAE={best_quantum['val_mae']:.6f})")
)

results_path = DATA_DIR / "results.xlsx"
results_df = None
if results_path.exists():
    results_df = pd.read_excel(results_path, engine="openpyxl")

st.header("1. Future Predictions (6 days)")

future_indices = test_info["future_indices"]
future_dates = [pd.to_datetime(test_df.iloc[idx]["date"]).strftime("%d/%m/%Y") for idx in future_indices]
future_surface = best_quantum["future_prices"]

for i, dt in enumerate(future_dates):
    prices_i = future_surface[i]
    with st.expander(f"{dt}", expanded=(i == 0)):
        col_a, col_b = st.columns([3, 1])
        with col_a:
            fig = plot_surface_heatmap(
                prices_i,
                title=f"{best_quantum['model_name']} Forecast - {dt}",
                zmin=prices.min(),
                zmax=prices.max(),
            )
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            st.metric("Mean Price", f"{prices_i.mean():.4f}")
            st.metric("Min Price", f"{prices_i.min():.4f}")
            st.metric("Max Price", f"{prices_i.max():.4f}")
            delta = prices_i.mean() - prices[-1].mean()
            st.metric("Delta vs last known", f"{delta:+.4f}")

st.header("2. Missing Data Imputation (2 dates)")
if results_df is None:
    st.info("results.xlsx not found, so missing-value outputs and download are not available yet.")
else:
    price_cols = [c for c in results_df.columns if c not in ("Type", "Date", "type", "date")]
    missing_indices = test_info["missing_indices"]
    missing_dates = [pd.to_datetime(test_df.iloc[idx]["date"]).strftime("%d/%m/%Y") for idx in missing_indices]

    for i, idx in enumerate(missing_indices):
        row = results_df.iloc[idx]
        imputed_prices = row[price_cols].values.astype(float)
        mask = test_info["missing_masks"][idx]
        n_missing = int((~mask).sum())

        with st.expander(f"{missing_dates[i]} - {n_missing} values imputed"):
            fig = plot_surface_heatmap(
                imputed_prices,
                title=f"Imputed Surface - {missing_dates[i]}",
                zmin=prices.min(),
                zmax=prices.max(),
            )
            st.plotly_chart(fig, use_container_width=True)

            missing_cols = np.where(~mask)[0]
            st.markdown("**Imputed cells:**")
            for col_idx in missing_cols:
                mat_idx = col_idx // N_TENORS
                ten_idx = col_idx % N_TENORS
                st.markdown(
                    f"- Tenor **{TENORS[ten_idx]}Y**, Maturity **{MATURITY_LABELS[mat_idx]}**: "
                    f"**{imputed_prices[col_idx]:.6f}**"
                )

st.header("3. Continuity Check")
st.markdown("Comparing best-quantum forecasts with recent history.")

import plotly.graph_objects as go

representative = [
    (0, 4, "Tenor 1Y, Mat 1Y"),
    (4, 8, "Tenor 5Y, Mat 3Y"),
    (9, 12, "Tenor 10Y, Mat 15Y"),
    (13, 15, "Tenor 30Y, Mat 30Y"),
]

for ten_idx, mat_idx, label in representative:
    flat_idx = mat_idx * N_TENORS + ten_idx
    hist_days = 30
    hist_prices = prices[-hist_days:, flat_idx]
    future_vals = [float(best_quantum["future_prices"][i, flat_idx]) for i in range(6)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(hist_days)),
        y=hist_prices,
        mode="lines",
        name="Historical",
        line=dict(color="#7030A0"),
    ))
    fig.add_trace(go.Scatter(
        x=list(range(hist_days, hist_days + 6)),
        y=future_vals,
        mode="lines+markers",
        name=f"{best_quantum['model_name']} Forecast",
        line=dict(color="#FF6B35", dash="dash"),
        marker=dict(size=8),
    ))
    fig.add_vline(x=hist_days - 0.5, line_dash="dot", line_color="gray", annotation_text="Forecast start")
    fig.update_layout(
        title=f"Continuity: {label}",
        xaxis_title="Day",
        yaxis_title="Price",
        height=340,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

st.header("4. Download Results")
if results_df is None:
    st.info("Run `python notebooks/05_final_predictions.py` to generate results.xlsx for submission download.")
else:
    with open(results_path, "rb") as f:
        st.download_button(
            label="📥 Download results.xlsx",
            data=f.read(),
            file_name="results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

if (TRAINED_DIR / "final_summary.json").exists():
    with st.expander("Model Summary JSON"):
        with open(TRAINED_DIR / "final_summary.json") as f:
            st.json(json.load(f))
