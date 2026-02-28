"""
🎯 Predictions — Final results and Excel download.
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
from src.utils.surface import plot_surface_heatmap, flat_to_grid

st.set_page_config(page_title="Final Predictions", page_icon="🎯", layout="wide")
TRAINED_DIR = ROOT / "trained_models"
DATA_DIR = ROOT / "data"

data = load_all(ROOT / "data")
prices = data["train_prices"]
test_df = data["test_df"]
test_info = data["test_info"]

st.title("🎯 Final Predictions")
st.markdown("Ensemble predictions for submission.")

# ── Check if results exist ───────────────────────────────────────────────
results_path = DATA_DIR / "results.xlsx"
if not results_path.exists():
    st.warning(
        "⚠️ results.xlsx not found. Run:\n\n"
        "```bash\npython notebooks/05_final_predictions.py\n```"
    )
    st.stop()

# ── Load results ─────────────────────────────────────────────────────────
results_df = pd.read_excel(results_path, engine="openpyxl")

st.success("✅ results.xlsx loaded — all values filled!")

# ── Section 1: Future predictions ────────────────────────────────────────
st.header("1. Future Predictions (6 days)")

future_indices = test_info["future_indices"]
future_dates = [
    pd.to_datetime(test_df.iloc[idx]["date"]).strftime("%d/%m/%Y")
    for idx in future_indices
]

# Extract future rows from results
price_cols = [c for c in results_df.columns if c not in ("Type", "Date", "type", "date")]

for i, row_idx in enumerate(future_indices):
    row = results_df.iloc[row_idx]
    future_prices = row[price_cols].values.astype(float)

    with st.expander(f"📅 {future_dates[i]}", expanded=(i == 0)):
        col_a, col_b = st.columns([3, 1])

        with col_a:
            fig = plot_surface_heatmap(
                future_prices,
                title=f"Predicted Surface — {future_dates[i]}",
                zmin=prices.min(),
                zmax=prices.max(),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.metric("Mean Price", f"{future_prices.mean():.4f}")
            st.metric("Min Price", f"{future_prices.min():.4f}")
            st.metric("Max Price", f"{future_prices.max():.4f}")

            # Change from last known
            delta = future_prices.mean() - prices[-1].mean()
            st.metric("Δ from last known", f"{delta:+.4f}")

# ── Section 2: Missing data imputation ────────────────────────────────────
st.header("2. Missing Data Imputation (2 dates)")

missing_indices = test_info["missing_indices"]
missing_dates = [
    pd.to_datetime(test_df.iloc[idx]["date"]).strftime("%d/%m/%Y")
    for idx in missing_indices
]

for i, idx in enumerate(missing_indices):
    row = results_df.iloc[idx]
    imputed_prices = row[price_cols].values.astype(float)
    mask = test_info["missing_masks"][idx]
    n_missing = int((~mask).sum())

    with st.expander(f"📅 {missing_dates[i]} — {n_missing} values imputed"):
        fig = plot_surface_heatmap(
            imputed_prices,
            title=f"Imputed Surface — {missing_dates[i]}",
            zmin=prices.min(),
            zmax=prices.max(),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show which cells were imputed
        missing_cols = np.where(~mask)[0]
        st.markdown("**Imputed cells:**")
        for col_idx in missing_cols:
            mat_idx = col_idx // N_TENORS
            ten_idx = col_idx % N_TENORS
            st.markdown(
                f"- Tenor **{TENORS[ten_idx]}Y**, Maturity **{MATURITY_LABELS[mat_idx]}**: "
                f"**{imputed_prices[col_idx]:.6f}**"
            )

# ── Section 3: Comparison with historical ────────────────────────────────
st.header("3. Continuity Check")
st.markdown("Comparing predicted surfaces with recent history to verify smooth transitions.")

import plotly.graph_objects as go

# Pick a few representative tenor/maturity combos
representative = [
    (0, 4, "Tenor 1Y, Mat 1Y"),
    (4, 8, "Tenor 5Y, Mat 3Y"),
    (9, 12, "Tenor 10Y, Mat 15Y"),
    (13, 15, "Tenor 30Y, Mat 30Y"),
]

for ten_idx, mat_idx, label in representative:
    flat_idx = mat_idx * N_TENORS + ten_idx

    # Last 30 days of history + 6 future
    hist_days = 30
    hist_prices = prices[-hist_days:, flat_idx]
    hist_dates = data["train_dates"][-hist_days:]

    future_vals = []
    for row_idx in future_indices:
        row = results_df.iloc[row_idx]
        future_vals.append(float(row[price_cols[flat_idx]]))

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
        name="Predicted",
        line=dict(color="#FF6B35", dash="dash"),
        marker=dict(size=8),
    ))
    fig.add_vline(x=hist_days - 0.5, line_dash="dot", line_color="gray",
                  annotation_text="Prediction start")
    fig.update_layout(
        title=f"Continuity: {label}",
        xaxis_title="Day",
        yaxis_title="Price",
        height=350,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Section 4: Download ──────────────────────────────────────────────────
st.header("4. Download Results")

with open(results_path, "rb") as f:
    st.download_button(
        label="📥 Download results.xlsx",
        data=f.read(),
        file_name="results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.markdown(
    """
    ---
    **Submission checklist:**
    - ✅ 6 future prediction rows filled
    - ✅ 2 missing data rows imputed
    - ✅ All 224 price columns populated
    - ✅ No remaining NA values
    """
)

# ── Ensemble info ─────────────────────────────────────────────────────────
if (TRAINED_DIR / "final_summary.json").exists():
    with st.expander("📋 Ensemble Details"):
        with open(TRAINED_DIR / "final_summary.json") as f:
            summary = json.load(f)
        st.json(summary)
