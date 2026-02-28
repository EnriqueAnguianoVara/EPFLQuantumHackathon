"""
⚛️ Quantum Reservoir — Photonic reservoir computing results.
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
from src.utils.surface import plot_surface_heatmap

st.set_page_config(page_title="Quantum Reservoir", page_icon="⚛️", layout="wide")
TRAINED_DIR = ROOT / "trained_models"

# ── Check if models exist ────────────────────────────────────────────────
if not (TRAINED_DIR / "qrc_ablation.json").exists():
    st.title("⚛️ Quantum Reservoir Computing")
    st.warning(
        "⚠️ No QRC results found. Run the training script first:\n\n"
        "```bash\npython notebooks/03_quantum_reservoir.py\n```"
    )
    st.stop()

data = load_all(ROOT / "data")
prices = data["train_prices"]

st.title("⚛️ Quantum Reservoir Computing")
st.markdown(
    "A **fixed** photonic circuit acts as a non-linear reservoir. "
    "Fock-space probability distributions serve as features for a classical Ridge readout. "
    "No quantum parameters are trained — only the classical readout layer."
)

# ── Section 1: Architecture ──────────────────────────────────────────────
st.header("1. Architecture")

st.markdown(
    """
    ```
    PCA window (20 × 6) → last step (6,) → [Fixed Photonic Circuit] → Fock probs (K dims) → Ridge → prediction (6,)
    ```

    The photonic circuit contains random but fixed interferometers. The input PCA
    components are encoded as phase shifts (angle encoding). The circuit propagates
    photons through beam splitters and phase shifters, producing a probability
    distribution over all possible output Fock states. This distribution is
    the reservoir's non-linear feature map.
    """
)

# ── Section 2: Ablation study ────────────────────────────────────────────
st.header("2. Ablation Study — Modes × Photons")

with open(TRAINED_DIR / "qrc_ablation.json") as f:
    ablation = json.load(f)

# Filter valid results
valid = [r for r in ablation if "error" not in r]

if valid:
    df_abl = pd.DataFrame(valid)
    display_cols = ["n_modes", "n_photons", "q_output_size", "orig_MAE", "orig_R2", "time_seconds"]
    df_show = df_abl[display_cols].rename(columns={
        "n_modes": "Modes",
        "n_photons": "Photons",
        "q_output_size": "Fock Dim",
        "orig_MAE": "MAE",
        "orig_R2": "R²",
        "time_seconds": "Time (s)",
    })

    st.dataframe(
        df_show.style.format({
            "MAE": "{:.6f}",
            "R²": "{:.4f}",
            "Time (s)": "{:.1f}",
        }).highlight_min(subset=["MAE"], color="#2d5a27")
        .highlight_max(subset=["R²"], color="#2d5a27"),
        use_container_width=True,
    )

    # Bar chart
    import plotly.graph_objects as go

    labels = [f"{r['n_modes']}m/{r['n_photons']}p" for r in valid]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=[r["orig_MAE"] for r in valid],
        name="MAE", marker_color="#7030A0",
    ))
    fig.update_layout(title="MAE by Configuration", yaxis_title="MAE", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"**Best configuration:** {valid[np.argmin([r['orig_MAE'] for r in valid])]['n_modes']} modes, "
        f"{valid[np.argmin([r['orig_MAE'] for r in valid])]['n_photons']} photons "
        f"(Fock dim = {valid[np.argmin([r['orig_MAE'] for r in valid])]['q_output_size']})"
    )

# ── Section 3: Memory capacity ───────────────────────────────────────────
st.header("3. Reservoir Memory Capacity")

if (TRAINED_DIR / "qrc_memory_capacity.npy").exists():
    mc = np.load(TRAINED_DIR / "qrc_memory_capacity.npy")

    import plotly.graph_objects as go
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Bar(
        x=list(range(1, len(mc) + 1)),
        y=mc,
        marker_color="#7030A0",
    ))
    fig_mc.update_layout(
        title=f"Memory Capacity per Lag (Total: {mc.sum():.2f})",
        xaxis_title="Lag (days)",
        yaxis_title="R² at lag",
        height=400,
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    st.markdown(
        f"The reservoir retains information about the past **{int((mc > 0.1).sum())}** days "
        f"with R² > 0.1. Total memory capacity: **{mc.sum():.2f}**."
    )

# ── Section 4: Future predictions ────────────────────────────────────────
st.header("4. Rolling Future Predictions")

if (TRAINED_DIR / "qrc_future_prices.npy").exists():
    future = np.load(TRAINED_DIR / "qrc_future_prices.npy")
    future_dates = ["24/12/2051", "26/12/2051", "27/12/2051",
                    "29/12/2051", "30/12/2051", "01/01/2052"]

    day = st.selectbox("Predicted day", range(6),
                       format_func=lambda i: future_dates[i])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Last known (23/12/2051)**")
        fig1 = plot_surface_heatmap(prices[-1], title="Last Known",
                                     zmin=prices.min(), zmax=prices.max())
        st.plotly_chart(fig1, use_container_width=True)
    with col_b:
        st.markdown(f"**QRC Predicted ({future_dates[day]})**")
        fig2 = plot_surface_heatmap(future[day], title="QRC Predicted",
                                     zmin=prices.min(), zmax=prices.max())
        st.plotly_chart(fig2, use_container_width=True)
