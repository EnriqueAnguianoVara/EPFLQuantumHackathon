"""
Quantum Approaches - unified page for all quantum models.
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
from src.data.preprocessing import normalize, denormalize
from src.utils.surface import plot_surface_heatmap

st.set_page_config(page_title="Quantum Approaches", page_icon="⚛️", layout="wide")
TRAINED_DIR = ROOT / "trained_models"

data = load_all(ROOT / "data")
prices = data["train_prices"]

st.title("⚛️ Quantum Approaches")
st.markdown(
    "Unified view of QRC, QKGP, QRLSTM and Quantum/Classical autoencoder results."
)

# -----------------------------------------------------------------------------
# 1. Quantum forecast models summary (QRC / QKGP / QRLSTM)
# -----------------------------------------------------------------------------
st.header("1. Forecast Quantum Models")

rows = []

if (TRAINED_DIR / "qrc_ablation.json").exists():
    with open(TRAINED_DIR / "qrc_ablation.json") as f:
        ablation = json.load(f)
    valid = [r for r in ablation if "error" not in r]
    if valid:
        best_qrc = min(valid, key=lambda r: r["orig_MAE"])
        rows.append({
            "Model": "QRC",
            "Val MAE": best_qrc.get("orig_MAE", np.nan),
            "Val RMSE": best_qrc.get("orig_RMSE", np.nan),
            "Val R2": best_qrc.get("orig_R2", np.nan),
            "Config": f"{best_qrc['n_modes']}m/{best_qrc['n_photons']}p/{best_qrc.get('encoding', 'last')}",
        })

if (TRAINED_DIR / "quantum_extra_summary.json").exists():
    with open(TRAINED_DIR / "quantum_extra_summary.json") as f:
        extra = json.load(f)
    for name in ("QKGP", "QRLSTM"):
        item = extra.get(name, {})
        if item.get("status") == "ok":
            rows.append({
                "Model": name,
                "Val MAE": item.get("val_MAE", np.nan),
                "Val RMSE": item.get("val_RMSE", np.nan),
                "Val R2": item.get("val_R2", np.nan),
                "Config": "default",
            })

if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("No quantum forecast metrics found yet. Run `python notebooks/03_quantum_reservoir.py`.")


# -----------------------------------------------------------------------------
# 2. Future surfaces from all available quantum forecasts
# -----------------------------------------------------------------------------
st.header("2. Future Surfaces")

future_map = {
    "QRC": TRAINED_DIR / "qrc_future_prices.npy",
    "QAE": TRAINED_DIR / "qae_future_prices.npy",
    "QKGP": TRAINED_DIR / "qkgp_future_prices.npy",
    "QRLSTM": TRAINED_DIR / "qrlstm_future_prices.npy",
}
future_available = {name: np.load(path) for name, path in future_map.items() if path.exists()}

if future_available:
    future_dates = ["24/12/2051", "26/12/2051", "27/12/2051", "29/12/2051", "30/12/2051", "01/01/2052"]
    model_name = st.selectbox("Quantum model", list(future_available.keys()))
    day = st.selectbox("Predicted day", range(6), format_func=lambda i: future_dates[i])

    c1, c2 = st.columns(2)
    with c1:
        fig_last = plot_surface_heatmap(prices[-1], title="Last Known", zmin=prices.min(), zmax=prices.max())
        st.plotly_chart(fig_last, use_container_width=True)
    with c2:
        fig_pred = plot_surface_heatmap(
            future_available[model_name][day],
            title=f"{model_name} Predicted ({future_dates[day]})",
            zmin=prices.min(),
            zmax=prices.max(),
        )
        st.plotly_chart(fig_pred, use_container_width=True)
else:
    st.info("No quantum future prediction artifacts found yet.")


# -----------------------------------------------------------------------------
# 3. QRC diagnostics
# -----------------------------------------------------------------------------
st.header("3. QRC Diagnostics")
if (TRAINED_DIR / "qrc_memory_capacity.npy").exists():
    import plotly.graph_objects as go

    mc = np.load(TRAINED_DIR / "qrc_memory_capacity.npy")
    fig_mc = go.Figure(go.Bar(x=list(range(1, len(mc) + 1)), y=mc, marker_color="#7030A0"))
    fig_mc.update_layout(
        title=f"Memory Capacity per Lag (Total: {mc.sum():.2f})",
        xaxis_title="Lag (days)",
        yaxis_title="R2 at lag",
        height=380,
    )
    st.plotly_chart(fig_mc, use_container_width=True)
else:
    st.info("QRC memory capacity artifact not found.")


# -----------------------------------------------------------------------------
# 4. Quantum vs Classical Autoencoder
# -----------------------------------------------------------------------------
st.header("4. Quantum vs Classical Autoencoder")

if not (TRAINED_DIR / "autoencoder_comparison.json").exists():
    st.info("Run `python notebooks/04_quantum_autoencoder.py` to generate autoencoder metrics.")
else:
    with open(TRAINED_DIR / "autoencoder_comparison.json") as f:
        comparison = json.load(f)

    q = comparison.get("quantum", {})
    c = comparison.get("classical", {})

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Quantum Autoencoder")
        st.metric("Reconstruction MAE", f"{q.get('reconstruction_val_MAE', np.nan):.6f}")
        st.metric("Reconstruction R2", f"{q.get('reconstruction_val_R2', np.nan):.6f}")
        st.metric("Trainable Params", f"{q.get('n_params', 'n/a')}")
    with col2:
        st.subheader("Classical Autoencoder")
        st.metric("Reconstruction MAE", f"{c.get('reconstruction_val_MAE', np.nan):.6f}")
        st.metric("Reconstruction R2", f"{c.get('reconstruction_val_R2', np.nan):.6f}")
        st.metric("Trainable Params", f"{c.get('n_params', 'n/a')}")

    st.caption("Classical autoencoder is shown here in the same section as the quantum autoencoder.")

    if (TRAINED_DIR / "q_train_losses.npy").exists() and (TRAINED_DIR / "c_train_losses.npy").exists():
        import plotly.graph_objects as go

        q_train = np.load(TRAINED_DIR / "q_train_losses.npy")
        q_val = np.load(TRAINED_DIR / "q_val_losses.npy")
        c_train = np.load(TRAINED_DIR / "c_train_losses.npy")
        c_val = np.load(TRAINED_DIR / "c_val_losses.npy")

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=q_train, name="Quantum Train", line=dict(color="#7030A0")))
        fig.add_trace(go.Scatter(y=q_val, name="Quantum Val", line=dict(color="#7030A0", dash="dash")))
        fig.add_trace(go.Scatter(y=c_train, name="Classical Train", line=dict(color="#FF6B35")))
        fig.add_trace(go.Scatter(y=c_val, name="Classical Val", line=dict(color="#FF6B35", dash="dash")))
        fig.update_layout(title="Autoencoder Loss Curves", xaxis_title="Epoch", yaxis_title="MSE Loss", height=420)
        st.plotly_chart(fig, use_container_width=True)

    try:
        import torch
        from src.models.quantum.autoencoder import QuantumAutoencoder

        with open(TRAINED_DIR / "scalers.pkl", "rb") as f:
            prep = pickle.load(f)
        scaler = prep["scaler"]
        all_norm, _ = normalize(prices, scaler)

        @st.cache_resource
        def load_q_autoencoder():
            model = QuantumAutoencoder(
                input_dim=224,
                pre_quantum_dim=6,
                n_modes=6,
                n_photons=3,
                latent_dim=comparison.get("latent_dim", 8),
            )
            model.load_state_dict(
                torch.load(str(TRAINED_DIR / "q_autoencoder_weights.pt"), map_location="cpu", weights_only=True)
            )
            model.eval()
            return model

        if (TRAINED_DIR / "q_autoencoder_weights.pt").exists():
            st.subheader("Quantum Autoencoder Reconstruction (sample day)")
            model = load_q_autoencoder()
            day = st.selectbox(
                "Select day",
                range(len(prices)),
                format_func=lambda i: data["train_dates"][i],
                index=0,
                key="qae_day",
            )

            with torch.no_grad():
                x = torch.tensor(all_norm[day:day + 1], dtype=torch.float32)
                recon_norm = model(x).numpy()
            recon = denormalize(recon_norm, scaler)[0]

            c1, c2 = st.columns(2)
            with c1:
                fig_o = plot_surface_heatmap(prices[day], title="Original", zmin=prices.min(), zmax=prices.max())
                st.plotly_chart(fig_o, use_container_width=True)
            with c2:
                fig_r = plot_surface_heatmap(recon, title="Reconstructed", zmin=prices.min(), zmax=prices.max())
                st.plotly_chart(fig_r, use_container_width=True)
    except Exception as exc:
        st.info(f"Live reconstruction unavailable: {exc}")
