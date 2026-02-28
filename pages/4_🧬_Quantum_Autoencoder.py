"""
🧬 Quantum Autoencoder — Surface compression and latent space analysis.
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

st.set_page_config(page_title="Quantum Autoencoder", page_icon="🧬", layout="wide")
TRAINED_DIR = ROOT / "trained_models"

# ── Check if results exist ───────────────────────────────────────────────
if not (TRAINED_DIR / "autoencoder_comparison.json").exists():
    st.title("🧬 Quantum Autoencoder")
    st.warning(
        "⚠️ No autoencoder results found. Run:\n\n"
        "```bash\npython notebooks/04_quantum_autoencoder.py\n```"
    )
    st.stop()

data = load_all(ROOT / "data")
prices = data["train_prices"]

with open(TRAINED_DIR / "scalers.pkl", "rb") as f:
    prep = pickle.load(f)
scaler = prep["scaler"]

all_norm, _ = normalize(prices, scaler)

st.title("🧬 Quantum Autoencoder")
st.markdown(
    "A **quantum bottleneck** compresses the 224-dimensional swaption surface "
    "into a compact latent space. The photonic circuit operates in a Hilbert "
    "space exponentially larger than its input, enabling richer representations."
)

# ── Section 1: Architecture ──────────────────────────────────────────────
st.header("1. Architecture")

st.markdown(
    """
    ```
    Surface (224) → Encoder (224→64→16→6) → QuantumLayer (6 modes, 3 photons)
                                              → LexGrouping → Latent z (8 dims)
                                              → Decoder (8→16→64→224) → Reconstructed (224)
    ```

    The quantum bottleneck maps 6 classical features into a Fock space of ~20-56
    dimensions via photon interference, then collapses back to 8 latent features
    via LexGrouping. This non-linear transformation cannot be replicated by a
    classical layer of the same width.
    """
)

# ── Section 2: Quantum vs Classical comparison ───────────────────────────
st.header("2. Quantum vs Classical Bottleneck")

with open(TRAINED_DIR / "autoencoder_comparison.json") as f:
    comparison = json.load(f)

col1, col2 = st.columns(2)

with col1:
    st.subheader("⚛️ Quantum")
    q = comparison["quantum"]
    st.metric("Reconstruction MAE", f"{q['reconstruction_val_MAE']:.6f}")
    st.metric("Reconstruction R²", f"{q['reconstruction_val_R2']:.6f}")
    st.metric("Trainable Params", f"{q['n_params']}")
    st.metric("Epochs Trained", f"{q['epochs_trained']}")

with col2:
    st.subheader("📐 Classical")
    c = comparison["classical"]
    st.metric("Reconstruction MAE", f"{c['reconstruction_val_MAE']:.6f}")
    st.metric("Reconstruction R²", f"{c['reconstruction_val_R2']:.6f}")
    st.metric("Trainable Params", f"{c['n_params']}")
    st.metric("Epochs Trained", f"{c['epochs_trained']}")

# Determine winner
if q['reconstruction_val_MAE'] < c['reconstruction_val_MAE']:
    advantage = (c['reconstruction_val_MAE'] - q['reconstruction_val_MAE']) / c['reconstruction_val_MAE'] * 100
    st.success(f"✅ Quantum bottleneck reduces MAE by **{advantage:.1f}%** vs classical")
else:
    st.info("Classical bottleneck achieves lower MAE in this configuration.")

# Training curves
st.subheader("Training Curves")

if (TRAINED_DIR / "q_train_losses.npy").exists():
    import plotly.graph_objects as go

    q_train = np.load(TRAINED_DIR / "q_train_losses.npy")
    q_val = np.load(TRAINED_DIR / "q_val_losses.npy")
    c_train = np.load(TRAINED_DIR / "c_train_losses.npy")
    c_val = np.load(TRAINED_DIR / "c_val_losses.npy")

    fig_curves = go.Figure()
    fig_curves.add_trace(go.Scatter(y=q_train, name="Quantum Train", line=dict(color="#7030A0")))
    fig_curves.add_trace(go.Scatter(y=q_val, name="Quantum Val", line=dict(color="#7030A0", dash="dash")))
    fig_curves.add_trace(go.Scatter(y=c_train, name="Classical Train", line=dict(color="#FF6B35")))
    fig_curves.add_trace(go.Scatter(y=c_val, name="Classical Val", line=dict(color="#FF6B35", dash="dash")))
    fig_curves.update_layout(
        title="Autoencoder Loss Curves",
        xaxis_title="Epoch",
        yaxis_title="MSE Loss",
        height=450,
    )
    st.plotly_chart(fig_curves, use_container_width=True)

# ── Section 3: Reconstruction visualization ──────────────────────────────
st.header("3. Surface Reconstruction")

try:
    import torch
    from src.models.quantum.autoencoder import QuantumAutoencoder

    @st.cache_resource
    def load_q_autoencoder():
        model = QuantumAutoencoder(
            input_dim=224, pre_quantum_dim=6,
            n_modes=6, n_photons=3,
            latent_dim=comparison.get("latent_dim", 8),
        )
        model.load_state_dict(
            torch.load(str(TRAINED_DIR / "q_autoencoder_weights.pt"),
                       map_location="cpu", weights_only=True)
        )
        model.eval()
        return model

    model = load_q_autoencoder()

    day = st.selectbox("Select day", range(len(prices)),
                       format_func=lambda i: data["train_dates"][i],
                       index=0)

    surface_norm = torch.tensor(all_norm[day:day+1], dtype=torch.float32)
    with torch.no_grad():
        recon_norm = model(surface_norm).numpy()

    original = prices[day]
    recon = denormalize(recon_norm, scaler)[0]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Original**")
        fig_orig = plot_surface_heatmap(original, title="Original",
                                         zmin=prices.min(), zmax=prices.max())
        st.plotly_chart(fig_orig, use_container_width=True)
    with col_b:
        st.markdown("**Reconstructed**")
        fig_recon = plot_surface_heatmap(recon, title="Reconstructed",
                                          zmin=prices.min(), zmax=prices.max())
        st.plotly_chart(fig_recon, use_container_width=True)

    recon_err = np.abs(original - recon).mean()
    st.metric("Reconstruction MAE (this day)", f"{recon_err:.6f}")

except Exception as e:
    st.info(f"Cannot load model for live reconstruction: {e}")

# ── Section 4: Latent space visualization ────────────────────────────────
st.header("4. Latent Space Exploration")

if (TRAINED_DIR / "latent_vectors.npy").exists():
    Z = np.load(TRAINED_DIR / "latent_vectors.npy")

    # PCA of latent space for 2D visualization
    from sklearn.decomposition import PCA
    pca_2d = PCA(n_components=2)
    Z_2d = pca_2d.fit_transform(Z)

    import plotly.graph_objects as go

    # Color by time (day index)
    fig_latent = go.Figure(
        data=go.Scatter(
            x=Z_2d[:, 0],
            y=Z_2d[:, 1],
            mode="markers",
            marker=dict(
                size=5,
                color=list(range(len(Z))),
                colorscale="Viridis",
                colorbar=dict(title="Day Index"),
            ),
            text=[data["train_dates"][i] for i in range(len(Z))],
            hovertemplate="Date: %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
        )
    )
    fig_latent.update_layout(
        title=f"Quantum Latent Space (PCA of {Z.shape[1]}D → 2D)",
        xaxis_title="Latent PC1",
        yaxis_title="Latent PC2",
        height=500,
    )
    st.plotly_chart(fig_latent, use_container_width=True)

    st.markdown(
        "Each point represents one trading day projected into the quantum latent space. "
        "The temporal gradient (early→late days shown by color) reveals how the market "
        "structure evolves over time. Clusters indicate similar market regimes."
    )

    # Latent time series
    st.subheader("Latent Components Over Time")
    n_show = st.slider("Components to show", 1, Z.shape[1], min(4, Z.shape[1]))

    fig_z_ts = go.Figure()
    for i in range(n_show):
        fig_z_ts.add_trace(go.Scatter(
            x=data["train_dates"],
            y=Z[:, i],
            name=f"z_{i+1}",
            mode="lines",
        ))
    fig_z_ts.update_layout(
        title="Latent Dimensions Over Time",
        xaxis_title="Date",
        yaxis_title="Value",
        height=400,
        hovermode="x unified",
    )
    st.plotly_chart(fig_z_ts, use_container_width=True)

# ── Section 5: Future predictions ────────────────────────────────────────
st.header("5. Future Predictions")

if (TRAINED_DIR / "qae_future_prices.npy").exists():
    future = np.load(TRAINED_DIR / "qae_future_prices.npy")
    future_dates = ["24/12/2051", "26/12/2051", "27/12/2051",
                    "29/12/2051", "30/12/2051", "01/01/2052"]

    day_sel = st.selectbox("Predicted day", range(6),
                           format_func=lambda i: future_dates[i],
                           key="qae_day")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Last known (23/12/2051)**")
        fig1 = plot_surface_heatmap(prices[-1], title="Last Known",
                                     zmin=prices.min(), zmax=prices.max())
        st.plotly_chart(fig1, use_container_width=True)
    with col_b:
        st.markdown(f"**QAE Predicted ({future_dates[day_sel]})**")
        fig2 = plot_surface_heatmap(future[day_sel], title="QAE Predicted",
                                     zmin=prices.min(), zmax=prices.max())
        st.plotly_chart(fig2, use_container_width=True)
