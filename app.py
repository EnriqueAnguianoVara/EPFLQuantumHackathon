"""
Quantum Swaptions — Home Page

Streamlit entry point for the Quandela EPFL Hackathon 2026 challenge.
"""

import streamlit as st

st.set_page_config(
    page_title="Quantum Swaptions",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚛️ Quantum Swaptions")
st.sidebar.markdown("**Quandela × MILA × AMF**")
st.sidebar.markdown("EPFL Hackathon 2026")
st.sidebar.divider()
st.sidebar.markdown("Navigate using the pages above ☝️")

# ── Main content ─────────────────────────────────────────────────────────
st.title("⚛️ Quantum Swaptions Pricing")
st.markdown("### Predicting Interest Rate Derivatives with Photonic Quantum Computing")

st.divider()

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown(
        """
        #### The Challenge

        Swaptions — options on interest rate swaps — are central to risk management
        and derivatives trading. Their prices form a **volatility surface** that evolves
        daily across 14 tenors and 16 maturities (224 price points per day).

        We tackle two tasks:

        **1. Predict future swaption prices** (6 trading days ahead)

        **2. Impute missing data** in the historical surface

        #### Our Approach

        We combine **photonic quantum circuits** (via Quandela's MerLin & Perceval)
        with classical time-series methods in a hybrid architecture:

        - **Quantum Reservoir Computing** — fixed photonic circuits as non-linear feature extractors
        - **Quantum Autoencoder** — a quantum bottleneck learns a compressed representation of the volatility surface
        - **Classical Baselines** — Ridge, XGBoost, LSTM for rigorous comparison
        """
    )

with col2:
    st.markdown(
        """
        #### Architecture Overview

        ```
        ┌─────────────────────┐
        │  Swaption Surface   │
        │  494 days × 224 pts │
        └────────┬────────────┘
                 │
            ┌────▼────┐
            │ Encoder │ (classical)
            │ 224 → 6 │
            └────┬────┘
                 │
          ┌──────▼──────┐
          │  Quantum    │
          │  Bottleneck │ (MerLin)
          │  6 modes    │
          │  3 photons  │
          └──────┬──────┘
                 │
            ┌────▼────┐
            │ Decoder │ (classical)
            │ → 224   │
            └────┬────┘
                 │
          ┌──────▼──────┐
          │  Temporal   │
          │  Predictor  │
          └─────────────┘
        ```
        """
    )

st.divider()

# Key numbers
c1, c2, c3, c4 = st.columns(4)
c1.metric("Training Days", "494")
c2.metric("Surface Points", "224")
c3.metric("Future Predictions", "6 days")
c4.metric("Missing Imputations", "2 days")

st.divider()

st.markdown(
    """
    #### 📍 Navigation

    Use the sidebar to explore each section:

    | Page | Description |
    |------|-------------|
    | 📊 **Market Explorer** | Interactive visualisation of the swaption surface and dataset |
    | 🔧 **Classical Baselines** | Ridge, XGBoost, LSTM benchmark results |
    | ⚛️ **Quantum Reservoir** | Photonic reservoir computing with MerLin |
    | 🧬 **Quantum Autoencoder** | Quantum bottleneck autoencoder for surface compression |
    | 📈 **Comparison** | Head-to-head quantum vs classical evaluation |
    | 🎯 **Predictions** | Final results and Excel download |

    ---
    *Built with [Quandela MerLin](https://www.quandela.com/) & [Perceval](https://perceval.quandela.net/)*
    """
)
