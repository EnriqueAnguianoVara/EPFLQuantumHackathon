"""
📊 Market Explorer — Interactive EDA of the swaption dataset.
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_all, TENORS, MATURITIES, MATURITY_LABELS, N_TENORS, N_MATURITIES
from src.data.preprocessing import normalize, pca_reduce
from src.utils.surface import (
    flat_to_grid,
    plot_surface_heatmap,
    plot_surface_3d,
    plot_time_series,
    plot_pca_variance,
    plot_correlation_matrix,
    plot_pca_components_on_surface,
)

st.set_page_config(page_title="Market Explorer", page_icon="📊", layout="wide")

# ── Load data ────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_all(ROOT / "data")

data = get_data()
prices = data["train_prices"]
dates = data["train_dates"]
n_days = prices.shape[0]

st.title("📊 Market Explorer")
st.markdown("Interactive exploration of the swaption volatility surface dataset.")

# ═══════════════════════════════════════════════════════════════════════════
# Section 1: Surface heatmap with date slider
# ═══════════════════════════════════════════════════════════════════════════
st.header("1. Volatility Surface Over Time")

col_slider, col_view = st.columns([1, 3])

with col_slider:
    day_idx = st.slider(
        "Select trading day",
        min_value=0,
        max_value=n_days - 1,
        value=0,
        format="Day %d",
    )
    st.markdown(f"**Date:** {dates[day_idx]}")
    view_3d = st.checkbox("Show 3D view", value=False)

with col_view:
    surface_flat = prices[day_idx]
    title = f"Swaption Surface — {dates[day_idx]}"

    if view_3d:
        fig = plot_surface_3d(surface_flat, title=title)
    else:
        fig = plot_surface_heatmap(
            surface_flat,
            title=title,
            zmin=prices.min(),
            zmax=prices.max(),
        )
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# Section 2: Time series of selected tenor/maturity
# ═══════════════════════════════════════════════════════════════════════════
st.header("2. Time Series by Tenor & Maturity")

col_t, col_m = st.columns(2)
with col_t:
    selected_tenors = st.multiselect(
        "Select Tenors",
        options=TENORS,
        default=[1, 5, 10, 30],
    )
with col_m:
    selected_mat_idx = st.selectbox(
        "Select Maturity",
        options=list(range(N_MATURITIES)),
        format_func=lambda i: MATURITY_LABELS[i],
        index=4,  # default: 1Y
    )

if selected_tenors:
    col_indices = []
    col_names = []
    for tenor in selected_tenors:
        tenor_idx = TENORS.index(tenor)
        flat_idx = selected_mat_idx * N_TENORS + tenor_idx
        col_indices.append(flat_idx)
        col_names.append(f"Tenor {tenor}Y")

    fig_ts = plot_time_series(
        prices,
        dates,
        col_indices,
        col_names,
        title=f"Swaption Prices — Maturity {MATURITY_LABELS[selected_mat_idx]}",
    )
    st.plotly_chart(fig_ts, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# Section 3: Dataset statistics
# ═══════════════════════════════════════════════════════════════════════════
st.header("3. Dataset Statistics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Trading Days", f"{n_days}")
c2.metric("Price Columns", f"{prices.shape[1]}")
c3.metric("Min Price", f"{prices.min():.4f}")
c4.metric("Max Price", f"{prices.max():.4f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Mean Price", f"{prices.mean():.4f}")
c6.metric("Std Dev", f"{prices.std():.4f}")
c7.metric("Grid Shape", f"{N_TENORS} × {N_MATURITIES}")
c8.metric("Date Range", f"{dates[0]} → {dates[-1]}")

# Correlation matrix
with st.expander("📐 Correlation Matrix (subsampled every 10th column)"):
    fig_corr = plot_correlation_matrix(prices, step=10)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown(
        "Adjacent columns (same tenor region or maturity region) are highly "
        "correlated, confirming the surface is smooth and low-dimensional."
    )

# ═══════════════════════════════════════════════════════════════════════════
# Section 4: PCA Analysis
# ═══════════════════════════════════════════════════════════════════════════
st.header("4. PCA — Intrinsic Dimensionality")

st.markdown(
    """
    The 224 price columns are highly redundant. PCA reveals the **true dimensionality**
    of the swaption surface. In interest rate markets, ~3–5 factors (level, slope,
    curvature...) typically explain >99% of variance.
    """
)

@st.cache_data
def compute_pca(prices):
    prices_norm, scaler = normalize(prices)
    n_components = min(20, prices.shape[1])
    reduced, reducer = pca_reduce(prices_norm, n_components)
    return prices_norm, scaler, reduced, reducer

prices_norm, scaler, reduced, reducer = compute_pca(prices)

col_pca1, col_pca2 = st.columns([2, 1])

with col_pca1:
    fig_var = plot_pca_variance(reducer.cumulative_variance)
    st.plotly_chart(fig_var, use_container_width=True)

with col_pca2:
    cv = reducer.cumulative_variance
    for n in [1, 2, 3, 4, 5, 6, 10]:
        if n <= len(cv):
            st.markdown(f"**{n} components:** {cv[n-1]:.2%}")

    # Find n for 95% and 99%
    n95 = int(np.searchsorted(cv, 0.95)) + 1
    n99 = int(np.searchsorted(cv, 0.99)) + 1
    st.divider()
    st.markdown(f"🎯 **95% variance:** {n95} components")
    st.markdown(f"🎯 **99% variance:** {n99} components")

# PCA component visualisation
st.subheader("PCA Components as Surface Patterns")
st.markdown(
    "Each principal component represents a **mode of variation** of the surface. "
    "PC1 is typically the overall level, PC2 the slope, PC3 the curvature."
)

n_show = st.slider("Components to show", 1, 6, 4)
fig_pc = plot_pca_components_on_surface(reducer.components, n_show)
st.plotly_chart(fig_pc, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# Section 5: Test template overview
# ═══════════════════════════════════════════════════════════════════════════
st.header("5. Test Template Overview")

test_df = data["test_df"]
test_info = data["test_info"]

st.markdown(f"**Future prediction rows:** {len(test_info['future_indices'])}")
st.markdown(f"**Missing data rows:** {len(test_info['missing_indices'])}")

st.dataframe(
    test_df[["type", "date"]],
    use_container_width=True,
    hide_index=False,
)

if test_info["missing_masks"]:
    st.subheader("Missing values detail")
    price_cols = [c for c in test_df.columns if c not in ("type", "date")]
    for idx, mask in test_info["missing_masks"].items():
        n_missing = int((~mask).sum())
        missing_cols = [price_cols[j] for j in range(len(mask)) if not mask[j]]
        date = test_df.iloc[idx]["date"]
        st.markdown(f"**Row {idx}** (Date: {date}): **{n_missing}** missing values")
        for mc in missing_cols:
            st.markdown(f"  - `{mc}`")
