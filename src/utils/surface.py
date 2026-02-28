"""
Utilities for working with the swaption volatility surface.

Converts between the flat 224-column representation and the
structured 14 (tenors) × 16 (maturities) grid. Provides Plotly-based
visualisation helpers.
"""

from typing import Optional, List

import numpy as np

from src.data.loader import TENORS, MATURITIES, MATURITY_LABELS, N_TENORS, N_MATURITIES


def _import_plotly():
    """Lazy import of plotly — only needed for plotting, not for reshape utilities."""
    import plotly.graph_objects as go
    return go


# ---------------------------------------------------------------------------
# Reshape helpers
# ---------------------------------------------------------------------------
def flat_to_grid(flat: np.ndarray) -> np.ndarray:
    """
    Reshape a flat 224-vector into a (14 tenors, 16 maturities) grid.

    The original Excel column order is maturity-major:
      for each maturity: all 14 tenors
    So flat[0..13]  = maturity 0.083, tenors 1..30
       flat[14..27] = maturity 0.25,  tenors 1..30
       etc.

    Returns shape (N_TENORS=14, N_MATURITIES=16) where:
      grid[i, j] = price for TENORS[i], MATURITIES[j]
    """
    assert flat.shape[-1] == N_TENORS * N_MATURITIES, (
        f"Expected {N_TENORS * N_MATURITIES} values, got {flat.shape[-1]}"
    )
    # flat is ordered: [mat0_ten0, mat0_ten1, ..., mat0_ten13, mat1_ten0, ...]
    # Reshape to (16 maturities, 14 tenors) then transpose to (14 tenors, 16 maturities)
    grid = flat.reshape(N_MATURITIES, N_TENORS)  # (16, 14)
    return grid.T  # (14, 16)


def grid_to_flat(grid: np.ndarray) -> np.ndarray:
    """
    Reshape a (14, 16) grid back to a flat 224-vector.

    Inverse of flat_to_grid.
    """
    assert grid.shape == (N_TENORS, N_MATURITIES), (
        f"Expected ({N_TENORS}, {N_MATURITIES}), got {grid.shape}"
    )
    return grid.T.reshape(-1)  # (16,14) → flatten


def batch_flat_to_grid(batch: np.ndarray) -> np.ndarray:
    """Convert (N, 224) → (N, 14, 16)."""
    N = batch.shape[0]
    grids = np.zeros((N, N_TENORS, N_MATURITIES))
    for i in range(N):
        grids[i] = flat_to_grid(batch[i])
    return grids


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_surface_heatmap(
    flat_or_grid: np.ndarray,
    title: str = "Swaption Surface",
    colorscale: str = "Viridis",
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
) -> "go.Figure":
    """
    Plot a swaption surface as a heatmap.

    Parameters
    ----------
    flat_or_grid : (224,) flat vector or (14, 16) grid
    title : plot title
    colorscale : Plotly colorscale
    zmin, zmax : color range (if None, auto)
    """
    go = _import_plotly()

    if flat_or_grid.ndim == 1:
        grid = flat_to_grid(flat_or_grid)
    else:
        grid = flat_or_grid

    fig = go.Figure(
        data=go.Heatmap(
            z=grid,
            x=MATURITY_LABELS,
            y=[str(t) for t in TENORS],
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Price"),
            hovertemplate=(
                "Maturity: %{x}<br>"
                "Tenor: %{y}<br>"
                "Price: %{z:.4f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Maturity",
        yaxis_title="Tenor",
        yaxis=dict(autorange="reversed"),  # tenor 1 at top
        width=800,
        height=500,
    )
    return fig


def plot_surface_3d(
    flat_or_grid: np.ndarray,
    title: str = "Swaption Surface 3D",
    colorscale: str = "Viridis",
) -> "go.Figure":
    """Plot the surface as a 3D surface plot."""
    go = _import_plotly()

    if flat_or_grid.ndim == 1:
        grid = flat_to_grid(flat_or_grid)
    else:
        grid = flat_or_grid

    mat_vals = np.array(MATURITIES)
    ten_vals = np.array(TENORS)

    fig = go.Figure(
        data=go.Surface(
            z=grid,
            x=mat_vals,
            y=ten_vals,
            colorscale=colorscale,
            colorbar=dict(title="Price"),
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Maturity (years)",
            yaxis_title="Tenor (years)",
            zaxis_title="Price",
        ),
        width=800,
        height=600,
    )
    return fig


def plot_time_series(
    prices: np.ndarray,
    dates: list,
    columns: List[int],
    col_names: Optional[List[str]] = None,
    title: str = "Swaption Price Time Series",
) -> "go.Figure":
    """
    Plot time series for selected columns.

    Parameters
    ----------
    prices : (N, 224)
    dates : list of dates
    columns : list of column indices to plot
    col_names : optional human-readable names
    """
    go = _import_plotly()
    fig = go.Figure()
    for i, col_idx in enumerate(columns):
        name = col_names[i] if col_names else f"Col {col_idx}"
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=prices[:, col_idx],
                mode="lines",
                name=name,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        width=900,
        height=500,
        hovermode="x unified",
    )
    return fig


def plot_pca_variance(
    cumulative_variance: np.ndarray,
    title: str = "PCA — Cumulative Explained Variance",
) -> "go.Figure":
    """Plot cumulative explained variance from PCA."""
    go = _import_plotly()
    n = len(cumulative_variance)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(range(1, n + 1)),
            y=cumulative_variance,
            marker_color="#7030A0",
            text=[f"{v:.1%}" for v in cumulative_variance],
            textposition="outside",
        )
    )
    # Add 95% and 99% lines
    fig.add_hline(y=0.95, line_dash="dash", line_color="red",
                  annotation_text="95%", annotation_position="top right")
    fig.add_hline(y=0.99, line_dash="dash", line_color="orange",
                  annotation_text="99%", annotation_position="top right")
    fig.update_layout(
        title=title,
        xaxis_title="Number of Components",
        yaxis_title="Cumulative Explained Variance",
        yaxis=dict(range=[0, 1.05]),
        width=700,
        height=450,
    )
    return fig


def plot_correlation_matrix(
    prices: np.ndarray,
    title: str = "Correlation Matrix (subset)",
    step: int = 10,
) -> "go.Figure":
    """
    Plot correlation matrix of price columns (subsampled for readability).

    Parameters
    ----------
    prices : (N, 224)
    step : show every `step`-th column
    """
    go = _import_plotly()
    subset = prices[:, ::step]
    corr = np.corrcoef(subset.T)
    n = corr.shape[0]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
        )
    )
    fig.update_layout(
        title=title,
        width=600,
        height=600,
    )
    return fig


def plot_pca_components_on_surface(
    components: np.ndarray,
    n_show: int = 4,
) -> "go.Figure":
    """
    Visualize the first N PCA components as heatmaps on the tenor×maturity grid.

    Parameters
    ----------
    components : (n_components, 224) — PCA component vectors
    n_show : number of components to show
    """
    go = _import_plotly()
    from plotly.subplots import make_subplots

    n_show = min(n_show, components.shape[0])
    fig = make_subplots(
        rows=1,
        cols=n_show,
        subplot_titles=[f"PC{i+1}" for i in range(n_show)],
    )

    for i in range(n_show):
        grid = flat_to_grid(components[i])
        fig.add_trace(
            go.Heatmap(
                z=grid,
                x=MATURITY_LABELS,
                y=[str(t) for t in TENORS],
                colorscale="RdBu_r",
                showscale=(i == n_show - 1),
                hovertemplate="Maturity: %{x}<br>Tenor: %{y}<br>Loading: %{z:.4f}<extra></extra>",
            ),
            row=1,
            col=i + 1,
        )

    fig.update_layout(
        title="PCA Components on Swaption Surface",
        height=400,
        width=250 * n_show,
    )
    # Reverse y-axis on all subplots
    for i in range(n_show):
        fig.update_yaxes(autorange="reversed", row=1, col=i + 1)

    return fig
