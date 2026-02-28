"""
🔧 Classical Baselines — Ridge, XGBoost, LSTM benchmark results.

Loads pre-trained models from trained_models/ and displays
metrics, predictions vs actuals, and training curves.
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
from src.utils.surface import flat_to_grid, plot_surface_heatmap

st.set_page_config(page_title="Classical Baselines", page_icon="🔧", layout="wide")

TRAINED_DIR = ROOT / "trained_models"

# ── Load data & preprocessing ────────────────────────────────────────────
@st.cache_data
def load_data():
    data = load_all(ROOT / "data")
    return data

@st.cache_resource
def load_preprocessing():
    with open(TRAINED_DIR / "scalers.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_models():
    models = {}
    if (TRAINED_DIR / "ridge_model.pkl").exists():
        with open(TRAINED_DIR / "ridge_model.pkl", "rb") as f:
            models["Ridge"] = pickle.load(f)
    if (TRAINED_DIR / "xgboost_model.pkl").exists():
        with open(TRAINED_DIR / "xgboost_model.pkl", "rb") as f:
            models["XGBoost"] = pickle.load(f)
    if (TRAINED_DIR / "baselines_summary.json").exists():
        with open(TRAINED_DIR / "baselines_summary.json", "r") as f:
            models["_summary"] = json.load(f)
    return models

# ── Check if models exist ────────────────────────────────────────────────
if not (TRAINED_DIR / "scalers.pkl").exists():
    st.title("🔧 Classical Baselines")
    st.warning(
        "⚠️ No trained models found. Run the training script first:\n\n"
        "```bash\npython notebooks/02_classical_baselines.py\n```"
    )
    st.stop()

data = load_data()
prep = load_preprocessing()
models = load_models()

prices = data["train_prices"]
dates = data["train_dates"]
scaler = prep["scaler"]
pca_reducer = prep["pca_reducer"]
config = prep["config"]

N_PCA = config["n_pca"]
WINDOW = config["window_size"]

# ── Prepare validation data ──────────────────────────────────────────────
@st.cache_data
def prepare_val():
    split_idx = int(len(prices) * (1 - config["val_ratio"]))
    val_raw = prices[split_idx:]
    val_dates = dates[split_idx:]

    all_norm, _ = normalize(prices, scaler)
    all_pca, _ = pca_reduce(all_norm, reducer=pca_reducer)

    val_norm, _ = normalize(val_raw, scaler)
    val_pca, _ = pca_reduce(val_norm, reducer=pca_reducer)

    X_val, Y_val = create_flat_windows(val_pca, window_size=WINDOW, horizon=1)

    return {
        "split_idx": split_idx,
        "val_raw": val_raw,
        "val_dates": val_dates,
        "val_pca": val_pca,
        "X_val": X_val,
        "Y_val": Y_val,
        "all_pca": all_pca,
    }

val_data = prepare_val()

# ══════════════════════════════════════════════════════════════════════════
st.title("🔧 Classical Baselines")
st.markdown(
    "Benchmark models operating in PCA space (6 components, 99.93% variance). "
    "All models use a 20-day lookback window to predict the next day's surface."
)

# ── Section 1: Summary metrics ───────────────────────────────────────────
st.header("1. Validation Metrics Comparison")

if "_summary" in models:
    summary = models["_summary"]

    metrics_df = pd.DataFrame(summary).T
    metrics_df.index.name = "Model"

    # Format nicely
    st.dataframe(
        metrics_df.style.format({
            "MAE": "{:.6f}",
            "RMSE": "{:.6f}",
            "MAPE (%)": "{:.2f}",
            "R²": "{:.6f}",
            "Max Error": "{:.6f}",
        }).highlight_min(axis=0, subset=["MAE", "RMSE", "Max Error"], color="#2d5a27")
        .highlight_max(axis=0, subset=["R²"], color="#2d5a27"),
        use_container_width=True,
    )

    # Bar chart
    import plotly.graph_objects as go

    fig = go.Figure()
    model_names = list(summary.keys())
    for metric in ["MAE", "RMSE"]:
        values = [summary[m][metric] for m in model_names]
        fig.add_trace(go.Bar(name=metric, x=model_names, y=values))

    fig.update_layout(
        title="MAE & RMSE by Model (original price space)",
        barmode="group",
        yaxis_title="Error",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Run `python notebooks/02_classical_baselines.py` to generate metrics.")

# ── Section 2: Predictions vs Actuals ────────────────────────────────────
st.header("2. Predictions vs Actuals (Validation Set)")

available_models = [k for k in models if not k.startswith("_")]
if available_models:
    selected_model = st.selectbox("Select model", available_models)
    model = models[selected_model]

    X_val = val_data["X_val"]
    Y_val = val_data["Y_val"]
    val_dates_for_plot = val_data["val_dates"][WINDOW:]

    Y_pred_pca = model.predict(X_val)

    # Convert both to original space
    Y_val_norm = pca_reducer.inverse_transform(Y_val)
    Y_pred_norm = pca_reducer.inverse_transform(Y_pred_pca)
    Y_val_orig = denormalize(Y_val_norm, scaler)
    Y_pred_orig = denormalize(Y_pred_norm, scaler)

    # Let user pick which column to plot
    col_options = list(range(0, 224, 14))  # One per maturity group
    from src.data.loader import TENORS, MATURITY_LABELS, N_TENORS

    col_sel_mat = st.selectbox(
        "Select maturity",
        range(len(MATURITY_LABELS)),
        format_func=lambda i: MATURITY_LABELS[i],
        index=4,
    )
    col_sel_ten = st.selectbox(
        "Select tenor",
        range(len(TENORS)),
        format_func=lambda i: f"{TENORS[i]}Y",
        index=0,
    )
    col_idx = col_sel_mat * N_TENORS + col_sel_ten

    import plotly.graph_objects as go

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=val_dates_for_plot,
        y=Y_val_orig[:, col_idx],
        mode="lines",
        name="Actual",
        line=dict(color="#7030A0"),
    ))
    fig_pred.add_trace(go.Scatter(
        x=val_dates_for_plot,
        y=Y_pred_orig[:, col_idx],
        mode="lines",
        name=f"{selected_model} Prediction",
        line=dict(color="#FF6B35", dash="dash"),
    ))
    fig_pred.update_layout(
        title=f"{selected_model} — Tenor {TENORS[col_sel_ten]}Y, Maturity {MATURITY_LABELS[col_sel_mat]}",
        xaxis_title="Date",
        yaxis_title="Price",
        height=450,
        hovermode="x unified",
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # Residuals
    residuals = Y_val_orig[:, col_idx] - Y_pred_orig[:, col_idx]
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (this series)", f"{np.mean(np.abs(residuals)):.6f}")
    c2.metric("Max residual", f"{np.max(np.abs(residuals)):.6f}")
    c3.metric("Std residual", f"{np.std(residuals):.6f}")

    # ── Section 3: Error heatmap ─────────────────────────────────────────
    st.header("3. Error by Surface Region")
    st.markdown("MAE at each (tenor, maturity) cell, averaged over the validation period.")

    err_grid = surface_error_grid(Y_val_orig, Y_pred_orig)
    fig_err = plot_surface_heatmap(
        err_grid,
        title=f"{selected_model} — MAE by Surface Cell",
        colorscale="Reds",
    )
    st.plotly_chart(fig_err, use_container_width=True)

else:
    st.info("No trained models found.")

# ── Section 4: Rolling future predictions ────────────────────────────────
st.header("4. Rolling Future Predictions (6 days)")

if (TRAINED_DIR / "ridge_future_prices.npy").exists():
    future_prices = np.load(TRAINED_DIR / "ridge_future_prices.npy")

    future_dates = ["24/12/2051", "26/12/2051", "27/12/2051",
                    "29/12/2051", "30/12/2051", "01/01/2052"]

    day_idx = st.selectbox("Predicted day", range(6),
                           format_func=lambda i: future_dates[i])

    fig_future = plot_surface_heatmap(
        future_prices[day_idx],
        title=f"Predicted Surface — {future_dates[day_idx]}",
        zmin=prices.min(),
        zmax=prices.max(),
    )
    st.plotly_chart(fig_future, use_container_width=True)

    # Compare with last known day
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Last known day (23/12/2051)**")
        fig_last = plot_surface_heatmap(
            prices[-1],
            title="Last Known",
            zmin=prices.min(),
            zmax=prices.max(),
        )
        st.plotly_chart(fig_last, use_container_width=True)
    with col_b:
        st.markdown(f"**Predicted ({future_dates[day_idx]})**")
        fig_pred_day = plot_surface_heatmap(
            future_prices[day_idx],
            title="Predicted",
            zmin=prices.min(),
            zmax=prices.max(),
        )
        st.plotly_chart(fig_pred_day, use_container_width=True)

else:
    st.info("Run `python notebooks/02_classical_baselines.py` to generate predictions.")

# ── Section 5: LSTM training curves ──────────────────────────────────────
st.header("5. Model Details")

with st.expander("🔍 Ridge Regression"):
    if "Ridge" in models:
        params = models["Ridge"].get_params()
        st.json(params)
    else:
        st.info("Ridge model not found.")

with st.expander("🔍 XGBoost"):
    if "XGBoost" in models:
        params = models["XGBoost"].get_params()
        st.json(params)
    else:
        st.info("XGBoost model not found.")

with st.expander("🔍 LSTM Training Curves"):
    if (TRAINED_DIR / "lstm_config.pkl").exists():
        with open(TRAINED_DIR / "lstm_config.pkl", "rb") as f:
            lstm_info = pickle.load(f)
        st.json(lstm_info)
    else:
        st.info(
            "LSTM info not found. The LSTM training curves will be shown here "
            "after running `python notebooks/02_classical_baselines.py`."
        )
