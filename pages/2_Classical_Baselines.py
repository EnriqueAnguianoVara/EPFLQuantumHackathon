"""
Classical Baselines - Ridge, XGBoost, LSTM and Naive.
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

from src.data.loader import load_all, TENORS, MATURITY_LABELS, N_TENORS
from src.data.preprocessing import normalize, pca_reduce, denormalize, create_flat_windows
from src.evaluation.metrics import surface_error_grid
from src.models.classical.lstm import LSTMBaseline
from src.utils.surface import plot_surface_heatmap

st.set_page_config(page_title="Classical Baselines", layout="wide")

TRAINED_DIR = ROOT / "trained_models"


@st.cache_data
def load_data():
    return load_all(ROOT / "data")


@st.cache_resource
def load_preprocessing():
    with open(TRAINED_DIR / "scalers.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_models(n_pca: int, window_size: int):
    models = {}
    if (TRAINED_DIR / "ridge_model.pkl").exists():
        with open(TRAINED_DIR / "ridge_model.pkl", "rb") as f:
            models["Ridge"] = pickle.load(f)
    if (TRAINED_DIR / "xgboost_model.pkl").exists():
        with open(TRAINED_DIR / "xgboost_model.pkl", "rb") as f:
            models["XGBoost"] = pickle.load(f)
    if (TRAINED_DIR / "lstm_weights.pt").exists() and (TRAINED_DIR / "lstm_config.pkl").exists():
        with open(TRAINED_DIR / "lstm_config.pkl", "rb") as f:
            cfg = pickle.load(f)
        lstm = LSTMBaseline(
            n_pca=n_pca,
            window_size=window_size,
            hidden_dim=int(cfg.get("hidden_dim", 32)),
            num_layers=int(cfg.get("num_layers", 1)),
            dropout=float(cfg.get("dropout", 0.0)),
            device="cpu",
        )
        lstm.load(str(TRAINED_DIR / "lstm_weights.pt"))
        models["LSTM"] = lstm
    if (TRAINED_DIR / "baselines_summary.json").exists():
        with open(TRAINED_DIR / "baselines_summary.json", "r") as f:
            models["_summary"] = json.load(f)
    return models


if not (TRAINED_DIR / "scalers.pkl").exists():
    st.title("Classical Baselines")
    st.warning("No trained models found. Run `python notebooks/02_classical_baselines.py`.")
    st.stop()

data = load_data()
prep = load_preprocessing()

prices = data["train_prices"]
dates = data["train_dates"]
scaler = prep["scaler"]
pca_reducer = prep["pca_reducer"]
config = prep["config"]
N_PCA = int(config["n_pca"])
WINDOW = int(config["window_size"])

models = load_models(N_PCA, WINDOW)


@st.cache_data
def prepare_val():
    split_idx = int(len(prices) * (1 - config["val_ratio"]))
    val_raw = prices[split_idx:]
    val_dates = dates[split_idx:]

    val_norm, _ = normalize(val_raw, scaler)
    val_pca, _ = pca_reduce(val_norm, reducer=pca_reducer)
    X_val, Y_val = create_flat_windows(val_pca, window_size=WINDOW, horizon=1)
    return {"val_dates": val_dates, "X_val": X_val, "Y_val": Y_val}


val_data = prepare_val()

st.title("Classical Baselines")
st.markdown(
    "Benchmark models in PCA space (6 components) with 20-day lookback and 1-step forecast."
)

st.header("1. Validation Metrics Comparison")
if "_summary" in models:
    summary = models["_summary"]
    metrics_df = pd.DataFrame(summary).T
    metrics_df.index.name = "Model"
    r2_col = "R²" if "R²" in metrics_df.columns else ("RÂ²" if "RÂ²" in metrics_df.columns else None)
    if r2_col and r2_col != "R²":
        metrics_df["R²"] = metrics_df[r2_col]

    cols = [c for c in ["MAE", "RMSE", "MAPE (%)", "R²", "Max Error"] if c in metrics_df.columns]
    st.dataframe(
        metrics_df[cols].style.format({
            "MAE": "{:.6f}",
            "RMSE": "{:.6f}",
            "MAPE (%)": "{:.2f}",
            "R²": "{:.6f}",
            "Max Error": "{:.6f}",
        }).highlight_min(axis=0, subset=[c for c in ["MAE", "RMSE", "MAPE (%)", "Max Error"] if c in cols], color="#2d5a27")
        .highlight_max(axis=0, subset=[c for c in ["R²"] if c in cols], color="#2d5a27"),
        use_container_width=True,
    )
else:
    st.info("Run `python notebooks/02_classical_baselines.py` to generate metrics.")

st.header("2. Predictions vs Actuals (Validation Set)")
available_models = [k for k in models if not k.startswith("_")]
if "Naive" not in available_models:
    available_models.append("Naive")
if available_models:
    selected_model = st.selectbox("Select model", available_models)
    model = models.get(selected_model)

    X_val = val_data["X_val"]
    Y_val = val_data["Y_val"]
    val_dates_for_plot = val_data["val_dates"][WINDOW:]

    if selected_model == "Naive":
        # Persistence baseline in PCA space: copy the last step from each input window.
        Y_pred_pca = X_val[:, -N_PCA:]
    else:
        Y_pred_pca = model.predict(X_val)
    Y_val_orig = denormalize(pca_reducer.inverse_transform(Y_val), scaler)
    Y_pred_orig = denormalize(pca_reducer.inverse_transform(Y_pred_pca), scaler)

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
        title=f"{selected_model} - Tenor {TENORS[col_sel_ten]}Y, Maturity {MATURITY_LABELS[col_sel_mat]}",
        xaxis_title="Date",
        yaxis_title="Price",
        height=450,
        hovermode="x unified",
    )
    st.plotly_chart(
        fig_pred,
        use_container_width=True,
        key=f"val_series_{selected_model}_{col_sel_mat}_{col_sel_ten}",
    )

    residuals = Y_val_orig[:, col_idx] - Y_pred_orig[:, col_idx]
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (this series)", f"{np.mean(np.abs(residuals)):.6f}")
    c2.metric("Max residual", f"{np.max(np.abs(residuals)):.6f}")
    c3.metric("Std residual", f"{np.std(residuals):.6f}")

    st.header("3. Error by Surface Region")
    err_grid = surface_error_grid(Y_val_orig, Y_pred_orig)
    fig_err = plot_surface_heatmap(
        err_grid,
        title=f"{selected_model} - MAE by Surface Cell",
        colorscale="Reds",
    )
    st.plotly_chart(fig_err, use_container_width=True, key=f"val_error_grid_{selected_model}")
else:
    st.info("No trained classical models found.")

st.header("4. Rolling Future Predictions (Ridge Direct 6-step)")
if (TRAINED_DIR / "ridge_future_prices.npy").exists():
    future_prices = np.load(TRAINED_DIR / "ridge_future_prices.npy")
    future_dates = ["24/12/2051", "26/12/2051", "27/12/2051", "29/12/2051", "30/12/2051", "01/01/2052"]
    day_idx = st.selectbox("Predicted day", range(6), format_func=lambda i: future_dates[i], key="ridge_future_day")

    col_a, col_b = st.columns(2)
    with col_a:
        fig_last = plot_surface_heatmap(prices[-1], title="Last Known", zmin=prices.min(), zmax=prices.max())
        st.plotly_chart(fig_last, use_container_width=True, key=f"ridge_last_{day_idx}")
    with col_b:
        fig_pred_day = plot_surface_heatmap(
            future_prices[day_idx],
            title=f"Ridge Predicted ({future_dates[day_idx]})",
            zmin=prices.min(),
            zmax=prices.max(),
        )
        st.plotly_chart(fig_pred_day, use_container_width=True, key=f"ridge_pred_{day_idx}")
else:
    st.info("Run `python notebooks/02_classical_baselines.py` to generate ridge forecasts.")

st.header("5. Naive Persistence Baseline")
if "_summary" in models and "naive_persistence" in models["_summary"]:
    naive = models["_summary"]["naive_persistence"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Val MAE", f"{naive.get('MAE', np.nan):.6f}")
    c2.metric("Val RMSE", f"{naive.get('RMSE', np.nan):.6f}")
    r2_val = naive.get("R²", naive.get("RÂ²", np.nan))
    c3.metric("Val R2", f"{r2_val:.6f}")

if (TRAINED_DIR / "naive_future_prices.npy").exists():
    naive_future = np.load(TRAINED_DIR / "naive_future_prices.npy")
    future_dates = ["24/12/2051", "26/12/2051", "27/12/2051", "29/12/2051", "30/12/2051", "01/01/2052"]
    day_idx = st.selectbox("Naive predicted day", range(6), format_func=lambda i: future_dates[i], key="naive_future_day")

    col_a, col_b = st.columns(2)
    with col_a:
        fig_last = plot_surface_heatmap(prices[-1], title="Last Known", zmin=prices.min(), zmax=prices.max())
        st.plotly_chart(fig_last, use_container_width=True, key=f"naive_last_{day_idx}")
    with col_b:
        fig_naive = plot_surface_heatmap(
            naive_future[day_idx],
            title=f"Naive Predicted ({future_dates[day_idx]})",
            zmin=prices.min(),
            zmax=prices.max(),
        )
        st.plotly_chart(fig_naive, use_container_width=True, key=f"naive_pred_{day_idx}")
else:
    st.info("Naive forecast artifact not found (`naive_future_prices.npy`).")
