"""
Model comparison page.
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

from src.data.loader import load_all, TENORS, MATURITY_LABELS
from src.utils.surface import plot_surface_heatmap, flat_to_grid

st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")
TRAINED_DIR = ROOT / "trained_models"

data = load_all(ROOT / "data")
prices = data["train_prices"]

st.title("📈 Model Comparison")
st.markdown("Head-to-head evaluation of classical and quantum approaches.")


def _extract_r2(metrics: dict) -> float:
    if "R2" in metrics:
        return metrics["R2"]
    if "R²" in metrics:
        return metrics["R²"]
    if "RÂ²" in metrics:
        return metrics["RÂ²"]
    for k, v in metrics.items():
        if str(k).startswith("R"):
            return v
    return np.nan


results = {}

if (TRAINED_DIR / "baselines_summary.json").exists():
    with open(TRAINED_DIR / "baselines_summary.json") as f:
        baselines = json.load(f)
    for name, m in baselines.items():
        results[name.replace("_", " ").title()] = {
            "MAE": m.get("MAE", np.nan),
            "RMSE": m.get("RMSE", np.nan),
            "R2": _extract_r2(m),
        }

if (TRAINED_DIR / "qrc_ablation.json").exists():
    with open(TRAINED_DIR / "qrc_ablation.json") as f:
        ablation = json.load(f)
    valid = [r for r in ablation if "error" not in r]
    if valid:
        best_qrc = min(valid, key=lambda r: r["orig_MAE"])
        results["QRC"] = {
            "MAE": best_qrc.get("orig_MAE", np.nan),
            "RMSE": best_qrc.get("orig_RMSE", np.nan),
            "R2": best_qrc.get("orig_R2", np.nan),
        }

if (TRAINED_DIR / "quantum_extra_summary.json").exists():
    with open(TRAINED_DIR / "quantum_extra_summary.json") as f:
        extra_q = json.load(f)
    for name in ("QKGP", "QRLSTM"):
        item = extra_q.get(name, {})
        if item.get("status") == "ok":
            results[name] = {
                "MAE": item.get("val_MAE", np.nan),
                "RMSE": item.get("val_RMSE", np.nan),
                "R2": item.get("val_R2", np.nan),
            }

ae_comp = {}
if (TRAINED_DIR / "autoencoder_comparison.json").exists():
    with open(TRAINED_DIR / "autoencoder_comparison.json") as f:
        ae_comp = json.load(f)
    if "quantum" in ae_comp:
        results["QAE"] = {
            "MAE": ae_comp["quantum"].get("reconstruction_val_MAE", np.nan),
            "RMSE": ae_comp["quantum"].get("reconstruction_val_RMSE", np.nan),
            "R2": ae_comp["quantum"].get("reconstruction_val_R2", np.nan),
        }
    if "classical" in ae_comp:
        results["Classical AE"] = {
            "MAE": ae_comp["classical"].get("reconstruction_val_MAE", np.nan),
            "RMSE": ae_comp["classical"].get("reconstruction_val_RMSE", np.nan),
            "R2": ae_comp["classical"].get("reconstruction_val_R2", np.nan),
        }

if not results:
    st.warning("No model results found. Run the training notebooks first.")
    st.stop()

forecast_models = [
    "Ridge",
    "Naive Persistence",
    "Xgboost",
    "Lstm",
    "QRC",
    "QKGP",
    "QRLSTM",
]
ae_models = ["QAE", "Classical AE"]

df_results = pd.DataFrame(results).T
df_results.index.name = "Model"
for col in ["MAE", "RMSE", "R2"]:
    if col not in df_results.columns:
        df_results[col] = np.nan

df_forecast = df_results.loc[[m for m in forecast_models if m in df_results.index]].copy()
df_ae = df_results.loc[[m for m in ae_models if m in df_results.index]].copy()

st.header("1. Forecast Metrics")
st.caption("Only 1-step-ahead forecasting models are compared here.")
classical_forecast = [m for m in ["Ridge", "Naive Persistence", "Xgboost", "Lstm"] if m in df_forecast.index]
quantum_forecast = [m for m in ["QRC", "QKGP", "QRLSTM"] if m in df_forecast.index]

best_classical = min(classical_forecast, key=lambda m: df_forecast.loc[m, "MAE"]) if classical_forecast else None
best_quantum = min(quantum_forecast, key=lambda m: df_forecast.loc[m, "MAE"]) if quantum_forecast else None

df_forecast_show = df_forecast[["MAE", "RMSE", "R2"]].copy()
df_forecast_show.insert(
    0,
    "Segment",
    ["Classical" if m in classical_forecast else "Quantum" for m in df_forecast_show.index],
)
df_forecast_show.insert(
    1,
    "Best",
    [
        "Best Classical" if m == best_classical
        else ("Best Quantum" if m == best_quantum else "")
        for m in df_forecast_show.index
    ],
)


def _highlight_best_rows(row):
    if row.name == best_classical:
        return ["background-color: #2d5a27"] * len(row)
    if row.name == best_quantum:
        return ["background-color: #2b4c7e"] * len(row)
    return [""] * len(row)


st.dataframe(
    df_forecast_show.style.apply(_highlight_best_rows, axis=1).format({
        "MAE": "{:.6f}",
        "RMSE": "{:.6f}",
        "R2": "{:.6f}",
    }),
    use_container_width=True,
)

c1, c2 = st.columns(2)
if best_classical is not None:
    c1.metric(
        "Best Classical (MAE)",
        best_classical,
        f"MAE={df_forecast.loc[best_classical, 'MAE']:.6f}",
    )
if best_quantum is not None:
    c2.metric(
        "Best Quantum (MAE)",
        best_quantum,
        f"MAE={df_forecast.loc[best_quantum, 'MAE']:.6f}",
    )

import plotly.graph_objects as go

model_names = list(df_forecast.index)
mae_vals = [df_forecast.loc[m, "MAE"] for m in model_names]
r2_vals = [df_forecast.loc[m, "R2"] for m in model_names]
one_minus_r2_vals = [max(1e-8, 1.0 - float(v)) for v in r2_vals]

col1, col2 = st.columns(2)
with col1:
    fig_mae = go.Figure(go.Bar(
        x=model_names, y=mae_vals,
        marker_color=["#FF6B35" if "Q" not in m and "Quantum" not in m else "#7030A0" for m in model_names],
    ))
    fig_mae.update_layout(title="MAE by Model", yaxis_title="MAE", height=380)
    st.plotly_chart(fig_mae, use_container_width=True)

with col2:
    fig_r2 = go.Figure(go.Bar(
        x=model_names, y=r2_vals,
        marker_color=["#FF6B35" if "Q" not in m and "Quantum" not in m else "#7030A0" for m in model_names],
    ))
    fig_r2.update_layout(
        title="R2 by Model (zoomed)",
        yaxis_title="R2",
        yaxis=dict(range=[max(0.9, min(r2_vals) - 0.01), 1.001]),
        height=380,
    )
    st.plotly_chart(fig_r2, use_container_width=True)

fig_gap = go.Figure(go.Bar(
    x=model_names,
    y=one_minus_r2_vals,
    marker_color=["#FF6B35" if "Q" not in m and "Quantum" not in m else "#7030A0" for m in model_names],
))
fig_gap.update_layout(
    title="1 - R2 by Model (lower is better, log scale)",
    yaxis_title="1 - R2",
    yaxis_type="log",
    height=360,
)
st.plotly_chart(fig_gap, use_container_width=True)

st.header("2. Autoencoder Reconstruction Metrics")
st.caption("Separate task: surface reconstruction, not temporal forecasting.")
if not df_ae.empty:
    st.dataframe(
        df_ae[["MAE", "RMSE", "R2"]].style.format({
            "MAE": "{:.6f}",
            "RMSE": "{:.6f}",
            "R2": "{:.6f}",
        }).highlight_min(axis=0, subset=["MAE", "RMSE"], color="#2d5a27")
        .highlight_max(axis=0, subset=["R2"], color="#2d5a27"),
        use_container_width=True,
    )
else:
    st.info("Autoencoder comparison artifact not found.")

st.header("3. Future Prediction Surfaces")
future_files = {
    "Ridge": "ridge_future_prices.npy",
    "QRC": "qrc_future_prices.npy",
    "QAE": "qae_future_prices.npy",
    "QKGP": "qkgp_future_prices.npy",
    "QRLSTM": "qrlstm_future_prices.npy",
    "Naive": "naive_future_prices.npy",
}
future_preds = {}
for name, fname in future_files.items():
    p = TRAINED_DIR / fname
    if p.exists():
        future_preds[name] = np.load(p)

if future_preds:
    future_dates = ["24/12/2051", "26/12/2051", "27/12/2051", "29/12/2051", "30/12/2051", "01/01/2052"]
    day = st.selectbox("Select day", range(6), format_func=lambda i: future_dates[i])
    all_models = list(future_preds.keys())
    models_to_show = st.multiselect(
        "Models to display",
        all_models,
        default=all_models,
    )
    charts_per_row = st.radio(
        "Charts per row",
        [2, 3],
        horizontal=True,
        index=1,
    )

    surfaces = [("Last Known", prices[-1])]
    for name in models_to_show:
        surfaces.append((name, future_preds[name][day]))

    for row_start in range(0, len(surfaces), charts_per_row):
        row_items = surfaces[row_start : row_start + charts_per_row]
        cols = st.columns(charts_per_row)
        for i, (name, surface) in enumerate(row_items):
            with cols[i]:
                fig = plot_surface_heatmap(
                    surface,
                    title=name,
                    zmin=prices.min(),
                    zmax=prices.max(),
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"future_surface_{day}_{name}_{row_start+i}",
                )

    if len(future_preds) >= 2:
        all_preds = np.stack(list(future_preds.values()))
        std_across = all_preds[:, day, :].std(axis=0)
        std_grid = flat_to_grid(std_across)
        fig_std = go.Figure(go.Heatmap(
            z=std_grid,
            x=MATURITY_LABELS,
            y=[str(t) for t in TENORS],
            colorscale="Reds",
            colorbar=dict(title="Std Dev"),
        ))
        fig_std.update_layout(
            title=f"Prediction Disagreement - {future_dates[day]}",
            xaxis_title="Maturity",
            yaxis_title="Tenor",
            yaxis=dict(autorange="reversed"),
            height=380,
        )
        st.plotly_chart(fig_std, use_container_width=True)
else:
    st.info("No future prediction artifacts found.")

st.header("4. Auxiliary RW Validation (non-official)")
st.caption(
    "This section compares forecasts against synthetic RW future surfaces. "
    "It is an additional sanity check and not part of the official benchmark."
)

rw_summary_path = TRAINED_DIR / "rw_validation_summary.json"
if rw_summary_path.exists():
    with open(rw_summary_path, encoding="utf-8") as f:
        rw_summary = json.load(f)

    rw_results = rw_summary.get("results", {})
    if rw_results:
        df_rw = pd.DataFrame(rw_results).T
        for col in ["MAE", "RMSE", "R2", "MAE_ratio_vs_naive", "MAE_delta_vs_naive"]:
            if col not in df_rw.columns:
                df_rw[col] = np.nan

        rw_classical = [m for m in ["Ridge", "Naive Persistence", "XGBoost", "LSTM", "Xgboost", "Lstm"] if m in df_rw.index]
        rw_quantum = [m for m in ["QRC", "QKGP", "QRLSTM", "QAE"] if m in df_rw.index]
        rw_best_classical = min(rw_classical, key=lambda m: df_rw.loc[m, "MAE"]) if rw_classical else None
        rw_best_quantum = min(rw_quantum, key=lambda m: df_rw.loc[m, "MAE"]) if rw_quantum else None

        df_rw_show = df_rw.copy()
        df_rw_show.insert(
            0,
            "Segment",
            ["Classical" if m in rw_classical else ("Quantum" if m in rw_quantum else "Other") for m in df_rw_show.index],
        )
        df_rw_show.insert(
            1,
            "Best",
            [
                "Best Classical" if m == rw_best_classical
                else ("Best Quantum" if m == rw_best_quantum else "")
                for m in df_rw_show.index
            ],
        )

        def _highlight_rw_rows(row):
            if row.name == rw_best_classical:
                return ["background-color: #2d5a27"] * len(row)
            if row.name == rw_best_quantum:
                return ["background-color: #2b4c7e"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_rw_show[["Segment", "Best", "MAE", "RMSE", "R2", "MAE_ratio_vs_naive", "MAE_delta_vs_naive"]]
            .style.apply(_highlight_rw_rows, axis=1)
            .format({
                "MAE": "{:.6f}",
                "RMSE": "{:.6f}",
                "R2": "{:.6f}",
                "MAE_ratio_vs_naive": "{:.3f}",
                "MAE_delta_vs_naive": "{:+.6f}",
            }),
            use_container_width=True,
        )

        c1, c2, c3 = st.columns(3)
        if rw_best_classical is not None:
            c1.metric("Best Classical vs RW", rw_best_classical, f"MAE={df_rw.loc[rw_best_classical, 'MAE']:.6f}")
        if rw_best_quantum is not None:
            c2.metric("Best Quantum vs RW", rw_best_quantum, f"MAE={df_rw.loc[rw_best_quantum, 'MAE']:.6f}")
        c3.metric("RW Days x Points", f"{rw_summary.get('n_days', '?')} x {rw_summary.get('n_points', '?')}")
else:
    st.info(
        "No RW auxiliary summary found. Run:\n"
        "`python scripts/evaluate_vs_rw.py --rw-file \"C:\\Users\\jorge\\Downloads\\rw_6days.xlsx\"`"
    )
