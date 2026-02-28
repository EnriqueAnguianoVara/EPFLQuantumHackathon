"""
Submission-ready Streamlit home page for judges.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
TRAINED_DIR = ROOT / "trained_models"


def _safe_load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _best_forecast_model():
    rows = []

    baselines = _safe_load_json(TRAINED_DIR / "baselines_summary.json") or {}
    for k, v in baselines.items():
        mae = v.get("MAE")
        if mae is not None:
            rows.append((k.replace("_", " ").title(), float(mae)))

    qrc_ablation = _safe_load_json(TRAINED_DIR / "qrc_ablation.json") or []
    valid = [r for r in qrc_ablation if "error" not in r and "orig_MAE" in r]
    if valid:
        rows.append(("QRC", float(min(valid, key=lambda r: r["orig_MAE"])["orig_MAE"])))

    extra_q = _safe_load_json(TRAINED_DIR / "quantum_extra_summary.json") or {}
    for name in ("QKGP", "QRLSTM"):
        item = extra_q.get(name, {})
        if item.get("status") == "ok" and "val_MAE" in item:
            rows.append((name, float(item["val_MAE"])))

    if not rows:
        return None, None
    best = min(rows, key=lambda x: x[1])
    return best[0], best[1]


st.set_page_config(
    page_title="Quantum Swaptions - Judge Presentation",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚛️ Quantum Swaptions - Judge Presentation")
st.caption("EPFL/Quandela challenge submission view.")

st.sidebar.markdown("## Judge Navigation")
st.sidebar.markdown("Use pages in order: 1 → 2 → 3 → 5 → 6")

col_a, col_b, col_c = st.columns(3)
col_a.metric("Dataset", "494 x 224")
col_b.metric("Forecast Horizon", "6 days")
col_c.metric("Missing Imputations", "2 dates")

best_name, best_mae = _best_forecast_model()
if best_name is not None:
    st.success(f"Current best validation MAE (official workflow): {best_name} ({best_mae:.6f})")

st.markdown("### What judges should review")
st.markdown(
    """
1. **Market Explorer**: data structure and volatility surface behavior.
2. **Classical Baselines**: Ridge/XGBoost/LSTM/Naive reference.
3. **Quantum Approaches**: QRC, QKGP, QRLSTM, QAE analysis.
4. **Comparison**: side-by-side metrics and auxiliary RW sanity check (non-official).
5. **Predictions**: final 6-day forecast surfaces and submission output.
"""
)

st.markdown("### Submission notes")
st.info(
    "Primary benchmark results are based on the official challenge workflow. "
    "RW-based analysis is shown only as auxiliary sanity check."
)

st.markdown(
    """
---
Built for presentation clarity: concise metrics, interpretable plots, and direct traceability to generated artifacts.
"""
)
