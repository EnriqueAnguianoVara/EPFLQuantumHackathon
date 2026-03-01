"""Streamlit home page for judge-facing project review."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
LOGO_PATH = ROOT / "assets" / "Logo_INVIQTVS.jpeg"

st.set_page_config(
    page_title="Quantum Swaptions - INVIQTVS",
    layout="wide",
    initial_sidebar_state="expanded",
)

if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), use_container_width=True)

top_left, top_mid, top_right = st.columns([1, 2, 1])
with top_mid:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)

st.title("Quantum Swaptions - Judge Presentation")
st.caption("Team INVIQTVS | EPFL/Quandela challenge submission view.")

st.sidebar.markdown("## Judge Navigation")
st.sidebar.markdown("Use pages in order: 1 -> 2 -> 3 -> 5 -> 6")

col_a, col_b, col_c = st.columns(3)
col_a.metric("Dataset", "494 x 224")
col_b.metric("Forecast Horizon", "6 days")
col_c.metric("Missing Imputations", "2 dates")

st.markdown("### What judges should review")
st.markdown(
    """
1. **Market Explorer**: data structure and volatility surface behavior.
2. **Classical Baselines**: Ridge/XGBoost/LSTM/Naive reference.
3. **Quantum Approaches**: QRC, QKGP, QRLSTM, QAE analysis.
4. **Comparison**: side-by-side metrics and auxiliary RW sanity check.
5. **Predictions**: final 6-day forecast surfaces and submission output.
"""
)

st.markdown("### Team INVIQTVS")
st.markdown(
    """
- **Enrique Anguiano Vara** - [enriqueangianov@gmail.com](mailto:enriqueangianov@gmail.com) (BSc student, UAM)
- **Jorge Benito Diaz** - [jorgebenito.diaz22@gmail.com](mailto:jorgebenito.diaz22@gmail.com) (MSc student, UAM; First Stage Researcher, UPM)
- **Manuel Esparcia Cantos** - [manuelesparciacantos@gmail.com](mailto:manuelesparciacantos@gmail.com) (MSc student, UAM)
- **Ignacio Lopez Leis** - [nacholopezleis@gmail.com](mailto:nacholopezleis@gmail.com) (MSc student, UAM)
"""
)

st.markdown(
    """
---
Built for presentation clarity: concise metrics, interpretable plots, and direct traceability to generated artifacts.
"""
)
