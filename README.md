# EPFL Quantum Hackathon 2026 - INVIQTVS

Submission repository for the EPFL x Quandela quantum challenge.

## Scope

This project benchmarks classical and quantum approaches for swaption surface forecasting and reconstruction:

- Classical forecasting baselines: Ridge, XGBoost, LSTM, Naive persistence
- Quantum forecasting approaches: QRC, QKGP, QRLSTM
- Reconstruction approaches: Quantum autoencoder and classical autoencoder
- Judge-facing Streamlit app to inspect datasets, metrics, and final forecasts

## Repository Structure

```
.
|-- app.py                       # Streamlit home page
|-- pages/                       # Streamlit pages
|-- src/
|   |-- data/                    # Loaders and preprocessing
|   |-- models/                  # Classical and quantum models
|   |-- evaluation/              # Metrics and diagnostics
|   `-- utils/                   # Surface utilities and Excel writer
|-- notebooks/                   # Executable pipeline scripts (.py format)
|-- scripts/                     # Helper scripts (e.g. RW validation)
|-- trained_models/              # Generated artifacts
|-- data/                        # Challenge datasets + output workbook
|-- run_pipeline.py              # Full end-to-end pipeline runner
|-- requirements.txt             # Core dependencies
`-- requirements-optional.txt    # Optional/experimental dependencies
```

## Python Version

Recommended: Python `3.10`, `3.11`, or `3.12`.

Notes:
- `merlinquantum` requires Python `<3.13`.
- Python 3.11 is the safest choice for cross-platform setup.

## Quick Start

### Windows (PowerShell)

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS (bash/zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional packages (only for extra experiments):

```bash
pip install -r requirements-optional.txt
```

## Run the Pipeline

```bash
python run_pipeline.py
```

This executes:

1. `notebooks/02_classical_baselines.py`
2. `notebooks/03_quantum_reservoir.py`
3. `notebooks/04_quantum_autoencoder.py`
4. `notebooks/05_final_predictions.py`

Artifacts are stored in `trained_models/` and `data/results.xlsx`.

## Run Streamlit

```bash
streamlit run app.py
```

## Validation Scripts

```bash
python tests/test_phase1.py
python tests/test_phase2.py
python tests/test_phase3.py
```

## Reproducibility Notes

- The official benchmark is based on challenge data only.
- The RW validation (`scripts/evaluate_vs_rw.py`) is auxiliary and does not replace the official benchmark.
- Trained artifacts are saved as `.pkl`, `.pt`, `.npy`, and `.json` files under `trained_models/`.

## Archived Material

Legacy and experimental assets that are not part of the official execution flow
(`run_pipeline.py` + `streamlit run app.py`) were moved to `archive/`.
See `archive/README.md` for details.
