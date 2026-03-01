# Setup and Verification Guide

This guide prepares a clean environment for running the repository on Windows and Linux.

## 1. Create and activate a virtual environment

### Windows (PowerShell)

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Linux / macOS (bash/zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2. Install dependencies

Install core dependencies:

```bash
pip install -r requirements.txt
```

Optional dependencies (only for extended/experimental modules):

```bash
pip install -r requirements-optional.txt
```

## 3. Run the full pipeline

From repository root:

```bash
python run_pipeline.py
```

Expected outputs:

- `trained_models/*.json`
- `trained_models/*.npy`
- `trained_models/*.pkl`
- `trained_models/*.pt`
- `data/results.xlsx`

## 4. Launch Streamlit

```bash
streamlit run app.py
```

If port 8501 is busy:

```bash
streamlit run app.py --server.port 8502
```

## 5. Run validation scripts

```bash
python tests/test_phase1.py
python tests/test_phase2.py
python tests/test_phase3.py
```

## 6. Common issues

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run commands from repository root. |
| `streamlit` command not found | Activate `.venv` first or run `python -m streamlit run app.py`. |
| `merlinquantum` installation fails | Use Python 3.10-3.12 and update `pip`. |
| Missing output artifacts in Streamlit | Run `python run_pipeline.py` first. |
| RW validation file not found | Provide a valid file path to `scripts/evaluate_vs_rw.py --rw-file ...`. |

## 7. Compatibility

- Windows: tested with PowerShell and Python virtual environments
- Linux/macOS: tested with standard `venv` and `bash/zsh`
- Recommended Python versions: `3.10`, `3.11`, `3.12`
