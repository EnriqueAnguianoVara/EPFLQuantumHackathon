# ⚛️ Quantum Swaptions Pricing

**Quandela × MILA × AMF — EPFL Hackathon 2026**

Predicting interest rate swaption prices using hybrid quantum-classical machine learning
with Quandela's MerLin photonic quantum computing framework.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
quantum-swaptions/
├── app.py                          # Streamlit entry point
├── pages/                          # Multi-page Streamlit app
├── src/
│   ├── data/                       # Data loading & preprocessing
│   │   ├── loader.py               # Parse train.xlsx, test_template.xlsx
│   │   └── preprocessing.py        # Normalization, PCA, windowing
│   ├── models/
│   │   ├── classical/              # Ridge, XGBoost, LSTM baselines
│   │   └── quantum/                # QRC, Quantum Autoencoder (MerLin)
│   ├── evaluation/                 # Metrics, walk-forward validation
│   └── utils/
│       ├── surface.py              # 224-flat ↔ 14×16 grid conversion & plots
│       └── excel_writer.py         # Fill test template with predictions
├── notebooks/                      # Development & training notebooks
├── trained_models/                 # Saved model weights
└── data/                           # Datasets
```

## Dataset

- **train.xlsx**: 494 trading days × 224 swaption prices (14 tenors × 16 maturities)
- **test_template.xlsx**: 6 future prediction rows + 2 missing data rows

## Key Finding

PCA analysis reveals the swaption surface is **extremely low-dimensional**:
- **3 components** explain **99.93%** of variance
- This justifies our quantum autoencoder approach with a 6-dimensional bottleneck

## Models

1. **Classical Baselines**: Ridge Regression, XGBoost, LSTM
2. **Quantum Reservoir Computing**: Fixed MerLin photonic circuit + Ridge readout
3. **Quantum Autoencoder**: Classical encoder → MerLin QuantumLayer bottleneck → Classical decoder

## Tech Stack

- **Quantum**: Quandela MerLin 0.2.2, Perceval
- **ML**: PyTorch, scikit-learn, XGBoost
- **Viz**: Streamlit, Plotly
