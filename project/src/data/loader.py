"""
Data loader for the Quandela Swaption Challenge.

Handles loading train.xlsx and test_template.xlsx, parsing the
'Tenor : X; Maturity : Y' column headers into structured metadata,
and returning clean DataFrames ready for modelling.
"""

import re
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

TENORS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
MATURITIES = [
    0.0833333333333333,  # 1 month
    0.25,                # 3 months
    0.5,                 # 6 months
    0.75,                # 9 months
    1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30,
]

N_TENORS = len(TENORS)       # 14
N_MATURITIES = len(MATURITIES)  # 16
N_PRICES = N_TENORS * N_MATURITIES  # 224

# Human-readable maturity labels for display
MATURITY_LABELS = [
    "1M", "3M", "6M", "9M",
    "1Y", "1.5Y", "2Y", "3Y", "4Y", "5Y",
    "7Y", "10Y", "15Y", "20Y", "25Y", "30Y",
]

_HEADER_RE = re.compile(
    r"Tenor\s*:\s*([\d.]+)\s*;\s*Maturity\s*:\s*([\d.]+)"
)


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------
def parse_header(header: str) -> Optional[Dict[str, float]]:
    """Parse 'Tenor : X; Maturity : Y' → {'tenor': X, 'maturity': Y}."""
    m = _HEADER_RE.match(header)
    if m is None:
        return None
    return {"tenor": float(m.group(1)), "maturity": float(m.group(2))}


def parse_all_headers(headers: List[str]) -> List[Dict[str, float]]:
    """Parse a list of column headers, returning only price columns."""
    results = []
    for h in headers:
        parsed = parse_header(h)
        if parsed is not None:
            results.append(parsed)
    return results


def get_price_column_names() -> List[str]:
    """Return the 224 price column names in the canonical order (tenor-major)."""
    cols = []
    for mat in MATURITIES:
        for ten in TENORS:
            cols.append(f"Tenor : {ten}; Maturity : {mat}")
    return cols


# ---------------------------------------------------------------------------
# Train data
# ---------------------------------------------------------------------------
def load_train(path: Optional[Path] = None) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Load train.xlsx.

    Returns
    -------
    df : pd.DataFrame
        Full dataframe with 'date' column and 224 price columns.
    prices : np.ndarray, shape (494, 224)
        Numpy array of prices only (float64).
    dates : list[str]
        List of date strings.
    """
    if path is None:
        path = DATA_DIR / "train.xlsx"

    df_raw = pd.read_excel(path, engine="openpyxl")

    # The first column is 'Date', remaining 224 are prices
    date_col = df_raw.columns[0]
    price_cols = [c for c in df_raw.columns if c != date_col]

    dates = df_raw[date_col].astype(str).tolist()
    prices = df_raw[price_cols].values.astype(np.float64)

    # Build clean dataframe
    df = pd.DataFrame(prices, columns=price_cols)
    df.insert(0, "date", pd.to_datetime(dates, dayfirst=True))

    return df, prices, dates


# ---------------------------------------------------------------------------
# Test template
# ---------------------------------------------------------------------------
def load_test_template(
    path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load test_template.xlsx.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with 'type', 'date', and 224 price columns.
        Price columns contain float where available, np.nan where 'NA'.
    info : dict
        {
            'future_indices': list[int],    # row indices for future prediction
            'missing_indices': list[int],   # row indices for missing data
            'missing_masks': dict[int, np.ndarray],  # bool mask per missing row (True=known)
        }
    """
    if path is None:
        path = DATA_DIR / "test_template.xlsx"

    df_raw = pd.read_excel(path, engine="openpyxl")

    # Column order: Type, 224 prices, Date
    type_col = df_raw.columns[0]   # 'Type'
    date_col = df_raw.columns[-1]  # 'Date'
    price_cols = list(df_raw.columns[1:-1])

    types = df_raw[type_col].astype(str).tolist()
    dates = pd.to_datetime(df_raw[date_col]).tolist()

    # Parse prices: 'NA' → np.nan, else float
    price_data = df_raw[price_cols].copy()
    for col in price_cols:
        price_data[col] = pd.to_numeric(price_data[col], errors="coerce")

    prices = price_data.values.astype(np.float64)

    # Build clean df
    df = pd.DataFrame(prices, columns=price_cols)
    df.insert(0, "type", types)
    df.insert(1, "date", dates)

    # Classify rows
    future_idx = [i for i, t in enumerate(types) if "future" in t.lower() or "Future" in t]
    missing_idx = [i for i, t in enumerate(types) if "missing" in t.lower() or "Missing" in t]

    # For missing rows, build boolean mask (True = value is known)
    missing_masks = {}
    for i in missing_idx:
        mask = ~np.isnan(prices[i])
        missing_masks[i] = mask

    info = {
        "future_indices": future_idx,
        "missing_indices": missing_idx,
        "missing_masks": missing_masks,
    }

    return df, info


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def load_all(
    data_dir: Optional[Path] = None,
) -> Dict:
    """
    Load everything and return a single dict with all data.

    Returns
    -------
    dict with keys:
        'train_df', 'train_prices', 'train_dates',
        'test_df', 'test_info',
        'tenors', 'maturities', 'n_prices'
    """
    if data_dir is not None:
        train_path = data_dir / "train.xlsx"
        test_path = data_dir / "test_template.xlsx"
    else:
        train_path = None
        test_path = None

    train_df, train_prices, train_dates = load_train(train_path)
    test_df, test_info = load_test_template(test_path)

    return {
        "train_df": train_df,
        "train_prices": train_prices,
        "train_dates": train_dates,
        "test_df": test_df,
        "test_info": test_info,
        "tenors": TENORS,
        "maturities": MATURITIES,
        "maturity_labels": MATURITY_LABELS,
        "n_tenors": N_TENORS,
        "n_maturities": N_MATURITIES,
        "n_prices": N_PRICES,
    }
