"""
Evaluation metrics for swaption price prediction.
"""

from typing import Dict, Optional

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Maximum absolute error."""
    return np.max(np.abs(y_true - y_pred))


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all metrics at once."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE (%)": mape(y_true, y_pred),
        "R²": r_squared(y_true, y_pred),
        "Max Error": max_error(y_true, y_pred),
    }


def surface_error_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_tenors: int = 14,
    n_maturities: int = 16,
) -> np.ndarray:
    """
    Compute MAE per (tenor, maturity) cell, averaged over time.

    Parameters
    ----------
    y_true : (N, 224)
    y_pred : (N, 224)

    Returns
    -------
    error_grid : (14, 16) — MAE at each grid point
    """
    abs_errors = np.abs(y_true - y_pred)  # (N, 224)
    mean_errors = abs_errors.mean(axis=0)  # (224,)
    # Reshape: columns are maturity-major (16 mats × 14 tenors each)
    grid = mean_errors.reshape(n_maturities, n_tenors).T  # (14, 16)
    return grid
