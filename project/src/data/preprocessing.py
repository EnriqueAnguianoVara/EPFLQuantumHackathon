"""
Preprocessing utilities for swaption surface data.

Provides normalization, PCA dimensionality reduction, sliding-window
construction, and temporal train/validation splitting.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
@dataclass
class Scaler:
    """Wraps sklearn StandardScaler with convenience methods for our data."""

    _scaler: StandardScaler = field(default_factory=StandardScaler)
    is_fitted: bool = False

    def fit(self, prices: np.ndarray) -> "Scaler":
        """Fit on (N, 224) price matrix."""
        self._scaler.fit(prices)
        self.is_fitted = True
        return self

    def transform(self, prices: np.ndarray) -> np.ndarray:
        return self._scaler.transform(prices)

    def inverse_transform(self, prices_norm: np.ndarray) -> np.ndarray:
        return self._scaler.inverse_transform(prices_norm)

    def fit_transform(self, prices: np.ndarray) -> np.ndarray:
        self.fit(prices)
        return self.transform(prices)

    @property
    def mean(self) -> np.ndarray:
        return self._scaler.mean_

    @property
    def std(self) -> np.ndarray:
        return self._scaler.scale_


def normalize(
    prices: np.ndarray,
    scaler: Optional[Scaler] = None,
) -> Tuple[np.ndarray, Scaler]:
    """
    Z-score normalize prices column-wise.

    Parameters
    ----------
    prices : (N, 224) array
    scaler : optional pre-fitted Scaler

    Returns
    -------
    prices_norm : (N, 224) array
    scaler : fitted Scaler
    """
    if scaler is None:
        scaler = Scaler()
        prices_norm = scaler.fit_transform(prices)
    else:
        prices_norm = scaler.transform(prices)
    return prices_norm, scaler


def denormalize(prices_norm: np.ndarray, scaler: Scaler) -> np.ndarray:
    """Reverse z-score normalization."""
    return scaler.inverse_transform(prices_norm)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
@dataclass
class PCAReducer:
    """PCA dimensionality reduction for the swaption surface."""

    n_components: int = 6
    _pca: PCA = field(init=False, default=None)
    is_fitted: bool = field(init=False, default=False)

    def __post_init__(self):
        self._pca = PCA(n_components=self.n_components)

    def fit(self, prices_norm: np.ndarray) -> "PCAReducer":
        self._pca.fit(prices_norm)
        self.is_fitted = True
        return self

    def transform(self, prices_norm: np.ndarray) -> np.ndarray:
        return self._pca.transform(prices_norm)

    def inverse_transform(self, reduced: np.ndarray) -> np.ndarray:
        return self._pca.inverse_transform(reduced)

    def fit_transform(self, prices_norm: np.ndarray) -> np.ndarray:
        self.fit(prices_norm)
        return self.transform(prices_norm)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self._pca.explained_variance_ratio_

    @property
    def cumulative_variance(self) -> np.ndarray:
        return np.cumsum(self._pca.explained_variance_ratio_)

    @property
    def components(self) -> np.ndarray:
        return self._pca.components_


def pca_reduce(
    prices_norm: np.ndarray,
    n_components: int = 6,
    reducer: Optional[PCAReducer] = None,
) -> Tuple[np.ndarray, PCAReducer]:
    """
    Reduce (N, 224) → (N, n_components) via PCA.

    Returns
    -------
    reduced : (N, n_components)
    reducer : fitted PCAReducer
    """
    if reducer is None:
        reducer = PCAReducer(n_components=n_components)
        reduced = reducer.fit_transform(prices_norm)
    else:
        reduced = reducer.transform(prices_norm)
    return reduced, reducer


# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------
def create_windows(
    data: np.ndarray,
    window_size: int = 20,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for time-series prediction.

    Parameters
    ----------
    data : (N, D) array — could be raw prices (224) or PCA-reduced (6)
    window_size : number of past steps as input
    horizon : number of future steps to predict

    Returns
    -------
    X : (N - window_size - horizon + 1, window_size, D) — input windows
    Y : (N - window_size - horizon + 1, horizon, D) — targets
    """
    N, D = data.shape
    n_samples = N - window_size - horizon + 1

    if n_samples <= 0:
        raise ValueError(
            f"Not enough data: N={N}, window={window_size}, horizon={horizon}"
        )

    X = np.zeros((n_samples, window_size, D))
    Y = np.zeros((n_samples, horizon, D))

    for i in range(n_samples):
        X[i] = data[i : i + window_size]
        Y[i] = data[i + window_size : i + window_size + horizon]

    return X, Y


def create_flat_windows(
    data: np.ndarray,
    window_size: int = 20,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Like create_windows but flattens X for use with linear models.

    Returns
    -------
    X : (n_samples, window_size * D) — flattened input
    Y : (n_samples, D) if horizon==1, else (n_samples, horizon * D)
    """
    X_3d, Y_3d = create_windows(data, window_size, horizon)
    n = X_3d.shape[0]
    X_flat = X_3d.reshape(n, -1)
    if horizon == 1:
        Y_flat = Y_3d.reshape(n, -1)
    else:
        Y_flat = Y_3d.reshape(n, -1)
    return X_flat, Y_flat


# ---------------------------------------------------------------------------
# Train / Validation split (temporal)
# ---------------------------------------------------------------------------
def train_val_split(
    prices: np.ndarray,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Split time series into train and validation, respecting temporal order.

    Returns
    -------
    train : (N_train, D)
    val : (N_val, D)
    split_index : int — index where validation starts
    """
    N = prices.shape[0]
    split = int(N * (1 - val_ratio))
    return prices[:split], prices[split:], split


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------
def full_pipeline(
    prices: np.ndarray,
    n_pca_components: int = 6,
    window_size: int = 20,
    horizon: int = 1,
    val_ratio: float = 0.15,
) -> dict:
    """
    Run the complete preprocessing pipeline.

    Returns dict with all intermediate results and fitted transformers.
    """
    # 1. Train/val split (on raw data)
    train_raw, val_raw, split_idx = train_val_split(prices, val_ratio)

    # 2. Normalize (fit on train only)
    train_norm, scaler = normalize(train_raw)
    val_norm, _ = normalize(val_raw, scaler)
    all_norm, _ = normalize(prices, scaler)

    # 3. PCA (fit on train only)
    train_pca, pca_reducer = pca_reduce(train_norm, n_pca_components)
    val_pca, _ = pca_reduce(val_norm, reducer=pca_reducer)
    all_pca, _ = pca_reduce(all_norm, reducer=pca_reducer)

    # 4. Windows (on PCA data)
    X_train, Y_train = create_flat_windows(train_pca, window_size, horizon)
    X_val, Y_val = create_flat_windows(val_pca, window_size, horizon)

    return {
        # Raw splits
        "train_raw": train_raw,
        "val_raw": val_raw,
        "split_idx": split_idx,
        # Normalized
        "train_norm": train_norm,
        "val_norm": val_norm,
        "all_norm": all_norm,
        "scaler": scaler,
        # PCA
        "train_pca": train_pca,
        "val_pca": val_pca,
        "all_pca": all_pca,
        "pca_reducer": pca_reducer,
        # Windowed (for models)
        "X_train": X_train,
        "Y_train": Y_train,
        "X_val": X_val,
        "Y_val": Y_val,
        # Config
        "window_size": window_size,
        "horizon": horizon,
        "n_pca_components": n_pca_components,
    }
