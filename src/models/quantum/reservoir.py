"""
Quantum Reservoir Computing (QRC) for swaption price prediction.

Uses a fixed (non-trainable) photonic circuit as a non-linear feature
extractor. The Fock-space probability distribution from the circuit
serves as high-dimensional reservoir features, which are fed to a
classical linear readout (Ridge regression).

This matches Quandela's slide 8 suggestion directly.
"""

from typing import Dict, Optional, List, Tuple
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

from merlin import LexGrouping
from src.models.quantum.circuits import build_reservoir_circuit


class QuantumReservoir:
    """
    Quantum Reservoir Computing model.

    Pipeline:
        1. Take a window of PCA data (window_size, n_pca)
        2. Flatten to (window_size * n_pca,)  — or use last step only
        3. Feed through fixed photonic circuit → Fock probabilities
        4. These probabilities are the reservoir features
        5. Train Ridge regression on features → target

    Parameters
    ----------
    n_modes : number of optical modes in the reservoir
    n_photons : number of photons
    n_pca : dimension of PCA input
    window_size : temporal lookback window
    encoding : 'last' (encode only last timestep) or 'flatten' (encode flattened window)
    readout_alpha : Ridge regularization
    seed : random seed for reservoir
    """

    def __init__(
        self,
        n_modes: int = 8,
        n_photons: int = 3,
        n_pca: int = 6,
        window_size: int = 20,
        encoding: str = "last",
        readout_alpha: float = 1.0,
        seed: int = 42,
    ):
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.n_pca = n_pca
        self.window_size = window_size
        self.encoding = encoding
        self.readout_alpha = readout_alpha
        self.seed = seed

        # Determine input size for the quantum layer
        if encoding == "last":
            self.q_input_size = n_pca
        elif encoding == "flatten":
            self.q_input_size = min(n_pca * window_size, n_modes)
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

        # Build the fixed reservoir circuit
        reservoir_info = build_reservoir_circuit(
            n_modes=n_modes,
            n_photons=n_photons,
            input_size=self.q_input_size,
            seed=seed,
        )
        self.q_layer = reservoir_info["layer"]
        self.q_output_size = reservoir_info["output_size"]
        self.circuit = reservoir_info["circuit"]

        # Classical readout
        self.readout = Ridge(alpha=readout_alpha)
        self.is_fitted = False

        # Store feature dimension info
        self.feature_dim = self.q_output_size

    def _encode_input(self, window: np.ndarray) -> torch.Tensor:
        """
        Encode a window of PCA data into quantum layer input.

        Parameters
        ----------
        window : (window_size, n_pca) or (batch, window_size, n_pca)

        Returns
        -------
        encoded : torch.Tensor suitable for the quantum layer
        """
        if window.ndim == 2:
            # Single window (window_size, n_pca)
            if self.encoding == "last":
                encoded = window[-1]  # (n_pca,)
            else:
                encoded = window.flatten()[:self.q_input_size]
            return torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
        elif window.ndim == 3:
            # Batch (batch, window_size, n_pca)
            batch_size = window.shape[0]
            if self.encoding == "last":
                encoded = window[:, -1, :]  # (batch, n_pca)
            else:
                encoded = window.reshape(batch_size, -1)[:, :self.q_input_size]
            return torch.tensor(encoded, dtype=torch.float32)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {window.ndim}D")

    def extract_features(self, window: np.ndarray) -> np.ndarray:
        """
        Extract reservoir features from a window of PCA data.

        Parameters
        ----------
        window : (window_size, n_pca) or (batch, window_size, n_pca)

        Returns
        -------
        features : (q_output_size,) or (batch, q_output_size)
        """
        x = self._encode_input(window)

        self.q_layer.eval()
        with torch.no_grad():
            features = self.q_layer(x)

        return features.cpu().numpy()

    def extract_all_features(
        self,
        pca_data: np.ndarray,
        window_size: Optional[int] = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Extract reservoir features for the entire time series.

        Parameters
        ----------
        pca_data : (N, n_pca) full PCA time series
        window_size : override default window size

        Returns
        -------
        features : (N - window_size, q_output_size)
        """
        ws = window_size or self.window_size
        N = pca_data.shape[0]
        n_samples = N - ws

        all_features = np.zeros((n_samples, self.q_output_size))

        t0 = time.time()
        for i in range(n_samples):
            window = pca_data[i : i + ws]
            feat = self.extract_features(window)
            all_features[i] = feat.squeeze()

            if verbose and (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n_samples - i - 1)
                print(f"    Extracting features: {i+1}/{n_samples} "
                      f"({elapsed:.1f}s elapsed, ~{eta:.1f}s remaining)")

        if verbose:
            print(f"    Done: {n_samples} features in {time.time()-t0:.1f}s")

        return all_features

    def fit(
        self,
        X_features: np.ndarray,
        Y_targets: np.ndarray,
    ) -> "QuantumReservoir":
        """
        Fit the classical readout on pre-extracted reservoir features.

        Parameters
        ----------
        X_features : (n_samples, q_output_size)
        Y_targets : (n_samples, n_pca)
        """
        self.readout.fit(X_features, Y_targets)
        self.is_fitted = True
        return self

    def fit_from_pca(
        self,
        train_pca: np.ndarray,
        verbose: bool = True,
    ) -> "QuantumReservoir":
        """
        End-to-end: extract features + fit readout.

        Parameters
        ----------
        train_pca : (N_train, n_pca) training PCA data
        """
        # Extract features
        features = self.extract_all_features(train_pca, verbose=verbose)

        # Build targets (next-step prediction)
        targets = train_pca[self.window_size:]  # (N_train - window, n_pca)

        # Align: features[i] predicts targets[i]
        assert features.shape[0] == targets.shape[0]

        self.fit(features, targets)
        return self

    def predict(self, X_features: np.ndarray) -> np.ndarray:
        """
        Predict from pre-extracted features.

        Parameters
        ----------
        X_features : (n_samples, q_output_size) or (q_output_size,)

        Returns
        -------
        Y_pred : (n_samples, n_pca)
        """
        if X_features.ndim == 1:
            X_features = X_features.reshape(1, -1)
        return self.readout.predict(X_features)

    def predict_from_window(self, window: np.ndarray) -> np.ndarray:
        """
        End-to-end: encode window → extract features → predict.

        Parameters
        ----------
        window : (window_size, n_pca)

        Returns
        -------
        Y_pred : (n_pca,)
        """
        features = self.extract_features(window)
        return self.predict(features).squeeze()

    def predict_rolling(
        self,
        history_pca: np.ndarray,
        n_steps: int,
    ) -> np.ndarray:
        """
        Rolling multi-step prediction.

        Parameters
        ----------
        history_pca : (N, n_pca) past data (at least window_size rows)
        n_steps : how many future steps

        Returns
        -------
        predictions : (n_steps, n_pca)
        """
        buffer = list(history_pca[-self.window_size:])
        predictions = []

        for _ in range(n_steps):
            window = np.array(buffer[-self.window_size:])
            pred = self.predict_from_window(window)
            predictions.append(pred)
            buffer.append(pred)

        return np.array(predictions)

    def get_memory_capacity(
        self,
        pca_data: np.ndarray,
        max_lag: int = 30,
    ) -> np.ndarray:
        """
        Compute memory capacity of the reservoir.

        Measures how well the reservoir features correlate with
        past inputs at different lags. Higher memory capacity =
        the reservoir retains more temporal information.

        Parameters
        ----------
        pca_data : (N, n_pca) PCA time series
        max_lag : maximum lag to test

        Returns
        -------
        mc : (max_lag,) array of memory capacity per lag
        """
        features = self.extract_all_features(pca_data, verbose=False)
        N = features.shape[0]

        mc = np.zeros(max_lag)

        for lag in range(1, max_lag + 1):
            if lag >= N:
                break

            # Target: input at time t-lag
            # Features: reservoir output at time t
            X = features[lag:]
            Y = pca_data[self.window_size:-lag] if lag < N - self.window_size else pca_data[self.window_size:self.window_size+X.shape[0]]

            min_len = min(X.shape[0], Y.shape[0])
            X = X[:min_len]
            Y = Y[:min_len]

            if min_len < 10:
                break

            # Fit ridge for this lag
            ridge = Ridge(alpha=1.0)
            ridge.fit(X, Y)
            Y_pred = ridge.predict(X)

            # R² score
            ss_res = np.sum((Y - Y_pred) ** 2)
            ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            mc[lag - 1] = max(0, r2)

        return mc

    def get_params(self) -> Dict:
        return {
            "n_modes": self.n_modes,
            "n_photons": self.n_photons,
            "n_pca": self.n_pca,
            "window_size": self.window_size,
            "encoding": self.encoding,
            "readout_alpha": self.readout_alpha,
            "seed": self.seed,
            "q_output_size": self.q_output_size,
            "feature_dim": self.feature_dim,
        }
