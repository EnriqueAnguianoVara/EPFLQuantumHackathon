"""
Quantum Kernel Gaussian Process (QK-GP) for swaption price prediction.

Uses a fixed or trainable photonic circuit as a quantum feature map to define
a kernel function k(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|². This kernel is then
used inside a classical Gaussian Process regressor, producing predictions
with calibrated uncertainty intervals.

References
----------
- QuaCK-TSF (2024): Quantum Kernel Gaussian Process Forecasting
- Large-scale Benchmark of Quantum Kernel Methods (2025)
- Havlíček et al. (2019): Supervised learning with quantum-enhanced feature maps

The key advantage over QRC + Ridge is that the GP provides:
    1. Calibrated uncertainty (σ) for each prediction
    2. Probabilistic forecasting (crucial for risk management)
    3. Kernel-based non-linearity without training the quantum circuit
"""

from typing import Dict, Optional, List, Tuple
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter
from sklearn.linear_model import Ridge

from merlin import LexGrouping
from src.models.quantum.circuits import build_reservoir_circuit, build_variational_circuit


# ---------------------------------------------------------------------------
# Quantum Kernel (sklearn-compatible)
# ---------------------------------------------------------------------------
class QuantumKernel(Kernel):
    """
    Quantum kernel using a photonic circuit as feature map.

    Computes k(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|² by:
        1. Encoding x_i into the circuit → probability vector p_i
        2. Encoding x_j into the circuit → probability vector p_j
        3. Computing the fidelity-based kernel from the distributions

    Three kernel variants are supported:
        - 'fidelity':  k = (Σ √(p_i · p_j))²   (Bhattacharyya / quantum fidelity)
        - 'inner':     k = Σ (p_i · p_j)         (linear kernel in Fock space)
        - 'rbf_fock':  k = exp(-γ ||p_i - p_j||²) (RBF on Fock probabilities)

    Parameters
    ----------
    q_layer : QuantumLayer (fixed circuit, no trainable params)
    kernel_type : 'fidelity', 'inner', or 'rbf_fock'
    gamma : RBF bandwidth (only used for 'rbf_fock')
    batch_size : batch size for quantum circuit evaluation
    """

    def __init__(
        self,
        q_layer: nn.Module,
        kernel_type: str = "fidelity",
        gamma: float = 1.0,
        batch_size: int = 64,
    ):
        self.q_layer = q_layer
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.batch_size = batch_size

        # Cache for feature vectors (avoid recomputing)
        self._cache: Dict[int, np.ndarray] = {}

    @property
    def hyperparameter_gamma(self):
        if self.kernel_type == "rbf_fock":
            return Hyperparameter("gamma", "numeric", (1e-5, 1e3))
        return Hyperparameter("gamma", "numeric", (1e-5, 1e3), fixed=True)

    def _get_fock_features(self, X: np.ndarray) -> np.ndarray:
        """
        Encode classical data through the quantum circuit to get
        Fock-space probability distributions.

        Parameters
        ----------
        X : (n_samples, input_dim) classical features

        Returns
        -------
        P : (n_samples, fock_dim) probability distributions
        """
        n = X.shape[0]
        all_probs = []

        self.q_layer.eval()
        with torch.no_grad():
            for i in range(0, n, self.batch_size):
                batch = torch.tensor(
                    X[i : i + self.batch_size], dtype=torch.float32
                )
                probs = self.q_layer(batch)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)

    def _compute_kernel_matrix(
        self, P_X: np.ndarray, P_Y: np.ndarray
    ) -> np.ndarray:
        """
        Compute kernel matrix from Fock probability vectors.

        Parameters
        ----------
        P_X : (n_x, fock_dim) Fock probabilities for X
        P_Y : (n_y, fock_dim) Fock probabilities for Y

        Returns
        -------
        K : (n_x, n_y) kernel matrix
        """
        if self.kernel_type == "fidelity":
            # Quantum fidelity: k(x,y) = (Σ √(p_x · p_y))²
            # Clamp to avoid negative values from numerical noise
            sqrt_X = np.sqrt(np.clip(P_X, 0, None))
            sqrt_Y = np.sqrt(np.clip(P_Y, 0, None))
            # (n_x, fock) @ (fock, n_y) → (n_x, n_y)
            K = (sqrt_X @ sqrt_Y.T) ** 2

        elif self.kernel_type == "inner":
            # Linear kernel in Fock space: k(x,y) = Σ p_x · p_y
            K = P_X @ P_Y.T

        elif self.kernel_type == "rbf_fock":
            # RBF on Fock probability vectors
            # ||p_x - p_y||² = ||p_x||² + ||p_y||² - 2 p_x·p_y
            sq_X = np.sum(P_X ** 2, axis=1, keepdims=True)  # (n_x, 1)
            sq_Y = np.sum(P_Y ** 2, axis=1, keepdims=True)  # (n_y, 1)
            dist_sq = sq_X + sq_Y.T - 2.0 * (P_X @ P_Y.T)
            dist_sq = np.clip(dist_sq, 0, None)
            K = np.exp(-self.gamma * dist_sq)

        else:
            raise ValueError(f"Unknown kernel_type: {self.kernel_type}")

        return K

    def __call__(self, X, Y=None, eval_gradient=False):
        """
        Compute kernel matrix K(X, Y).

        Follows sklearn Kernel interface.
        """
        P_X = self._get_fock_features(X)

        if Y is None:
            P_Y = P_X
        else:
            P_Y = self._get_fock_features(Y)

        K = self._compute_kernel_matrix(P_X, P_Y)

        if eval_gradient:
            # Fixed kernel — no gradient w.r.t. hyperparameters
            return K, np.zeros((*K.shape, 0))
        return K

    def diag(self, X):
        """Diagonal of kernel matrix K(X, X)."""
        P_X = self._get_fock_features(X)

        if self.kernel_type == "fidelity":
            # k(x, x) = (Σ √(p_x²))² = (Σ p_x)² = 1 (normalized probs)
            return np.sum(P_X, axis=1) ** 2

        elif self.kernel_type == "inner":
            return np.sum(P_X ** 2, axis=1)

        elif self.kernel_type == "rbf_fock":
            return np.ones(X.shape[0])

        else:
            raise ValueError(f"Unknown kernel_type: {self.kernel_type}")

    def is_stationary(self):
        return self.kernel_type == "rbf_fock"

    def get_params(self, deep=True):
        return {
            "q_layer": self.q_layer,
            "kernel_type": self.kernel_type,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
        }

    def __repr__(self):
        return (
            f"QuantumKernel(type={self.kernel_type}, gamma={self.gamma:.3f})"
        )


# ---------------------------------------------------------------------------
# Quantum Kernel GP Model
# ---------------------------------------------------------------------------
class QuantumKernelGP:
    """
    Quantum Kernel Gaussian Process for swaption price forecasting.

    Pipeline:
        1. Take a window of PCA data (window_size, n_pca)
        2. Encode via 'last' timestep or 'flatten' strategy
        3. Compute quantum kernel K(X_train, X_test) using photonic circuit
        4. GP regression with quantum kernel → prediction + uncertainty

    This model provides the same interface as QuantumReservoir for
    easy comparison, plus uncertainty quantification.

    Parameters
    ----------
    n_modes : number of optical modes in the quantum feature map
    n_photons : number of photons
    n_pca : dimension of PCA input
    window_size : temporal lookback window
    encoding : 'last' (encode only last timestep) or 'flatten'
    kernel_type : 'fidelity', 'inner', or 'rbf_fock'
    gamma : RBF bandwidth (for 'rbf_fock')
    alpha : GP noise level (regularization)
    seed : random seed for circuit
    circuit_type : 'reservoir' (fixed) or 'variational' (trainable feature map)
    circuit_depth : depth for variational circuit
    """

    def __init__(
        self,
        n_modes: int = 8,
        n_photons: int = 3,
        n_pca: int = 6,
        window_size: int = 20,
        encoding: str = "last",
        kernel_type: str = "fidelity",
        gamma: float = 1.0,
        alpha: float = 1e-4,
        seed: int = 42,
        circuit_type: str = "reservoir",
        circuit_depth: int = 2,
    ):
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.n_pca = n_pca
        self.window_size = window_size
        self.encoding = encoding
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.alpha = alpha
        self.seed = seed
        self.circuit_type = circuit_type
        self.circuit_depth = circuit_depth

        # Determine input size for the quantum layer
        if encoding == "last":
            self.q_input_size = n_pca
        elif encoding == "flatten":
            self.q_input_size = min(n_pca * window_size, n_modes)
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

        # Build the quantum circuit (feature map)
        if circuit_type == "reservoir":
            circuit_info = build_reservoir_circuit(
                n_modes=n_modes,
                n_photons=n_photons,
                input_size=self.q_input_size,
                seed=seed,
                depth=circuit_depth,
            )
            self.q_layer = circuit_info["layer"]
            self.q_output_size = circuit_info["output_size"]
        elif circuit_type == "variational":
            circuit_info = build_variational_circuit(
                n_modes=n_modes,
                n_photons=n_photons,
                input_size=self.q_input_size,
                n_layers=circuit_depth,
            )
            self.q_layer = circuit_info["layer"]
            self.q_output_size = circuit_info["raw_output_size"]
        else:
            raise ValueError(f"Unknown circuit_type: {circuit_type}")

        # Build quantum kernel
        self.quantum_kernel = QuantumKernel(
            q_layer=self.q_layer,
            kernel_type=kernel_type,
            gamma=gamma,
        )

        # GP regressors — one per PCA component (sklearn GP is single-output)
        self.gp_models: List[GaussianProcessRegressor] = []
        self.is_fitted = False

        # Stored training data for rolling predictions
        self._X_train: Optional[np.ndarray] = None
        self._Y_train: Optional[np.ndarray] = None

    def _encode_input(self, window: np.ndarray) -> np.ndarray:
        """
        Encode a window of PCA data into quantum layer input.

        Parameters
        ----------
        window : (window_size, n_pca) or (batch, window_size, n_pca)

        Returns
        -------
        encoded : (1, q_input_size) or (batch, q_input_size)
        """
        if window.ndim == 2:
            # Single window (window_size, n_pca)
            if self.encoding == "last":
                encoded = window[-1]  # (n_pca,)
            else:
                encoded = window.flatten()[: self.q_input_size]
            return encoded.reshape(1, -1)
        elif window.ndim == 3:
            # Batch (batch, window_size, n_pca)
            batch_size = window.shape[0]
            if self.encoding == "last":
                encoded = window[:, -1, :]  # (batch, n_pca)
            else:
                encoded = window.reshape(batch_size, -1)[:, : self.q_input_size]
            return encoded
        else:
            raise ValueError(f"Expected 2D or 3D input, got {window.ndim}D")

    def _build_windowed_dataset(
        self, pca_data: np.ndarray, window_size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build (X, Y) pairs from a PCA time series using sliding windows.

        Parameters
        ----------
        pca_data : (N, n_pca) full PCA time series
        window_size : override default window size

        Returns
        -------
        X : (N - ws, q_input_size) encoded inputs
        Y : (N - ws, n_pca) next-step targets
        """
        ws = window_size or self.window_size
        N = pca_data.shape[0]
        n_samples = N - ws

        X_list = []
        for i in range(n_samples):
            window = pca_data[i : i + ws]
            x_enc = self._encode_input(window)
            X_list.append(x_enc.squeeze(0))

        X = np.array(X_list)          # (n_samples, q_input_size)
        Y = pca_data[ws:]             # (n_samples, n_pca)

        return X, Y

    def fit(
        self,
        X_encoded: np.ndarray,
        Y_targets: np.ndarray,
        verbose: bool = True,
    ) -> "QuantumKernelGP":
        """
        Fit GP regressors on pre-encoded data.

        Fits one independent GP per PCA component, all sharing
        the same quantum kernel.

        Parameters
        ----------
        X_encoded : (n_samples, q_input_size) encoded inputs
        Y_targets : (n_samples, n_pca) targets
        """
        n_pca = Y_targets.shape[1]
        self.gp_models = []
        self._X_train = X_encoded.copy()
        self._Y_train = Y_targets.copy()

        t0 = time.time()

        # Pre-compute the kernel matrix once (shared across all GPs)
        if verbose:
            print(f"    Computing quantum kernel matrix "
                  f"({X_encoded.shape[0]}×{X_encoded.shape[0]})...")

        K_train = self.quantum_kernel(X_encoded)

        # Add noise term for numerical stability
        K_train += self.alpha * np.eye(K_train.shape[0])

        if verbose:
            elapsed_k = time.time() - t0
            print(f"    Kernel matrix computed in {elapsed_k:.1f}s")

        # Fit one GP per output dimension using pre-computed kernel
        for j in range(n_pca):
            gp = GaussianProcessRegressor(
                kernel=self.quantum_kernel,
                alpha=self.alpha,
                optimizer=None,  # Don't optimize kernel hyperparams
            )
            # Manually set the training data and pre-computed kernel
            gp.X_train_ = X_encoded
            gp.y_train_ = Y_targets[:, j]
            gp.K_inv_ = np.linalg.inv(K_train)
            gp.alpha_ = gp.K_inv_ @ Y_targets[:, j]
            gp._y_train_mean = np.mean(Y_targets[:, j])
            gp.log_marginal_likelihood_value_ = None

            self.gp_models.append(gp)

            if verbose and (j + 1) % max(1, n_pca // 4) == 0:
                print(f"    Fitted GP {j+1}/{n_pca}")

        self.is_fitted = True

        if verbose:
            print(f"    All {n_pca} GPs fitted in {time.time()-t0:.1f}s")

        return self

    def fit_from_pca(
        self,
        train_pca: np.ndarray,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> "QuantumKernelGP":
        """
        End-to-end: build windowed dataset + fit GPs.

        Parameters
        ----------
        train_pca : (N_train, n_pca) training PCA data
        max_samples : optional cap on training samples (GP scales as O(n³),
                      so for large datasets you may want to subsample)
        """
        if verbose:
            print(f"    Building windowed dataset from {train_pca.shape[0]} "
                  f"timesteps (window={self.window_size})...")

        X, Y = self._build_windowed_dataset(train_pca)

        # Subsample if needed (GP is O(n³) in memory and computation)
        if max_samples is not None and X.shape[0] > max_samples:
            if verbose:
                print(f"    Subsampling {X.shape[0]} → {max_samples} "
                      f"(GP complexity constraint)")
            # Use evenly spaced samples to preserve temporal coverage
            indices = np.linspace(0, X.shape[0] - 1, max_samples, dtype=int)
            X = X[indices]
            Y = Y[indices]

        return self.fit(X, Y, verbose=verbose)

    def predict(
        self,
        X_encoded: np.ndarray,
        return_std: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict from pre-encoded data.

        Parameters
        ----------
        X_encoded : (n_samples, q_input_size) or (q_input_size,)
        return_std : whether to return uncertainty estimates

        Returns
        -------
        Y_pred : (n_samples, n_pca) point predictions
        Y_std : (n_samples, n_pca) standard deviations (if return_std=True)
        """
        if X_encoded.ndim == 1:
            X_encoded = X_encoded.reshape(1, -1)

        n_samples = X_encoded.shape[0]
        n_pca = len(self.gp_models)

        # Compute kernel between test and training points
        K_test_train = self.quantum_kernel(X_encoded, self._X_train)

        Y_pred = np.zeros((n_samples, n_pca))
        Y_std = np.zeros((n_samples, n_pca)) if return_std else None

        for j, gp in enumerate(self.gp_models):
            # Mean prediction: K_*^T K^{-1} y
            mean = K_test_train @ gp.alpha_ + gp._y_train_mean
            # Correction: sklearn GP centers the targets
            Y_pred[:, j] = mean

            if return_std:
                # Variance: k(x*,x*) - K_*^T K^{-1} K_*
                k_diag = self.quantum_kernel.diag(X_encoded)
                v = K_test_train @ gp.K_inv_
                var = k_diag - np.sum(v * K_test_train, axis=1)
                var = np.clip(var, 0, None)
                Y_std[:, j] = np.sqrt(var)

        if return_std:
            return Y_pred, Y_std
        return Y_pred, None

    def predict_from_window(
        self,
        window: np.ndarray,
        return_std: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        End-to-end: encode window → predict.

        Parameters
        ----------
        window : (window_size, n_pca)

        Returns
        -------
        Y_pred : (n_pca,) point prediction
        Y_std : (n_pca,) uncertainty (if return_std=True)
        """
        x_enc = self._encode_input(window)
        pred, std = self.predict(x_enc, return_std=return_std)
        if return_std:
            return pred.squeeze(), std.squeeze()
        return pred.squeeze(), None

    def predict_rolling(
        self,
        history_pca: np.ndarray,
        n_steps: int,
        return_std: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Rolling multi-step prediction with uncertainty propagation.

        At each step, the GP prediction (and optionally its uncertainty)
        is appended to the buffer for the next step.

        Parameters
        ----------
        history_pca : (N, n_pca) past data (at least window_size rows)
        n_steps : how many future steps
        return_std : whether to return per-step uncertainty

        Returns
        -------
        predictions : (n_steps, n_pca) point predictions
        uncertainties : (n_steps, n_pca) standard deviations (if return_std)
        """
        buffer = list(history_pca[-self.window_size :])
        predictions = []
        uncertainties = []

        for _ in range(n_steps):
            window = np.array(buffer[-self.window_size :])
            pred, std = self.predict_from_window(
                window, return_std=return_std
            )
            predictions.append(pred)
            if return_std:
                uncertainties.append(std)
            buffer.append(pred)

        predictions = np.array(predictions)
        if return_std:
            return predictions, np.array(uncertainties)
        return predictions, None

    def get_memory_capacity(
        self,
        pca_data: np.ndarray,
        max_lag: int = 30,
    ) -> np.ndarray:
        """
        Compute memory capacity using the quantum kernel features.

        Mirrors the QuantumReservoir.get_memory_capacity interface
        for direct comparison.

        Parameters
        ----------
        pca_data : (N, n_pca) PCA time series
        max_lag : maximum lag to test

        Returns
        -------
        mc : (max_lag,) array of memory capacity per lag
        """
        # Extract Fock features (equivalent to reservoir features)
        X, _ = self._build_windowed_dataset(pca_data)
        features = self.quantum_kernel._get_fock_features(X)

        N = features.shape[0]
        mc = np.zeros(max_lag)

        for lag in range(1, max_lag + 1):
            if lag >= N:
                break

            X_feat = features[lag:]
            Y_target = pca_data[self.window_size : -lag] if lag < N else pca_data[
                self.window_size : self.window_size + X_feat.shape[0]
            ]

            min_len = min(X_feat.shape[0], Y_target.shape[0])
            X_feat = X_feat[:min_len]
            Y_target = Y_target[:min_len]

            if min_len < 10:
                break

            ridge = Ridge(alpha=1.0)
            ridge.fit(X_feat, Y_target)
            Y_pred = ridge.predict(X_feat)

            ss_res = np.sum((Y_target - Y_pred) ** 2)
            ss_tot = np.sum((Y_target - Y_target.mean(axis=0)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            mc[lag - 1] = max(0, r2)

        return mc

    def score(
        self,
        X_encoded: np.ndarray,
        Y_targets: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Parameters
        ----------
        X_encoded : (n_test, q_input_size)
        Y_targets : (n_test, n_pca)

        Returns
        -------
        dict with 'mse', 'rmse', 'mae', 'r2', 'mean_nlpd'
        """
        Y_pred, Y_std = self.predict(X_encoded, return_std=True)

        residuals = Y_targets - Y_pred
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((Y_targets - Y_targets.mean(axis=0)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Negative Log Predictive Density (probabilistic metric)
        # NLPD = 0.5 * log(2π σ²) + (y - μ)² / (2σ²)
        if Y_std is not None:
            var = Y_std ** 2 + 1e-10
            nlpd = 0.5 * np.log(2 * np.pi * var) + residuals ** 2 / (2 * var)
            mean_nlpd = np.mean(nlpd)
        else:
            mean_nlpd = float("nan")

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mean_nlpd": float(mean_nlpd),
        }

    def get_params(self) -> Dict:
        return {
            "n_modes": self.n_modes,
            "n_photons": self.n_photons,
            "n_pca": self.n_pca,
            "window_size": self.window_size,
            "encoding": self.encoding,
            "kernel_type": self.kernel_type,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "seed": self.seed,
            "circuit_type": self.circuit_type,
            "circuit_depth": self.circuit_depth,
            "q_output_size": self.q_output_size,
            "n_gp_models": len(self.gp_models),
        }


# ---------------------------------------------------------------------------
# Composite Kernel: Quantum + Classical (for ablation / boosting)
# ---------------------------------------------------------------------------
class HybridQuantumClassicalKernel(Kernel):
    """
    Weighted combination of a quantum kernel and a classical RBF kernel:

        k(x, y) = w_q · k_quantum(x, y) + (1 - w_q) · k_rbf(x, y)

    This allows measuring how much the quantum kernel contributes
    beyond a classical baseline. If w_q → 0 provides the best fit,
    the quantum kernel adds no value (no quantum advantage).

    Parameters
    ----------
    quantum_kernel : QuantumKernel instance
    w_quantum : weight of quantum kernel in [0, 1]
    gamma_rbf : bandwidth of classical RBF component
    """

    def __init__(
        self,
        quantum_kernel: QuantumKernel,
        w_quantum: float = 0.5,
        gamma_rbf: float = 1.0,
    ):
        self.quantum_kernel = quantum_kernel
        self.w_quantum = w_quantum
        self.gamma_rbf = gamma_rbf

    @property
    def hyperparameter_w_quantum(self):
        return Hyperparameter("w_quantum", "numeric", (0.0, 1.0))

    @property
    def hyperparameter_gamma_rbf(self):
        return Hyperparameter("gamma_rbf", "numeric", (1e-5, 1e3))

    def _classical_rbf(self, X, Y):
        """Compute classical RBF kernel."""
        sq_X = np.sum(X ** 2, axis=1, keepdims=True)
        sq_Y = np.sum(Y ** 2, axis=1, keepdims=True)
        dist_sq = sq_X + sq_Y.T - 2.0 * (X @ Y.T)
        dist_sq = np.clip(dist_sq, 0, None)
        return np.exp(-self.gamma_rbf * dist_sq)

    def __call__(self, X, Y=None, eval_gradient=False):
        K_q = self.quantum_kernel(X, Y)
        K_c = self._classical_rbf(X, Y if Y is not None else X)

        K = self.w_quantum * K_q + (1 - self.w_quantum) * K_c

        if eval_gradient:
            return K, np.zeros((*K.shape, 0))
        return K

    def diag(self, X):
        d_q = self.quantum_kernel.diag(X)
        d_c = np.ones(X.shape[0])  # RBF diagonal is always 1
        return self.w_quantum * d_q + (1 - self.w_quantum) * d_c

    def is_stationary(self):
        return False

    def get_params(self, deep=True):
        return {
            "quantum_kernel": self.quantum_kernel,
            "w_quantum": self.w_quantum,
            "gamma_rbf": self.gamma_rbf,
        }

    def __repr__(self):
        return (
            f"HybridKernel(w_q={self.w_quantum:.2f}, "
            f"quantum={self.quantum_kernel}, "
            f"gamma_rbf={self.gamma_rbf:.3f})"
        )


# ---------------------------------------------------------------------------
# Classical Kernel GP (baseline for ablation)
# ---------------------------------------------------------------------------
class ClassicalKernelGP:
    """
    Classical RBF Kernel GP with the same interface as QuantumKernelGP.

    Used as ablation baseline: if this performs equally well,
    the quantum kernel provides no advantage.
    """

    def __init__(
        self,
        n_pca: int = 6,
        window_size: int = 20,
        encoding: str = "last",
        gamma: float = 1.0,
        alpha: float = 1e-4,
    ):
        self.n_pca = n_pca
        self.window_size = window_size
        self.encoding = encoding
        self.gamma = gamma
        self.alpha = alpha

        if encoding == "last":
            self.input_size = n_pca
        else:
            self.input_size = n_pca * window_size

        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        self.kernel = RBF(length_scale=1.0 / np.sqrt(2 * gamma))
        self.gp_models: List[GaussianProcessRegressor] = []
        self.is_fitted = False
        self._X_train: Optional[np.ndarray] = None

    def _encode_input(self, window: np.ndarray) -> np.ndarray:
        if window.ndim == 2:
            if self.encoding == "last":
                return window[-1].reshape(1, -1)
            return window.flatten().reshape(1, -1)
        elif window.ndim == 3:
            if self.encoding == "last":
                return window[:, -1, :]
            return window.reshape(window.shape[0], -1)

    def fit_from_pca(
        self,
        train_pca: np.ndarray,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> "ClassicalKernelGP":
        ws = self.window_size
        N = train_pca.shape[0]
        n_samples = N - ws

        X_list = []
        for i in range(n_samples):
            window = train_pca[i : i + ws]
            x_enc = self._encode_input(window).squeeze(0)
            X_list.append(x_enc)

        X = np.array(X_list)
        Y = train_pca[ws:]

        if max_samples is not None and X.shape[0] > max_samples:
            indices = np.linspace(0, X.shape[0] - 1, max_samples, dtype=int)
            X = X[indices]
            Y = Y[indices]

        self._X_train = X.copy()
        n_pca = Y.shape[1]
        self.gp_models = []

        t0 = time.time()
        for j in range(n_pca):
            gp = GaussianProcessRegressor(
                kernel=self.kernel,
                alpha=self.alpha,
                optimizer=None,
            )
            gp.fit(X, Y[:, j])
            self.gp_models.append(gp)

        self.is_fitted = True
        if verbose:
            print(f"    Classical GP fitted in {time.time()-t0:.1f}s")

        return self

    def predict_from_window(
        self, window: np.ndarray, return_std: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        x_enc = self._encode_input(window)
        n_pca = len(self.gp_models)
        pred = np.zeros(n_pca)
        std = np.zeros(n_pca) if return_std else None

        for j, gp in enumerate(self.gp_models):
            if return_std:
                p, s = gp.predict(x_enc, return_std=True)
                pred[j] = p[0]
                std[j] = s[0]
            else:
                pred[j] = gp.predict(x_enc)[0]

        if return_std:
            return pred, std
        return pred, None

    def predict_rolling(
        self,
        history_pca: np.ndarray,
        n_steps: int,
        return_std: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        buffer = list(history_pca[-self.window_size :])
        predictions = []
        uncertainties = []

        for _ in range(n_steps):
            window = np.array(buffer[-self.window_size :])
            pred, std = self.predict_from_window(window, return_std=return_std)
            predictions.append(pred)
            if return_std:
                uncertainties.append(std)
            buffer.append(pred)

        predictions = np.array(predictions)
        if return_std:
            return predictions, np.array(uncertainties)
        return predictions, None

    def get_params(self) -> Dict:
        return {
            "n_pca": self.n_pca,
            "window_size": self.window_size,
            "encoding": self.encoding,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "n_gp_models": len(self.gp_models),
        }
