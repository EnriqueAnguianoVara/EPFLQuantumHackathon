"""
Quantum Autoencoder for swaption surface compression and prediction.

Architecture:
    Classical Encoder (224 → 64 → 16 → n_pca)
    → QuantumLayer bottleneck (n_pca modes, angle encoding)
    → LexGrouping (Fock probs → latent_dim)
    → Classical Decoder (latent_dim → 16 → 64 → 224)

The quantum bottleneck operates in a Hilbert space much larger than its
input dimension, enabling non-linear transformations that a classical
layer of the same width cannot express.
"""

from typing import Dict, Optional, Tuple, List
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from merlin import LexGrouping
from src.models.quantum.circuits import build_autoencoder_circuit


# ---------------------------------------------------------------------------
# Quantum Autoencoder (nn.Module)
# ---------------------------------------------------------------------------
class QuantumAutoencoder(nn.Module):
    """
    Hybrid quantum-classical autoencoder for swaption surfaces.

    Parameters
    ----------
    input_dim : surface dimension (224)
    pre_quantum_dim : dimension before quantum layer (default 6)
    n_modes : photonic modes in quantum bottleneck
    n_photons : photons in quantum circuit
    latent_dim : output dimension after LexGrouping
    circuit_depth : depth of trainable quantum circuit
    """

    def __init__(
        self,
        input_dim: int = 224,
        pre_quantum_dim: int = 6,
        n_modes: int = 6,
        n_photons: int = 3,
        latent_dim: int = 8,
        circuit_depth: int = 1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.pre_quantum_dim = pre_quantum_dim
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.latent_dim = latent_dim

        # Classical encoder: 224 → 64 → 16 → pre_quantum_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, pre_quantum_dim),
        )

        # Quantum bottleneck
        q_info = build_autoencoder_circuit(
            n_modes=n_modes,
            n_photons=n_photons,
            input_size=pre_quantum_dim,
            depth=circuit_depth,
        )
        self.quantum_layer = q_info["layer"]
        q_raw_output = q_info["raw_output_size"]

        # LexGrouping: collapse Fock probs → latent_dim
        self.grouping = LexGrouping(q_raw_output, latent_dim)

        # Classical decoder: latent_dim → 16 → 64 → 224
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode surface → latent vector.

        Parameters
        ----------
        x : (batch, 224) normalized surface

        Returns
        -------
        z : (batch, latent_dim) latent representation
        """
        h = self.encoder(x)           # (batch, pre_quantum_dim)
        q_out = self.quantum_layer(h)  # (batch, q_raw_output)
        z = self.grouping(q_out)       # (batch, latent_dim)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent → surface.

        Parameters
        ----------
        z : (batch, latent_dim)

        Returns
        -------
        x_hat : (batch, 224)
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: encode → decode.

        Parameters
        ----------
        x : (batch, 224)

        Returns
        -------
        x_hat : (batch, 224) reconstructed surface
        """
        z = self.encode(x)
        return self.decode(z)


# ---------------------------------------------------------------------------
# Classical Autoencoder (same architecture, no quantum — for ablation)
# ---------------------------------------------------------------------------
class ClassicalAutoencoder(nn.Module):
    """
    Same architecture as QuantumAutoencoder but with a classical bottleneck
    (Linear + Tanh) instead of QuantumLayer + LexGrouping.

    Used as ablation baseline to measure quantum advantage.
    """

    def __init__(
        self,
        input_dim: int = 224,
        pre_quantum_dim: int = 6,
        latent_dim: int = 8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, pre_quantum_dim),
        )

        # Classical bottleneck (replaces quantum layer + grouping)
        self.bottleneck = nn.Sequential(
            nn.Linear(pre_quantum_dim, latent_dim),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.bottleneck(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


# ---------------------------------------------------------------------------
# Autoencoder Trainer
# ---------------------------------------------------------------------------
class AutoencoderTrainer:
    """
    Training loop for QuantumAutoencoder or ClassicalAutoencoder.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.01,
        epochs: int = 200,
        batch_size: int = 32,
        patience: int = 20,
        device: Optional[str] = None,
    ):
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def fit(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Train the autoencoder.

        Parameters
        ----------
        train_data : (N_train, 224) normalized surfaces
        val_data : (N_val, 224) optional validation data

        Returns
        -------
        dict with training history
        """
        X_train = torch.tensor(train_data, dtype=torch.float32).to(self.device)
        has_val = val_data is not None
        if has_val:
            X_val = torch.tensor(val_data, dtype=torch.float32).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        n_samples = X_train.shape[0]

        self.train_losses = []
        self.val_losses = []

        t0 = time.time()

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            perm = torch.randperm(n_samples)
            X_shuf = X_train[perm]

            for i in range(0, n_samples, self.batch_size):
                batch = X_shuf[i : i + self.batch_size]

                optimizer.zero_grad()
                reconstructed = self.model(batch)
                loss = loss_fn(reconstructed, batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train = epoch_loss / n_batches
            self.train_losses.append(avg_train)

            # Validation
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_recon = self.model(X_val)
                    val_loss = loss_fn(val_recon, X_val).item()
                self.val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    if verbose:
                        print(f"    Early stopping at epoch {epoch+1}")
                    break
            else:
                self.val_losses.append(avg_train)

            if verbose and (epoch + 1) % 20 == 0:
                val_str = f", val={self.val_losses[-1]:.6f}" if has_val else ""
                print(f"    Epoch {epoch+1:3d}/{self.epochs}: "
                      f"train={avg_train:.6f}{val_str}")

        # Restore best
        if best_state is not None:
            self.model.load_state_dict(best_state)

        elapsed = time.time() - t0
        if verbose:
            print(f"    Training complete in {elapsed:.1f}s "
                  f"({len(self.train_losses)} epochs)")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": best_val_loss if has_val else min(self.train_losses),
            "epochs_trained": len(self.train_losses),
            "elapsed_seconds": elapsed,
        }

    def extract_latents(self, data: np.ndarray) -> np.ndarray:
        """
        Extract latent vectors for all surfaces.

        Parameters
        ----------
        data : (N, 224) normalized surfaces

        Returns
        -------
        Z : (N, latent_dim)
        """
        X = torch.tensor(data, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            Z = self.model.encode(X)
        return Z.cpu().numpy()

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        """Reconstruct surfaces through the autoencoder."""
        X = torch.tensor(data, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            X_hat = self.model(X)
        return X_hat.cpu().numpy()


# ---------------------------------------------------------------------------
# Temporal Predictor (models z_t evolution)
# ---------------------------------------------------------------------------
class TemporalPredictor:
    """
    Predicts evolution of latent vectors z_t.

    Given a window of past latents [z_{t-W}, ..., z_{t-1}],
    predicts z_t using Ridge regression.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        window_size: int = 20,
        alpha: float = 1.0,
    ):
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.is_fitted = False

    def fit(self, Z: np.ndarray) -> "TemporalPredictor":
        """
        Fit on latent time series.

        Parameters
        ----------
        Z : (N, latent_dim) latent vectors ordered by time
        """
        X, Y = [], []
        for t in range(self.window_size, len(Z)):
            X.append(Z[t - self.window_size : t].flatten())
            Y.append(Z[t])

        X = np.array(X)
        Y = np.array(Y)

        self.model.fit(X, Y)
        self.is_fitted = True
        return self

    def predict(self, Z_window: np.ndarray) -> np.ndarray:
        """
        Predict next latent from a window.

        Parameters
        ----------
        Z_window : (window_size, latent_dim)

        Returns
        -------
        z_next : (latent_dim,)
        """
        x = Z_window.flatten().reshape(1, -1)
        return self.model.predict(x)[0]

    def predict_rolling(
        self, Z_history: np.ndarray, n_steps: int
    ) -> np.ndarray:
        """
        Rolling multi-step prediction in latent space.

        Parameters
        ----------
        Z_history : (N, latent_dim) past latents
        n_steps : number of future steps

        Returns
        -------
        Z_future : (n_steps, latent_dim)
        """
        buffer = list(Z_history[-self.window_size:])
        predictions = []

        for _ in range(n_steps):
            window = np.array(buffer[-self.window_size:])
            z_next = self.predict(window)
            predictions.append(z_next)
            buffer.append(z_next)

        return np.array(predictions)


# ---------------------------------------------------------------------------
# Swaption Predictor (combines autoencoder + temporal model)
# ---------------------------------------------------------------------------
class SwaptionPredictor:
    """
    End-to-end predictor: uses the autoencoder to compress surfaces
    and the temporal model to forecast in latent space.
    """

    def __init__(
        self,
        autoencoder: nn.Module,
        temporal_predictor: TemporalPredictor,
        device: Optional[str] = None,
    ):
        self.autoencoder = autoencoder
        self.temporal = temporal_predictor

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.autoencoder.to(self.device)

    def predict_future(
        self,
        Z_history: np.ndarray,
        n_steps: int,
    ) -> np.ndarray:
        """
        Predict future surfaces.

        Parameters
        ----------
        Z_history : (N, latent_dim) past latent vectors
        n_steps : number of future days

        Returns
        -------
        surfaces : (n_steps, 224) predicted normalized surfaces
        """
        # Predict future latents
        Z_future = self.temporal.predict_rolling(Z_history, n_steps)

        # Decode each latent to a surface
        Z_tensor = torch.tensor(Z_future, dtype=torch.float32).to(self.device)
        self.autoencoder.eval()
        with torch.no_grad():
            surfaces = self.autoencoder.decode(Z_tensor)

        return surfaces.cpu().numpy()

    def impute_interpolation(
        self,
        Z_history: np.ndarray,
        idx_before: int,
        idx_after: int,
    ) -> np.ndarray:
        """
        Impute a missing surface by interpolating in latent space.

        Parameters
        ----------
        Z_history : (N, latent_dim) all latent vectors
        idx_before : index of the day before the missing date
        idx_after : index of the day after the missing date

        Returns
        -------
        surface : (224,) imputed normalized surface
        """
        z_interp = 0.5 * (Z_history[idx_before] + Z_history[idx_after])
        z_tensor = torch.tensor(z_interp, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.autoencoder.eval()
        with torch.no_grad():
            surface = self.autoencoder.decode(z_tensor)

        return surface.cpu().numpy().squeeze()

    def impute_masked(
        self,
        known_values: np.ndarray,
        known_mask: np.ndarray,
        z_init: Optional[np.ndarray] = None,
        n_steps: int = 500,
        lr: float = 0.01,
    ) -> np.ndarray:
        """
        Impute missing values by optimizing the latent vector.

        Finds z such that decode(z) matches known values at known positions.

        Parameters
        ----------
        known_values : (224,) surface with known values (NaN for missing)
        known_mask : (224,) boolean mask, True = known
        z_init : optional initialization for z
        n_steps : optimization steps
        lr : learning rate

        Returns
        -------
        surface : (224,) complete imputed surface
        """
        latent_dim = self.temporal.latent_dim

        if z_init is not None:
            z = torch.tensor(z_init, dtype=torch.float32, requires_grad=True)
        else:
            z = torch.randn(latent_dim, requires_grad=True)

        z = z.to(self.device)
        mask = torch.tensor(known_mask, dtype=torch.bool).to(self.device)
        target = torch.tensor(known_values, dtype=torch.float32).to(self.device)

        optimizer = torch.optim.Adam([z], lr=lr)

        self.autoencoder.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad_(False)

        for step in range(n_steps):
            optimizer.zero_grad()
            surface_hat = self.autoencoder.decode(z.unsqueeze(0)).squeeze()
            loss = ((surface_hat[mask] - target[mask]) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Re-enable gradients
        for param in self.autoencoder.parameters():
            param.requires_grad_(True)

        with torch.no_grad():
            result = self.autoencoder.decode(z.unsqueeze(0)).squeeze()

        return result.cpu().numpy()
