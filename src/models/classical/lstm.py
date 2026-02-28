"""
LSTM baseline for swaption price prediction.

Small LSTM operating in PCA space. Takes a window of past PCA components
as a sequence and predicts the next step.
"""

from typing import Dict, Optional, Tuple, List
import numpy as np

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Small LSTM for PCA time series prediction.

    Input:  (batch, window_size, n_pca)
    Output: (batch, n_pca)
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, window_size, input_dim)

        Returns
        -------
        (batch, input_dim)
        """
        # LSTM output: (batch, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)
        # Take last time step
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        return self.fc(last_hidden)  # (batch, input_dim)


class LSTMBaseline:
    """
    Wrapper around LSTMModel with training loop and prediction interface.

    Matches the same API as RidgeBaseline and GBTBaseline.
    """

    def __init__(
        self,
        n_pca: int = 6,
        window_size: int = 20,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        lr: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 15,
        device: Optional[str] = None,
    ):
        self.n_pca = n_pca
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = LSTMModel(
            input_dim=n_pca,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        self.is_fitted = False
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def _prepare_sequences(
        self, X_flat: np.ndarray
    ) -> torch.Tensor:
        """
        Reshape flat windows (n, window_size*n_pca) → (n, window_size, n_pca).
        """
        n = X_flat.shape[0]
        X_3d = X_flat.reshape(n, self.window_size, self.n_pca)
        return torch.tensor(X_3d, dtype=torch.float32).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
    ) -> "LSTMBaseline":
        """
        Train the LSTM with early stopping.

        Parameters
        ----------
        X_train : (n_samples, window_size * n_pca) — flattened windows
        Y_train : (n_samples, n_pca)
        X_val, Y_val : optional validation data (same format)
        """
        X_seq = self._prepare_sequences(X_train)
        Y_t = torch.tensor(Y_train, dtype=torch.float32).to(self.device)

        has_val = X_val is not None and Y_val is not None
        if has_val:
            X_val_seq = self._prepare_sequences(X_val)
            Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self.train_losses = []
        self.val_losses = []

        n_samples = X_seq.shape[0]

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            # Shuffle
            perm = torch.randperm(n_samples)
            X_shuf = X_seq[perm]
            Y_shuf = Y_t[perm]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuf[i : i + self.batch_size]
                Y_batch = Y_shuf[i : i + self.batch_size]

                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = loss_fn(pred, Y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches
            self.train_losses.append(avg_train_loss)

            # Validation
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val_seq)
                    val_loss = loss_fn(val_pred, Y_val_t).item()
                self.val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    break
            else:
                self.val_losses.append(avg_train_loss)

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict.

        Parameters
        ----------
        X : (n_samples, window_size * n_pca) or (window_size * n_pca,)

        Returns
        -------
        Y : (n_samples, n_pca)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_seq = self._prepare_sequences(X)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_seq)
        return pred.cpu().numpy()

    def get_params(self) -> Dict:
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "total_params": total_params,
            "trainable_params": trainable,
            "epochs_trained": len(self.train_losses),
            "final_train_loss": self.train_losses[-1] if self.train_losses else None,
            "best_val_loss": min(self.val_losses) if self.val_losses else None,
        }

    def predict_rolling(
        self,
        history_pca: np.ndarray,
        n_steps: int,
        window_size: int,
        n_pca: int,
    ) -> np.ndarray:
        """
        Rolling multi-step prediction.

        Parameters
        ----------
        history_pca : (N, n_pca)
        n_steps : future steps
        window_size : lookback
        n_pca : number of PCA components

        Returns
        -------
        predictions : (n_steps, n_pca)
        """
        buffer = list(history_pca[-window_size:])
        predictions = []

        self.model.eval()
        with torch.no_grad():
            for _ in range(n_steps):
                window = np.array(buffer[-window_size:])  # (window_size, n_pca)
                x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
                pred = self.model(x).cpu().numpy()[0]  # (n_pca,)
                predictions.append(pred)
                buffer.append(pred)

        return np.array(predictions)

    def save(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """Load model weights."""
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        self.is_fitted = True
