from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SURFACE_META_COLUMNS = ("Type", "Date")


@dataclass
class BenchmarkResult:
    name: str
    factor_mse: float
    surface_mse: float
    details: str


@dataclass(frozen=True)
class QuantumReservoirLSTMConfig:
    n_components: int = 12
    window: int = 8
    reservoir_dim: int = 96
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 120
    patience: int = 18
    validation_fraction: float = 0.2
    seed: int = 42
    leak: float = 0.65
    recurrent_scale: float = 0.35
    input_scale: float = 0.9
    entanglement_scale: float = 0.25


class QuantumReservoirFeatureMap:
    """Fixed quantum-inspired reservoir based on angle encoding and mixing.

    This does not require a quantum runtime, but it mirrors the usual pattern:
    - angle encode PCA factors into phases,
    - apply fixed nonlinear interactions,
    - keep a recurrent reservoir state with a fixed spectral radius.
    """

    def __init__(self, input_dim: int, config: QuantumReservoirLSTMConfig) -> None:
        self.input_dim = input_dim
        self.config = config
        rng = np.random.default_rng(config.seed)

        self.w_in = rng.normal(
            loc=0.0,
            scale=1.0 / math.sqrt(max(1, input_dim)),
            size=(input_dim, config.reservoir_dim),
        )
        self.bias = rng.uniform(-math.pi, math.pi, size=(config.reservoir_dim,))
        w_res = rng.normal(
            loc=0.0,
            scale=1.0 / math.sqrt(max(1, config.reservoir_dim)),
            size=(config.reservoir_dim, config.reservoir_dim),
        )
        eigvals = np.linalg.eigvals(w_res)
        radius = float(np.max(np.abs(eigvals)))
        if radius > 0:
            w_res = 0.92 * (w_res / radius)
        self.w_res = w_res

    def transform_sequence(self, sequence: np.ndarray) -> np.ndarray:
        state = np.zeros(self.config.reservoir_dim, dtype=np.float32)
        outputs: list[np.ndarray] = []
        for row in sequence:
            drive = self.config.input_scale * (row @ self.w_in) + self.bias
            phase_sin = np.sin(drive)
            phase_cos = np.cos(1.618 * drive)
            entangled = (
                phase_sin
                + 0.5 * phase_cos
                + self.config.entanglement_scale * phase_sin * np.roll(phase_cos, 1)
            )
            recurrent = state @ self.w_res
            updated = np.tanh(entangled + self.config.recurrent_scale * recurrent)
            state = ((1.0 - self.config.leak) * state + self.config.leak * updated).astype(
                np.float32
            )
            outputs.append(state.copy())
        return np.vstack(outputs)


class ReservoirLSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, output_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(sequence)
        last_state = outputs[:, -1, :]
        return self.head(last_state)


class QuantumReservoirLSTMForecaster:
    """Hybrid forecaster: fixed quantum-inspired reservoir + trainable LSTM."""

    def __init__(self, config: QuantumReservoirLSTMConfig | None = None) -> None:
        self.config = config or QuantumReservoirLSTMConfig()
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.reservoir_: QuantumReservoirFeatureMap | None = None
        self.model_: ReservoirLSTMRegressor | None = None
        self.surface_columns_: list[str] | None = None
        self.factor_history_: np.ndarray | None = None
        self.train_lookup_: pd.DataFrame | None = None
        self.validation_result_: BenchmarkResult | None = None
        self.training_history_: list[tuple[int, float, float]] = []

    @staticmethod
    def _surface_columns(df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in SURFACE_META_COLUMNS]

    def _to_surface(self, factors: np.ndarray) -> np.ndarray:
        if self.scaler_ is None or self.pca_ is None:
            raise RuntimeError("Model is not fitted.")
        z_values = self.pca_.inverse_transform(factors)
        return self.scaler_.inverse_transform(z_values)

    def _make_factor_sequences(
        self,
        factors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(factors) <= self.config.window:
            raise ValueError("Not enough observations for the selected window.")

        inputs = []
        targets = []
        anchors = []
        for t in range(self.config.window - 1, len(factors) - 1):
            current = factors[t - self.config.window + 1 : t + 1]
            inputs.append(current)
            anchors.append(current[-1])
            targets.append(factors[t + 1] - current[-1])
        return np.stack(inputs), np.stack(targets), np.stack(anchors)

    def _prepare_reservoir_features(self, factor_sequences: np.ndarray) -> np.ndarray:
        if self.reservoir_ is None:
            raise RuntimeError("Reservoir is not initialised.")
        transformed = [self.reservoir_.transform_sequence(seq) for seq in factor_sequences]
        return np.stack(transformed).astype(np.float32)

    def _split_sequences(
        self,
        X: np.ndarray,
        y_delta: np.ndarray,
        anchors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        split_idx = int((1.0 - self.config.validation_fraction) * len(X))
        split_idx = min(max(split_idx, 1), len(X) - 1)
        return (
            X[:split_idx],
            X[split_idx:],
            y_delta[:split_idx],
            y_delta[split_idx:],
            anchors[:split_idx],
            anchors[split_idx:],
        )

    def _train_model(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        anchor_val: np.ndarray,
    ) -> BenchmarkResult:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.model_ = ReservoirLSTMRegressor(
            input_dim=self.config.reservoir_dim,
            hidden_size=self.config.hidden_size,
            output_dim=self.config.n_components,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = nn.MSELoss()

        train_ds = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train.astype(np.float32)),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=min(self.config.batch_size, len(train_ds)),
            shuffle=True,
        )

        best_state = None
        best_metric = None
        best_factor = None
        wait = 0
        self.training_history_.clear()

        X_val_tensor = torch.from_numpy(X_val)
        for epoch in range(1, self.config.max_epochs + 1):
            self.model_.train()
            running_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                pred = self.model_(batch_X)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += float(loss.item()) * len(batch_X)

            self.model_.eval()
            with torch.no_grad():
                pred_delta = self.model_(X_val_tensor).cpu().numpy()
            pred_factors = pred_delta + anchor_val
            true_factors = y_val + anchor_val
            factor_mse = float(mean_squared_error(true_factors, pred_factors))
            pred_surface = self._to_surface(pred_factors)
            true_surface = self._to_surface(true_factors)
            surface_mse = float(mean_squared_error(true_surface, pred_surface))
            epoch_loss = running_loss / max(1, len(train_ds))
            self.training_history_.append((epoch, epoch_loss, surface_mse))

            if best_metric is None or surface_mse < best_metric:
                best_metric = surface_mse
                best_factor = factor_mse
                best_state = copy.deepcopy(self.model_.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= self.config.patience:
                    break

        if best_state is None or best_metric is None or best_factor is None:
            raise RuntimeError("Training did not produce a valid validation checkpoint.")

        self.model_.load_state_dict(best_state)
        details = (
            f"k={self.config.n_components}, window={self.config.window}, "
            f"reservoir_dim={self.config.reservoir_dim}, hidden={self.config.hidden_size}"
        )
        return BenchmarkResult(
            name="quantum_reservoir_lstm",
            factor_mse=best_factor,
            surface_mse=best_metric,
            details=details,
        )

    def fit(self, train_df: pd.DataFrame) -> "QuantumReservoirLSTMForecaster":
        train_df = train_df.copy()
        train_df["Date"] = pd.to_datetime(train_df["Date"], dayfirst=True)
        train_df = train_df.sort_values("Date").reset_index(drop=True)

        self.surface_columns_ = self._surface_columns(train_df)
        surface_values = train_df[self.surface_columns_].to_numpy(dtype=float)

        self.scaler_ = StandardScaler()
        z_values = self.scaler_.fit_transform(surface_values)
        self.pca_ = PCA(n_components=self.config.n_components, random_state=0)
        factors = self.pca_.fit_transform(z_values)
        self.factor_history_ = factors.copy()
        self.train_lookup_ = train_df.set_index("Date")[self.surface_columns_]
        self.reservoir_ = QuantumReservoirFeatureMap(self.config.n_components, self.config)

        factor_sequences, y_delta, anchors = self._make_factor_sequences(factors)
        reservoir_sequences = self._prepare_reservoir_features(factor_sequences)
        X_train, X_val, y_train, y_val, _, anchor_val = self._split_sequences(
            reservoir_sequences,
            y_delta,
            anchors,
        )
        self.validation_result_ = self._train_model(X_train, X_val, y_train.astype(np.float32), y_val, anchor_val)
        return self

    def _require_fitted(self) -> None:
        if (
            self.model_ is None
            or self.scaler_ is None
            or self.pca_ is None
            or self.reservoir_ is None
            or self.factor_history_ is None
            or self.surface_columns_ is None
            or self.train_lookup_ is None
        ):
            raise RuntimeError("The QuantumReservoirLSTMForecaster must be fitted first.")

    def _predict_next_factor(self, factor_window: np.ndarray) -> np.ndarray:
        self._require_fitted()
        reservoir_sequence = self.reservoir_.transform_sequence(factor_window).astype(np.float32)
        tensor = torch.from_numpy(reservoir_sequence).unsqueeze(0)
        self.model_.eval()
        with torch.no_grad():
            delta = self.model_(tensor).cpu().numpy()[0]
        return factor_window[-1] + delta

    def forecast_future_surfaces(self, steps: int) -> np.ndarray:
        self._require_fitted()
        factor_history = self.factor_history_.copy()
        forecasts = []
        for _ in range(steps):
            factor_window = factor_history[-self.config.window :]
            next_factor = self._predict_next_factor(factor_window)
            forecasts.append(next_factor)
            factor_history = np.vstack([factor_history, next_factor])
        return self._to_surface(np.vstack(forecasts))

    def impute_missing_row(self, row: pd.Series) -> pd.Series:
        self._require_fitted()
        filled = row.copy()
        date_value = pd.to_datetime(row["Date"])
        missing_mask = filled[self.surface_columns_].isna()
        if not bool(missing_mask.any()):
            return filled

        if date_value in self.train_lookup_.index:
            reference = self.train_lookup_.loc[date_value]
            current = pd.to_numeric(filled.loc[self.surface_columns_], errors="coerce")
            filled.loc[self.surface_columns_] = current.where(~current.isna(), reference)
            return filled

        earlier = self.train_lookup_[self.train_lookup_.index < date_value]
        later = self.train_lookup_[self.train_lookup_.index > date_value]
        if not earlier.empty and not later.empty:
            reference = 0.5 * (earlier.iloc[-1] + later.iloc[0])
        elif not earlier.empty:
            reference = earlier.iloc[-1]
        elif not later.empty:
            reference = later.iloc[0]
        else:
            reference = pd.Series(np.nan, index=self.surface_columns_)
        current = pd.to_numeric(filled.loc[self.surface_columns_], errors="coerce")
        filled.loc[self.surface_columns_] = current.where(~current.isna(), reference)
        return filled

    def build_submission(self, template_df: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        submission = template_df.copy()
        submission["Date"] = pd.to_datetime(submission["Date"])

        type_labels = submission["Type"].astype(str).str.lower()
        future_mask = type_labels.eq("future prediction")
        missing_mask = type_labels.str.contains("missing")

        future_count = int(future_mask.sum())
        if future_count:
            forecasts = self.forecast_future_surfaces(future_count)
            submission.loc[future_mask, self.surface_columns_] = forecasts

        for idx in submission.index[missing_mask]:
            submission.loc[idx, :] = self.impute_missing_row(submission.loc[idx])

        return submission


def evaluate_quantum_reservoir_lstm(train_df: pd.DataFrame) -> tuple[BenchmarkResult, QuantumReservoirLSTMForecaster]:
    model = QuantumReservoirLSTMForecaster().fit(train_df)
    if model.validation_result_ is None:
        raise RuntimeError("Quantum reservoir LSTM did not produce validation metrics.")
    return model.validation_result_, model


def locate_default_dataset_dir(project_root: Path) -> Path:
    return project_root / "Quandela_folder" / "CHALLENGE RESOURCES" / "DATASETS"


def load_default_train_template(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_dir = locate_default_dataset_dir(project_root)
    train_df = pd.read_excel(dataset_dir / "train.xlsx")
    template_df = pd.read_excel(dataset_dir / "test_template.xlsx")
    return train_df, template_df


def evaluate_notebook_baseline(train_df: pd.DataFrame) -> BenchmarkResult:
    train_df = train_df.copy()
    train_df["Date"] = pd.to_datetime(train_df["Date"], dayfirst=True)
    train_df = train_df.sort_values("Date").reset_index(drop=True)
    surface_cols = [c for c in train_df.columns if c != "Date"]
    surface_values = train_df[surface_cols].to_numpy(dtype=float)

    scaler = StandardScaler()
    z_values = scaler.fit_transform(surface_values)
    k = min(12, surface_values.shape[1], max(2, surface_values.shape[0] // 20))
    pca = PCA(n_components=k, random_state=0)
    factors = pca.fit_transform(z_values)
    window = min(20, max(5, len(factors) // 8))

    features = []
    targets = []
    for t in range(window - 1, len(factors) - 1):
        features.append(factors[t - window + 1 : t + 1].reshape(-1))
        targets.append(factors[t + 1])
    X = np.vstack(features)
    y = np.vstack(targets)

    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    best_alpha = None
    best_surface = None
    best_factor = None
    for alpha in (1e-3, 1e-2, 1e-1, 1, 10, 100):
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        factor_mse = float(mean_squared_error(y_val, pred))
        pred_surface = scaler.inverse_transform(pca.inverse_transform(pred))
        true_surface = scaler.inverse_transform(pca.inverse_transform(y_val))
        surface_mse = float(mean_squared_error(true_surface, pred_surface))
        if best_surface is None or surface_mse < best_surface:
            best_alpha = alpha
            best_surface = surface_mse
            best_factor = factor_mse

    return BenchmarkResult(
        name="notebook_baseline",
        factor_mse=float(best_factor),
        surface_mse=float(best_surface),
        details=f"k={k}, window={window}, alpha={best_alpha}",
    )
