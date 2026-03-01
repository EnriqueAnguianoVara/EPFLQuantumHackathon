"""
Ridge Regression baseline for swaption price prediction.

Operates in PCA space: takes a flattened window of past PCA components
as input and predicts the next step's PCA components.
"""

from typing import Optional, Dict
import numpy as np
from sklearn.linear_model import Ridge


class RidgeBaseline:
    """
    Ridge Regression on flattened PCA windows.

    Input:  (window_size * n_pca,) — flattened past PCA components
    Output: (horizon * n_pca,)     — predicted future PCA components
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.is_fitted = False

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "RidgeBaseline":
        """
        Fit the model.

        Parameters
        ----------
        X : (n_samples, window_size * n_pca)
        Y : (n_samples, horizon * n_pca)
        """
        self.model.fit(X, Y)
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
        Y : (n_samples, horizon * n_pca)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X)

    def get_params(self) -> Dict:
        return {"alpha": self.alpha, "n_features": self.model.n_features_in_}

    def predict_rolling(
        self,
        history_pca: np.ndarray,
        n_steps: int,
        window_size: int,
        n_pca: int,
    ) -> np.ndarray:
        """
        Rolling multi-step prediction (for forecasting future days).

        Parameters
        ----------
        history_pca : (N, n_pca) — past PCA data (at least window_size rows)
        n_steps : how many steps into the future to predict
        window_size : lookback window
        n_pca : number of PCA components

        Returns
        -------
        predictions : (n_steps, n_pca)
        """
        buffer = list(history_pca[-window_size:])
        predictions = []

        for _ in range(n_steps):
            x = np.array(buffer[-window_size:]).flatten().reshape(1, -1)
            y_pred = self.model.predict(x)[0]  # (n_pca,) or (horizon*n_pca,)

            # Take only the first step if multi-horizon
            step_pred = y_pred[:n_pca]
            predictions.append(step_pred)
            buffer.append(step_pred)

        return np.array(predictions)
