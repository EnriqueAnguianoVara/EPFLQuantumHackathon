"""
XGBoost / LightGBM baseline for swaption price prediction.

Trains one model per PCA component (multi-output via separate regressors).
Operates in PCA space with flattened windows as input.
"""

from typing import Optional, Dict, List
import numpy as np

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


class GBTBaseline:
    """
    Gradient Boosted Trees baseline.

    Trains one regressor per output dimension (PCA component).
    Supports both XGBoost and LightGBM backends.

    Input:  (window_size * n_pca,) — flattened past PCA components
    Output: (n_pca,)               — predicted next-step PCA components
    """

    def __init__(
        self,
        backend: str = "xgboost",
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        random_state: int = 42,
    ):
        self.backend = backend
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state
        self.models: List = []
        self.n_outputs: int = 0
        self.is_fitted = False

    def _make_model(self):
        if self.backend == "xgboost":
            if not HAS_XGB:
                raise ImportError("xgboost not installed. pip install xgboost")
            return xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=self.random_state,
                verbosity=0,
                n_jobs=-1,
            )
        elif self.backend == "lightgbm":
            if not HAS_LGB:
                raise ImportError("lightgbm not installed. pip install lightgbm")
            return lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "GBTBaseline":
        """
        Fit one model per output dimension.

        Parameters
        ----------
        X : (n_samples, window_size * n_pca)
        Y : (n_samples, n_pca) — single-step targets only
        """
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        self.n_outputs = Y.shape[1]
        self.models = []

        for i in range(self.n_outputs):
            model = self._make_model()
            model.fit(X, Y[:, i])
            self.models.append(model)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict all output dimensions.

        Parameters
        ----------
        X : (n_samples, window_size * n_pca) or (window_size * n_pca,)

        Returns
        -------
        Y : (n_samples, n_pca)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        preds = np.zeros((X.shape[0], self.n_outputs))
        for i, model in enumerate(self.models):
            preds[:, i] = model.predict(X)
        return preds

    def get_params(self) -> Dict:
        return {
            "backend": self.backend,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_outputs": self.n_outputs,
        }

    def feature_importance(self) -> np.ndarray:
        """
        Return (n_outputs, n_features) array of feature importances.
        """
        importances = []
        for model in self.models:
            if self.backend == "xgboost":
                importances.append(model.feature_importances_)
            elif self.backend == "lightgbm":
                importances.append(model.feature_importances_)
        return np.array(importances)

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
        n_steps : future steps to predict
        window_size : lookback
        n_pca : number of PCA components

        Returns
        -------
        predictions : (n_steps, n_pca)
        """
        buffer = list(history_pca[-window_size:])
        predictions = []

        for _ in range(n_steps):
            x = np.array(buffer[-window_size:]).flatten().reshape(1, -1)
            y_pred = self.predict(x)[0]
            predictions.append(y_pred)
            buffer.append(y_pred)

        return np.array(predictions)
