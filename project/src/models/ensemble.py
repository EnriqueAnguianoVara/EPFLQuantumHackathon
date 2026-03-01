"""
Ensemble methods for combining predictions from multiple models.

Supports simple averaging, weighted averaging, and optimal weight selection.
"""

from typing import Dict, List, Optional
import numpy as np
from src.evaluation.metrics import all_metrics


class Ensemble:
    """
    Combine predictions from multiple models.

    Parameters
    ----------
    method : 'mean', 'weighted', or 'optimal'
        - 'mean': simple average
        - 'weighted': user-specified weights
        - 'optimal': find weights that minimize validation MAE
    weights : optional dict of {model_name: weight} for 'weighted' method
    """

    def __init__(
        self,
        method: str = "mean",
        weights: Optional[Dict[str, float]] = None,
    ):
        self.method = method
        self.weights = weights or {}
        self.model_names: List[str] = []
        self.is_fitted = False

    def fit(
        self,
        predictions: Dict[str, np.ndarray],
        actuals: np.ndarray,
    ) -> "Ensemble":
        """
        Fit ensemble weights (only needed for 'optimal' method).

        Parameters
        ----------
        predictions : {model_name: (N, D) predictions}
        actuals : (N, D) ground truth
        """
        self.model_names = list(predictions.keys())

        if self.method == "optimal":
            self.weights = self._find_optimal_weights(predictions, actuals)
        elif self.method == "mean":
            n = len(self.model_names)
            self.weights = {name: 1.0 / n for name in self.model_names}
        # 'weighted' uses user-provided weights

        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        self.is_fitted = True
        return self

    def predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine predictions using fitted weights.

        Parameters
        ----------
        predictions : {model_name: (N, D) or (D,) predictions}

        Returns
        -------
        combined : same shape as individual predictions
        """
        arrays = []
        weights = []

        for name in predictions:
            w = self.weights.get(name, 0.0)
            if w > 0:
                arrays.append(predictions[name])
                weights.append(w)

        if not arrays:
            # Fallback: equal weight all
            arrays = list(predictions.values())
            weights = [1.0 / len(arrays)] * len(arrays)

        result = np.zeros_like(arrays[0], dtype=np.float64)
        for arr, w in zip(arrays, weights):
            result += w * arr.astype(np.float64)

        return result

    def _find_optimal_weights(
        self,
        predictions: Dict[str, np.ndarray],
        actuals: np.ndarray,
        n_grid: int = 21,
    ) -> Dict[str, float]:
        """
        Grid search for optimal weights (works for 2-3 models).
        """
        names = list(predictions.keys())
        n_models = len(names)

        if n_models == 1:
            return {names[0]: 1.0}

        best_mae = float("inf")
        best_weights = {name: 1.0 / n_models for name in names}

        if n_models == 2:
            for w1 in np.linspace(0, 1, n_grid):
                w2 = 1.0 - w1
                combo = w1 * predictions[names[0]] + w2 * predictions[names[1]]
                mae_val = np.mean(np.abs(actuals - combo))
                if mae_val < best_mae:
                    best_mae = mae_val
                    best_weights = {names[0]: w1, names[1]: w2}

        elif n_models == 3:
            for w1 in np.linspace(0, 1, n_grid):
                for w2 in np.linspace(0, 1 - w1, n_grid):
                    w3 = 1.0 - w1 - w2
                    combo = (w1 * predictions[names[0]] +
                             w2 * predictions[names[1]] +
                             w3 * predictions[names[2]])
                    mae_val = np.mean(np.abs(actuals - combo))
                    if mae_val < best_mae:
                        best_mae = mae_val
                        best_weights = {names[0]: w1, names[1]: w2, names[2]: w3}
        else:
            # Fallback to equal weights for >3 models
            best_weights = {name: 1.0 / n_models for name in names}

        return best_weights

    def get_params(self) -> Dict:
        return {
            "method": self.method,
            "weights": self.weights,
            "model_names": self.model_names,
        }
