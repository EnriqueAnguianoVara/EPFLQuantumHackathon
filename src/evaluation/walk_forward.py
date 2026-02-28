"""
Walk-forward validation for time series models.
"""

from typing import Callable, Dict, List

import numpy as np

from src.evaluation.metrics import all_metrics


def walk_forward_validation(
    data: np.ndarray,
    train_fn: Callable,
    predict_fn: Callable,
    initial_train_size: int = 300,
    step_size: int = 1,
    window_size: int = 20,
    horizon: int = 1,
) -> Dict:
    """
    Expanding-window walk-forward validation.

    Parameters
    ----------
    data : (N, D) — full dataset (normalized or PCA-reduced)
    train_fn : callable(X_train, Y_train) → model
        Trains a model on the given data.
    predict_fn : callable(model, X_input) → Y_pred
        Makes predictions with a trained model.
    initial_train_size : number of rows for first training set
    step_size : how many steps to advance the window each iteration
    window_size : lookback for input features
    horizon : prediction horizon

    Returns
    -------
    dict with:
        'predictions': list of (horizon, D) predictions
        'actuals': list of (horizon, D) actuals
        'metrics': dict of aggregated metrics
        'per_step_metrics': list of per-step metric dicts
    """
    N, D = data.shape
    all_preds = []
    all_actuals = []
    per_step = []

    t = initial_train_size
    while t + horizon <= N:
        # Build training data
        train_data = data[:t]

        # Build X, Y for training
        X_list, Y_list = [], []
        for i in range(window_size, len(train_data) - horizon + 1):
            X_list.append(train_data[i - window_size : i].flatten())
            Y_list.append(train_data[i : i + horizon].reshape(-1))

        if len(X_list) < 10:
            t += step_size
            continue

        X_train = np.array(X_list)
        Y_train = np.array(Y_list)

        # Train
        model = train_fn(X_train, Y_train)

        # Predict
        X_test = data[t - window_size : t].flatten().reshape(1, -1)
        Y_pred = predict_fn(model, X_test).reshape(horizon, D)
        Y_actual = data[t : t + horizon]

        all_preds.append(Y_pred)
        all_actuals.append(Y_actual)
        per_step.append(all_metrics(Y_actual, Y_pred))

        t += step_size

    # Aggregate
    preds_concat = np.concatenate(all_preds, axis=0)
    actuals_concat = np.concatenate(all_actuals, axis=0)
    agg_metrics = all_metrics(actuals_concat, preds_concat)

    return {
        "predictions": all_preds,
        "actuals": all_actuals,
        "metrics": agg_metrics,
        "per_step_metrics": per_step,
        "n_steps": len(all_preds),
    }
