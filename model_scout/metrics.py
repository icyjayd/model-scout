"""
metrics.py
Computes regression and classification metrics for Model Scout.
"""

import numpy as np
from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from scipy.stats import spearmanr
import warnings


def compute_metrics(task, y_true, y_pred, context=None):
    """Compute evaluation metrics for regression or classification tasks."""

    # --- Handle NaN-safe conversion ---
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if task == "regression":
        try:
            r2 = r2_score(y_true, y_pred)
            rmse = root_mean_squared_error(y_true, y_pred, squared=False)
        except Exception:
            r2, rmse = np.nan, np.nan

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            rho, p = spearmanr(y_true, y_pred)

        return {
            "r2": r2,
            "rmse": rmse,
            "rho": float(rho) if np.isfinite(rho) else np.nan,
            "p": float(p) if np.isfinite(p) else np.nan,
        }

    elif task == "classification":
        # convert predictions to discrete classes
        try:
            if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                y_class = np.argmax(y_pred, axis=1)
                y_prob = y_pred[:, 1] if y_pred.shape[1] > 1 else y_pred[:, 0]
            else:
                y_class = (y_pred >= 0.5).astype(int)
                y_prob = y_pred
        except Exception:
            y_class = np.round(y_pred).astype(int)
            y_prob = y_pred

        # compute conventional metrics
        try:
            acc = accuracy_score(y_true, y_class)
            f1 = f1_score(y_true, y_class, zero_division=0)
            prec = precision_score(y_true, y_class, zero_division=0)
            rec = recall_score(y_true, y_class, zero_division=0)
        except Exception:
            acc = f1 = prec = rec = np.nan

        # AUC (binary only)
        try:
            auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
        except Exception:
            auc = np.nan

        # Spearman correlation for consistency
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            rho, p = spearmanr(y_true, y_pred)

        return {
            "accuracy": acc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "auc": auc,
            "rho": float(rho) if np.isfinite(rho) else np.nan,
            "p": float(p) if np.isfinite(p) else np.nan,
        }

    else:
        raise ValueError(f"Unknown task type: {task}")
