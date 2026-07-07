
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-12
    return float(1 - ss_res / ss_tot)


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def slice_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute metrics on canonical H sub-ranges (Phase 8.2)."""
    slices = {
        "low_memory (H<0.4)": y_true < 0.4,
        "brownian (0.4<=H<0.6)": (y_true >= 0.4) & (y_true < 0.6),
        "persistent (H>=0.6)": y_true >= 0.6,
    }
    results = {}
    for name, mask in slices.items():
        if mask.sum() == 0:
            results[name] = None
            continue
        results[name] = summarize(y_true[mask], y_pred[mask])
        results[name]["n"] = int(mask.sum())
    return results
