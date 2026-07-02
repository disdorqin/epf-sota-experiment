"""
metrics.py — Evaluation metrics, matching fusion/metrics.py exactly.

Primary metric: smape_floor50 (floor at 50 for both y_true and y_pred).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def smape_floor50(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """
    sMAPE with floor at 50 for both true and predicted values.
    Matches fusion/metrics.py:smape_floor50 EXACTLY.
    """
    true_clip = np.where(y_true < 50.0, 50.0, y_true)
    pred_clip = np.where(y_pred < 50.0, 50.0, y_pred)
    denom = (np.abs(true_clip) + np.abs(pred_clip)) / 2.0
    denom = np.where(denom < eps, eps, denom)
    return float(np.mean(np.abs(pred_clip - true_clip) / denom) * 100.0)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def peak_mae_q90(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE on the top 10% highest-price hours (by true value)."""
    threshold = np.quantile(y_true, 0.9)
    mask = y_true >= threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def negative_price_hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    If true values include negatives, return the % of those hours
    where prediction also captured negativity (pred < 0).
    If no negative true values, return NaN.
    """
    neg_mask = y_true < 0
    if neg_mask.sum() == 0:
        return float("nan")
    return float((y_pred[neg_mask] < 0).sum() / neg_mask.sum() * 100.0)


def high_spike_mae_q90(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Same as peak_mae_q90 — MAE on top-decile true values."""
    return peak_mae_q90(y_true, y_pred)


def compute_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute all standard metrics and return as dict."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE_floor50": smape_floor50(y_true, y_pred),
        "peak_MAE_q90": peak_mae_q90(y_true, y_pred),
        "negative_price_hit_rate": negative_price_hit_rate(y_true, y_pred),
        "high_spike_MAE_q90": high_spike_mae_q90(y_true, y_pred),
    }
