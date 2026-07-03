"""
target_transform.py — Utility for target transformation (asinh, clip_low50, none).

Used by tuned day-ahead models to handle negative prices and spikes.
"""

from __future__ import annotations

import numpy as np


def apply_transform(y: np.ndarray, method: str = "none", scale: float = 100.0) -> np.ndarray:
    """
    Apply target transform.

    Methods:
      - "none":   y_out = y
      - "asinh":  y_out = arcsinh(y / scale), invert via sinh(y * scale)
      - "clip_low50": y_out = max(y, 50), invert by keeping pred if pred >= 50,
                      otherwise clip back to 50 (only for inference)
    """
    if method == "none":
        return y.copy()
    elif method == "asinh":
        return np.arcsinh(y / scale)
    elif method == "clip_low50":
        return np.where(y < 50.0, 50.0, y)
    else:
        raise ValueError(f"Unknown transform method: {method}")


def invert_transform(y_transformed: np.ndarray, method: str = "none", scale: float = 100.0) -> np.ndarray:
    """Invert target transform to get original price scale."""
    if method == "none":
        return y_transformed.copy()
    elif method == "asinh":
        return np.sinh(y_transformed) * scale
    elif method == "clip_low50":
        # clip_low50: during training we clipped y to 50;
        # during inference we don't clip the prediction (let model learn the clip)
        return y_transformed.copy()
    else:
        raise ValueError(f"Unknown transform method: {method}")
