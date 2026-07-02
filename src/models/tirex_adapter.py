"""
tirex_adapter.py — TiRex zero-shot adapter for electricity price forecasting.

Model name: "tirex_zero_shot"

Key design:
    - Zero-shot only (no training)
    - Uses TiRex (xLSTM-based foundation model) for time series forecasting
    - Default context window: 30 days × 24h = 720 points
    - Prediction length: 24 hours
    - If TiRex is unavailable, records failure and skips gracefully
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_resize(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Resize array to target_len, truncating or repeating as needed."""
    arr = np.asarray(arr).ravel()
    if len(arr) == target_len:
        return arr
    if len(arr) == 1:
        return np.full(target_len, arr[0])
    return np.resize(arr, target_len)


# ── Lazy import ──
_TIREX_AVAILABLE = False
_TIREX_ERROR: Optional[str] = None

try:
    import torch
    from tirex import load_model, ForecastModel
    _TIREX_AVAILABLE = True
except ImportError as e:
    _TIREX_ERROR = f"TiRex import failed: {type(e).__name__}: {e}"

DEFAULT_CONTEXT_LENGTH = 720
PREDICTION_LENGTH = 24


class TiRexAdapter:
    """
    TiRex zero-shot adapter.

    Uses TiRex foundation model for zero-shot time series forecasting.
    If TiRex is unavailable, records the reason and allows graceful degradation.
    """

    def __init__(
        self,
        model_name: str = "tirex_zero_shot",
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        prediction_length: int = PREDICTION_LENGTH,
        device: Optional[str] = None,
        hf_model_id: str = "NX-AI/TiRex",
    ):
        self.model_name = model_name
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.hf_model_id = hf_model_id

        # Auto-detect device
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self._model: Optional["ForecastModel"] = None
        self._unavailable_reason: Optional[str] = _TIREX_ERROR
        self._loaded: bool = False

    @property
    def is_available(self) -> bool:
        return _TIREX_AVAILABLE and self._loaded

    @property
    def unavailable_reason(self) -> Optional[str]:
        return self._unavailable_reason

    # ── Load ──

    def load(self) -> bool:
        """
        Load TiRex model from HuggingFace.

        Returns True if loaded successfully, False if unavailable.
        """
        if not _TIREX_AVAILABLE:
            self._unavailable_reason = _TIREX_ERROR
            logger.warning(f"TiRex not available: {self._unavailable_reason}")
            return False

        try:
            logger.info(f"Loading TiRex from {self.hf_model_id}...")
            self._model = load_model(self.hf_model_id)
            self._model.to(self.device)
            self._loaded = True
            logger.info(f"TiRex loaded on {self.device}")
            return True
        except Exception as e:
            self._unavailable_reason = f"TiRex load failed: {type(e).__name__}: {e}"
            logger.warning(self._unavailable_reason)
            return False

    # ── Prediction ──

    def predict_context(
        self,
        context_values: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Predict next 24 hours given context_values.

        Parameters
        ----------
        context_values : np.ndarray of shape (context_length,)

        Returns
        -------
        dict with y_pred (mean), and quantiles if available.
        """
        if not self.is_available:
            raise RuntimeError(f"TiRex not available: {self._unavailable_reason}")

        import torch

        context_tensor = torch.tensor(context_values, dtype=torch.float32).unsqueeze(0)  # (1, ctx_len)
        context_tensor = context_tensor.to(self.device)

        # TiRex returns (quantiles, mean)
        quantiles, mean = self._model.forecast(
            context=context_tensor,
            prediction_length=self.prediction_length,
        )

        # mean: (batch, horizon)
        mean_np = mean.squeeze(0).detach().cpu().numpy() if hasattr(mean, "detach") else np.asarray(mean).squeeze(0)
        if mean_np.ndim == 0:
            mean_np = np.full(self.prediction_length, float(mean_np))
        elif len(mean_np) != self.prediction_length:
            logger.warning(f"TiRex mean unexpected length {len(mean_np)}, truncating/padding to {self.prediction_length}")
            mean_np = np.resize(mean_np, self.prediction_length)

        # quantiles: (batch, num_quantiles, horizon) or (batch, horizon)
        result = {"y_pred": mean_np}

        if hasattr(quantiles, "detach"):
            q_np = quantiles.detach().cpu().numpy()
        else:
            q_np = np.asarray(quantiles)

        # Squeeze batch dim
        if q_np.ndim == 3 and q_np.shape[0] == 1:
            q_np = q_np.squeeze(0)  # (num_quantiles, horizon)

        if q_np.ndim == 2:
            n_q, h_q = q_np.shape
            logger.info(f"TiRex quantiles shape: ({n_q}, {h_q})")
            if h_q != self.prediction_length:
                logger.warning(f"TiRex quantile horizon {h_q} != prediction_length {self.prediction_length}")
            # Try to extract p10, p50, p90 by position
            if n_q >= 9:
                # Standard 9 quantiles [0.1..0.9]
                result["y_pred_p10"] = _safe_resize(q_np[0], self.prediction_length)
                result["y_pred_p50"] = _safe_resize(q_np[4], self.prediction_length)
                result["y_pred_p90"] = _safe_resize(q_np[8], self.prediction_length)
            elif n_q >= 3:
                result["y_pred_p10"] = _safe_resize(q_np[0], self.prediction_length)
                result["y_pred_p50"] = _safe_resize(q_np[n_q // 2], self.prediction_length)
                result["y_pred_p90"] = _safe_resize(q_np[-1], self.prediction_length)
        elif q_np.ndim == 1 and len(q_np) >= 3:
            # (num_quantiles,) — scalar per quantile, broadcast to horizon
            result["y_pred_p10"] = np.full(self.prediction_length, q_np[0])
            result["y_pred_p50"] = np.full(self.prediction_length, q_np[len(q_np) // 2])
            result["y_pred_p90"] = np.full(self.prediction_length, q_np[-1])

        return result

    def predict_day(
        self,
        full_df: pd.DataFrame,
        target_date: str,
        task: str = "dayahead",
        y_col: str = "y",
    ) -> pd.DataFrame:
        """
        Predict all 24 business hours for a given target day using zero-shot TiRex.
        """
        from ..common.output_schema import make_long_table
        from ..common.business_time import build_business_hour_grid

        if not self.is_available:
            raise RuntimeError(f"TiRex not available: {self._unavailable_reason}")

        day_dt = pd.Timestamp(target_date)

        # Context: up to context_length hours before target_date 01:00
        context_cutoff = day_dt + pd.Timedelta(hours=1)
        context_df = full_df[full_df["ds"] < context_cutoff].copy()
        if len(context_df) == 0:
            raise ValueError(f"No context data before {context_cutoff}")

        context_values = context_df[y_col].values[-self.context_length:]
        if len(context_values) < self.context_length:
            logger.warning(
                f"TiRex context window shorter than {self.context_length}: got {len(context_values)}. Padding."
            )
            pad_width = self.context_length - len(context_values)
            context_values = np.pad(context_values, (pad_width, 0), mode="edge")

        result_dict = self.predict_context(context_values)

        grid = build_business_hour_grid(target_date, task)
        grid["y_pred"] = result_dict["y_pred"]

        # Merge y_true
        target_ds_start = day_dt + pd.Timedelta(hours=1)
        target_ds_end = day_dt + pd.Timedelta(hours=24)
        truth = full_df[
            (full_df["ds"] >= target_ds_start) & (full_df["ds"] <= target_ds_end)
        ][["ds", y_col]].copy()
        truth.rename(columns={y_col: "y_true"}, inplace=True)
        grid = grid.merge(truth, on="ds", how="left")

        result = make_long_table(
            grid,
            model_name=self.model_name,
            task=task,
        )

        if "y_pred_p10" in result_dict:
            result["y_pred_p10"] = result_dict["y_pred_p10"]
        if "y_pred_p90" in result_dict:
            result["y_pred_p90"] = result_dict["y_pred_p90"]

        if len(result) != 24:
            logger.warning(
                f"tirex: {target_date} {task} — got {len(result)} rows, expected 24"
            )
        return result

    def get_manifest(self) -> dict:
        return {
            "model_name": self.model_name,
            "is_available": self.is_available,
            "unavailable_reason": self._unavailable_reason,
            "device": self.device,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "hf_model_id": self.hf_model_id,
        }
