"""
chronos_adapter.py — Chronos-2 / Chronos-Bolt zero-shot adapter for electricity price forecasting.

Model names:
    Primary:   "chronos2_zero_shot"
    Fallback:  "chronos_bolt_zero_shot"

Priority:
    1. amazon/chronos-2-small
    2. amazon/chronos-bolt-small (fallback)
    3. amazon/chronos-bolt-base  (if resources allow, configurable)

Key design:
    - Zero-shot only (no training, no fine-tuning)
    - Default context window: 30 days × 24h = 720 points
    - Prediction length: 24 hours
    - Output: median (p50) as y_pred; optionally p10, p90 in debug
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Model registry ──
CHRONOS_MODELS = {
    "chronos2": [
        "amazon/chronos-2-small",
    ],
    "chronos_bolt": [
        "amazon/chronos-bolt-small",
        "amazon/chronos-bolt-base",
    ],
}

DEFAULT_CONTEXT_LENGTH = 720   # 30 days × 24h
PREDICTION_LENGTH = 24
QUANTILES = [0.1, 0.5, 0.9]

# ── Lazy imports ──
_CHRONOS_PIPELINE_AVAILABLE = False

try:
    import torch
    from chronos import ChronosPipeline, ChronosBoltPipeline
    _CHRONOS_PIPELINE_AVAILABLE = True
except ImportError:
    _CHRONOS_PIPELINE_AVAILABLE = False


class ChronosAdapter:
    """
    Chronos zero-shot adapter.

    Automatically tries Chronos-2 first; on failure, falls back to Chronos-Bolt.
    """

    def __init__(
        self,
        model_name: str = "chronos2_zero_shot",
        fallback_model_name: str = "chronos_bolt_zero_shot",
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        prediction_length: int = PREDICTION_LENGTH,
        device: Optional[str] = None,
        model_id_chronos2: str = "amazon/chronos-2-small",
        model_id_bolt: str = "amazon/chronos-bolt-small",
    ):
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.context_length = context_length
        self.prediction_length = prediction_length

        # Auto-detect device
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self.model_id_chronos2 = model_id_chronos2
        self.model_id_bolt = model_id_bolt

        self._pipeline = None
        self._loaded_model_id: Optional[str] = None
        self._is_fallback: bool = False
        self._fallback_reason: Optional[str] = None
        self._quantiles = QUANTILES

    # ── Properties ──

    @property
    def is_fallback(self) -> bool:
        return self._is_fallback

    @property
    def fallback_reason(self) -> Optional[str]:
        return self._fallback_reason

    @property
    def loaded_model_id(self) -> Optional[str]:
        return self._loaded_model_id

    # ── Load / Fallback logic ──

    def load(self) -> str:
        """
        Load the model. Tries Chronos-2 first, then Chronos-Bolt.

        Returns the model_name actually loaded.
        """
        # Try Chronos-2
        if not self._is_fallback:
            try:
                self._load_chronos2()
                self._loaded_model_id = self.model_id_chronos2
                logger.info(f"Chronos-2 loaded: {self.model_id_chronos2} on {self.device}")
                return self.model_name
            except Exception as e:
                reason = f"Chronos-2 failed: {type(e).__name__}: {e}"
                logger.warning(reason)
                self._fallback_reason = reason
                self._is_fallback = True

        # Fallback to Chronos-Bolt
        if self._is_fallback:
            try:
                self._load_chronos_bolt()
                self._loaded_model_id = self.model_id_bolt
                logger.info(f"Chronos-Bolt loaded: {self.model_id_bolt} on {self.device}")
                return self.fallback_model_name
            except Exception as e:
                reason = f"Chronos-Bolt also failed: {type(e).__name__}: {e}"
                logger.error(reason)
                self._fallback_reason = reason
                raise RuntimeError(f"Both Chronos-2 and Chronos-Bolt failed. Last error: {e}")

        return self.model_name

    def _load_chronos2(self):
        """Load Chronos-2 pipeline directly via HuggingFace."""
        if not _CHRONOS_PIPELINE_AVAILABLE:
            raise ImportError("chronos-forecasting not installed. Run: pip install chronos-forecasting")

        self._pipeline = ChronosPipeline.from_pretrained(
            self.model_id_chronos2,
            device_map=self.device if self.device == "cpu" else "auto",
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
        )

    def _load_chronos_bolt(self):
        """Load Chronos-Bolt pipeline via HuggingFace (same package)."""
        if not _CHRONOS_PIPELINE_AVAILABLE:
            raise ImportError("chronos-forecasting not installed. Run: pip install chronos-forecasting")

        self._pipeline = ChronosBoltPipeline.from_pretrained(
            self.model_id_bolt,
            device_map=self.device if self.device == "cpu" else "auto",
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
        )

    # ── Prediction ──

    def predict_context(
        self,
        context_values: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Predict next 24 hours given context_values (array of historical prices).

        Robustly handles multiple output shapes:
          - (batch, quantile, horizon) → pick p50 quantile
          - (batch, sample, horizon)   → median over samples
          - (batch, horizon)           → direct as p50

        Always returns y_pred (p50); also returns p10/p50/p90 when available.
        """
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        context_tensor = torch.tensor(context_values, dtype=torch.float32).unsqueeze(0)

        forecast = self._pipeline.predict(
            context_tensor,
            prediction_length=self.prediction_length,
        )

        # ── Robust tensor → numpy ──
        if hasattr(forecast, "detach"):
            forecast_np = forecast.squeeze(0).detach().cpu().numpy()
        else:
            forecast_np = np.asarray(forecast).squeeze(0)

        # ── Handle multiple output shapes ──
        result: dict[str, np.ndarray] = {}

        if forecast_np.ndim == 2:
            n_dim1 = forecast_np.shape[0]
            n_dim2 = forecast_np.shape[1]

            if n_dim2 == self.prediction_length:
                # shape (quantile|sample, horizon)
                # Try to identify if this is quantiles (9) or samples
                if n_dim1 == 9:
                    # Chronos-Bolt: 9 quantiles [0.1..0.9]
                    result["y_pred"] = forecast_np[4]       # p50 at index 4
                    result["y_pred_p10"] = forecast_np[0]
                    result["y_pred_p50"] = forecast_np[4]
                    result["y_pred_p90"] = forecast_np[8]
                else:
                    # Unknown dim1 → median across all
                    result["y_pred"] = np.median(forecast_np, axis=0)
                    result["y_pred_p10"] = np.percentile(forecast_np, 10, axis=0)
                    result["y_pred_p50"] = np.median(forecast_np, axis=0)
                    result["y_pred_p90"] = np.percentile(forecast_np, 90, axis=0)
        elif forecast_np.ndim == 1 and len(forecast_np) == self.prediction_length:
            # direct (horizon,) output
            result["y_pred"] = forecast_np
            result["y_pred_p50"] = forecast_np
        else:
            # fallback: just use as-is
            result["y_pred"] = forecast_np.ravel()

        return result

    def predict_day(
        self,
        full_df: pd.DataFrame,
        target_date: str,
        task: str = "dayahead",
        y_col: str = "y",
    ) -> pd.DataFrame:
        """
        Predict all 24 business hours for a given target day using zero-shot Chronos.

        Steps:
        1. Extract context window (default 720 points) ending before target_date 01:00
        2. Run Chronos inference to get 24-point forecast
        3. Build long-table output

        Parameters
        ----------
        full_df : full DataFrame with 'ds' and y_col (historical data)
        target_date : str YYYY-MM-DD
        task : "dayahead" or "realtime"
        y_col : name of the target price column

        Returns
        -------
        DataFrame with 24 rows, standard long-table format.
        """
        from ..common.output_schema import make_long_table
        from ..common.business_time import build_business_hour_grid

        day_dt = pd.Timestamp(target_date)

        # Context: up to context_length hours before target_date 01:00
        context_cutoff = day_dt + pd.Timedelta(hours=1)  # target day 01:00
        context_df = full_df[full_df["ds"] < context_cutoff].copy()
        if len(context_df) == 0:
            raise ValueError(f"No context data before {context_cutoff}")

        # Take the last context_length points
        context_values = context_df[y_col].values[-self.context_length:]
        if len(context_values) < self.context_length:
            logger.warning(
                f"Context window shorter than {self.context_length}: got {len(context_values)}. Padding."
            )
            # Pad with first value
            pad_width = self.context_length - len(context_values)
            context_values = np.pad(
                context_values, (pad_width, 0), mode="edge"
            )

        # Run inference
        result_dict = self.predict_context(context_values)

        # Build 24-row grid
        grid = build_business_hour_grid(target_date, task)
        grid["y_pred"] = result_dict["y_pred"]

        # Merge y_true if available
        target_ds_start = day_dt + pd.Timedelta(hours=1)
        target_ds_end = day_dt + pd.Timedelta(hours=24)
        truth = full_df[
            (full_df["ds"] >= target_ds_start)
            & (full_df["ds"] <= target_ds_end)
        ][["ds", y_col]].copy()
        truth.rename(columns={y_col: "y_true"}, inplace=True)
        grid = grid.merge(truth, on="ds", how="left")

        # Build long table
        result = make_long_table(
            grid,
            model_name=self.model_name if not self._is_fallback else self.fallback_model_name,
            task=task,
        )

        # Attach debug quantile columns
        if "y_pred_p10" in result_dict:
            result["y_pred_p10"] = result_dict["y_pred_p10"]
        if "y_pred_p90" in result_dict:
            result["y_pred_p90"] = result_dict["y_pred_p90"]

        if len(result) != 24:
            logger.warning(
                f"chronos: {target_date} {task} — got {len(result)} rows, expected 24"
            )
        return result

    # ── Get fallback status as dict ──

    def get_manifest(self) -> dict:
        return {
            "model_name": self.model_name,
            "fallback_model_name": self.fallback_model_name,
            "loaded_model_id": self._loaded_model_id,
            "is_fallback": self._is_fallback,
            "fallback_reason": self._fallback_reason,
            "device": self.device,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
        }
