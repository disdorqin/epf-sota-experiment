"""
pair_fusion.py — Pairwise model fusion for CatBoost + TabPFN.

Supports 3 fusion strategies:
1. simple_average        — 0.5 * (catboost + tabpfn)
2. inverse_smape_weight  — weight by inverse of past N-day sMAPE,
                               computed separately for each prediction day
                               using ONLY data from previous days (no leakage).
3. period_best           — for each (task, period, target_day), pick the
                               historically better model over the past N days.

All rolling computations use EXPANDING window: for prediction day T,
weights are computed from all available data with target_day < T,
up to N distinct days back. First N days fall back to simple average.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Helper: metrics (local, matching src/common/metrics.py) ──


def _smape_raw(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Raw sMAPE (not floored), used for weighting."""
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < 1e-8, 1e-8, denom)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)


def _align_catboost_tabpfn(
    cb_df: pd.DataFrame,
    tp_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align CatBoost and TabPFN predictions row-by-row.

    Merge key: ds (sufficient for hourly data); also uses task/target_day/period
    when available in both dataframes.
    Returns a merged DataFrame with _cb / _tp suffixes.
    """
    merge_keys = ["ds", "y_true"]
    for col in ["task", "target_day", "hour_business", "period"]:
        if col in cb_df.columns and col in tp_df.columns:
            merge_keys.append(col)
    merged = cb_df.merge(
        tp_df,
        on=merge_keys,
        suffixes=("_cb", "_tp"),
        how="inner",
    )
    if len(merged) == 0:
        logger.warning(f"Align: 0 rows after merge on {merge_keys}")
        logger.warning(f"  cb columns: {list(cb_df.columns)[:15]}")
        logger.warning(f"  tp columns: {list(tp_df.columns)[:15]}")
    return merged


# ── Rolling sMAPE computation (no future data) ──


def _compute_rolling_smape(
    merged: pd.DataFrame,
    past_days: int,
) -> pd.DataFrame:
    """
    For each prediction day T, compute the sMAPE of each model
    using ONLY data from previous `past_days` distinct target_days.

    Returns a DataFrame with one row per (task, period, target_day) group:
      target_day, task, period, cb_smape, tp_smape
    If a row has NaN sMAPE, the caller should fall back to simple average.
    """
    merged = merged.copy()
    merged["target_day_dt"] = pd.to_datetime(merged["target_day"])
    merged = merged.sort_values("target_day_dt").reset_index(drop=True)

    all_days = sorted(merged["target_day_dt"].unique())

    rows = []
    for day in all_days:
        # Past days for this `day`: all distinct target_days < day, up to `past_days`
        past_days_list = [d for d in all_days if d < day]
        if len(past_days_list) > past_days:
            past_days_list = past_days_list[-past_days:]
        past_mask = merged["target_day_dt"].isin(past_days_list)

        if past_mask.sum() == 0:
            # No history yet → record NaN weights for all (task, period) on this day
            for task_val in merged["task"].unique():
                if "period" in merged.columns:
                    pers = merged.loc[
                        (merged["task"] == task_val) & (merged["target_day_dt"] == day),
                        "period"
                    ].unique()
                    if len(pers) == 0:
                        pers = [None]
                else:
                    pers = [None]
                for period_val in pers:
                    rows.append({
                        "target_day_dt": day,
                        "task": task_val,
                        "period": period_val,
                        "cb_smape": np.nan,
                        "tp_smape": np.nan,
                    })
            continue

        # Compute per (task, period)
        for task_val in merged["task"].unique():
            task_past = past_mask & (merged["task"] == task_val)
            if task_past.sum() == 0:
                continue
            if "period" in merged.columns:
                periods = merged.loc[task_past, "period"].unique()
            else:
                periods = [None]
            for period_val in periods:
                if period_val is not None:
                    group_mask = task_past & (merged["period"] == period_val)
                else:
                    group_mask = task_past
                if group_mask.sum() < 2:
                    cb_smape = np.nan
                    tp_smape = np.nan
                else:
                    y_true_g = merged.loc[group_mask, "y_true"].values
                    y_pred_cb_g = merged.loc[group_mask, "y_pred_cb"].values
                    y_pred_tp_g = merged.loc[group_mask, "y_pred_tp"].values
                    valid = ~(
                        np.isnan(y_true_g) | np.isnan(y_pred_cb_g) | np.isnan(y_pred_tp_g)
                    )
                    if valid.sum() < 2:
                        cb_smape = np.nan
                        tp_smape = np.nan
                    else:
                        cb_smape = _smape_raw(y_true_g[valid], y_pred_cb_g[valid])
                        tp_smape = _smape_raw(y_true_g[valid], y_pred_tp_g[valid])
                rows.append({
                    "target_day_dt": day,
                    "task": task_val,
                    "period": period_val,
                    "cb_smape": cb_smape,
                    "tp_smape": tp_smape,
                })

    result = pd.DataFrame(rows)
    if "target_day_dt" in result.columns:
        result["target_day"] = result["target_day_dt"].dt.strftime("%Y-%m-%d")
    return result


def _add_weights_to_merged(
    merged: pd.DataFrame,
    rolling: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge rolling sMAPE weights onto merged DataFrame.
    Adds columns: w_cb, w_tp (weights, sum=1.0; NaN = fallback to simple avg).
    """
    merge_cols = ["target_day", "task"]
    if "period" in merged.columns and "period" in rolling.columns:
        merge_cols.append("period")
    # Ensure target_day is string in both
    merged = merged.copy()
    merged["target_day"] = pd.to_datetime(merged["target_day"]).dt.strftime("%Y-%m-%d")
    rolling = rolling.copy()
    if "target_day_dt" in rolling.columns and "target_day" not in rolling.columns:
        rolling["target_day"] = rolling["target_day_dt"].dt.strftime("%Y-%m-%d")

    merged_w = merged.merge(
        rolling[merge_cols + ["cb_smape", "tp_smape"]],
        on=merge_cols,
        how="left",
    )
    # Compute weights (vectorized)
    cb_s = merged_w["cb_smape"].values
    tp_s = merged_w["tp_smape"].values
    w_cb = np.full(len(merged_w), 0.5)
    w_tp = np.full(len(merged_w), 0.5)
    valid = ~(np.isnan(cb_s) | np.isnan(tp_s)) & (cb_s > 0) & (tp_s > 0)
    if valid.sum() > 0:
        inv_cb = 1.0 / cb_s[valid]
        inv_tp = 1.0 / tp_s[valid]
        total = inv_cb + inv_tp
        w_cb[valid] = inv_cb / total
        w_tp[valid] = inv_tp / total
    merged_w["w_cb"] = w_cb
    merged_w["w_tp"] = w_tp
    return merged_w


# ── Fusion method 1: simple_average ──


def simple_average(
    cb_df: pd.DataFrame,
    tp_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simple average: y_pred = 0.5 * catboost + 0.5 * tabpfn.
    """
    merged = _align_catboost_tabpfn(cb_df, tp_df)
    if len(merged) == 0:
        logger.error("simple_average: 0 aligned rows, returning empty")
        result = cb_df.copy()
        result["y_pred"] = np.nan
        result["model_name"] = "fused_simple_average"
        return result

    result = merged.copy()
    result["y_pred"] = 0.5 * merged["y_pred_cb"].values + 0.5 * merged["y_pred_tp"].values
    # Remove _cb / _tp duplicate columns
    result = result.loc[:, ~result.columns.str.endswith(("_cb", "_tp"))]
    # If y_pred_cb / y_pred_tp remain as non-suffix columns, drop them
    for drop_col in ["y_pred_cb", "y_pred_tp"]:
        if drop_col in result.columns:
            result = result.drop(columns=[drop_col])
    result["model_name"] = "fused_simple_average"
    return result


# ── Fusion method 2: inverse_smape_weight ──


def inverse_smape_weight(
    cb_df: pd.DataFrame,
    tp_df: pd.DataFrame,
    past_days: int = 7,
) -> pd.DataFrame:
    """
    Weight predictions by inverse of past N-day sMAPE.

    For each prediction day T, compute sMAPE using only data from days < T
    (up to `past_days` distinct days back). No future data leakage.

    Returns a DataFrame with columns from cb_df plus fused y_pred.
    """
    merged = _align_catboost_tabpfn(cb_df, tp_df)
    if len(merged) == 0:
        logger.error("inverse_smape_weight: 0 aligned rows, returning empty")
        result = cb_df.copy()
        result["y_pred"] = np.nan
        result["model_name"] = "fused_inverse_smape_weight"
        return result

    rolling = _compute_rolling_smape(merged, past_days)
    merged_w = _add_weights_to_merged(merged, rolling)

    merged_w["y_pred"] = (
        merged_w["w_cb"] * merged_w["y_pred_cb"]
        + merged_w["w_tp"] * merged_w["y_pred_tp"]
    )

    result = merged_w.copy()
    result = result.loc[:, ~result.columns.str.endswith(("_cb", "_tp"))]
    for drop_col in ["y_pred_cb", "y_pred_tp", "w_cb", "w_tp", "cb_smape", "tp_smape"]:
        if drop_col in result.columns:
            result = result.drop(columns=[drop_col])
    result["model_name"] = "fused_inverse_smape_weight"
    return result


# ── Fusion method 3: period_best ──


def period_best(
    cb_df: pd.DataFrame,
    tp_df: pd.DataFrame,
    past_days: int = 7,
) -> pd.DataFrame:
    """
    For each (task, period, target_day), pick the historically better model
    over the past N days (no future data).

    Returns a DataFrame with columns from cb_df plus fused y_pred.
    """
    merged = _align_catboost_tabpfn(cb_df, tp_df)
    if len(merged) == 0:
        logger.error("period_best: 0 aligned rows, returning empty")
        result = cb_df.copy()
        result["y_pred"] = np.nan
        result["model_name"] = "fused_period_best"
        return result

    rolling = _compute_rolling_smape(merged, past_days)

    merge_cols = ["target_day", "task"]
    if "period" in merged.columns and "period" in rolling.columns:
        merge_cols.append("period")

    merged["target_day"] = pd.to_datetime(merged["target_day"]).dt.strftime("%Y-%m-%d")
    rolling = rolling.copy()
    if "target_day_dt" in rolling.columns and "target_day" not in rolling.columns:
        rolling["target_day"] = rolling["target_day_dt"].dt.strftime("%Y-%m-%d")

    merged_w = merged.merge(
        rolling[merge_cols + ["cb_smape", "tp_smape"]],
        on=merge_cols,
        how="left",
    )

    # Pick better model (vectorized)
    cb_better = (
        merged_w["cb_smape"].notna()
        & merged_w["tp_smape"].notna()
        & (merged_w["cb_smape"] <= merged_w["tp_smape"])
    )
    fallback = merged_w["cb_smape"].isna() | merged_w["tp_smape"].isna()

    y_pred = np.where(
        fallback,
        0.5 * merged_w["y_pred_cb"] + 0.5 * merged_w["y_pred_tp"],
        np.where(
            cb_better,
            merged_w["y_pred_cb"],
            merged_w["y_pred_tp"],
        ),
    )
    merged_w["y_pred"] = y_pred

    result = merged_w.copy()
    result = result.loc[:, ~result.columns.str.endswith(("_cb", "_tp"))]
    for drop_col in ["y_pred_cb", "y_pred_tp", "cb_smape", "tp_smape"]:
        if drop_col in result.columns:
            result = result.drop(columns=[drop_col])
    result["model_name"] = "fused_period_best"
    return result


# ── Load predictions from directory ──


def load_model_predictions(pred_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Load prediction CSVs from a directory.

    Expected filenames:
      catboost_sota_dayahead.csv, catboost_sota_realtime.csv,
      tabpfn_ts_sota_dayahead.csv, tabpfn_ts_sota_realtime.csv, ...
    """
    results = {}
    if not pred_dir.exists():
        logger.warning(f"Prediction directory not found: {pred_dir}")
        return results
    for csv_path in sorted(pred_dir.glob("*.csv")):
        name = csv_path.stem
        try:
            df = pd.read_csv(str(csv_path), encoding="utf-8-sig")
            results[name] = df
            logger.info(f"  Loaded {name}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"  Failed to load {csv_path}: {e}")
    return results
