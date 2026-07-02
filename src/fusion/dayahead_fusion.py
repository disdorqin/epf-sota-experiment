"""
dayahead_fusion.py — Day-ahead fusion methods, building on pair_fusion.py.

Adds 5 fusion strategies specific to day-ahead:
1. simple_average          — 0.5 * (catboost + tabpfn)  [reuse pair_fusion]
2. inverse_smape_period    — weight by inverse sMAPE per period (not per task+period)
3. inverse_smape_hour      — weight by inverse sMAPE per hour_business
4. winner_by_period        — per period, pick historically better model
5. winner_by_hour          — per hour_business, pick historically better model
6. ridge_stacking          — Ridge regression on past 7 days' predictions

All rolling computations use EXPANDING window with max `past_days` history,
computed SEPARATELY for each prediction day using ONLY earlier data (no leakage).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.fusion.pair_fusion import (
    _add_weights_to_merged,
    _align_catboost_tabpfn,
    _compute_rolling_smape,
    _smape_raw,
)

logger = logging.getLogger(__name__)

# ── Helper: compute rolling sMAPE grouped by period only ────────────────────────


def _compute_rolling_smape_by_period(
    merged: pd.DataFrame,
    past_days: int,
) -> pd.DataFrame:
    """
    For each prediction day T, compute sMAPE per (period) using only
    data from previous `past_days` distinct target_days.

    Returns DataFrame: target_day, period, cb_smape, tp_smape
    """
    merged = merged.copy()
    merged["target_day_dt"] = pd.to_datetime(merged["target_day"])
    merged = merged.sort_values("target_day_dt").reset_index(drop=True)
    all_days = sorted(merged["target_day_dt"].unique())

    rows = []
    for day in all_days:
        past_days_list = [d for d in all_days if d < day]
        if len(past_days_list) > past_days:
            past_days_list = past_days_list[-past_days:]
        past_mask = merged["target_day_dt"].isin(past_days_list)

        if past_mask.sum() == 0:
            pers = merged.loc[merged["target_day_dt"] == day, "period"].unique()
            for p in pers:
                rows.append({"target_day_dt": day, "period": p, "cb_smape": np.nan, "tp_smape": np.nan})
            continue

        for period_val in merged.loc[past_mask, "period"].unique():
            group_mask = past_mask & (merged["period"] == period_val)
            if group_mask.sum() < 2:
                cb_smape = np.nan
                tp_smape = np.nan
            else:
                y_true_g = merged.loc[group_mask, "y_true"].values
                y_pred_cb_g = merged.loc[group_mask, "y_pred_cb"].values
                y_pred_tp_g = merged.loc[group_mask, "y_pred_tp"].values
                valid = ~(np.isnan(y_true_g) | np.isnan(y_pred_cb_g) | np.isnan(y_pred_tp_g))
                if valid.sum() < 2:
                    cb_smape = np.nan
                    tp_smape = np.nan
                else:
                    cb_smape = _smape_raw(y_true_g[valid], y_pred_cb_g[valid])
                    tp_smape = _smape_raw(y_true_g[valid], y_pred_tp_g[valid])
            rows.append({"target_day_dt": day, "period": period_val, "cb_smape": cb_smape, "tp_smape": tp_smape})

    result = pd.DataFrame(rows)
    if "target_day_dt" in result.columns:
        result["target_day"] = result["target_day_dt"].dt.strftime("%Y-%m-%d")
    return result


# ── Helper: compute rolling sMAPE grouped by hour_business ─────────────────────


def _compute_rolling_smape_by_hour(
    merged: pd.DataFrame,
    past_days: int,
) -> pd.DataFrame:
    """
    For each prediction day T, compute sMAPE per (hour_business) using only
    data from previous `past_days` distinct target_days.

    Returns DataFrame: target_day, hour_business, cb_smape, tp_smape
    """
    merged = merged.copy()
    merged["target_day_dt"] = pd.to_datetime(merged["target_day"])
    merged = merged.sort_values("target_day_dt").reset_index(drop=True)
    all_days = sorted(merged["target_day_dt"].unique())

    rows = []
    for day in all_days:
        past_days_list = [d for d in all_days if d < day]
        if len(past_days_list) > past_days:
            past_days_list = past_days_list[-past_days:]
        past_mask = merged["target_day_dt"].isin(past_days_list)

        if past_mask.sum() == 0:
            hours = merged.loc[merged["target_day_dt"] == day, "hour_business"].unique()
            for h in hours:
                rows.append({"target_day_dt": day, "hour_business": h, "cb_smape": np.nan, "tp_smape": np.nan})
            continue

        for hour_val in merged.loc[past_mask, "hour_business"].unique():
            group_mask = past_mask & (merged["hour_business"] == hour_val)
            if group_mask.sum() < 2:
                cb_smape = np.nan
                tp_smape = np.nan
            else:
                y_true_g = merged.loc[group_mask, "y_true"].values
                y_pred_cb_g = merged.loc[group_mask, "y_pred_cb"].values
                y_pred_tp_g = merged.loc[group_mask, "y_pred_tp"].values
                valid = ~(np.isnan(y_true_g) | np.isnan(y_pred_cb_g) | np.isnan(y_pred_tp_g))
                if valid.sum() < 2:
                    cb_smape = np.nan
                    tp_smape = np.nan
                else:
                    cb_smape = _smape_raw(y_true_g[valid], y_pred_cb_g[valid])
                    tp_smape = _smape_raw(y_true_g[valid], y_pred_tp_g[valid])
            rows.append({"target_day_dt": day, "hour_business": hour_val, "cb_smape": cb_smape, "tp_smape": tp_smape})

    result = pd.DataFrame(rows)
    if "target_day_dt" in result.columns:
        result["target_day"] = result["target_day_dt"].dt.strftime("%Y-%m-%d")
    return result


# ── Fusion method: inverse_smape_period ─────────────────────────────────────────


def inverse_smape_period(
    cb_df: pd.DataFrame,
    tp_df: pd.DataFrame,
    past_days: int = 7,
) -> pd.DataFrame:
    """
    Weight by inverse sMAPE, computed per period (not per task+period).

    For each prediction day T, compute sMAPE per period using only
    data from days < T (up to `past_days` back). No future leakage.

    Returns DataFrame with fused y_pred and model_name='fused_inverse_smape_period_dayahead'.
    """
    merged = _align_catboost_tabpfn(cb_df, tp_df)
    if len(merged) == 0:
        logger.error("inverse_smape_period: 0 aligned rows")
        result = cb_df.copy()
        result["y_pred"] = np.nan
        result["model_name"] = "fused_inverse_smape_period_dayahead"
        return result

    rolling = _compute_rolling_smape_by_period(merged, past_days)

    # Merge weights onto merged
    merged = merged.copy()
    merged["target_day"] = pd.to_datetime(merged["target_day"]).dt.strftime("%Y-%m-%d")
    rolling = rolling.copy()
    if "target_day" not in rolling.columns and "target_day_dt" in rolling.columns:
        rolling["target_day"] = rolling["target_day_dt"].dt.strftime("%Y-%m-%d")

    merged_w = merged.merge(
        rolling[["target_day", "period", "cb_smape", "tp_smape"]],
        on=["target_day", "period"],
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

    merged_w["y_pred"] = w_cb * merged_w["y_pred_cb"] + w_tp * merged_w["y_pred_tp"]

    result = merged_w.copy()
    result = result.loc[:, ~result.columns.str.endswith(("_cb", "_tp"))]
    for drop_col in ["y_pred_cb", "y_pred_tp", "cb_smape", "tp_smape"]:
        if drop_col in result.columns:
            result = result.drop(columns=[drop_col])
    result["model_name"] = "fused_inverse_smape_period_dayahead"
    return result


# ── Fusion method: inverse_smape_hour ───────────────────────────────────────────


def inverse_smape_hour(
    cb_df: pd.DataFrame,
    tp_df: pd.DataFrame,
    past_days: int = 7,
) -> pd.DataFrame:
    """
    Weight by inverse sMAPE, computed per hour_business.

    For each prediction day T, compute sMAPE per hour_business using only
    data from days < T (up to `past_days` back). No future leakage.

    Returns DataFrame with fused y_pred and model_name='fused_inverse_smape_hour_dayahead'.
    """
    merged = _align_catboost_tabpfn(cb_df, tp_df)
    if len(merged) == 0:
        logger.error("inverse_smape_hour: 0 aligned rows")
        result = cb_df.copy()
        result["y_pred"] = np.nan
        result["model_name"] = "fused_inverse_smape_hour_dayahead"
        return result

    rolling = _compute_rolling_smape_by_hour(merged, past_days)

    merged = merged.copy()
    merged["target_day"] = pd.to_datetime(merged["target_day"]).dt.strftime("%Y-%m-%d")
    rolling = rolling.copy()
    if "target_day" not in rolling.columns and "target_day_dt" in rolling.columns:
        rolling["target_day"] = rolling["target_day_dt"].dt.strftime("%Y-%m-%d")

    merged_w = merged.merge(
        rolling[["target_day", "hour_business", "cb_smape", "tp_smape"]],
        on=["target_day", "hour_business"],
        how="left",
    )

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

    merged_w["y_pred"] = w_cb * merged_w["y_pred_cb"] + w_tp * merged_w["y_pred_tp"]

    result = merged_w.copy()
    result = result.loc[:, ~result.columns.str.endswith(("_cb", "_tp"))]
    for drop_col in ["y_pred_cb", "y_pred_tp", "cb_smape", "tp_smape"]:
        if drop_col in result.columns:
            result = result.drop(columns=[drop_col])
    result["model_name"] = "fused_inverse_smape_hour_dayahead"
    return result


# ── Fusion method: winner_by_period ─────────────────────────────────────────────


def winner_by_period(
    cb_df: pd.DataFrame,
    tp_df: pd.DataFrame,
    past_days: int = 7,
) -> pd.DataFrame:
    """
    For each period, pick the historically better model over past N days.

    No future leakage: for prediction day T, only use data from days < T.

    Returns DataFrame with fused y_pred and model_name='fused_winner_by_period_dayahead'.
    """
    merged = _align_catboost_tabpfn(cb_df, tp_df)
    if len(merged) == 0:
        logger.error("winner_by_period: 0 aligned rows")
        result = cb_df.copy()
        result["y_pred"] = np.nan
        result["model_name"] = "fused_winner_by_period_dayahead"
        return result

    rolling = _compute_rolling_smape_by_period(merged, past_days)

    merged = merged.copy()
    merged["target_day"] = pd.to_datetime(merged["target_day"]).dt.strftime("%Y-%m-%d")
    rolling = rolling.copy()
    if "target_day" not in rolling.columns and "target_day_dt" in rolling.columns:
        rolling["target_day"] = rolling["target_day_dt"].dt.strftime("%Y-%m-%d")

    merged_w = merged.merge(
        rolling[["target_day", "period", "cb_smape", "tp_smape"]],
        on=["target_day", "period"],
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
        np.where(cb_better, merged_w["y_pred_cb"], merged_w["y_pred_tp"]),
    )
    merged_w["y_pred"] = y_pred

    result = merged_w.copy()
    result = result.loc[:, ~result.columns.str.endswith(("_cb", "_tp"))]
    for drop_col in ["y_pred_cb", "y_pred_tp", "cb_smape", "tp_smape"]:
        if drop_col in result.columns:
            result = result.drop(columns=[drop_col])
    result["model_name"] = "fused_winner_by_period_dayahead"
    return result


# ── Fusion method: winner_by_hour ──────────────────────────────────────────────


def winner_by_hour(
    cb_df: pd.DataFrame,
    tp_df: pd.DataFrame,
    past_days: int = 7,
) -> pd.DataFrame:
    """
    For each hour_business, pick the historically better model over past N days.

    No future leakage: for prediction day T, only use data from days < T.

    Returns DataFrame with fused y_pred and model_name='fused_winner_by_hour_dayahead'.
    """
    merged = _align_catboost_tabpfn(cb_df, tp_df)
    if len(merged) == 0:
        logger.error("winner_by_hour: 0 aligned rows")
        result = cb_df.copy()
        result["y_pred"] = np.nan
        result["model_name"] = "fused_winner_by_hour_dayahead"
        return result

    rolling = _compute_rolling_smape_by_hour(merged, past_days)

    merged = merged.copy()
    merged["target_day"] = pd.to_datetime(merged["target_day"]).dt.strftime("%Y-%m-%d")
    rolling = rolling.copy()
    if "target_day" not in rolling.columns and "target_day_dt" in rolling.columns:
        rolling["target_day"] = rolling["target_day_dt"].dt.strftime("%Y-%m-%d")

    merged_w = merged.merge(
        rolling[["target_day", "hour_business", "cb_smape", "tp_smape"]],
        on=["target_day", "hour_business"],
        how="left",
    )

    cb_better = (
        merged_w["cb_smape"].notna()
        & merged_w["tp_smape"].notna()
        & (merged_w["cb_smape"] <= merged_w["tp_smape"])
    )
    fallback = merged_w["cb_smape"].isna() | merged_w["tp_smape"].isna()

    y_pred = np.where(
        fallback,
        0.5 * merged_w["y_pred_cb"] + 0.5 * merged_w["y_pred_tp"],
        np.where(cb_better, merged_w["y_pred_cb"], merged_w["y_pred_tp"]),
    )
    merged_w["y_pred"] = y_pred

    result = merged_w.copy()
    result = result.loc[:, ~result.columns.str.endswith(("_cb", "_tp"))]
    for drop_col in ["y_pred_cb", "y_pred_tp", "cb_smape", "tp_smape"]:
        if drop_col in result.columns:
            result = result.drop(columns=[drop_col])
    result["model_name"] = "fused_winner_by_hour_dayahead"
    return result


# ── Fusion method: ridge_stacking ───────────────────────────────────────────────


def ridge_stacking(
    cb_df: pd.DataFrame,
    tp_df: pd.DataFrame,
    past_days: int = 7,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Ridge regression stacking: for each target_day, train a Ridge model on the
    past N days' predictions (cb_pred, tp_pred) → y_true.

    No future leakage: for prediction day T, only use data from days < T.

    If fewer than 3 historical days available, fall back to simple_average.

    Returns DataFrame with fused y_pred and model_name='fused_ridge_stacking_dayahead'.
    """
    merged = _align_catboost_tabpfn(cb_df, tp_df)
    if len(merged) == 0:
        logger.error("ridge_stacking: 0 aligned rows")
        result = cb_df.copy()
        result["y_pred"] = np.nan
        result["model_name"] = "fused_ridge_stacking_dayahead"
        return result

    merged = merged.copy()
    merged["target_day_dt"] = pd.to_datetime(merged["target_day"])
    merged = merged.sort_values("target_day_dt").reset_index(drop=True)
    all_days = sorted(merged["target_day_dt"].unique())

    result_rows = []
    for day in all_days:
        past_days_list = [d for d in all_days if d < day]
        if len(past_days_list) > past_days:
            past_days_list = past_days_list[-past_days:]
        past_mask = merged["target_day_dt"].isin(past_days_list)

        if past_mask.sum() < 3:
            # Fallback to simple average
            day_mask = merged["target_day_dt"] == day
            for idx in merged.loc[day_mask].index:
                result_rows.append({
                    **{c: merged.at[idx, c] for c in merged.columns if c not in ("y_pred_cb", "y_pred_tp")},
                    "y_pred": 0.5 * merged.at[idx, "y_pred_cb"] + 0.5 * merged.at[idx, "y_pred_tp"],
                })
            continue

        X_train = np.column_stack([
            merged.loc[past_mask, "y_pred_cb"].values,
            merged.loc[past_mask, "y_pred_tp"].values,
        ])
        y_train = merged.loc[past_mask, "y_true"].values

        # Remove rows with NaN in X or y
        valid = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        if valid.sum() < 3:
            day_mask = merged["target_day_dt"] == day
            for idx in merged.loc[day_mask].index:
                result_rows.append({
                    **{c: merged.at[idx, c] for c in merged.columns if c not in ("y_pred_cb", "y_pred_tp")},
                    "y_pred": 0.5 * merged.at[idx, "y_pred_cb"] + 0.5 * merged.at[idx, "y_pred_tp"],
                })
            continue

        X_train = X_train[valid]
        y_train = y_train[valid]

        try:
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
        except Exception as e:
            logger.warning(f"Ridge fit failed for {day}: {e}, falling back to simple average")
            day_mask = merged["target_day_dt"] == day
            for idx in merged.loc[day_mask].index:
                result_rows.append({
                    **{c: merged.at[idx, c] for c in merged.columns if c not in ("y_pred_cb", "y_pred_tp")},
                    "y_pred": 0.5 * merged.at[idx, "y_pred_cb"] + 0.5 * merged.at[idx, "y_pred_tp"],
                })
            continue

        # Predict for this day
        day_mask = merged["target_day_dt"] == day
        X_test = np.column_stack([
            merged.loc[day_mask, "y_pred_cb"].values,
            merged.loc[day_mask, "y_pred_tp"].values,
        ])
        y_pred_test = model.predict(X_test)

        for i, idx in enumerate(merged.loc[day_mask].index):
            row = {c: merged.at[idx, c] for c in merged.columns if c not in ("y_pred_cb", "y_pred_tp")}
            row["y_pred"] = y_pred_test[i]
            result_rows.append(row)

    if len(result_rows) == 0:
        logger.error("ridge_stacking: no result rows produced")
        result = cb_df.copy()
        result["y_pred"] = np.nan
        result["model_name"] = "fused_ridge_stacking_dayahead"
        return result

    result = pd.DataFrame(result_rows)
    # Drop any remaining _cb / _tp columns
    result = result.loc[:, ~result.columns.str.endswith(("_cb", "_tp"))]
    result["model_name"] = "fused_ridge_stacking_dayahead"
    return result


# ── Load predictions from directory (day-ahead only) ────────────────────────────


def load_dayahead_predictions(pred_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Load day-ahead prediction CSVs from a directory.

    Only loads files with 'dayahead' in the name (or all if filtering is problematic).
    Returns dict: model_name -> DataFrame (dayahead rows only).
    """
    results = {}
    if not pred_dir.exists():
        logger.warning(f"Prediction directory not found: {pred_dir}")
        return results

    for csv_path in sorted(pred_dir.glob("*.csv")):
        name = csv_path.stem
        try:
            df = pd.read_csv(str(csv_path), encoding="utf-8-sig")
            # Filter to dayahead only
            if "task" in df.columns:
                df = df[df["task"] == "dayahead"].copy()
            if len(df) == 0:
                logger.info(f"  {name}: 0 dayahead rows, skipping")
                continue
            results[name] = df
            logger.info(f"  Loaded {name} (dayahead): {len(df)} rows")
        except Exception as e:
            logger.warning(f"  Failed to load {csv_path}: {e}")
    return results
