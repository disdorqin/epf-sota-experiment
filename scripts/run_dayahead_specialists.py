"""
run_dayahead_specialists.py — Day-ahead specialist walk-forward for sMAPE < 8%.

Models:
  1. catboost_dayahead_tuned     — tuned CatBoost (walk-forward, per date)
  2. catboost_period_specialist    — one CatBoost per period, trained ONCE on all
                                  historical data before start_date, then predicts all dates.

Features: standard + enhanced day-ahead features (lags, rolling stats, calendar)
Selection metric: sMAPE_floor50 on validation

Usage:
    python scripts/run_dayahead_specialists.py ^
        --start 2026-02-01 --end 2026-03-02 ^
        --models catboost_dayahead_tuned,catboost_period_specialist ^
        --target dayahead ^
        --output-root outputs/dayahead_specialists_30d

Resume (default: on):
    The script checks existing output CSVs at startup and skips already-done
    (model_name, target_date) pairs. To force re-run, use --no-resume.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from src.common.data_loader  import load_data
from src.common.metrics     import compute_all_metrics
from src.common.output_schema import make_long_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Enhanced Feature Builder ───────────────────────────────────────────────────

SPRING_FESTIVAL_2026 = pd.Timestamp("2026-02-17")
SPRING_FESTIVAL_WINDOW = 10   # days before + after


def build_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build standard + enhanced day-ahead features. No future leakage."""
    df = df.copy()

    # Standard time features
    adjusted = df["ds"] - pd.Timedelta(seconds=1)
    df["hour"] = adjusted.dt.hour + 1
    df["month"] = adjusted.dt.month
    df["day_of_week"] = adjusted.dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # ── Enhanced lags ──
    for lag_h in [24, 48, 72, 168, 336]:
        df[f"lag_{lag_h}h"] = df["y"].shift(lag_h)

    # Smart lag (same as original)
    df["lag_price_target"] = np.where(
        df["day_of_week"] < 5, df["lag_168h"], df["lag_48h"]
    )
    df["lag_price_week"] = df["lag_168h"]
    for c in ["lag_price_target", "lag_price_week"]:
        df[c] = df[c].ffill().fillna(0)

    # ── Rolling same-hour statistics (leak-proof: shift 24 to use only past data) ──
    for window in [7, 14]:
        df[f"same_hour_mean_{window}d"] = (
            df.groupby("hour")["y"]
            .transform(lambda x: x.shift(24).rolling(window=window * 24, min_periods=1).mean())
        )
        df[f"same_hour_std_{window}d"] = (
            df.groupby("hour")["y"]
            .transform(lambda x: x.shift(24).rolling(window=window * 24, min_periods=1).std())
        )
    df["same_hour_min_7d"] = (
        df.groupby("hour")["y"]
        .transform(lambda x: x.shift(24).rolling(window=7 * 24, min_periods=1).min())
    )
    df["same_hour_max_7d"] = (
        df.groupby("hour")["y"]
        .transform(lambda x: x.shift(24).rolling(window=7 * 24, min_periods=1).max())
    )

    # Fill NaN rolling stats
    roll_cols = [c for c in df.columns if "same_hour" in c]
    for c in roll_cols:
        df[c] = df[c].ffill().fillna(0)

    # ── Price momentum: (lag_24h - lag_168h) / lag_168h ──
    df["price_momentum_24_168"] = np.where(
        df["lag_168h"].abs() > 1,
        (df["lag_24h"] - df["lag_168h"]) / df["lag_168h"].abs(),
        0,
    )

    # ── Physical features ──
    safe_load = df["load"].replace(0, 1)
    df["net_load"] = df["load"] - df["wind"] - df["solar"]
    df["solar_ratio"] = df["solar"] / safe_load
    df["net_load_sq"] = (df["net_load"] / 1000) ** 2

    if "bidding_space_raw" in df.columns and df["bidding_space_raw"].notna().sum() > 0:
        df["bidding_space"] = df["bidding_space_raw"]
    else:
        df["bidding_space"] = df["net_load"] - df["interconnect"]
    df["space_ratio"] = df["bidding_space"] / safe_load
    df["wind_ratio"] = df["wind"] / safe_load
    df["renew_penetration"] = (df["wind"] + df["solar"]) / safe_load
    df["ramp_load"] = df["load"].diff().fillna(0)
    df["ramp_solar"] = df["solar"].diff().fillna(0)

    # ── Rolling rank features (30-day window, within-day) ──
    df["net_load_rank_30d"] = (
        df.groupby("hour")["net_load"]
        .transform(lambda x: x.shift(24).rolling(30 * 24, min_periods=1).rank(pct=True))
    )
    df["bidding_space_rank_30d"] = (
        df.groupby("hour")["bidding_space"]
        .transform(lambda x: x.shift(24).rolling(30 * 24, min_periods=1).rank(pct=True))
    )

    # ── Calendar features ──
    df["date_only"] = df["ds"].dt.date
    df["is_spring_festival_window"] = (
        (df["ds"] >= SPRING_FESTIVAL_2026 - timedelta(days=SPRING_FESTIVAL_WINDOW)) &
        (df["ds"] <= SPRING_FESTIVAL_2026 + timedelta(days=SPRING_FESTIVAL_WINDOW))
    ).astype(int)
    df["days_to_spring_festival"] = (SPRING_FESTIVAL_2026 - df["ds"]).dt.days
    df["days_after_spring_festival"] = (df["ds"] - SPRING_FESTIVAL_2026).dt.days
    df["is_month_start"] = (adjusted.dt.day <= 3).astype(int)
    df["is_month_end"] = (adjusted.dt.day >= 28).astype(int)

    # ── D-day latest-info features (same as original) ──
    mask_morning = (df["hour"] >= 1) & (df["hour"] <= 15)
    df_morning = df[mask_morning].copy()

    def calc_trend(x):
        return x.iloc[-1] - x.iloc[0] if len(x) >= 2 else 0

    stats_basic = df_morning.groupby("date_only")["y"].agg(
        morning_mean="mean", morning_std="std",
    )
    mask_noon = (df_morning["hour"] >= 11) & (df_morning["hour"] <= 15)
    stats_noon = df_morning[mask_noon].groupby("date_only")["y"].agg(
        noon_min="min", morning_trend=calc_trend,
    )
    daily_feats = pd.concat([stats_basic, stats_noon], axis=1).reset_index()
    cols_to_shift = ["morning_mean", "noon_min", "morning_std", "morning_trend"]
    daily_feats[cols_to_shift] = daily_feats[cols_to_shift].shift(1)
    daily_feats["is_info_fresh"] = daily_feats["morning_mean"].notna().astype(int)
    for c in cols_to_shift:
        daily_feats[c] = daily_feats[c].ffill().fillna(0)

    df = df.merge(daily_feats, on="date_only", how="left")

    # Clean up intermediate cols
    drop_cols = ["date_only", "lag_24h", "lag_48h", "lag_72h", "lag_168h", "lag_336h"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Fill all remaining NaNs
    df = df.ffill().fillna(0)
    return df


ENHANCED_FEATURE_COLUMNS = [
    "hour", "month", "day_of_week", "is_weekend",
    "lag_price_target", "lag_price_week",
    "same_hour_mean_7d", "same_hour_mean_14d", "same_hour_std_7d",
    "same_hour_min_7d", "same_hour_max_7d",
    "price_momentum_24_168",
    "load", "wind", "solar", "interconnect",
    "bidding_space", "space_ratio",
    "net_load", "solar_ratio", "net_load_sq",
    "wind_ratio", "renew_penetration", "ramp_load", "ramp_solar",
    "net_load_rank_30d", "bidding_space_rank_30d",
    "is_spring_festival_window", "days_to_spring_festival", "days_after_spring_festival",
    "is_month_start", "is_month_end",
    "morning_mean", "noon_min", "morning_std", "morning_trend", "is_info_fresh",
]


# ── CatBoost imports (lazy) ───────────────────────────────────────────────────

def _get_catboost():
    from catboost import CatBoostRegressor
    return CatBoostRegressor


# ── Specialist Training ─────────────────────────────────────────────────────────

def train_catboost_specialist(train_df: pd.DataFrame, val_df: pd.DataFrame,
                               hour_filter: Optional[int] = None,
                               period_filter: Optional[str] = None,
                               use_smape_selection: bool = True) -> tuple:
    """Train a CatBoost specialist. Uses sMAPE_floor50 for model selection if use_smape_selection."""
    CB = _get_catboost()

    df = train_df.copy()
    if hour_filter is not None:
        df = df[df["hour_business"] == hour_filter]
    if period_filter is not None:
        df = df[df["period"] == period_filter]

    if len(df) < 200:
        return None, None

    # Feature columns (minus target and metadata)
    exclude = {"ds", "y", "y_true", "y_pred", "hour_business", "period",
               "business_day", "target_day", "task", "model_name", "date_only"}
    feature_cols = [c for c in df.columns if c not in exclude and c != "y" and df[c].dtype in (np.float64, np.int64, np.float32, np.int32)]

    X_train = df[feature_cols].values
    y_train = df["y"].values.astype(float)

    # If val_df provided, use it for early stopping / model selection
    eval_mode = False
    if val_df is not None and len(val_df) > 50:
        vdf = val_df.copy()
        if hour_filter is not None:
            vdf = vdf[vdf["hour_business"] == hour_filter]
        if period_filter is not None:
            vdf = vdf[vdf["period"] == period_filter]
        if len(vdf) >= 50:
            eval_mode = True
            X_val = vdf[feature_cols].values
            y_val = vdf["y"].values.astype(float)

    # Try multiple settings and pick by sMAPE
    best_model = None
    best_smape = float("inf")

    configs = [
        {"depth": 6, "learning_rate": 0.05, "iterations": 1000, "l2_leaf_reg": 3.0},
        {"depth": 8, "learning_rate": 0.03, "iterations": 1500, "l2_leaf_reg": 5.0},
        {"depth": 10, "learning_rate": 0.02, "iterations": 2000, "l2_leaf_reg": 8.0},
    ]

    for cfg in configs:
        model = CB(
            **cfg,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            verbose=50,
            thread_count=-1,
        )
        if eval_mode:
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
        else:
            model.fit(X_train, y_train, verbose=False)

        if use_smape_selection and eval_mode:
            yp = model.predict(X_val)
            s = _smape(y_val, yp)
            if s < best_smape:
                best_smape = s
                best_model = model
        else:
            if best_model is None:
                best_model = model

    return best_model, feature_cols


def _smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.maximum(denom, 1e-8)
    return float(np.mean(200 * np.abs(y_true - y_pred) / denom))


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict_catboost_specialist(model, feature_cols, feat_df: pd.DataFrame,
                                 target_date: str, task: str,
                                 hour_filter=None, period_filter=None) -> Optional[pd.DataFrame]:
    """Predict for a target date using the specialist model."""
    target_dt = pd.Timestamp(target_date)
    pred_df = feat_df[feat_df["ds"].between(target_dt, target_dt + timedelta(hours=23))].copy()

    if pred_df.empty:
        return None

    # Filter to specialist's domain
    if hour_filter is not None:
        pred_df = pred_df[pred_df["hour_business"] == hour_filter]
    if period_filter is not None:
        pred_df = pred_df[pred_df["period"] == period_filter]

    if pred_df.empty:
        return None

    X_pred = pred_df[feature_cols].values
    pred_df["y_pred"] = model.predict(X_pred)
    pred_df["y_true"] = pred_df["y"].values

    result = make_long_table(
        pred_df, model_name="catboost_specialist", task=task,
    )
    return result


# ── Resume helpers ─────────────────────────────────────────────────────────────

def _load_done_dates(output_root: Path, model_name: str, task: str = "dayahead") -> set:
    """
    Load already-completed (model_name, target_date) pairs from existing output CSV.
    Returns set of target_day strings that are already done.
    """
    pred_dir = output_root / "predictions"
    done = set()
    if not pred_dir.exists():
        return done

    # Check final CSV
    final_path = pred_dir / f"{model_name}_{task}.csv"
    if final_path.exists():
        try:
            df = pd.read_csv(str(final_path), encoding="utf-8-sig", usecols=["target_day"])
            done.update(df["target_day"].astype(str).unique())
        except Exception:
            pass

    # Also check partial CSV
    partial_path = pred_dir / "partial" / f"{model_name}_{task}_partial.csv"
    if partial_path.exists():
        try:
            df = pd.read_csv(str(partial_path), encoding="utf-8-sig", usecols=["target_day"])
            done.update(df["target_day"].astype(str).unique())
        except Exception:
            pass

    return done


def _append_partial(pred_df: pd.DataFrame, output_root: Path, model_name: str, task: str) -> None:
    """Append predictions to partial CSV (saved after each date)."""
    partial_dir = output_root / "predictions" / "partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    path = partial_dir / f"{model_name}_{task}_partial.csv"

    if path.exists():
        existing = pd.read_csv(str(path), encoding="utf-8-sig")
        combined = pd.concat([existing, pred_df], ignore_index=True)
        # Deduplicate by (target_day, hour_business) if those cols exist
        dedup_cols = [c for c in ["target_day", "hour_business"] if c in combined.columns]
        if dedup_cols:
            combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
    else:
        combined = pred_df
    combined.to_csv(str(path), index=False, encoding="utf-8-sig")


def _merge_partial(output_root: Path, model_name: str, task: str) -> None:
    """Merge partial CSVs into final output CSV."""
    partial_dir = output_root / "predictions" / "partial"
    path = partial_dir / f"{model_name}_{task}_partial.csv"
    if not path.exists():
        return
    pred_dir = output_root / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    final_path = pred_dir / f"{model_name}_{task}.csv"
    df = pd.read_csv(str(path), encoding="utf-8-sig")
    # Deduplicate
    dedup_cols = [c for c in ["target_day", "hour_business"] if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols, keep="last")
    df.to_csv(str(final_path), index=False, encoding="utf-8-sig")
    logger.info(f"Merged partial -> {final_path} ({len(df)} rows)")


# ── Metrics ─────────────────────────────────────────────────────────────────────

def compute_hour_metrics(all_preds: list) -> pd.DataFrame:
    rows = []
    for df in all_preds:
        model = df["model_name"].iloc[0]
        task  = df["task"].iloc[0]
        for hour, grp in df.groupby("hour_business"):
            yt = grp["y_true"].values
            yp = grp["y_pred"].values
            v  = ~(np.isnan(yt) | np.isnan(yp))
            if v.sum() < 2:
                continue
            m = compute_all_metrics(yt[v], yp[v])
            m["model_name"]    = model
            m["task"]          = task
            m["hour_business"] = hour
            m["n"]             = int(v.sum())
            rows.append(m)
    return pd.DataFrame(rows)


def compute_period_metrics(all_preds: list) -> pd.DataFrame:
    rows = []
    for df in all_preds:
        model  = df["model_name"].iloc[0]
        task   = df["task"].iloc[0]
        for period, grp in df.groupby("period"):
            yt = grp["y_true"].values
            yp = grp["y_pred"].values
            v  = ~(np.isnan(yt) | np.isnan(yp))
            if v.sum() < 2:
                continue
            m = compute_all_metrics(yt[v], yp[v])
            m["model_name"] = model
            m["task"]       = task
            m["period"]     = period
            m["n"]          = int(v.sum())
            rows.append(m)
    return pd.DataFrame(rows)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Day-ahead specialist walk-forward")
    p.add_argument("--data-path",      type=str, default=None)
    p.add_argument("--start",          type=str, default="2026-02-01")
    p.add_argument("--end",            type=str, default="2026-03-02")
    p.add_argument("--output-root",    type=str, default="outputs/dayahead_specialists_30d")
    p.add_argument("--models",          type=str, default="catboost_period_specialist,catboost_dayahead_tuned")
    p.add_argument("--no-resume",      action="store_true", help="Disable resume (re-run all dates)")
    return p.parse_args()


def main():
    args = parse_args()

    data_path = args.data_path
    if data_path is None:
        from src.common.repo_paths import get_data_path
        data_path = str(get_data_path())

    output_root   = Path(args.output_root)
    resume        = not args.no_resume
    for sub in ["predictions", "metrics", "reports", "debug"]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    models         = [m.strip() for m in args.models.split(",") if m.strip()]
    expected_dates = pd.date_range(start=args.start, end=args.end).strftime("%Y-%m-%d").tolist()

    logger.info(f"Specialist config: {models}, dates: {args.start} -> {args.end} ({len(expected_dates)} days)")
    logger.info(f"Resume: {resume}")

    # Load data
    raw = load_data(data_path, target="dayahead")

    # Build enhanced features (ONCE)
    feat_df = build_enhanced_features(raw)
    logger.info(f"Enhanced features: {len(feat_df)} rows, {len(ENHANCED_FEATURE_COLUMNS)} columns")

    # Build business hour and period columns (needed for filtering)
    adjusted = feat_df["ds"] - pd.Timedelta(seconds=1)
    feat_df["hour_business"] = adjusted.dt.hour + 1
    feat_df["period"] = np.select(
        [feat_df["hour_business"].between(1, 8),
         feat_df["hour_business"].between(9, 16)],
        ["1_8", "9_16"],
        default="17_24",
    )

    all_predictions = []
    run_manifest = {
        "start": args.start, "end": args.end,
        "models": models,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "failed_dates": [],
    }

    for model_name in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_name}")

        # ── Load done dates (resume) ──
        done_dates = set()
        if resume:
            done_dates = _load_done_dates(output_root, model_name, task="dayahead")
            if len(done_dates) > 0:
                logger.info(f"  Resume: {len(done_dates)} dates already done, skipping")

        remaining_dates = [d for d in expected_dates if d not in done_dates]
        if len(remaining_dates) == 0:
            logger.info(f"  All dates already done for {model_name}, skipping.")
            # Still need to load existing predictions for metrics
            final_path = output_root / "predictions" / f"{model_name}_dayahead.csv"
            if final_path.exists():
                df = pd.read_csv(str(final_path), encoding="utf-8-sig")
                all_predictions.append(df)
            continue

        logger.info(f"  Remaining dates: {len(remaining_dates)}")

        # ── Category: catboost_period_specialist (train ONCE per period) ──
        if model_name == "catboost_period_specialist":
            # Train 3 models (one per period) using all data before start_date
            start_dt = pd.Timestamp(args.start)
            train_all = feat_df[feat_df["ds"] < start_dt].copy()

            period_models = {}
            feature_cols = None
            for period in ["1_8", "9_16", "17_24"]:
                logger.info(f"  Training period specialist: {period}")
                train_period = train_all[train_all["period"] == period].copy()
                if len(train_period) < 200:
                    logger.warning(f"  Not enough data for {period}: {len(train_period)} rows")
                    continue

                # Validation: last 30 days of train_all
                val_start = start_dt - timedelta(days=30)
                val_df = train_all[train_all["ds"].between(val_start, start_dt - timedelta(hours=1))].copy()
                val_period = val_df[val_df["period"] == period].copy() if len(val_df) > 50 else None

                model, fc = train_catboost_specialist(
                    train_period, val_period, period_filter=period,
                )
                if model is None:
                    logger.warning(f"  Failed to train {period}")
                    continue

                period_models[period] = model
                feature_cols = fc
                logger.info(f"  Trained {period}: {len(train_period)} samples")

            if len(period_models) == 0:
                logger.error("  No period models trained successfully")
                continue

            # Predict all remaining dates with the 3 models
            logger.info(f"  Predicting {len(remaining_dates)} dates with period models...")
            for target_date_str in remaining_dates:
                target_dt = pd.Timestamp(target_date_str)
                try:
                    pred_df = feat_df[feat_df["ds"].between(target_dt, target_dt + timedelta(hours=23))].copy()
                    if pred_df.empty:
                        continue

                    # Predict per period
                    all_period_preds = []
                    for period, model in period_models.items():
                        period_df = pred_df[pred_df["period"] == period].copy()
                        if period_df.empty:
                            continue
                        X_pred = period_df[feature_cols].values
                        period_df["y_pred"] = model.predict(X_pred)
                        period_df["y_true"] = period_df["y"].values
                        all_period_preds.append(period_df)

                    if len(all_period_preds) == 0:
                        continue

                    combined_pred = pd.concat(all_period_preds, ignore_index=True)
                    result = make_long_table(combined_pred, model_name="catboost_period_specialist", task="dayahead")

                    if result is not None and len(result) > 0:
                        all_predictions.append(result)
                        # Append to partial CSV
                        _append_partial(result, output_root, model_name, "dayahead")

                except Exception as e:
                    logger.error(f"    FAILED {target_date_str}: {e}")
                    run_manifest["failed_dates"].append(f"{target_date_str}:{model_name}:{str(e)[:100]}")

            # Merge partial -> final
            _merge_partial(output_root, model_name, "dayahead")

        # ── Category: catboost_dayahead_tuned (walk-forward, per date) ──
        elif model_name == "catboost_dayahead_tuned":
            CB = _get_catboost()
            exclude = {"ds", "y", "y_true", "y_pred", "hour_business", "period",
                       "business_day", "target_day", "task", "model_name", "date_only"}

            for target_date_str in remaining_dates:
                target_dt = pd.Timestamp(target_date_str)
                try:
                    train_df = feat_df[feat_df["ds"] < target_dt].copy()
                    if len(train_df) < 2000:
                        continue

                    val_start = target_dt - timedelta(days=30)
                    val_df = feat_df[feat_df["ds"].between(val_start, target_dt - timedelta(hours=1))].copy()

                    feat_cols = [c for c in train_df.columns if c not in exclude and
                                 train_df[c].dtype in (np.float64, np.int64, np.float32, np.int32)]
                    X_tr, y_tr = train_df[feat_cols].values, train_df["y"].values.astype(float)

                    # Train with validation early stopping
                    if len(val_df) > 50:
                        X_val = val_df[feat_cols].values
                        y_val = val_df["y"].values.astype(float)

                        model = CB(
                            depth=8, learning_rate=0.03, iterations=2000,
                            l2_leaf_reg=3.0, loss_function="RMSE",
                            eval_metric="RMSE", random_seed=42,
                            thread_count=-1, verbose=100,
                        )
                        model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                                  early_stopping_rounds=50, verbose=False)
                    else:
                        model = CB(
                            depth=8, learning_rate=0.03, iterations=1500,
                            l2_leaf_reg=3.0, random_seed=42, thread_count=-1,
                        )
                        model.fit(X_tr, y_tr, verbose=False)

                    # Predict
                    pred_df = feat_df[feat_df["ds"].between(target_dt, target_dt + timedelta(hours=23))].copy()
                    if pred_df.empty:
                        continue
                    X_pred = pred_df[feat_cols].values
                    pred_df["y_pred"] = model.predict(X_pred).flatten()
                    pred_df["y_true"] = pred_df["y"].values
                    result = make_long_table(pred_df, model_name="catboost_dayahead_tuned", task="dayahead")

                    if result is not None and len(result) > 0:
                        all_predictions.append(result)
                        _append_partial(result, output_root, model_name, "dayahead")

                except Exception as e:
                    logger.error(f"    FAILED {target_date_str}: {e}")
                    run_manifest["failed_dates"].append(f"{target_date_str}:{model_name}:{str(e)[:100]}")

            # Merge partial -> final
            _merge_partial(output_root, model_name, "dayahead")

        else:
            logger.warning(f"Unknown model: {model_name}, skipping")

    # ── Save final predictions (also load any skipped models' existing CSVs) ──
    # Reload all final CSVs to ensure we have everything
    all_predictions = []
    pred_dir = output_root / "predictions"
    if pred_dir.exists():
        for csv_path in sorted(pred_dir.glob("*.csv")):
            if "partial" in str(csv_path):
                continue
            try:
                df = pd.read_csv(str(csv_path), encoding="utf-8-sig")
                if "task" in df.columns:
                    df = df[df["task"] == "dayahead"].copy()
                if len(df) > 0:
                    all_predictions.append(df)
                    logger.info(f"Loaded existing: {csv_path.name} ({len(df)} rows)")
            except Exception as e:
                logger.warning(f"Failed to load {csv_path}: {e}")

    # ── Compute metrics ──
    if all_predictions:
        # Hour metrics
        hour_metrics = compute_hour_metrics(all_predictions)
        if not hour_metrics.empty:
            hour_metrics.to_csv(str(output_root / "metrics" / "hour_metrics.csv"), index=False, encoding="utf-8-sig")

        # Period metrics
        period_metrics = compute_period_metrics(all_predictions)
        if not period_metrics.empty:
            period_metrics.to_csv(str(output_root / "metrics" / "period_metrics.csv"), index=False, encoding="utf-8-sig")

        # Overall summary
        rows = []
        for df in all_predictions:
            model = df["model_name"].iloc[0]
            task  = df["task"].iloc[0] if "task" in df.columns else "dayahead"
            yt = df["y_true"].values
            yp = df["y_pred"].values
            v  = ~(np.isnan(yt) | np.isnan(yp))
            if v.sum() < 2:
                continue
            m = compute_all_metrics(yt[v], yp[v])
            m["model_name"] = model
            m["task"]       = task
            m["n"]          = int(v.sum())
            rows.append(m)
        summary = pd.DataFrame(rows)
        summary.to_csv(str(output_root / "metrics" / "summary.csv"), index=False, encoding="utf-8-sig")

        print("\n" + "=" * 60)
        print("SPECIALIST SUMMARY (DAY-AHEAD)")
        print("=" * 60)
        cols = ["model_name", "MAE", "RMSE", "sMAPE_floor50", "peak_MAE_q90", "negative_price_hit_rate", "n"]
        if all(c in summary.columns for c in cols):
            print(summary[cols].to_string(index=False))
        else:
            print(summary.to_string(index=False))
        print("=" * 60)
    else:
        logger.warning("No predictions generated!")

    # ── Run manifest ──
    run_manifest["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_manifest["total_predictions"] = len(all_predictions)
    with open(str(output_root / "debug" / "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, ensure_ascii=False, indent=2)

    logger.info(f"Done. Output: {output_root}")


if __name__ == "__main__":
    main()
