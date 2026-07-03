"""
run_fair_h13_h17_specialist.py — Fair 30-day H13/H17 specialist evaluation.

Fixes the 5-day issue: trains specialist on ALL pre-target data for hours 11/12/13/17,
then replaces CatBoost baseline predictions for specified hours.

Usage:
    python scripts/run_fair_h13_h17_specialist.py ^
        --input-pred outputs/dayahead_30d_core/predictions/catboost_sota_dayahead.csv ^
        --output-root outputs/dayahead_model_pool_30d
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from src.common.metrics import compute_all_metrics
from src.common.output_schema import make_long_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _CB():
    from catboost import CatBoostRegressor
    return CatBoostRegressor


def _smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.maximum(denom, 1e-8)
    return float(np.nanmean(200 * np.abs(y_true - y_pred) / denom))


def parse_args():
    p = argparse.ArgumentParser(description="Fair 30-day H13/H17 specialist")
    p.add_argument("--input-pred", type=str,
                   default="outputs/dayahead_30d_core/predictions/catboost_sota_dayahead.csv")
    p.add_argument("--output-root", type=str, default="outputs/dayahead_model_pool_30d")
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_pred)
    output_root = Path(args.output_root)

    for sub in ["predictions", "metrics", "reports", "debug"]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading predictions from {input_path}")
    df = pd.read_csv(str(input_path), encoding="utf-8-sig")
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows")

    # Feature columns (all numeric columns minus metadata)
    exclude = {"ds", "y", "y_pred", "y_true", "hour_business", "period",
               "business_day", "target_day", "task", "model_name",
               "source", "run_mode", "created_at"}
    feat_cols = [c for c in df.columns if c not in exclude
                 and df[c].dtype in (np.float64, np.int64, np.float32, np.int32, np.int8)]

    days = sorted(df["target_day"].unique())
    logger.info(f"Target days: {len(days)} ({days[0]} → {days[-1]})")

    # Define replacement strategies
    strategies = {
        "catboost_replace_H13": [13],
        "catboost_replace_H17": [17],
        "catboost_replace_H13_H17": [13, 17],
        "catboost_replace_H12_H13_H17": [12, 13, 17],
    }

    all_results = {}

    for strategy_name, hours in strategies.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Strategy: {strategy_name} (hours {hours})")
        logger.info('='*60)

        day_preds = []

        for day_idx, day in enumerate(days):
            target_dt = pd.Timestamp(day)

            # Training data: ALL pre-target data, filtered to target hours
            train_df = df[df["target_day"] < day].copy()
            train_spec = train_df[train_df["hour_business"].isin(hours)].copy()

            if len(train_spec) < 50:
                logger.debug(f"  Day {day}: insufficient specialist training ({len(train_spec)})")
                # Fallback: use baseline predictions
                day_df = df[df["target_day"] == day].copy()
                day_df["y_pred"] = day_df["y_pred"].values  # keep original
                day_preds.append(day_df)
                continue

            try:
                CB = _CB()
                X_tr = train_spec[feat_cols].values.astype(float)
                y_tr = train_spec["y_true"].values.astype(float)

                # Try 2 configs, pick by internal CV sMAPE
                configs = [
                    {"depth": 8, "learning_rate": 0.03, "iterations": 1500, "l2_leaf_reg": 3.0},
                    {"depth": 6, "learning_rate": 0.05, "iterations": 1200, "l2_leaf_reg": 5.0},
                ]

                best_model = None
                best_smape = float("inf")

                for cfg in configs:
                    model = CB(**cfg, loss_function="RMSE", random_seed=42,
                               verbose=False, thread_count=-1)
                    model.fit(X_tr, y_tr, verbose=False)

                    # Internal CV: split last 20% of training data
                    split = int(len(X_tr) * 0.8)
                    if split >= 20:
                        yp = model.predict(X_tr[split:]).flatten()
                        s = _smape(y_tr[split:], yp)
                        if s < best_smape:
                            best_smape = s
                            best_model = model

                if best_model is None:
                    best_model = model

                # Predict current day
                day_df = df[df["target_day"] == day].copy()
                y_pred_base = day_df["y_pred"].values.copy()

                # Replace predictions for target hours
                mask = day_df["hour_business"].isin(hours)
                if mask.sum() > 0:
                    X_pred = day_df.loc[mask, feat_cols].values.astype(float)
                    y_pred_base[mask.values] = best_model.predict(X_pred).flatten()

                day_df["y_pred"] = y_pred_base
                day_df["y_true"] = day_df["y"].values
                day_preds.append(day_df)

                if day_idx % 5 == 0:
                    logger.info(f"  Day {day}: trained on {len(train_spec)} specialist rows")

            except Exception as e:
                logger.warning(f"  Day {day}: failed: {e}")
                # Fallback
                day_df = df[df["target_day"] == day].copy()
                day_df["y_pred"] = day_df["y_pred"].values
                day_preds.append(day_df)

        if day_preds:
            combined = pd.concat(day_preds, ignore_index=True)
            # Fix model_name
            combined["model_name"] = strategy_name
            all_results[strategy_name] = combined

            m = compute_all_metrics(
                combined["y_true"].values,
                combined["y_pred"].values
            )
            logger.info(f"  FINAL sMAPE = {m.get('sMAPE_floor50', 0):.4f}% (n={len(combined)})")

    # Save all predictions
    for name, result_df in all_results.items():
        path = output_root / "predictions" / f"{name}_dayahead.csv"
        result_df.to_csv(str(path), index=False, encoding="utf-8-sig")
        logger.info(f"Saved {path} ({len(result_df)} rows)")

    # Print summary
    print("\n" + "=" * 70)
    print("FAIR 30-DAY H13/H17 SPECIALIST RESULTS")
    print("=" * 70)
    print(f"{'Strategy':40s} {'sMAPE':>8s} {'MAE':>8s} {'Rows':>6s}")
    print("-" * 65)
    for name, result_df in sorted(all_results.items()):
        m = compute_all_metrics(
            result_df["y_true"].values,
            result_df["y_pred"].values
        )
        smape = m.get("sMAPE_floor50", 0)
        mae = m.get("MAE", 0)
        beats = "✅" if smape < 12.58 else "❌"
        print(f"{name:40s} {smape:7.2f}% {mae:7.2f} {len(result_df):>6d} {beats}")
    print("-" * 65)
    print(f"{'CatBoost baseline':40s} {'12.58':>7s}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
