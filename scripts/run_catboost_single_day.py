"""
run_catboost_single_day.py — Train CatBoost and predict a single day.

Usage:
    python scripts/run_catboost_single_day.py ^
        --data-path "path\to\shandong_pmos_hourly.csv" ^
        --target-date 2026-02-15 ^
        --task dayahead
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── Ensure src is on path (use absolute path via os.path.abspath) ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)  # models/
sys.path.insert(0, _PROJECT_DIR)

from src.common.data_loader import load_data
from src.common.feature_builder import build_features
from src.models.catboost_adapter import CatBoostAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run CatBoost single-day prediction")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to CSV data. If omitted, reads from configs/paths.yaml.")
    parser.add_argument("--target-date", type=str, default="2026-02-15", help="YYYY-MM-DD")
    parser.add_argument("--task", type=str, default="dayahead", choices=["dayahead", "realtime"])
    parser.add_argument("--output-dir", type=str, default="outputs/catboost_single_day")
    parser.add_argument("--train-months", type=int, default=12, help="Months of training data")
    parser.add_argument("--device", type=str, default="CPU", choices=["CPU", "GPU"])
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve data path: --data-path > configs/paths.yaml > error
    data_path = args.data_path
    if data_path is None:
        try:
            from src.common.repo_paths import get_data_path
            data_path = str(get_data_path())
            logger.info(f"Data path from configs/paths.yaml: {data_path}")
        except FileNotFoundError as e:
            logger.error(f"Cannot resolve data path. Provide --data-path or check configs/paths.yaml.\n{e}")
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_str = "日前电价" if args.task == "dayahead" else "实时电价"

    # ── 1. Load data ──
    logger.info(f"Loading data from {data_path}")
    df = load_data(data_path, target=args.task)
    logger.info(f"Data loaded: {len(df)} rows, {df['ds'].min()} ~ {df['ds'].max()}")

    # ── 2. Build features ──
    logger.info("Building features...")
    full_df = build_features(df)

    # ── 3. Split train / eval ──
    target_dt = pd.Timestamp(args.target_date)
    train_end = target_dt  # exclusive
    train_df = full_df[full_df["ds"] < train_end].copy()

    # Validation: last 30 days before target
    val_start = target_dt - pd.DateOffset(days=30)
    val_df = full_df[
        (full_df["ds"] >= val_start) & (full_df["ds"] < train_end)
    ].copy()

    logger.info(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")

    # ── 4. Train CatBoost ──
    logger.info("Training CatBoostRegressor...")
    adapter = CatBoostAdapter(task_type=args.device)
    manifest = adapter.train(train_df, eval_df=val_df)
    logger.info(f"Training complete. Best iteration: {manifest.get('best_iteration')}")

    # ── 5. Predict target day ──
    logger.info(f"Predicting {args.target_date} ({args.task})...")
    result = adapter.predict_day(full_df, args.target_date, task=args.task)

    # ── 6. Save outputs ──
    # Prediction CSV
    pred_path = output_dir / f"catboost_sota_{args.task}_{args.target_date}.csv"
    result.to_csv(str(pred_path), index=False, encoding="utf-8-sig")
    logger.info(f"Prediction saved to {pred_path}")

    # Model file
    model_path = output_dir / f"catboost_sota_{args.task}.cbm"
    adapter.save_model(model_path)

    # Feature importance
    fi = adapter.get_feature_importance()
    fi_path = output_dir / f"feature_importance_{args.task}.csv"
    fi.to_csv(str(fi_path), index=False, encoding="utf-8-sig")
    logger.info(f"Feature importance saved to {fi_path}")

    # Training manifest
    manifest_path = output_dir / "training_manifest.json"
    with open(str(manifest_path), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info(f"Manifest saved to {manifest_path}")

    # ── 7. Summary ──
    logger.info(f"\n{'='*60}")
    logger.info(f"CatBoost Single-Day Prediction Complete")
    logger.info(f"  Target:   {args.target_date} ({args.task})")
    logger.info(f"  Rows:     {len(result)}")
    logger.info(f"  Columns:  {list(result.columns)}")
    if "hour_business" in result.columns:
        logger.info(f"  Hours:    {sorted(result['hour_business'].unique())}")
    logger.info(f"{'='*60}")

    # Print first few rows
    print("\n--- Prediction (first 5 rows) ---")
    cols_show = [c for c in ["ds", "hour_business", "period", "y_pred", "y_true"] if c in result.columns]
    print(result[cols_show].head())
    print("---")


if __name__ == "__main__":
    main()
