"""
run_tabpfn_single_day.py — Train TabPFN-TS and predict a single day.

Usage:
    python scripts/run_tabpfn_single_day.py ^
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
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from src.common.data_loader import load_data
from src.common.feature_builder import build_features
from src.models.tabpfn_ts_adapter import TabPFNTSAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run TabPFN-TS single-day prediction")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to CSV data. If omitted, reads from configs/paths.yaml.")
    parser.add_argument("--target-date", type=str, default="2026-02-15", help="YYYY-MM-DD")
    parser.add_argument("--task", type=str, default="dayahead", choices=["dayahead", "realtime"])
    parser.add_argument("--output-dir", type=str, default="outputs/tabpfn_single_day")
    parser.add_argument("--max-train-rows", type=int, default=50000, help="Max training rows")
    parser.add_argument("--device", type=str, default="cpu", help="cpu / cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = args.data_path
    if data_path is None:
        from src.common.repo_paths import get_data_path
        data_path = str(get_data_path())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from {data_path}")
    df = load_data(data_path, target=args.task)
    full_df = build_features(df)
    logger.info(f"Data: {len(full_df)} rows")

    target_dt = pd.Timestamp(args.target_date)
    train_df = full_df[full_df["ds"] < target_dt].copy()
    val_df = full_df[
        (full_df["ds"] >= target_dt - pd.DateOffset(days=30)) & (full_df["ds"] < target_dt)
    ].copy()
    logger.info(f"Train: {len(train_df)} rows")

    logger.info("Training TabPFN-TS...")
    adapter = TabPFNTSAdapter(max_train_rows=args.max_train_rows, device=args.device)
    manifest = adapter.train(train_df, eval_df=val_df)

    logger.info(f"Predicting {args.target_date} ({args.task})...")
    result = adapter.predict_day(full_df, args.target_date, task=args.task)

    pred_path = output_dir / f"tabpfn_ts_sota_{args.task}_{args.target_date}.csv"
    result.to_csv(str(pred_path), index=False, encoding="utf-8-sig")
    logger.info(f"Prediction saved to {pred_path}")

    manifest_path = output_dir / "training_manifest.json"
    with open(str(manifest_path), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(f"Rows: {len(result)}, Hours: {sorted(result['hour_business'].unique())}")
    cols = [c for c in ["ds", "hour_business", "period", "y_pred", "y_true"] if c in result.columns]
    print(result[cols].head())
    logger.info("✅ TabPFN-TS single-day complete!")


if __name__ == "__main__":
    main()
