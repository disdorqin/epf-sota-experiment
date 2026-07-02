"""
run_tirex_single_day.py — TiRex zero-shot single-day prediction.

Usage:
    python scripts/run_tirex_single_day.py ^
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
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from src.common.data_loader import load_data
from src.models.tirex_adapter import TiRexAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run TiRex single-day zero-shot prediction")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to CSV data. If omitted, reads from configs/paths.yaml.")
    parser.add_argument("--target-date", type=str, default="2026-02-15", help="YYYY-MM-DD")
    parser.add_argument("--task", type=str, default="dayahead", choices=["dayahead", "realtime"])
    parser.add_argument("--context-days", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="outputs/tirex_single_day")
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
    logger.info(f"Data: {len(df)} rows")

    adapter = TiRexAdapter(context_length=args.context_days * 24)
    loaded = adapter.load()
    if not loaded:
        logger.error(f"TiRex unavailable: {adapter.unavailable_reason}")
        manifest = adapter.get_manifest()
        manifest_path = output_dir / "inference_manifest.json"
        with open(str(manifest_path), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"❌ TiRex unavailable: {adapter.unavailable_reason}")
        sys.exit(1)

    logger.info(f"Predicting {args.target_date} ({args.task})...")
    result = adapter.predict_day(df, args.target_date, task=args.task, y_col="y")

    pred_path = output_dir / f"tirex_zero_shot_{args.task}_{args.target_date}.csv"
    result.to_csv(str(pred_path), index=False, encoding="utf-8-sig")
    logger.info(f"Prediction saved to {pred_path}")

    manifest = adapter.get_manifest()
    manifest["target_date"] = args.target_date
    manifest_path = output_dir / "inference_manifest.json"
    with open(str(manifest_path), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(f"Rows: {len(result)}, Hours: {sorted(result['hour_business'].unique())}")
    cols = [c for c in ["ds", "hour_business", "period", "y_pred", "y_true"] if c in result.columns]
    print(result[cols].head())
    logger.info("✅ TiRex single-day complete!")


if __name__ == "__main__":
    main()
