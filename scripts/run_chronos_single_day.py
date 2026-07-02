"""
run_chronos_single_day.py — Zero-shot Chronos single-day prediction.

Usage:
    python scripts/run_chronos_single_day.py ^
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

# ── Ensure src on path (absolute via os.path) ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from src.common.data_loader import load_data
from src.models.chronos_adapter import ChronosAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Chronos single-day zero-shot prediction")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to CSV data. If omitted, reads from configs/paths.yaml.")
    parser.add_argument("--target-date", type=str, default="2026-02-15", help="YYYY-MM-DD")
    parser.add_argument("--task", type=str, default="dayahead", choices=["dayahead", "realtime"])
    parser.add_argument("--context-days", type=int, default=30, help="Context window in days")
    parser.add_argument("--output-dir", type=str, default="outputs/chronos_single_day")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu / None=auto")
    parser.add_argument("--no-fallback", action="store_true",
                        help="If set, do NOT fall back to Chronos-Bolt on failure")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve data path
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

    context_length = args.context_days * 24

    # ── 1. Load data ──
    logger.info(f"Loading data from {data_path}")
    df = load_data(data_path, target=args.task)
    logger.info(f"Data loaded: {len(df)} rows, {df['ds'].min()} ~ {df['ds'].max()}")

    # ── 2. Load Chronos (with optional fallback) ──
    logger.info("Loading Chronos model...")
    adapter = ChronosAdapter(
        context_length=context_length,
        device=args.device,
    )

    try:
        loaded_name = adapter.load()
        logger.info(f"Loaded model: {loaded_name}")
        if adapter.is_fallback:
            logger.warning(f"USED FALLBACK: {adapter.fallback_reason}")
    except Exception as e:
        logger.error(f"Failed to load Chronos: {e}")
        if args.no_fallback:
            logger.error("Fallback disabled. Exiting.")
            sys.exit(1)
        raise

    # ── 3. Predict target day ──
    logger.info(f"Predicting {args.target_date} ({args.task})...")
    try:
        result = adapter.predict_day(df, args.target_date, task=args.task, y_col="y")
    except Exception as e:
        logger.error(f"Chronos prediction failed for {args.target_date}: {e}")
        raise

    # ── 4. Save outputs ──
    model_tag = adapter.fallback_model_name if adapter.is_fallback else adapter.model_name

    # Prediction CSV
    pred_path = output_dir / f"{model_tag}_{args.task}_{args.target_date}.csv"
    result.to_csv(str(pred_path), index=False, encoding="utf-8-sig")
    logger.info(f"Prediction saved to {pred_path}")

    # Manifest
    manifest = adapter.get_manifest()
    manifest["target_date"] = args.target_date
    manifest["task"] = args.task
    manifest["result_rows"] = len(result)
    manifest_path = output_dir / "inference_manifest.json"
    with open(str(manifest_path), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info(f"Manifest saved to {manifest_path}")

    # Debug file with quantiles if available
    if "y_pred_p10" in result.columns:
        debug_path = output_dir / f"debug_{model_tag}_{args.task}_{args.target_date}.csv"
        result.to_csv(str(debug_path), index=False, encoding="utf-8-sig")
        logger.info(f"Debug (with quantiles) saved to {debug_path}")

    # ── 5. Summary ──
    logger.info(f"\n{'='*60}")
    logger.info(f"Chronos Single-Day Prediction Complete")
    logger.info(f"  Model:    {model_tag}")
    logger.info(f"  Fallback: {adapter.is_fallback}")
    if adapter.is_fallback:
        logger.info(f"  Reason:   {adapter.fallback_reason}")
    logger.info(f"  Target:   {args.target_date} ({args.task})")
    logger.info(f"  Rows:     {len(result)}")
    if "hour_business" in result.columns:
        logger.info(f"  Hours:    {sorted(result['hour_business'].unique())}")
    logger.info(f"{'='*60}")

    print("\n--- Prediction (first 5 rows) ---")
    cols_show = [c for c in ["ds", "hour_business", "period", "y_pred", "y_true"] if c in result.columns]
    print(result[cols_show].head())
    print("---")


if __name__ == "__main__":
    main()
