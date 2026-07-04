#!/usr/bin/env python3
"""
run_dayahead_model_zoo.py — Run day-ahead model zoo and output unified predictions.

This script runs the specified day-ahead models and outputs unified
long-table predictions for downstream fusion.

Usage:
  python scripts/run_dayahead_model_zoo.py --models default
  python scripts/run_dayahead_model_zoo.py --models cfg05
  python scripts/run_dayahead_model_zoo.py --models cfg05,best_two_average
"""

import sys, os, argparse, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from pathlib import Path
from src.registry.dayahead_models import (
    DAYAHEAD_MODELS, INVALID_MODELS, DEFAULT_FUSION_POOL,
    raise_if_invalid, get_model_info,
)
from src.common.metrics import smape_floor50, compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_OUT_DIR = Path("outputs/dayahead_model_zoo_30d")


def _ensure_output_dir():
    """Create output directory structure."""
    for d in ["predictions", "metrics", "reports"]:
        (_OUT_DIR / d).mkdir(parents=True, exist_ok=True)


def _standardize_schema(df, model_name):
    """Standardize output schema for a model's predictions."""
    required_cols = [
        "task", "model_name", "target_day", "business_day",
        "ds", "hour_business", "period", "y_true", "y_pred",
    ]
    # Ensure all required columns exist
    for col in required_cols:
        if col not in df.columns:
            if col == "task":
                df[col] = "dayahead"
            elif col == "model_name":
                df[col] = model_name
            else:
                raise ValueError(f"Missing required column: {col}")
    return df[required_cols]


def _run_cfg05():
    """Run cfg05 champion model."""
    logger.info("Running cfg05 champion...")
    runner = Path("scripts/run_champion_cfg05.py")
    if not runner.exists():
        raise FileNotFoundError(f"Runner not found: {runner}")
    # Run the champion script
    import subprocess
    result = subprocess.run(
        [sys.executable, str(runner)],
        cwd=str(Path(__file__).parent.parent),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error(f"cfg05 runner failed: {result.stderr}")
        raise RuntimeError("cfg05 runner failed")
    # Read output
    out_path = Path("outputs/dayahead_champion_cfg05_30d/predictions/cfg05_dayahead.csv")
    if not out_path.exists():
        raise FileNotFoundError(f"cfg05 output not found: {out_path}")
    df = pd.read_csv(str(out_path), encoding="utf-8-sig")
    df["model_name"] = "cfg05"
    df["task"] = "dayahead"
    logger.info(f"cfg05: {len(df)} rows")
    return df


def _run_best_two_average():
    """Run best_two_average model (average of trial_02 + trial_24)."""
    logger.info("Running best_two_average (trial_02 + trial_24)...")
    # Try to read existing predictions
    p02_path = Path("outputs/dayahead_lgbm_stage2_30d/predictions/trial_02_w150_nl255_lr0.03_dayahead.csv")
    p24_path = Path("outputs/dayahead_lgbm_stage2_30d/predictions/trial_24_w150_nl255_lr0.03_dayahead.csv")
    if not p02_path.exists() or not p24_path.exists():
        raise FileNotFoundError(
            f"best_two_average needs trial_02 and trial_24 predictions. "
            f"Please run Stage2 first."
        )
    df02 = pd.read_csv(str(p02_path), encoding="utf-8-sig")
    df24 = pd.read_csv(str(p24_path), encoding="utf-8-sig")
    # Merge on key columns
    key_cols = ["target_day", "business_day", "ds", "hour_business", "period"]
    merged = df02[key_cols + ["y_true", "y_pred"]].rename(columns={"y_pred": "y_pred_02"})
    merged = merged.merge(
        df24[key_cols + ["y_pred"]].rename(columns={"y_pred": "y_pred_24"}),
        on=key_cols, how="inner",
    )
    merged["y_pred"] = (merged["y_pred_02"] + merged["y_pred_24"]) / 2.0
    merged["model_name"] = "best_two_average"
    merged["task"] = "dayahead"
    logger.info(f"best_two_average: {len(merged)} rows")
    return merged.drop(columns=["y_pred_02", "y_pred_24"])


def _run_stage3_business_fixed():
    """Run stage3 with correct business-day mapping."""
    logger.info("Running stage3_business_fixed...")
    # Try to read existing predictions
    out_path = Path("outputs/dayahead_lgbm_stage3_30d/predictions/stage3_business_fixed_dayahead.csv")
    if out_path.exists():
        df = pd.read_csv(str(out_path), encoding="utf-8-sig")
        df["model_name"] = "stage3_business_fixed"
        df["task"] = "dayahead"
        logger.info(f"stage3_business_fixed: {len(df)} rows (from cached output)")
        return df
    # If not available, need to run Stage3 with correct mapping
    # This requires running the full Stage3 pipeline
    raise NotImplementedError(
        "stage3_business_fixed runner not yet implemented. "
        "Please provide predictions at: outputs/dayahead_lgbm_stage3_30d/predictions/stage3_business_fixed_dayahead.csv"
    )


def _run_catboost_spike_residual():
    """Run CatBoost spike residual model."""
    logger.info("Running catboost_spike_residual...")
    # Try to read existing predictions
    out_path = Path("outputs/dayahead_catboost_spike_residual/predictions/catboost_spike_residual_dayahead.csv")
    if out_path.exists():
        df = pd.read_csv(str(out_path), encoding="utf-8-sig")
        df["model_name"] = "catboost_spike_residual"
        df["task"] = "dayahead"
        logger.info(f"catboost_spike_residual: {len(df)} rows (from cached output)")
        return df
    raise NotImplementedError(
        "catboost_spike_residual runner not yet implemented. "
        "Please provide predictions at: outputs/dayahead_catboost_spike_residual/predictions/catboost_spike_residual_dayahead.csv"
    )


def _run_catboost_sota():
    """Run CatBoost sota baseline."""
    logger.info("Running catboost_sota...")
    # Try to read existing predictions
    out_path = Path("outputs/dayahead_catboost_sota/predictions/catboost_sota_dayahead.csv")
    if out_path.exists():
        df = pd.read_csv(str(out_path), encoding="utf-8-sig")
        df["model_name"] = "catboost_sota"
        df["task"] = "dayahead"
        logger.info(f"catboost_sota: {len(df)} rows (from cached output)")
        return df
    raise NotImplementedError(
        "catboost_sota runner not yet implemented. "
        "Please provide predictions at: outputs/dayahead_catboost_sota/predictions/catboost_sota_dayahead.csv"
    )


def run_model(model_id):
    """Run a single model and return predictions DataFrame."""
    raise_if_invalid(model_id)
    if model_id == "cfg05":
        return _run_cfg05()
    elif model_id == "best_two_average":
        return _run_best_two_average()
    elif model_id == "stage3_business_fixed":
        return _run_stage3_business_fixed()
    elif model_id == "catboost_spike_residual":
        return _run_catboost_spike_residual()
    elif model_id == "catboost_sota":
        return _run_catboost_sota()
    else:
        raise ValueError(f"Unknown model_id: {model_id}")


def main():
    parser = argparse.ArgumentParser(description="Run day-ahead model zoo")
    parser.add_argument(
        "--models", type=str, default="default",
        help="Comma-separated model IDs, or 'default' for DEFAULT_FUSION_POOL",
    )
    args = parser.parse_args()

    # ── Parse model list ──
    if args.models == "default":
        model_ids = DEFAULT_FUSION_POOL
    else:
        model_ids = [m.strip() for m in args.models.split(",")]
    logger.info(f"Models to run: {model_ids}")

    # ── Validate all models before running ──
    for mid in model_ids:
        raise_if_invalid(mid)
    logger.info("All models validated (not blacklisted)")

    # ── Run models ──
    _ensure_output_dir()
    all_preds = []
    metrics_rows = []
    for mid in model_ids:
        try:
            df = run_model(mid)
            df = _standardize_schema(df, mid)
            all_preds.append(df)
            # Calculate metrics
            smape = smape_floor50(df["y_true"].values, df["y_pred"].values)
            m = compute_all_metrics(df["y_true"].values, df["y_pred"].values)
            m["model_name"] = mid
            m["smape_floor50"] = smape
            metrics_rows.append(m)
            logger.info(f"  {mid}: sMAPE_floor50 = {smape:.4f}% ({len(df)} rows)")
        except NotImplementedError as e:
            logger.warning(f"  {mid}: {e}")
            continue
        except Exception as e:
            logger.error(f"  {mid}: FAILED: {e}")
            continue

    if not all_preds:
        logger.error("No models ran successfully!")
        return

    # ── Save unified predictions ──
    unified = pd.concat(all_preds, ignore_index=True)
    out_path = _OUT_DIR / "predictions" / "model_zoo_unified.csv"
    unified.to_csv(str(out_path), index=False, encoding="utf-8-sig")
    logger.info(f"Saved unified predictions: {out_path}")

    # ── Save per-model predictions ──
    for mid in model_ids:
        mid_preds = unified[unified["model_name"] == mid]
        if len(mid_preds) > 0:
            mid_path = _OUT_DIR / "predictions" / f"{mid}_dayahead.csv"
            mid_preds.to_csv(str(mid_path), index=False, encoding="utf-8-sig")
            logger.info(f"Saved {mid}: {mid_path}")

    # ── Save metrics ──
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(str(_OUT_DIR / "metrics" / "model_zoo_summary.csv"), index=False, encoding="utf-8-sig")
    logger.info(f"Saved metrics: {_OUT_DIR / 'metrics' / 'model_zoo_summary.csv'}")

    # ── Print summary ──
    print()
    print("=" * 65)
    print("DAY-AHEAD MODEL ZOO SUMMARY")
    print("=" * 65)
    print()
    for mid in model_ids:
        mid_metrics = [m for m in metrics_rows if m["model_name"] == mid]
        if mid_metrics:
            m = mid_metrics[0]
            print(f"  {mid}: sMAPE_floor50 = {m['smape_floor50']:.4f}%")
        else:
            print(f"  {mid}: NOT RUN")
    print()
    print("=" * 65)
    print(f"Unified predictions: {len(unified)} rows")
    print(f"Output: {_OUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
