"""
run_pair_fusion.py — Run pairwise fusion on CatBoost + TabPFN 30-day predictions.

Usage (as required by user spec):
    python scripts/run_pair_fusion.py ^
        --input-root outputs/catboost_tabpfn_30d ^
        --lookback-days 7

Expected directory layout under --input-root:
    predictions/
        catboost_sota_dayahead.csv
        catboost_sota_realtime.csv
        tabpfn_ts_sota_dayahead.csv
        tabpfn_ts_sota_realtime.csv
    fusion/              (output)
        fused_simple_average.csv
        fused_inverse_smape_weight.csv
        fused_period_best.csv
        fusion_metrics.csv
    metrics/
        model_target_metrics.csv   (optional, used for fallback weights)

If 30d predictions are not found yet, prints:
    "30d predictions not found yet. Run walk-forward first."
and exits with code 1.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Ensure src on path (absolute, robust to spaces) ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from src.common.metrics import compute_all_metrics
from src.fusion.pair_fusion import (
    load_model_predictions,
    simple_average,
    inverse_smape_weight,
    period_best,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _check_predictions_exist(pred_dir: Path) -> bool:
    """Check that at least one CatBoost and one TabPFN CSV exist."""
    cb_files = list(pred_dir.glob("catboost_sota_*.csv"))
    tp_files = list(pred_dir.glob("tabpfn_ts_sota_*.csv"))
    return len(cb_files) > 0 and len(tp_files) > 0


def _safe_metric(df: pd.DataFrame, metric_name: str) -> float | None:
    """Safely extract a single metric value from a metrics DataFrame."""
    if metric_name not in df.columns:
        return None
    vals = df[metric_name].dropna()
    if len(vals) == 0:
        return None
    return float(vals.iloc[0])


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    pred_dir = input_root / "predictions"
    fusion_dir = input_root / "fusion"
    metrics_dir = input_root / "metrics"
    fusion_dir.mkdir(parents=True, exist_ok=True)

    # ── Early exit if predictions not found ──
    if not pred_dir.exists() or not _check_predictions_exist(pred_dir):
        print("30d predictions not found yet. Run walk-forward first.")
        logger.error(f"Prediction directory not ready: {pred_dir}")
        sys.exit(1)

    logger.info(f"Loading predictions from {pred_dir}")
    preds = load_model_predictions(pred_dir)
    if len(preds) == 0:
        print("30d predictions not found yet. Run walk-forward first.")
        sys.exit(1)

    logger.info(f"Loaded {len(preds)} prediction files: {sorted(preds.keys())}")

    # ── Identify model keys ──
    cb_keys = sorted([k for k in preds if k.startswith("catboost_sota")])
    tp_keys = sorted([k for k in preds if k.startswith("tabpfn_ts_sota")])
    logger.info(f"CatBoost keys: {cb_keys}")
    logger.info(f"TabPFN keys: {tp_keys}")

    if len(cb_keys) == 0 or len(tp_keys) == 0:
        print("30d predictions not found yet. Run walk-forward first.")
        logger.error("Missing CatBoost or TabPFN prediction files.")
        sys.exit(1)

    # ── Load metrics for reference (optional) ──
    metrics_path = metrics_dir / "model_target_metrics.csv"
    if metrics_path.exists():
        single_metrics = pd.read_csv(str(metrics_path), encoding="utf-8-sig")
        logger.info(f"Loaded metrics: {len(single_metrics)} rows")
    else:
        single_metrics = None
        logger.info("No metrics file found — will compute weights from predictions only.")

    # ── Run fusion per task_suffix ──
    fusion_results = []  # list of DataFrames, each with model_name column

    task_suffixes = ["dayahead", "realtime"]

    for task_suffix in task_suffixes:
        cb_key = f"catboost_sota_{task_suffix}"
        tp_key = f"tabpfn_ts_sota_{task_suffix}"

        if cb_key not in preds or tp_key not in preds:
            logger.warning(f"Skipping '{task_suffix}': missing predictions ({cb_key}, {tp_key})")
            continue

        cb_df = preds[cb_key].copy()
        tp_df = preds[tp_key].copy()

        # ── Normalize column names ──
        # Ensure y_pred exists (some CSVs may have different names)
        for df_ in [cb_df, tp_df]:
            if "y_pred" not in df_.columns and "prediction" in df_.columns:
                df_["y_pred"] = df_["prediction"]
            if "y_true" not in df_.columns and "actual" in df_.columns:
                df_["y_true"] = df_["actual"]

        # Add model_name if missing
        if "model_name" not in cb_df.columns:
            cb_df["model_name"] = "catboost_sota"
        if "model_name" not in tp_df.columns:
            tp_df["model_name"] = "tabpfn_ts_sota"

        # Ensure task column
        if "task" not in cb_df.columns:
            cb_df["task"] = task_suffix
        if "task" not in tp_df.columns:
            tp_df["task"] = task_suffix

        logger.info(f"[{task_suffix}] CatBoost: {len(cb_df)} rows, TabPFN: {len(tp_df)} rows")

        # ── 1. Simple average ──
        logger.info(f"[{task_suffix}] Computing simple_average...")
        fused_avg = simple_average(cb_df, tp_df)
        fused_avg["task"] = task_suffix
        fusion_results.append(fused_avg)

        # ── 2. Inverse sMAPE weight ──
        logger.info(f"[{task_suffix}] Computing inverse_smape_weight (lookback={args.lookback_days})...")
        fused_isw = inverse_smape_weight(cb_df, tp_df, past_days=args.lookback_days)
        fused_isw["task"] = task_suffix
        fusion_results.append(fused_isw)

        # ── 3. Period best ──
        logger.info(f"[{task_suffix}] Computing period_best (lookback={args.lookback_days})...")
        fused_pb = period_best(cb_df, tp_df, past_days=args.lookback_days)
        fused_pb["task"] = task_suffix
        fusion_results.append(fused_pb)

        # Also append single-model predictions (for metrics comparison)
        cb_copy = cb_df.copy()
        cb_copy["model_name"] = "catboost_sota"
        fusion_results.append(cb_copy)
        tp_copy = tp_df.copy()
        tp_copy["model_name"] = "tabpfn_ts_sota"
        fusion_results.append(tp_copy)

    if len(fusion_results) == 0:
        logger.error("No fusion results produced. Check prediction files.")
        sys.exit(1)

    # ── Save fused predictions ──
    combined = pd.concat(fusion_results, ignore_index=True)
    # Deduplicate: keep only needed columns for output
    out_cols = [c for c in combined.columns if not (c.endswith("_cb") or c.endswith("_tp"))]
    combined_out = combined[out_cols].copy()

    # Save per fusion method
    for model_name, grp in combined_out.groupby("model_name"):
        out_path = fusion_dir / f"{model_name}.csv"
        grp.to_csv(str(out_path), index=False, encoding="utf-8-sig")
        logger.info(f"Saved {out_path} ({len(grp)} rows)")

    # ── Compute fusion metrics ──
    logger.info("Computing fusion metrics...")
    metric_rows = []

    for model_name, grp in combined_out.groupby("model_name"):
        if "y_true" not in grp.columns or "y_pred" not in grp.columns:
            logger.warning(f"  {model_name}: missing y_true/y_pred, skipping metrics")
            continue
        y_true = grp["y_true"].values
        y_pred = grp["y_pred"].values
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        if valid.sum() < 2:
            logger.warning(f"  {model_name}: insufficient valid values ({valid.sum()}), skipping")
            continue
        m = compute_all_metrics(y_true[valid], y_pred[valid])
        m["model_name"] = model_name
        m["task"] = grp["task"].iloc[0] if "task" in grp.columns else "unknown"
        m["n"] = int(valid.sum())
        # Also add task-level breakdown
        metric_rows.append(m)

    # Also compute per-task metrics (required: model_name + task in output)
    for (model_name, task_val), grp in combined_out.groupby(["model_name", "task"]):
        y_true = grp["y_true"].values
        y_pred = grp["y_pred"].values
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        if valid.sum() < 2:
            continue
        m = compute_all_metrics(y_true[valid], y_pred[valid])
        m["model_name"] = model_name
        m["task"] = task_val
        m["n"] = int(valid.sum())
        # Avoid duplicates: only add if not already present
        already = any(
            r["model_name"] == model_name and r.get("task") == task_val
            for r in metric_rows
        )
        if not already:
            metric_rows.append(m)

    fusion_metrics = pd.DataFrame(metric_rows)

    # Ensure required columns exist
    required_cols = ["model_name", "task", "MAE", "RMSE", "sMAPE_floor50",
                     "peak_MAE_q90", "negative_price_hit_rate", "n"]
    for col in required_cols:
        if col not in fusion_metrics.columns:
            fusion_metrics[col] = np.nan

    fusion_metrics_path = fusion_dir / "fusion_metrics.csv"
    fusion_metrics.to_csv(str(fusion_metrics_path), index=False, encoding="utf-8-sig")
    logger.info(f"Fusion metrics saved to {fusion_metrics_path} ({len(fusion_metrics)} rows)")

    # ── Print summary ──
    print("\n" + "=" * 72)
    print("FUSION METRICS SUMMARY")
    print("=" * 72)
    summary_cols = [c for c in required_cols if c in fusion_metrics.columns]
    print(fusion_metrics[summary_cols].to_string(index=False))
    print("=" * 72)
    print(f"\nOutput directory: {fusion_dir}")

    logger.info("Done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pairwise fusion on CatBoost + TabPFN 30-day predictions"
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="outputs/catboost_tabpfn_30d",
        help="Root directory containing predictions/ and metrics/ subdirectories",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="Number of past days for rolling weight calculation (default: 7)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
