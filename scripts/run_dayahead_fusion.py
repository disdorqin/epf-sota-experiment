"""
run_dayahead_fusion.py — Day-ahead fusion runner script.

Reads day-ahead predictions from input roots, runs fusion methods,
and outputs fused predictions + metrics.

Usage:
    python scripts/run_dayahead_fusion.py ^
        --input-root outputs/dayahead_30d_core ^
        --extra-input-root outputs/dayahead_specialists_30d ^
        --output-root outputs/dayahead_fusion_30d ^
        --lookback-days 7
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `import src` works
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Import fusion methods ─────────────────────────────────────────────────────

try:
    from src.fusion.pair_fusion import (
        simple_average as pf_simple_average,
        inverse_smape_weight as pf_inverse_smape_weight,
        period_best as pf_period_best,
        load_model_predictions as pf_load,
    )
    from src.fusion.dayahead_fusion import (
        inverse_smape_period as daf_inverse_smape_period,
        inverse_smape_hour as daf_inverse_smape_hour,
        winner_by_period as daf_winner_by_period,
        winner_by_hour as daf_winner_by_hour,
        ridge_stacking as daf_ridge_stacking,
        load_dayahead_predictions as daf_load,
    )
    from src.common.metrics import smape_floor50, mae, rmse, peak_mae_q90, negative_price_hit_rate
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)


# ── Find input root ───────────────────────────────────────────────────────────


def _find_input_root(cli_root: str) -> Path:
    candidates = [
        Path(cli_root),
        Path("outputs/dayahead_30d_core"),
        Path("outputs/catboost_tabpfn_30d"),
        Path("outputs/sota_walkforward"),
        Path("outputs/sota_walkforward_7d"),
    ]
    for p in candidates:
        pred_dir = p / "predictions"
        if pred_dir.exists():
            csvs = list(pred_dir.glob("*dayahead*.csv")) + list(pred_dir.glob("*catboost*sota*.csv")) + list(pred_dir.glob("*tabpfn*.csv"))
            if len(csvs) >= 1:
                logger.info(f"Found input root: {p}")
                return p
    return None


# ── Compute metrics for a prediction DataFrame ────────────────────────────────


def _compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute metrics for a prediction DataFrame.
    Returns dict with keys: model_name, task, MAE, RMSE, sMAPE_floor50,
                          peak_MAE_q90, negative_price_hit_rate, n
    """
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    model_name = df["model_name"].iloc[0] if "model_name" in df.columns else "unknown"
    task = df["task"].iloc[0] if "task" in df.columns else "dayahead"

    return {
        "model_name": model_name,
        "task": task,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE_floor50": smape_floor50(y_true, y_pred),
        "peak_MAE_q90": peak_mae_q90(y_true, y_pred),
        "negative_price_hit_rate": negative_price_hit_rate(y_true, y_pred),
        "n": len(df),
    }


# ── Main fusion logic ──────────────────────────────────────────────────────────


def _run_fusion(input_root: Path, extra_input_root: Path | None, output_root: Path, lookback_days: int) -> None:
    """
    Run all fusion methods and save results.
    """
    pred_dir = input_root / "predictions"
    extra_pred_dir = extra_input_root / "predictions" if extra_input_root else None

    # ── Load base predictions ─────────────────────────────────────────────────
    logger.info("Loading base predictions...")
    base_preds = {}
    if pred_dir.exists():
        base_preds = daf_load(pred_dir)
    if len(base_preds) == 0:
        # Try pair_fusion loader
        base_preds = pf_load(pred_dir)

    # Also load from extra root
    extra_preds = {}
    if extra_pred_dir and extra_pred_dir.exists():
        logger.info("Loading extra predictions (specialists)...")
        extra_preds = daf_load(extra_pred_dir)
        if len(extra_preds) == 0:
            extra_preds = pf_load(extra_pred_dir)

    all_preds = {**base_preds, **extra_preds}

    # ── Find catboost and tabpfn base models ────────────────────────────────
    cb_name = None
    tp_name = None
    for name in all_preds:
        if "catboost" in name and "sota" in name:
            cb_name = name
        if "tabpfn" in name and "sota" in name:
            tp_name = name

    if cb_name is None or tp_name is None:
        logger.error(f"Cannot find base catboost and tabpfn predictions. Found: {list(all_preds.keys())}")
        print("day-ahead predictions not found yet. Run day-ahead walk-forward first.")
        sys.exit(1)

    cb_df = all_preds[cb_name]
    tp_df = all_preds[tp_name]
    logger.info(f"Base models: {cb_name} ({len(cb_df)} rows), {tp_name} ({len(tp_df)} rows)")

    # ── Create output directories ─────────────────────────────────────────────
    fusion_dir = output_root / "fusion"
    fusion_dir.mkdir(parents=True, exist_ok=True)

    # ── Run fusion methods ───────────────────────────────────────────────────
    fusion_methods = []

    # 1. simple_average (reuse pair_fusion)
    logger.info("Running simple_average...")
    fused_simple = pf_simple_average(cb_df, tp_df)
    fused_simple["model_name"] = "fused_simple_average_dayahead"
    fusion_methods.append(("fused_simple_average_dayahead", fused_simple))

    # 2. inverse_smape_period
    logger.info("Running inverse_smape_period...")
    try:
        fused_period = daf_inverse_smape_period(cb_df, tp_df, past_days=lookback_days)
        fusion_methods.append(("fused_inverse_smape_period_dayahead", fused_period))
    except Exception as e:
        logger.warning(f"inverse_smape_period failed: {e}")

    # 3. inverse_smape_hour
    logger.info("Running inverse_smape_hour...")
    try:
        fused_hour = daf_inverse_smape_hour(cb_df, tp_df, past_days=lookback_days)
        fusion_methods.append(("fused_inverse_smape_hour_dayahead", fused_hour))
    except Exception as e:
        logger.warning(f"inverse_smape_hour failed: {e}")

    # 4. winner_by_period
    logger.info("Running winner_by_period...")
    try:
        fused_winner_period = daf_winner_by_period(cb_df, tp_df, past_days=lookback_days)
        fusion_methods.append(("fused_winner_by_period_dayahead", fused_winner_period))
    except Exception as e:
        logger.warning(f"winner_by_period failed: {e}")

    # 5. winner_by_hour
    logger.info("Running winner_by_hour...")
    try:
        fused_winner_hour = daf_winner_by_hour(cb_df, tp_df, past_days=lookback_days)
        fusion_methods.append(("fused_winner_by_hour_dayahead", fused_winner_hour))
    except Exception as e:
        logger.warning(f"winner_by_hour failed: {e}")

    # 6. ridge_stacking
    logger.info("Running ridge_stacking...")
    try:
        fused_ridge = daf_ridge_stacking(cb_df, tp_df, past_days=lookback_days)
        fusion_methods.append(("fused_ridge_stacking_dayahead", fused_ridge))
    except Exception as e:
        logger.warning(f"ridge_stacking failed: {e}")

    # ── Also add base models to metrics ─────────────────────────────────────
    all_models_for_metrics = []
    for name, df in all_preds.items():
        df = df.copy()
        df["model_name"] = name
        all_models_for_metrics.append(df)
    for name, df in fusion_methods:
        all_models_for_metrics.append(df)

    # ── Save fused predictions ──────────────────────────────────────────────
    logger.info("Saving fused predictions...")
    for name, df in fusion_methods:
        out_path = fusion_dir / f"{name}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"  Saved {name}: {len(df)} rows -> {out_path}")

    # ── Compute and save fusion metrics ─────────────────────────────────────
    logger.info("Computing fusion metrics...")
    metrics_rows = []
    for df in all_models_for_metrics:
        # Only dayahead task
        if "task" in df.columns:
            df = df[df["task"] == "dayahead"].copy()
        if len(df) == 0:
            continue
        m = _compute_metrics(df)
        metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = fusion_dir / "fusion_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved fusion metrics: {metrics_path}")

    # ── Print summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Day-Ahead Fusion Summary")
    print(f"{'='*60}")
    for _, row in metrics_df.iterrows():
        print(f"  {row['model_name']:<45} sMAPE={row['sMAPE_floor50']:.2f}%  MAE={row['MAE']:.2f}")
    print(f"{'='*60}\n")

    # ── Save base predictions for reference ──────────────────────────────────
    for name, df in all_preds.items():
        out_path = fusion_dir / f"__base__{name}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Day-ahead fusion runner")
    parser.add_argument("--input-root", type=str, default="outputs/dayahead_30d_core", help="Primary input root (base models)")
    parser.add_argument("--extra-input-root", type=str, default=None, help="Extra input root (specialists)")
    parser.add_argument("--output-root", type=str, default="outputs/dayahead_fusion_30d", help="Output root")
    parser.add_argument("--lookback-days", type=int, default=7, help="Past days for rolling weights (default: 7)")
    args = parser.parse_args()

    input_root = _find_input_root(args.input_root)
    if input_root is None:
        print("day-ahead predictions not found yet. Run day-ahead walk-forward first.")
        sys.exit(1)

    extra_input_root = Path(args.extra_input_root) if args.extra_input_root else None
    if extra_input_root and not extra_input_root.exists():
        logger.warning(f"Extra input root not found: {extra_input_root}, ignoring")
        extra_input_root = None

    output_root = Path(args.output_root)

    print(f"Input root:  {input_root}")
    print(f"Output root: {output_root}")
    print(f"Lookback:    {args.lookback_days} days")
    print()

    _run_fusion(input_root, extra_input_root, output_root, args.lookback_days)


if __name__ == "__main__":
    main()
