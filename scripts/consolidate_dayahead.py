"""
consolidate_dayahead.py — Consolidate CatBoost + TabPFN 30-day day-ahead results.

Usage:
    python scripts/consolidate_dayahead.py

Reads from:
    outputs/catboost_30d/
    outputs/tabpfn_30d/

Writes to:
    outputs/dayahead_30d_core/
"""

from __future__ import annotations

import logging
import os
import sys
import shutil
from pathlib import Path

import pandas as pd
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from src.common.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CB_DIR = Path("outputs/catboost_30d")
TP_DIR = Path("outputs/tabpfn_30d")
OUT_DIR = Path("outputs/dayahead_30d_core")


def main():
    # Check both runs complete
    cb_preds = CB_DIR / "predictions" / "catboost_sota_dayahead.csv"
    tp_preds = TP_DIR / "predictions" / "tabpfn_ts_sota_dayahead.csv"

    missing = []
    if not cb_preds.exists():
        missing.append(f"CatBoost: {cb_preds}")
    if not tp_preds.exists():
        missing.append(f"TabPFN: {tp_preds}")

    if missing:
        logger.warning("Missing prediction files:")
        for m in missing:
            logger.warning(f"  {m}")
        logger.warning("One or both runs are still in progress.")
        return

    # Copy predictions
    for src in [cb_preds, tp_preds]:
        dst = OUT_DIR / "predictions" / src.name
        shutil.copy2(str(src), str(dst))
        logger.info(f"Copied {src.name} → {dst}")

    # Copy metrics
    for sub in ["model_target_metrics.csv", "daily_metrics.csv", "model_period_metrics.csv", "summary.csv"]:
        for src_dir in [CB_DIR, TP_DIR]:
            src = src_dir / "metrics" / sub
            if src.exists():
                dst = OUT_DIR / "metrics" / f"{src_dir.name}_{sub}"
                shutil.copy2(str(src), str(dst))
                logger.info(f"Copied {src_dir.name}/{sub} → {dst}")

    # Copy manifests
    for src_dir in [CB_DIR, TP_DIR]:
        src = src_dir / "debug" / "run_manifest.json"
        if src.exists():
            dst = OUT_DIR / "debug" / f"{src_dir.name}_run_manifest.json"
            shutil.copy2(str(src), str(dst))

    # Build combined metrics
    cb = pd.read_csv(str(cb_preds), encoding="utf-8-sig")
    tp = pd.read_csv(str(tp_preds), encoding="utf-8-sig")

    # Summary
    rows = []
    for name, df in [("catboost_sota", cb), ("tabpfn_ts_sota", tp)]:
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        if valid.sum() < 2:
            continue
        m = compute_all_metrics(y_true[valid], y_pred[valid])
        m["model_name"] = name
        m["task"] = "dayahead"
        m["n"] = int(valid.sum())
        rows.append(m)

    summary = pd.DataFrame(rows)
    summary.to_csv(str(OUT_DIR / "metrics" / "combined_summary.csv"), index=False, encoding="utf-8-sig")
    logger.info("Saved combined_summary.csv")

    # Print
    print("\n" + "=" * 60)
    print("30-DAY DAY-AHEAD FINAL SUMMARY")
    print("=" * 60)
    cols = ["model_name", "MAE", "RMSE", "sMAPE_floor50", "peak_MAE_q90", "negative_price_hit_rate", "n"]
    print(summary[cols].to_string(index=False))
    print("=" * 60)

    # Verify rows
    print(f"\nCatBoost predictions: {len(cb)} rows (expected 720)")
    print(f"TabPFN predictions:   {len(tp)} rows (expected 720)")
    print(f"\n✅ Consolidation complete → {OUT_DIR}")


if __name__ == "__main__":
    main()
