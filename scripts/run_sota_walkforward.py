"""
run_sota_walkforward.py — Walk-forward evaluation for SOTA single models.

Usage:
    python scripts/run_sota_walkforward.py ^
        --source-repo "D:\...\electricity_forecast_model2.0" ^
        --data-path "D:\...\shandong_pmos_hourly.csv" ^
        --start 2026-02-01 ^
        --end 2026-02-03 ^
        --target both ^
        --models catboost_sota,chronos2_zero_shot ^
        --output-root outputs\sota_walkforward
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# ── Ensure src on path (absolute via os.path) ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from src.common.data_loader import load_data
from src.common.feature_builder import build_features
from src.common.metrics import compute_all_metrics
from src.common.output_schema import make_long_table
from src.models.catboost_adapter import CatBoostAdapter
from src.models.chronos_adapter import ChronosAdapter
from src.models.tabpfn_ts_adapter import TabPFNTSAdapter
from src.models.tirex_adapter import TiRexAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Walk-forward SOTA model evaluation")
    parser.add_argument("--source-repo", type=str, default=None,
                        help="Path to original repo (for baseline loading if needed)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to CSV data. If omitted, reads from configs/paths.yaml.")
    parser.add_argument("--start", type=str, default="2026-02-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-02-03", help="End date (YYYY-MM-DD)")
    parser.add_argument("--target", type=str, default="both", choices=["dayahead", "realtime", "both"])
    parser.add_argument("--models", type=str, default="catboost_sota,chronos2_zero_shot",
                        help="Comma-separated model names: catboost_sota, chronos2_zero_shot")
    parser.add_argument("--output-root", type=str, default="outputs/sota_walkforward")
    parser.add_argument("--train-window", type=str, default="all", choices=["all", "months"],
                        help="'all' = use all available history; 'months' = use --train-months")
    parser.add_argument("--train-months", type=int, default=12,
                        help="Training window in months (only used when --train-window=months)")
    parser.add_argument("--device", type=str, default="CPU", help="CatBoost device: CPU/GPU")
    parser.add_argument("--max-train-rows", type=int, default=50000,
                        help="Max training rows for TabPFN-TS")
    parser.add_argument("--allow-model-skip", action="store_true",
                        help="Allow models to be skipped if unavailable (e.g. TiRex)")
    return parser.parse_args()


def _resolve_models(model_str: str) -> list[str]:
    return [m.strip() for m in model_str.split(",") if m.strip()]


def _load_baseline_output(source_repo: Optional[str], task: str, model: str) -> Optional[pd.DataFrame]:
    """Try to load original LightGBM or TimesFM output from the source repo."""
    if not source_repo:
        return None
    repo = Path(source_repo)
    # Try to find existing outputs from original pipelines
    candidates = []
    if model == "lightgbm":
        candidates = [
            repo / "fusion_runs" / "dayahead" / "lightgbm_output.csv",
            repo / "lightGBM" / "outputs" / f"lightgbm_{task}.csv",
            repo / "outputs" / f"lightgbm_{task}.csv",
        ]
    elif model == "timesfm":
        candidates = [
            repo / "fusion_runs" / "dayahead" / "timesfm_output.csv",
            repo / "TimesFM" / "output" / f"timesfm_{task}.csv",
            repo / "outputs" / f"timesfm_{task}.csv",
        ]

    for c in candidates:
        if c.exists():
            logger.info(f"Found baseline output: {c}")
            df = pd.read_csv(str(c), encoding="utf-8-sig")
            return df
    return None


def _save_predictions(all_preds: list[pd.DataFrame], output_dir: Path):
    """Save predictions per model/task."""
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    by_key = {}
    for df in all_preds:
        key = f"{df['model_name'].iloc[0]}_{df['task'].iloc[0]}"
        if key not in by_key:
            by_key[key] = []
        by_key[key].append(df)

    for key, dfs in by_key.items():
        combined = pd.concat(dfs, ignore_index=True)
        path = pred_dir / f"{key}.csv"
        combined.to_csv(str(path), index=False, encoding="utf-8-sig")
        logger.info(f"Saved predictions: {path} ({len(combined)} rows)")


def _compute_period_metrics(all_preds: list[pd.DataFrame]) -> pd.DataFrame:
    """Compute metrics grouped by model, task, period."""
    rows = []
    for df in all_preds:
        model = df["model_name"].iloc[0]
        task = df["task"].iloc[0]
        for period, grp in df.groupby("period"):
            y_true = grp["y_true"].values
            y_pred = grp["y_pred"].values
            valid = ~(np.isnan(y_true) | np.isnan(y_pred))
            if valid.sum() < 2:
                continue
            metrics = compute_all_metrics(y_true[valid], y_pred[valid])
            metrics["model_name"] = model
            metrics["task"] = task
            metrics["period"] = period
            metrics["n"] = int(valid.sum())
            rows.append(metrics)
    return pd.DataFrame(rows)


def _compute_target_metrics(all_preds: list[pd.DataFrame]) -> pd.DataFrame:
    """Compute metrics grouped by model, task."""
    rows = []
    for df in all_preds:
        model = df["model_name"].iloc[0]
        task = df["task"].iloc[0]
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        if valid.sum() < 2:
            continue
        metrics = compute_all_metrics(y_true[valid], y_pred[valid])
        metrics["model_name"] = model
        metrics["task"] = task
        metrics["n"] = int(valid.sum())
        rows.append(metrics)
    return pd.DataFrame(rows)


def _compute_daily_metrics(all_preds: list[pd.DataFrame]) -> pd.DataFrame:
    """Compute daily metrics per model/task/target_day."""
    rows = []
    for df in all_preds:
        model = df["model_name"].iloc[0]
        task = df["task"].iloc[0]
        for target_day, grp in df.groupby("target_day"):
            y_true = grp["y_true"].values
            y_pred = grp["y_pred"].values
            valid = ~(np.isnan(y_true) | np.isnan(y_pred))
            if valid.sum() < 2:
                continue
            metrics = compute_all_metrics(y_true[valid], y_pred[valid])
            metrics["model_name"] = model
            metrics["task"] = task
            metrics["target_day"] = target_day
            metrics["n"] = int(valid.sum())
            rows.append(metrics)
    return pd.DataFrame(rows)


def _check_nan_and_missing(all_preds: list[pd.DataFrame], expected_dates: list[str]) -> dict:
    """Check for NaN rows and missing dates."""
    issues = {}
    for df in all_preds:
        model = df["model_name"].iloc[0]
        task = df["task"].iloc[0]
        key = f"{model}_{task}"

        # NaN check
        nan_mask = df["y_pred"].isna() | df["y_true"].isna()
        nan_count = int(nan_mask.sum())
        nan_dates = sorted(df.loc[nan_mask, "target_day"].unique().tolist()) if nan_count > 0 else []

        # Missing dates
        present_dates = set(df["target_day"].unique())
        missing_dates = sorted(set(expected_dates) - present_dates)

        issues[key] = {
            "nan_count": nan_count,
            "nan_dates": nan_dates,
            "missing_dates": missing_dates,
            "total_rows": len(df),
        }
    return issues


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

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "predictions").mkdir(parents=True, exist_ok=True)
    (output_root / "metrics").mkdir(parents=True, exist_ok=True)
    (output_root / "debug").mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)

    tasks = ["dayahead", "realtime"] if args.target == "both" else [args.target]
    models = _resolve_models(args.models)
    expected_dates = pd.date_range(start=args.start, end=args.end).strftime("%Y-%m-%d").tolist()

    logger.info(f"Walk-forward config:")
    logger.info(f"  Data:      {data_path}")
    logger.info(f"  Range:     {args.start} → {args.end}")
    logger.info(f"  Tasks:     {tasks}")
    logger.info(f"  Models:    {models}")
    logger.info(f"  Output:    {output_root}")

    # ── Load data ──
    logger.info("Loading data...")
    raw_realtime = load_data(data_path, target="realtime")
    raw_dayahead = load_data(data_path, target="dayahead")

    # ── Walk-forward loop ──
    all_predictions = []
    chronos_fallback_info = {}
    full_feature_df = None  # Will be built when needed
    run_manifest = {
        "start": args.start,
        "end": args.end,
        "tasks": tasks,
        "models": models,
        "train_window": args.train_window,
        "train_months": args.train_months if args.train_window == "months" else "all",
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "chronos_fallback": {},
        "failed_dates": [],
        "baselines_loaded": {},
    }

    for task in tasks:
        raw_df = raw_dayahead if task == "dayahead" else raw_realtime

        for model_name in models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting model={model_name} task={task}")

            # ── Initialize model ──
            if model_name == "catboost_sota":
                adapter = CatBoostAdapter(task_type=args.device)
            elif model_name in ("chronos2_zero_shot", "chronos_bolt_zero_shot"):
                adapter = ChronosAdapter()
                try:
                    loaded_name = adapter.load()
                    if adapter.is_fallback:
                        chronos_fallback_info[model_name] = {
                            "is_fallback": True,
                            "reason": adapter.fallback_reason,
                            "loaded": loaded_name,
                        }
                        run_manifest["chronos_fallback"][model_name] = adapter.fallback_reason
                        logger.warning(f"Chronos fallback: {adapter.fallback_reason}")
                except Exception as e:
                    logger.error(f"Failed to load Chronos: {e}. Skipping.")
                    run_manifest["failed_dates"].append(f"model_load_failed:{model_name}:{task}:{e}")
                    continue
            elif model_name == "tabpfn_ts_sota":
                adapter = TabPFNTSAdapter(max_train_rows=args.max_train_rows, device="cpu")
            elif model_name == "tirex_zero_shot":
                adapter = TiRexAdapter()
                loaded = adapter.load()
                if not loaded:
                    reason = adapter.unavailable_reason or "unknown"
                    logger.warning(f"TiRex unavailable: {reason}")
                    run_manifest["tirex_unavailable_reason"] = reason
                    if args.allow_model_skip:
                        logger.warning("Skipping TiRex (--allow-model-skip is set).")
                        continue
                    else:
                        logger.error("TiRex unavailable and --allow-model-skip not set. Exiting.")
                        sys.exit(1)
            else:
                logger.error(f"Unknown model: {model_name}")
                continue

            # ── For CatBoost: train once, then predict each day ──
            if model_name == "catboost_sota":
                # Build features on full data
                full_feature_df = build_features(raw_df)

            # ── Loop over target dates ──
            for target_date_str in expected_dates:
                target_dt = pd.Timestamp(target_date_str)
                logger.info(f"  → {target_date_str} ({task})")

                try:
                    if model_name == "catboost_sota":
                        # Determine training window
                        if args.train_window == "months":
                            train_start = target_dt - pd.DateOffset(months=args.train_months)
                            train_df = full_feature_df[
                                (full_feature_df["ds"] >= train_start) & (full_feature_df["ds"] < target_dt)
                            ].copy()
                        else:
                            # 'all' = use all data before target date
                            train_df = full_feature_df[full_feature_df["ds"] < target_dt].copy()

                        if len(train_df) < 2000:
                            logger.warning(f"    Insufficient training data ({len(train_df)} rows). Skipping.")
                            run_manifest["failed_dates"].append(f"{target_date_str}:{model_name}:no_train_data")
                            continue

                        val_start = target_dt - pd.DateOffset(days=30)
                        val_df = full_feature_df[
                            (full_feature_df["ds"] >= val_start) & (full_feature_df["ds"] < target_dt)
                        ].copy()

                        adapter.train(train_df, eval_df=val_df)
                        result = adapter.predict_day(full_feature_df, target_date_str, task=task)

                    elif model_name in ("chronos2_zero_shot", "chronos_bolt_zero_shot"):
                        result = adapter.predict_day(raw_df, target_date_str, task=task, y_col="y")

                    elif model_name == "tabpfn_ts_sota":
                        # Build features on full data (once)
                        if "full_feature_df" not in dir() or full_feature_df is None:
                            full_feature_df = build_features(raw_df)
                        # Determine training window
                        if args.train_window == "months":
                            train_start = target_dt - pd.DateOffset(months=args.train_months)
                            train_df = full_feature_df[
                                (full_feature_df["ds"] >= train_start) & (full_feature_df["ds"] < target_dt)
                            ].copy()
                        else:
                            train_df = full_feature_df[full_feature_df["ds"] < target_dt].copy()
                        if len(train_df) < 2000:
                            logger.warning(f"    Insufficient training data ({len(train_df)} rows). Skipping.")
                            run_manifest["failed_dates"].append(f"{target_date_str}:{model_name}:no_train_data")
                            continue
                        adapter.train(train_df)
                        result = adapter.predict_day(full_feature_df, target_date_str, task=task)

                    elif model_name == "tirex_zero_shot":
                        result = adapter.predict_day(raw_df, target_date_str, task=task, y_col="y")

                    else:
                        continue

                    # Validate
                    if len(result) != 24:
                        logger.warning(f"    Expected 24 rows, got {len(result)}")
                    if "hour_business" in result.columns:
                        hb = sorted(result["hour_business"].unique())
                        if hb != list(range(1, 25)):
                            logger.warning(f"    hour_business range unexpected: {hb}")

                    all_predictions.append(result)

                except Exception as e:
                    logger.error(f"    FAILED: {e}")
                    logger.error(traceback.format_exc())
                    run_manifest["failed_dates"].append(f"{target_date_str}:{model_name}:{task}:{str(e)[:100]}")

    # ── Try to load baseline outputs from source repo ──
    baseline_map = {}
    if args.source_repo:
        repo = Path(args.source_repo)
        logger.info("Checking for original baseline outputs...")
        for task in tasks:
            for sota_model in models:
                baseline_model = None
                if sota_model == "catboost_sota":
                    baseline_model = "lightgbm"
                elif "chronos" in sota_model:
                    baseline_model = "timesfm"
                if baseline_model:
                    candidates = []
                    if baseline_model == "lightgbm":
                        candidates = [
                            repo / "fusion_runs" / task / "lightgbm_output.csv",
                            repo / "lightGBM" / "outputs" / f"lightgbm_{task}.csv",
                            repo / "outputs" / f"lightgbm_{task}.csv",
                        ]
                    elif baseline_model == "timesfm":
                        candidates = [
                            repo / "fusion_runs" / task / "timesfm_output.csv",
                            repo / "TimesFM" / "output" / f"timesfm_{task}.csv",
                            repo / "outputs" / f"timesfm_{task}.csv",
                        ]
                    for c in candidates:
                        if c.exists():
                            logger.info(f"  Baseline found: {c}")
                            baseline_map[f"{baseline_model}_{task}"] = str(c)
                            break
        run_manifest["baselines_loaded"] = {
            k: "found" for k in baseline_map
        } if baseline_map else {"status": "not_found"}
    else:
        run_manifest["baselines_loaded"] = {"status": "no_source_repo"}

    # Save baseline info to debug
    baseline_path = output_root / "debug" / "baselines_found.json"
    with open(str(baseline_path), "w", encoding="utf-8") as f:
        json.dump(run_manifest["baselines_loaded"], f, ensure_ascii=False, indent=2)

    # ── Save all predictions ──
    _save_predictions(all_predictions, output_root)

    # ── Compute metrics ──
    if all_predictions:
        logger.info(f"Computing metrics from {len(all_predictions)} prediction sets...")
        daily_metrics = _compute_daily_metrics(all_predictions)
        if not daily_metrics.empty:
            daily_metrics.to_csv(str(output_root / "metrics" / "daily_metrics.csv"),
                                 index=False, encoding="utf-8-sig")
        else:
            logger.warning("daily_metrics is empty — skipping save")

        period_metrics = _compute_period_metrics(all_predictions)
        if not period_metrics.empty:
            period_metrics.to_csv(str(output_root / "metrics" / "model_period_metrics.csv"),
                                  index=False, encoding="utf-8-sig")
        else:
            logger.warning("period_metrics is empty — skipping save")

        target_metrics = _compute_target_metrics(all_predictions)
        if not target_metrics.empty:
            target_metrics.to_csv(str(output_root / "metrics" / "model_target_metrics.csv"),
                                  index=False, encoding="utf-8-sig")

            # Summary: best sMAPE per model/task
            required_cols = ["model_name", "task"]
            missing = [c for c in required_cols if c not in target_metrics.columns]
            if not missing:
                summary = target_metrics.groupby(required_cols).agg(
                    avg_MAE=("MAE", "mean"),
                    avg_RMSE=("RMSE", "mean"),
                    avg_sMAPE=("sMAPE_floor50", "mean"),
                    avg_peak_MAE=("peak_MAE_q90", "mean"),
                    avg_neg_hit_rate=("negative_price_hit_rate", "mean"),
                    total_n=("n", "sum"),
                ).reset_index()
                summary.to_csv(str(output_root / "metrics" / "summary.csv"),
                               index=False, encoding="utf-8-sig")
                logger.info(f"\n{'='*60}")
                logger.info("SUMMARY:")
                print(summary.to_string())
            else:
                logger.warning(f"Cannot group summary — missing columns: {missing}")
        else:
            logger.warning("target_metrics is empty — skipping summary")

    # ── Try to load baseline outputs from source repo ──
    baseline_map = {}
    if args.source_repo:
        repo = Path(args.source_repo)
        logger.info("Checking for original baseline outputs...")
        for task in tasks:
            for sota_model in models:
                baseline_model = None
                if sota_model == "catboost_sota":
                    baseline_model = "lightgbm"
                elif "chronos" in sota_model:
                    baseline_model = "timesfm"
                if baseline_model:
                    candidates = []
                    if baseline_model == "lightgbm":
                        candidates = [
                            repo / "fusion_runs" / task / "lightgbm_output.csv",
                            repo / "lightGBM" / "outputs" / f"lightgbm_{task}.csv",
                            repo / "outputs" / f"lightgbm_{task}.csv",
                        ]
                    elif baseline_model == "timesfm":
                        candidates = [
                            repo / "fusion_runs" / task / "timesfm_output.csv",
                            repo / "TimesFM" / "output" / f"timesfm_{task}.csv",
                            repo / "outputs" / f"timesfm_{task}.csv",
                        ]
                    for c in candidates:
                        if c.exists():
                            logger.info(f"  Baseline found: {c}")
                            baseline_map[f"{baseline_model}_{task}"] = str(c)
                            break
        run_manifest["baselines_loaded"] = {
            k: "found" for k in baseline_map
        } if baseline_map else {"status": "not_found"}
    else:
        run_manifest["baselines_loaded"] = {"status": "no_source_repo"}

    # Save baseline info to debug
    baseline_path = output_root / "debug" / "baselines_found.json"
    with open(str(baseline_path), "w", encoding="utf-8") as f:
        json.dump(run_manifest["baselines_loaded"], f, ensure_ascii=False, indent=2)

    # ── Check issues ──
    issues = _check_nan_and_missing(all_predictions, expected_dates)
    issues_path = output_root / "debug" / "data_quality_issues.json"
    with open(str(issues_path), "w", encoding="utf-8") as f:
        json.dump(issues, f, ensure_ascii=False, indent=2)

    # ── Save run manifest ──
    run_manifest["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_manifest["total_predictions"] = len(all_predictions)
    run_manifest["n_failed_dates"] = len(run_manifest["failed_dates"])
    manifest_path = output_root / "debug" / "run_manifest.json"
    with open(str(manifest_path), "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, ensure_ascii=False, indent=2)
    logger.info(f"Run manifest saved to {manifest_path}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Walk-forward complete. Output: {output_root}")
    if run_manifest["failed_dates"]:
        logger.warning(f"Failed dates: {len(run_manifest['failed_dates'])}")
        for fd in run_manifest["failed_dates"][:10]:
            logger.warning(f"  {fd}")
    baseline_status = run_manifest.get("baselines_loaded", {})
    if baseline_status.get("status") == "not_found":
        logger.info("Baseline outputs not found in source repo.")
    elif baseline_status.get("status") == "no_source_repo":
        logger.info("No --source-repo specified. Skipping baseline search.")
    elif any("found" in str(v) for v in baseline_status.values()):
        logger.info(f"Baselines loaded: {list(baseline_status.keys())}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
