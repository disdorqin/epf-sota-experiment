"""
run_dayahead_tabular_model_search.py — Search LightGBM/XGBoost day-ahead models
with multiple training windows and configs.

Usage:
    python scripts/run_dayahead_tabular_model_search.py ^
        --start 2026-02-01 --end 2026-03-02 ^
        --models lightgbm,xgboost ^
        --windows 30d,45d,60d
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

import argparse

from src.common.data_loader import load_data
from src.common.metrics import compute_all_metrics
from src.common.feature_builder_dayahead import build_features_dayahead as build_enhanced_features

_OUTPUT_ROOT = os.path.join(_PROJECT_DIR, "outputs", "dayahead_tabular_search_30d")

WINDOW_DAYS = {
    "30d": 30, "45d": 45, "60d": 60, "90d": 90, "all": 99999,
}

LGB_CONFIGS = ["gbdt_default", "dart_regularized", "high_leaf_regularized"]
XGB_CONFIGS = ["squared_error_default", "absolute_error", "regularized_deep"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--start", type=str, default="2026-02-01")
    p.add_argument("--end", type=str, default="2026-03-02")
    p.add_argument("--output-root", type=str, default="outputs/dayahead_tabular_search_30d")
    p.add_argument("--models", type=str, default="lightgbm,xgboost",
                    help="Comma-separated: lightgbm,xgboost")
    p.add_argument("--windows", type=str, default="30d,45d,60d,90d,all",
                    help="Comma-separated training windows")
    p.add_argument("--no-resume", action="store_true",
                    help="Force re-run all dates")
    return p.parse_args()


def _load_or_create_features(data_path: str | None,
                              start_date: str, end_date: str) -> pd.DataFrame:
    """Load and build features for the full date range."""
    if data_path is None or data_path.endswith(".yaml"):
        import yaml
        yaml_path = data_path or os.path.join(_PROJECT_DIR, "configs", "paths.yaml")
        with open(yaml_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        data_path = cfg.get("default_data", os.path.join(_PROJECT_DIR, "..",
                          "electricity_forecast_model2.0_exp", "data", "shandong_pmos_hourly.csv"))

    logger.info(f"Data path: {data_path}")
    logger.info("Loading data...")
    df = load_data(data_path, target="dayahead")
    logger.info(f"  Raw data: {len(df)} rows, {df['ds'].min()} ~ {df['ds'].max()}")

    # Build features
    logger.info("Building features (enhanced day-ahead)...")
    df = build_enhanced_features(df)
    logger.info(f"  Features built: {len(df)} rows, {len(df.columns)} columns")

    # Filter to date range
    df["ds"] = pd.to_datetime(df["ds"])
    start = pd.Timestamp(start_date) - pd.Timedelta(days=180)  # buffer for lags
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    df = df[(df["ds"] >= start) & (df["ds"] < end)].copy()
    logger.info(f"  After filter: {len(df)} rows")

    return df


def _get_train_val_split(df: pd.DataFrame, target_date: str,
                          window_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and validation for a given target date.
    
    Train: [target - window_days, target - 1] (all hours)
    Val:   [target - 7, target - 1] (recent days for config selection)
    """
    target_dt = pd.Timestamp(target_date)
    train_start = target_dt - pd.Timedelta(days=window_days)
    train_end = target_dt - pd.Timedelta(hours=1)

    train_mask = (df["ds"] >= train_start) & (df["ds"] < train_end)
    train_df = df[train_mask].copy()

    # Validation = last 7 days of training period
    val_start = target_dt - pd.Timedelta(days=min(7, window_days))
    val_mask = (df["ds"] >= val_start) & (df["ds"] < train_end)
    val_df = df[val_mask].copy()

    return train_df, val_df


def _compute_feature_importance(model, adapter, feature_cols):
    """Extract feature importance from model."""
    try:
        importance = model.feature_importance(importance_type="gain")
        if hasattr(model, "feature_name"):
            names = model.feature_name()
        else:
            available = [c for c in feature_cols]
            names = available[:len(importance)]
        imp_df = pd.DataFrame({"feature": names, "importance": importance})
        imp_df = imp_df.sort_values("importance", ascending=False)
        return imp_df.head(20).to_dict("records")
    except Exception:
        return []


def run_model_search(df: pd.DataFrame, model_type: str,
                      windows: list[str], start_date: str, end_date: str,
                      resume: bool = True) -> dict:
    """Run walk-forward search for a model type across windows and configs."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Model: {model_type}")
    logger.info(f"{'='*60}")

    dates = pd.date_range(start_date, end_date, freq="D")
    dates_str = [d.strftime("%Y-%m-%d") for d in dates]

    # Shorter dates for fast smoke test
    if len(dates_str) > 5:
        logger.info(f"  Running {len(dates_str)} days ({dates_str[0]} ~ {dates_str[-1]})")

    if model_type == "lightgbm":
        from src.models.lightgbm_dayahead_adapter import LightGBMDayaheadAdapter, LGB_CONFIGS as LGB_CONFIG_MAP
        configs = LGB_CONFIGS
        adapter_cls = LightGBMDayaheadAdapter
        # Inject GPU (+ faster training) params
        gpu_params = {"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0,
                       "num_leaves": 63, "min_data_in_leaf": 30}
    elif model_type == "xgboost":
        from src.models.xgboost_dayahead_adapter import XGBoostDayaheadAdapter, XGB_CONFIGS as XGB_CONFIG_MAP
        configs = XGB_CONFIGS
        adapter_cls = XGBoostDayaheadAdapter
        gpu_params = {"tree_method": "gpu_hist", "gpu_id": 0, "max_depth": 6, "min_child_weight": 10}
    else:
        raise ValueError(f"Unknown model: {model_type}")

    all_results = []  # list of {window, config, date, metrics}
    all_imp = []
    best_per_window = {}  # {window: {config: overall_smape}}

    for window_label in windows:
        window_days = WINDOW_DAYS.get(window_label, 30)
        logger.info(f"\n  --- Window: {window_label} ({window_days}d) ---")

        for config_name in configs:
            logger.info(f"    Config: {config_name}")

            day_results = []
            n_done = 0

            for target_date in dates_str:
                # Skip if already done (resume)
                if resume:
                    check_path = os.path.join(
                        _OUTPUT_ROOT, "predictions",
                        f"{model_type}_dayahead_sota_{config_name}_w{window_label}_{target_date}.csv"
                    )
                    if os.path.exists(check_path):
                        n_done += 1
                        continue

                try:
                    train_df, val_df = _get_train_val_split(df, target_date, window_days)
                    if len(train_df) < 100 or len(val_df) < 24:
                        if n_done == 0:
                            logger.warning(f"      {target_date}: insufficient train data ({len(train_df)} rows)")
                        continue

                    # Build params: base config + GPU override + faster training
                    base_params = LGB_CONFIG_MAP.get(config_name, {}) if model_type == "lightgbm" else \
                                  XGB_CONFIG_MAP.get(config_name, {})
                    model_params = {**base_params, **gpu_params,
                                    "num_boost_round": 500, "early_stopping_rounds": 20}
                    # Reduce rounds for speed
                    if "feature_fraction" not in model_params and model_type == "lightgbm":
                        model_params["feature_fraction"] = 0.8
                        model_params["bagging_fraction"] = 0.8
                        model_params["bagging_freq"] = 4

                    adapter = adapter_cls(config_name=config_name, model_params=model_params)
                    manifest = adapter.train(train_df, val_df)

                    # Predict on validation → config selection metric
                    # Also predict on target day
                    day_out = adapter.predict_day(df, target_date, task="dayahead")

                    if len(day_out) > 0:
                        day_metrics = compute_all_metrics(
                            day_out["y_true"].values,
                            day_out["y_pred"].values
                        )
                        day_results.append({
                            "date": target_date,
                            "smape": day_metrics["sMAPE_floor50"],
                            "mae": day_metrics["MAE"],
                            "rmse": day_metrics["RMSE"],
                        })
                        n_done += 1

                        # Save per-day prediction
                        os.makedirs(os.path.join(_OUTPUT_ROOT, "predictions"), exist_ok=True)
                        day_out.to_csv(check_path if resume else os.devnull,
                                        index=False, encoding="utf-8-sig")

                    if n_done % 5 == 0 and n_done > 0:
                        logger.info(f"      {target_date}: done ({n_done}/{len(dates_str)}), "
                                     f"avg smape so far: {np.mean([r['smape'] for r in day_results]):.2f}%")

                except Exception as e:
                    logger.warning(f"      {target_date}: failed ({e})")
                    continue

            # Compute overall config+window metrics
            if day_results:
                avg_smape = np.mean([r["smape"] for r in day_results])
                all_results.append({
                    "model": model_type,
                    "window": window_label,
                    "config": config_name,
                    "sMAPE_floor50": avg_smape,
                    "n_days": len(day_results),
                    "avg_MAE": np.mean([r["mae"] for r in day_results]),
                    "avg_RMSE": np.mean([r["rmse"] for r in day_results]),
                })
                logger.info(f"    -> {config_name} @ {window_label}: {avg_smape:.4f}% ({len(day_results)} days)")
            else:
                logger.warning(f"    -> {config_name} @ {window_label}: NO RESULTS")

    return {"results": all_results, "importances": all_imp}


def save_predictions(df: pd.DataFrame, model_type: str, config_name: str,
                      window_label: str, dates_str: list[str]):
    """Merge per-day predictions into single CSV."""
    all_days = []
    for d in dates_str:
        check_path = os.path.join(
            _OUTPUT_ROOT, "predictions",
            f"{model_type}_dayahead_sota_{config_name}_w{window_label}_{d}.csv"
        )
        if os.path.exists(check_path):
            all_days.append(pd.read_csv(check_path))
        # Clean up temp file
        try:
            os.remove(check_path)
        except OSError:
            pass

    if all_days:
        merged = pd.concat(all_days, ignore_index=True)
        merged = merged.sort_values(["target_day", "hour_business"]).reset_index(drop=True)
        out_path = os.path.join(
            _OUTPUT_ROOT, "predictions",
            f"{model_type}_dayahead_sota_dayahead.csv"
        )
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"  Saved merged predictions: {out_path} ({len(merged)} rows)")
        return merged
    return pd.DataFrame()


def generate_report(all_results: list[dict],
                    baseline_smape: float = 12.58,
                    corrected_smape: float = 12.47) -> str:
    """Generate final markdown report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    lines = [
        "# Day-Ahead Tabular Model Search Report",
        f"> Generated: {now}",
        f"> Baseline: catboost_sota = {baseline_smape:.2f}%",
        f"> Reference: spike_residual_corrected = {corrected_smape:.2f}%",
        f"> Target: < 8%",
        "",
    ]

    # Best per model
    if all_results:
        df_results = pd.DataFrame(all_results)
        lines.append("## Ranking")
        lines.append("")
        lines.append("| Rank | Model | Window | Config | sMAPE | Days | vs 12.47% | vs 12.58% |")
        lines.append("|---|---|---|---|---|---|---|---|")
        sorted_df = df_results.sort_values("sMAPE_floor50")
        for i, (_, r) in enumerate(sorted_df.iterrows()):
            vs_corrected = r["sMAPE_floor50"] - corrected_smape
            vs_baseline = r["sMAPE_floor50"] - baseline_smape
            c_corrected = "✅" if vs_corrected < 0 else "❌"
            c_baseline = "✅" if vs_baseline < 0 else "❌"
            lines.append(
                f"| {i+1} | {r['model']} | {r['window']} | {r['config']} | "
                f"{r['sMAPE_floor50']:.4f}% | {int(r['n_days'])} | "
                f"{c_corrected} {vs_corrected:+.2f}pp | {c_baseline} {vs_baseline:+.2f}pp |"
            )

        lines.append("")

        # Best config
        best_row = sorted_df.iloc[0]
        best_model = best_row["model"]
        best_smape = best_row["sMAPE_floor50"]

        lines.append("## 结论")
        lines.append("")
        lines.append(f"| 问题 | 回答 |")
        lines.append(f"|---|---|")
        lines.append(f"| Best 模型 | {best_model} ({best_row['window']}, {best_row['config']}) |")
        lines.append(f"| Best sMAPE | {best_smape:.2f}% |")
        lines.append(f"| 优于 spike_residual 12.47% | {'✅' if best_smape < 12.47 else '❌'} |")
        lines.append(f"| 低于 12% | {'✅' if best_smape < 12 else '❌'} |")
        lines.append(f"| 低于 10% | {'✅' if best_smape < 10 else '❌'} |")
        lines.append(f"| 低于 8% | {'✅' if best_smape < 8 else '❌'} |")
        lines.append(f"| 最优窗口 | {best_row['window']} |")
        lines.append(f"| 建议进入模型池 | {'✅' if best_smape < 12.47 else '❌'} |")
        lines.append(f"| 需要 N-BEATSx | {'✅ 当前非GBDT路线未突破' if best_smape >= 12 else '否'} |")

    return "\n".join(lines)


def main():
    args = parse_args()
    os.makedirs(os.path.join(_OUTPUT_ROOT, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(_OUTPUT_ROOT, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(_OUTPUT_ROOT, "reports"), exist_ok=True)

    windows = [w.strip() for w in args.windows.split(",")]
    models = [m.strip() for m in args.models.split(",")]

    logger.info(f"Models: {models}")
    logger.info(f"Windows: {windows}")
    logger.info(f"Date range: {args.start} ~ {args.end}")

    # Load features once
    df = _load_or_create_features(args.data_path, args.start, args.end)

    # Run search for each model
    all_results = []
    for model_type in models:
        result = run_model_search(df, model_type, windows, args.start, args.end,
                                   resume=not args.no_resume)
        all_results.extend(result["results"])

    # Save config search results
    if all_results:
        csv_path = os.path.join(_OUTPUT_ROOT, "metrics", "config_search_results.csv")
        pd.DataFrame(all_results).to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"Config search results saved: {csv_path}")

    # Save summary
    if all_results:
        summary_path = os.path.join(_OUTPUT_ROOT, "metrics", "summary.csv")
        df_summary = pd.DataFrame(all_results)
        df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        logger.info(f"Summary saved: {summary_path}")

    # Generate report
    report = generate_report(all_results)
    report_path = os.path.join(_OUTPUT_ROOT, "reports", "dayahead_tabular_search_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved: {report_path}")

    # Print summary
    logger.info("\n" + report)


if __name__ == "__main__":
    main()
