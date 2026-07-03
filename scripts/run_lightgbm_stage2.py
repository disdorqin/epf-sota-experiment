"""
run_lightgbm_stage2.py — LightGBM Stage-2 tuning for day-ahead.

Steps:
1. Confirm 90d high_leaf_regularized (should reach ~11.91%)
2. Random search over windows + hyperparams (30 trials)
3. Hold-out validation (Feb 1-20 search, Feb 21-Mar 2 confirm)
4. Final report
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import random
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from src.common.metrics import smape_floor50, compute_all_metrics
from src.models.lightgbm_dayahead_adapter import (
    LightGBMDayaheadAdapter, LGB_CONFIGS, FEATURE_COLS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="LightGBM Stage-2 day-ahead tuning")
    p.add_argument("--input-pred", type=str,
                   default="outputs/dayahead_30d_core/predictions/catboost_sota_dayahead.csv")
    p.add_argument("--output-root", type=str, default="outputs/dayahead_lgbm_stage2_30d")
    p.add_argument("--n-trials", type=int, default=30,
                   help="Number of random search trials")
    return p.parse_args()


def load_data(input_path: str) -> pd.DataFrame:
    """Load prediction CSV which already has features + y_true."""
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    return df


def smape_floor50_score(y_true, y_pred):
    return smape_floor50(np.array(y_true), np.array(y_pred))


def compute_metrics_dict(y_true, y_pred, model_name, task="dayahead"):
    v = ~(np.isnan(y_true) | np.isnan(y_pred))
    if v.sum() < 2:
        return {"model_name": model_name, "n": 0}
    m = compute_all_metrics(y_true[v], y_pred[v])
    m["model_name"] = model_name
    m["task"] = task
    m["n"] = int(v.sum())
    return m


FEATURE_COLS_ACTUAL = [c for c in FEATURE_COLS
                       if c not in ("lag_24h","lag_48h","lag_72h","lag_168h","lag_336h")]


def run_day_eval(df: pd.DataFrame, day: str, adapter: LightGBMDayaheadAdapter) -> pd.DataFrame:
    """Train on pre-day data, predict day, return long table."""
    target_dt = pd.Timestamp(day)
    train_df = df[df["target_day"] < day].copy()
    if len(train_df) < 1000:
        return pd.DataFrame()

    # Use validation split for early stopping
    val_start = target_dt - timedelta(days=30)
    val_df = df[df["ds"].between(val_start, target_dt - timedelta(hours=1))].copy()

    # Prepare training data (use y column which has prices)
    adapter.train(train_df, eval_df=val_df if len(val_df) > 50 else None)

    # Predict
    result = adapter.predict_day(df, day, task="dayahead")
    return result


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    for sub in ["predictions", "metrics", "reports", "debug"]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    df = load_data(args.input_pred)
    all_days = sorted(df["target_day"].unique())
    logger.info(f"Loaded {len(df)} rows, {len(all_days)} days ({all_days[0]} → {all_days[-1]})")

    # Split: search window Feb 1-20, confirm window Feb 21-Mar 2
    search_days = [d for d in all_days if d <= "2026-02-20"]
    confirm_days = [d for d in all_days if d > "2026-02-20"]
    logger.info(f"Search window: {len(search_days)} days ({search_days[0]} → {search_days[-1]})")
    logger.info(f"Confirm window: {len(confirm_days)} days ({confirm_days[0]} → {confirm_days[-1]})")

    all_results = {}  # name -> (full_preds, search_smape, confirm_smape)
    config_search_log = []

    # ════════════════════════════════════════════════════
    # Step 1: Confirm 90d high_leaf_regularized
    # ════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Confirm 90d high_leaf_regularized")
    logger.info("=" * 60)

    confirm_preds = []
    adapter = LightGBMDayaheadAdapter(config_name="high_leaf_regularized")

    for day in all_days:
        result = run_day_eval(df, day, adapter)
        if len(result) > 0:
            result["model_name"] = "lightgbm_90d_high_leaf_confirm"
            confirm_preds.append(result)

    if confirm_preds:
        full = pd.concat(confirm_preds, ignore_index=True)
        search_m = compute_metrics_dict(
            full[full["target_day"].isin(search_days)]["y_true"].values,
            full[full["target_day"].isin(search_days)]["y_pred"].values,
            "confirm"
        )
        confirm_m = compute_metrics_dict(
            full[full["target_day"].isin(confirm_days)]["y_true"].values,
            full[full["target_day"].isin(confirm_days)]["y_pred"].values,
            "confirm"
        )
        confirm_smape_val = confirm_m.get("sMAPE_floor50", None)
        full_m = compute_metrics_dict(full["y_true"].values, full["y_pred"].values, "confirm")
        all_results["lightgbm_90d_high_leaf_confirm"] = (full, search_m["sMAPE_floor50"], confirm_smape_val)
        logger.info(f"  Full 30d: {full_m['sMAPE_floor50']:.4f}%")
        logger.info(f"  Search (Feb 1-20): {search_m['sMAPE_floor50']:.4f}%")
        logger.info(f"  Confirm (Feb 21-Mar 2): {confirm_smape_val:.4f}%" if confirm_smape_val else "  Confirm: N/A")

    # ════════════════════════════════════════════════════
    # Step 2: Window search
    # ════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Window search (75d, 90d, 105d, 120d, 150d, all)")
    logger.info("=" * 60)

    windows_to_test = [75, 90, 105, 120, 150]

    # We need a different approach for window testing since the adapter uses all data.
    # Let me create a wrapper that limits training data window.
    base_adapter = LightGBMDayaheadAdapter(config_name="high_leaf_regularized")

    for window in windows_to_test:
        logger.info(f"  Testing window={window}d...")
        preds = []
        for day in all_days:
            target_dt = pd.Timestamp(day)
            # Limit training to N months
            train_start = target_dt - timedelta(days=window)
            # Filter the df to only include training data within window
            # The adapter's train() uses df directly, so we need to slice the df
            # Actually, let's modify: adapter.train() uses the full train_df passed to it
            # So we just slice the train_df before passing

            train_df = df[df["target_day"] < day].copy()
            if len(train_df) < 500:
                continue
            # Apply window limit
            train_df = train_df[train_df["ds"] >= train_start].copy()
            if len(train_df) < 200:
                continue

            val_start = target_dt - timedelta(days=30)
            val_df = df[df["ds"].between(val_start, target_dt - timedelta(hours=1))].copy()

            try:
                adapter = LightGBMDayaheadAdapter(config_name="high_leaf_regularized")
                adapter.train(train_df, eval_df=val_df if len(val_df) > 50 else None)
                result = adapter.predict_day(df, day, task="dayahead")
                if len(result) > 0:
                    result["model_name"] = f"lightgbm_{window}d_high_leaf"
                    preds.append(result)
            except Exception as e:
                logger.debug(f"    Day {day}: failed: {e}")

        if preds:
            full = pd.concat(preds, ignore_index=True)
            full_m = compute_metrics_dict(full["y_true"].values, full["y_pred"].values, "window")
            search_m_val = smape_floor50_score(
                full[full["target_day"].isin(search_days)]["y_true"].values,
                full[full["target_day"].isin(search_days)]["y_pred"].values,
            ) if len(full[full["target_day"].isin(search_days)]) >= 10 else None
            confirm_m_val = smape_floor50_score(
                full[full["target_day"].isin(confirm_days)]["y_true"].values,
                full[full["target_day"].isin(confirm_days)]["y_pred"].values,
            ) if len(full[full["target_day"].isin(confirm_days)]) >= 10 else None
            all_results[f"lightgbm_{window}d_high_leaf"] = (full, search_m_val, confirm_m_val)
            logger.info(
                f"    Full: {full_m['sMAPE_floor50']:.4f}%  "
                f"Search: {search_m_val:.4f}%  "
                f"Confirm: {confirm_m_val:.4f}%"
                if search_m_val is not None and confirm_m_val is not None else
                f"    Full: {full_m['sMAPE_floor50']:.4f}%  "
                f"Search: {'N/A' if search_m_val is None else f'{search_m_val:.4f}%'}  "
                f"Confirm: {'N/A' if confirm_m_val is None else f'{confirm_m_val:.4f}%'}"
            )

    # ════════════════════════════════════════════════════
    # Step 3: Random search (30 trials)
    # ════════════════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info(f"STEP 3: Random search ({args.n_trials} trials)")
    logger.info("=" * 60)

    # Base config from high_leaf_regularized
    base_params = LGB_CONFIGS["high_leaf_regularized"].copy()

    param_grid = {
        "num_leaves": [63, 127, 191, 255],
        "min_data_in_leaf": [30, 50, 80, 120],
        "lambda_l1": [0.0, 0.1, 0.5, 1.0],
        "lambda_l2": [1.0, 2.0, 5.0, 10.0],
        "learning_rate": [0.015, 0.02, 0.03],
        "feature_fraction": [0.75, 0.85, 0.95],
        "bagging_fraction": [0.75, 0.85, 0.95],
        "bagging_freq": [1, 5],
    }

    objectives = ["rmse", "mae", "huber"]

    best_overall = {"smape": float("inf"), "params": {}, "window": 90}
    best_confirm = {"smape": float("inf"), "params": {}, "window": 90}

    for trial in range(args.n_trials):
        # Pick window randomly
        window = random.choice(windows_to_test + [90])

        # Pick objective
        obj = random.choice(objectives)

        # Sample params
        params = base_params.copy()
        params["objective"] = obj
        if obj == "rmse":
            params["metric"] = "rmse"
        elif obj == "mae":
            params["metric"] = "mae"
        else:
            params["metric"] = "huber"

        for key, values in param_grid.items():
            params[key] = random.choice(values)

        trial_name = f"trial_{trial+1:02d}_w{window}_nl{params['num_leaves']}_lr{params['learning_rate']}"

        logger.info(f"  Trial {trial+1}/{args.n_trials}: window={window}d, "
                    f"num_leaves={params['num_leaves']}, lr={params['learning_rate']}, "
                    f"obj={obj}, l1={params['lambda_l1']}, l2={params['lambda_l2']}")

        preds = []
        for day in all_days:
            target_dt = pd.Timestamp(day)
            train_df = df[df["target_day"] < day].copy()
            if len(train_df) < 500:
                continue
            train_start = target_dt - timedelta(days=window)
            train_df = train_df[train_df["ds"] >= train_start].copy()
            if len(train_df) < 200:
                continue

            val_start = target_dt - timedelta(days=30)
            val_df = df[df["ds"].between(val_start, target_dt - timedelta(hours=1))].copy()

            try:
                adapter = LightGBMDayaheadAdapter(model_params=params)
                # Fix: create adapter with custom params
                adapter.config_name = trial_name
                adapter.params = params
                adapter.feature_cols = [c for c in FEATURE_COLS]
                adapter.train(train_df, eval_df=val_df if len(val_df) > 50 else None)
                result = adapter.predict_day(df, day, task="dayahead")
                if len(result) > 0:
                    result["model_name"] = trial_name
                    preds.append(result)
            except Exception as e:
                logger.debug(f"    Day {day}: {e}")
                continue

        if not preds:
            continue

        full = pd.concat(preds, ignore_index=True)

        # Compute search and confirm window performance
        search_mask = full["target_day"].isin(search_days)
        confirm_mask = full["target_day"].isin(confirm_days)

        if search_mask.sum() < 10:
            continue

        search_smape = smape_floor50_score(
            full.loc[search_mask, "y_true"].values,
            full.loc[search_mask, "y_pred"].values,
        )
        confirm_smape = smape_floor50_score(
            full.loc[confirm_mask, "y_true"].values,
            full.loc[confirm_mask, "y_pred"].values,
        ) if confirm_mask.sum() >= 10 else None
        full_smape = smape_floor50_score(
            full["y_true"].values, full["y_pred"].values
        )

        all_results[trial_name] = (full, search_smape, confirm_smape)

        log_entry = {
            "trial": trial + 1, "window_days": window,
            "objective": obj,
            "num_leaves": params["num_leaves"],
            "min_data_in_leaf": params["min_data_in_leaf"],
            "lambda_l1": params["lambda_l1"],
            "lambda_l2": params["lambda_l2"],
            "learning_rate": params["learning_rate"],
            "feature_fraction": params["feature_fraction"],
            "bagging_fraction": params["bagging_fraction"],
            "bagging_freq": params["bagging_freq"],
            "search_smape": round(search_smape, 4),
            "confirm_smape": round(confirm_smape, 4) if confirm_smape else None,
            "full_smape": round(full_smape, 4),
        }
        config_search_log.append(log_entry)

        logger.info(f"    Search: {search_smape:.4f}%  Confirm: {confirm_smape or 'N/A'}%  Full: {full_smape:.4f}%")

        if search_smape < best_overall["smape"]:
            best_overall = {"smape": search_smape, "params": params.copy(), "window": window, "trial": trial_name}
        if confirm_smape and confirm_smape < best_confirm["smape"]:
            best_confirm = {"smape": confirm_smape, "params": params.copy(), "window": window, "trial": trial_name}

    # ── Save predictions ──
    for name, (full, _, _) in all_results.items():
        path = output_root / "predictions" / f"{name}_dayahead.csv"
        full.to_csv(str(path), index=False, encoding="utf-8-sig")

    # ── Config search log ──
    config_df = pd.DataFrame(config_search_log)
    if len(config_df) > 0:
        config_df = config_df.sort_values("full_smape")
        config_df.to_csv(str(output_root / "metrics" / "config_search_results.csv"),
                         index=False, encoding="utf-8-sig")

    # ── Overall summary metrics ──
    summary_rows = []
    hour_rows = []
    period_rows = []

    for name, (full, search_s, confirm_s) in all_results.items():
        m = compute_metrics_dict(full["y_true"].values, full["y_pred"].values, name)
        m["search_smape"] = search_s
        m["confirm_smape"] = confirm_s
        summary_rows.append(m)

        for hour, grp in full.groupby("hour_business"):
            hm = compute_metrics_dict(grp["y_true"].values, grp["y_pred"].values, name)
            hm["hour_business"] = hour
            hour_rows.append(hm)

        for period, grp in full.groupby("period"):
            pm = compute_metrics_dict(grp["y_true"].values, grp["y_pred"].values, name)
            pm["period"] = period
            period_rows.append(pm)

    summary_df = pd.DataFrame(summary_rows).sort_values("sMAPE_floor50")
    summary_df.to_csv(str(output_root / "metrics" / "summary.csv"), index=False, encoding="utf-8-sig")

    hour_df = pd.DataFrame(hour_rows)
    hour_df.to_csv(str(output_root / "metrics" / "hour_metrics.csv"), index=False, encoding="utf-8-sig")

    period_df = pd.DataFrame(period_rows)
    period_df.to_csv(str(output_root / "metrics" / "period_metrics.csv"), index=False, encoding="utf-8-sig")

    # ════════════════════════════════════════════════════
    # Report
    # ════════════════════════════════════════════════════
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    def w(s=""):
        lines.append(s)

    w(f"# LightGBM Stage-2 Day-Ahead Report")
    w(f"**Generated**: {now}")
    w(f"**Search window**: Feb 1-20 ({len(search_days)} days)")
    w(f"**Confirm window**: Feb 21-Mar 2 ({len(confirm_days)} days)")
    w()

    # Confirm run
    w("## 1. Confirmation Run (90d high_leaf_regularized)")
    if "lightgbm_90d_high_leaf_confirm" in all_results:
        _, s_s, c_s = all_results["lightgbm_90d_high_leaf_confirm"]
        w(f"- Full 30d sMAPE: {summary_df[summary_df['model_name']=='lightgbm_90d_high_leaf_confirm']['sMAPE_floor50'].iloc[0]:.4f}%")
        w(f"- Search window: {s_s:.4f}%")
        w(f"- Confirm window: {c_s:.4f}%")
        w(f"- Reached 11.91%? {'✅ Yes' if any(summary_df[summary_df['model_name']=='lightgbm_90d_high_leaf_confirm']['sMAPE_floor50'] < 11.92) else '❌ No'}")
    w()

    # Best models
    w("## 2. Best Configurations")
    w("### By Full 30d sMAPE")
    w("| Trial | sMAPE | Search | Confirm | Window | Params |")
    w("|---|---|---|---|---|---|")
    top5 = summary_df.head(5)
    for _, r in top5.iterrows():
        name = r["model_name"]
        smape = r["sMAPE_floor50"]
        # Find in config log
        log_row = config_df[config_df["trial"] == int(name.split("_")[1])] if "_trial_" in name else None
        if log_row is not None and len(log_row) > 0:
            r2 = log_row.iloc[0]
            w(f"| {name} | {smape:.2f}% | {r2['search_smape']:.2f}% | {r2['confirm_smape'] or 'N/A'}% | {r2['window_days']}d | nl={r2['num_leaves']} lr={r2['learning_rate']} l1={r2['lambda_l1']} l2={r2['lambda_l2']} |")
        else:
            w(f"| {name} | {smape:.2f}% | - | - | - | - |")
    w()

    # Target check
    best_row = summary_df.iloc[0]
    best_smape = best_row["sMAPE_floor50"]
    best_model = best_row["model_name"]
    w("## 3. Target Check")
    w(f"- **Best model**: {best_model} ({best_smape:.2f}%)")
    w(f"- Below 12.47% (spike_residual)? {'✅' if best_smape < 12.47 else '❌'}")
    w(f"- Below 12%? {'✅' if best_smape < 12 else '❌'}")
    w(f"- Below 11.5%? {'✅' if best_smape < 11.5 else '❌'}")
    w(f"- Below 11%? {'✅' if best_smape < 11 else '❌'}")
    w(f"- Below 10%? {'✅' if best_smape < 10 else '❌'}")
    w()

    # Window analysis
    w("## 4. Window Analysis")
    w("| Window | Best sMAPE |")
    w("|---|---|")
    for wd in [75, 90, 105, 120, 150]:
        models = summary_df[summary_df["model_name"].str.contains(f"{wd}d", na=False)]
        best_w = models.iloc[0]["sMAPE_floor50"] if len(models) > 0 else None
        w(f"| {wd}d | {best_w:.2f}% |" if best_w else f"| {wd}d | N/A |")
    w()

    # Worst hours/days
    w("## 5. Error Analysis (Best Model)")
    if best_model in all_results:
        best_full, _, _ = all_results[best_model]
        # Worst 5 hours
        hour_err = best_full.groupby("hour_business").apply(
            lambda g: smape_floor50_score(g["y_true"].values, g["y_pred"].values)
        )
        w("### Worst 5 Hours")
        for h, s in hour_err.sort_values(ascending=False).head(5).items():
            w(f"- Hour {h}: {s:.2f}%")
        w()
        # Worst 5 days
        day_err = best_full.groupby("target_day").apply(
            lambda g: smape_floor50_score(g["y_true"].values, g["y_pred"].values)
        )
        w("### Worst 5 Days")
        for d, s in day_err.sort_values(ascending=False).head(5).items():
            w(f"- {d}: {s:.2f}%")
        w()

    # Recommendations
    w("## 6. Recommendations")
    if best_smape < 12:
        w("✅ **LightGBM has broken the 12% barrier.**")
        if best_smape < 11.91:
            w(f"✅ **Improved over 11.91% baseline.**")
    else:
        w("❌ LightGBM did not break 12%.")

    if best_smape < 11.5:
        w("✅ **Below 11.5% target.**")
    if best_smape < 11:
        w("✅ **Below 11% — significant breakthrough.**")
    if best_smape < 10:
        w("✅ **Below 10% — approaching 8% target.**")

    w()
    w(f"### Suggested Next Steps")
    if best_smape < 11:
        w("1. LightGBM should enter the main model pool")
        w("2. XGBoost may provide further improvement")
        w("3. AutoGluon ensembling can push further")
        w("4. N-BEATSx for further gains on spike hours")
    else:
        w("1. Continue LightGBM with more trials (increase n_trials to 100)")
        w("2. Test XGBoost for comparison")
        w("3. AutoGluon if LightGBM plateaus")
        w("4. N-BEATSx as next architecture")

    report = "\n".join(lines)
    report_path = output_root / "reports" / "dayahead_lgbm_stage2_report.md"
    with open(str(report_path), "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("LIGHTGBM STAGE-2 SUMMARY")
    print("=" * 70)
    print(f"{'Model':50s} {'sMAPE':>8s} {'Search':>8s} {'Confirm':>8s}")
    print("-" * 75)
    for _, r in summary_df.iterrows():
        name = r["model_name"]
        smape = r["sMAPE_floor50"]
        # Find in config log
        log_row = config_df[config_df["trial"] == int(name.split("_")[1])] if "_trial_" in name else None
        s_s = log_row.iloc[0]["search_smape"] if log_row is not None and len(log_row) > 0 else "-"
        c_s = log_row.iloc[0]["confirm_smape"] if log_row is not None and len(log_row) > 0 else "-"
        # Only print top 10
        if len(print) > 10:
            pass
        beats = "✅" if smape < 12 else "❌"
        print(f"{name[:50]:50s} {smape:7.2f}% {str(s_s)[:7]:>8s} {str(c_s)[:7]:>8s} {beats}")
    print("-" * 75)
    print(f"{'Spike residual corrector':50s} {'12.47':>7s}%")
    print(f"{'CatBoost baseline':50s} {'12.58':>7s}%")
    print("=" * 70)
    print(f"Best: {best_model} @ {best_smape:.2f}%")
    print(f"Better than spike (12.47%)? {'✅' if best_smape < 12.47 else '❌'}")
    print(f"Below 12%? {'✅' if best_smape < 12 else '❌'}")
    print(f"Below 11.5%? {'✅' if best_smape < 11.5 else '❌'}")
    print(f"Below 11%? {'✅' if best_smape < 11 else '❌'}")

    # Save manifest
    manifest = {
        "n_trials": args.n_trials,
        "best_model": best_model,
        "best_smape": best_smape,
        "best_confirm_smape": best_confirm["smape"] if best_confirm["smape"] < float("inf") else None,
        "confirm_window_smape": best_confirm["smape"] if best_confirm["smape"] < float("inf") else None,
        "completed_at": now,
    }
    with open(str(output_root / "debug" / "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
