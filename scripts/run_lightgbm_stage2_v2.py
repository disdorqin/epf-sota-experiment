"""
run_lightgbm_stage2_v2.py — LightGBM Stage-2 tuning, loads raw data from scratch.

Usage:
    python scripts/run_lightgbm_stage2_v2.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import random
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from src.common.metrics import smape_floor50, compute_all_metrics
from src.common.data_loader import load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="LightGBM Stage-2 v2 (correct training data)")
    p.add_argument("--output-root", type=str, default="outputs/dayahead_lgbm_stage2_30d")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--data-path", type=str, default=None)
    return p.parse_args()


FEATURE_COLS = [
    "hour", "month", "day_of_week", "is_weekend",
    "lag_price_target", "lag_price_week",
    "load", "wind", "solar", "interconnect", "bidding_space", "space_ratio",
    "net_load", "solar_ratio", "net_load_sq", "wind_ratio", "renew_penetration",
    "ramp_load", "ramp_solar", "morning_mean", "noon_min", "morning_std",
    "morning_trend", "is_info_fresh",
]


def build_enhanced_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build features from raw data. Returns df with ds, y, features."""
    from src.common.feature_builder import build_features
    df = build_features(raw)

    # Add hour_business and period
    adjusted = df["ds"] - pd.Timedelta(seconds=1)
    df["hour_business"] = adjusted.dt.hour + 1
    df["period"] = np.select(
        [df["hour_business"].between(1, 8),
         df["hour_business"].between(9, 16)],
        ["1_8", "9_16"],
        default="17_24",
    )

    # Additional features
    df["target_day"] = df["ds"].dt.date.astype(str)
    df["business_day"] = df["target_day"]

    # Spring festival features
    sf = pd.Timestamp("2026-02-17")
    ds = df["ds"]
    df["is_spring_festival_window"] = (
        (ds >= sf - timedelta(days=10)) & (ds <= sf + timedelta(days=10))
    ).astype(int)
    df["days_to_spring_festival"] = (sf - ds).dt.days
    df["is_month_start"] = (ds.dt.day <= 3).astype(int)
    df["is_month_end"] = (ds.dt.day >= 28).astype(int)

    # Rolling same-hour stats (use shift 24 to avoid leakage)
    for wd, suf in [(7, "7d"), (14, "14d")]:
        wh = wd * 24
        sh = df.groupby("hour_business")["y"].transform(lambda x: x.shift(24))
        df[f"same_hour_mean_{suf}"] = sh.rolling(wh, min_periods=1).mean()
        df[f"same_hour_std_{suf}"] = sh.rolling(wh, min_periods=1).std()
        df[f"same_hour_max_{suf}"] = sh.rolling(wh, min_periods=1).max()
        df[f"same_hour_min_{suf}"] = sh.rolling(wh, min_periods=1).min()

    # Price momentum
    df["lag_24h"] = df["y"].shift(24)
    df["lag_168h"] = df["y"].shift(168)
    denom = df["lag_168h"].abs().replace(0, 1e-8)
    df["price_momentum_24_168"] = (df["lag_24h"] - df["lag_168h"]) / denom

    # Volatility
    for window, suf in [(24, "24h"), (168, "168h")]:
        df[f"price_volatility_{suf}"] = df["y"].shift(24).rolling(window, min_periods=2).std()

    df = df.ffill().fillna(0).reset_index(drop=True)
    return df


def smape_floor50_score(y_true, y_pred):
    return smape_floor50(np.array(y_true), np.array(y_pred))


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    for sub in ["predictions", "metrics", "reports", "debug"]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    # ── Load raw data and build features ──
    data_path = args.data_path
    if data_path is None:
        from src.common.repo_paths import get_data_path
        data_path = str(get_data_path())

    raw = load_data(data_path, target="dayahead")
    df = build_enhanced_features(raw)
    df = df[df["ds"] >= "2025-11-01"].reset_index(drop=True)
    logger.info(f"Full feature dataframe: {len(df)} rows ({df['ds'].min()} → {df['ds'].max()})")

    import lightgbm as lgb

    # Evaluation days and split
    all_days = sorted(df[(df["ds"] >= "2026-02-01") & (df["ds"] <= "2026-03-02 23:00:00")]["target_day"].unique())
    search_days = [d for d in all_days if d <= "2026-02-20"]
    confirm_days = [d for d in all_days if d > "2026-02-20"]
    logger.info(f"Evaluation: {len(all_days)} days ({all_days[0]} → {all_days[-1]})")
    logger.info(f"Search: {len(search_days)} days, Confirm: {len(confirm_days)} days")

    # Feature columns
    exclude = {"ds", "y", "target_day", "business_day", "hour_business", "period",
               "date_only", "lag_24h", "lag_168h"}
    feat_cols = [c for c in df.columns if c not in exclude
                 and df[c].dtype in (np.float64, np.int64, np.float32, np.int32, np.int8)]
    logger.info(f"Feature columns: {len(feat_cols)}")

    PARAM_GRID = {
        "num_leaves": [63, 127, 191, 255],
        "min_data_in_leaf": [30, 50, 80, 120],
        "lambda_l1": [0.0, 0.1, 0.5, 1.0],
        "lambda_l2": [1.0, 2.0, 5.0, 10.0],
        "learning_rate": [0.015, 0.02, 0.03],
        "feature_fraction": [0.75, 0.85, 0.95],
        "bagging_fraction": [0.75, 0.85, 0.95],
        "bagging_freq": [1, 5],
    }
    OBJECTIVES = ["rmse", "mae", "huber"]
    WINDOWS = [75, 90, 105, 120, 150]

    all_results = {}  # name -> (full_pred_df, search_smape, confirm_smape)
    config_log = []

    def evaluate_config(params, window, config_name):
        """Train rolling and evaluate."""
        preds = []
        for day in all_days:
            target_dt = pd.Timestamp(day)
            # Training: all data before day, limited by window
            train_df_all = df[df["target_day"] < day].copy()
            if len(train_df_all) < 200:
                continue
            train_start = target_dt - timedelta(days=window)
            train_df = train_df_all[train_df_all["ds"] >= train_start].copy()
            if len(train_df) < 100:
                continue

            # Validation: last 30 days
            val_start = target_dt - timedelta(days=30)
            val_df = train_df_all[train_df_all["ds"].between(val_start, target_dt - timedelta(hours=1))].copy()

            X_tr = train_df[feat_cols].values.astype(float)
            y_tr = train_df["y"].values.astype(float)

            try:
                if len(val_df) >= 50:
                    X_val = val_df[feat_cols].values.astype(float)
                    y_val = val_df["y"].values.astype(float)
                    model = lgb.train(
                        params, lgb.Dataset(X_tr, y_tr),
                        num_boost_round=2000,
                        valid_sets=[lgb.Dataset(X_val, y_val)],
                        valid_names=["eval"],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
                    )
                else:
                    model = lgb.train(params, lgb.Dataset(X_tr, y_tr),
                                      num_boost_round=2000,
                                      callbacks=[lgb.log_evaluation(0)])

                # Predict current day
                day_df = df[df["target_day"] == day].copy()
                X_pred = day_df[feat_cols].values.astype(float)
                day_df["y_pred"] = model.predict(X_pred)
                day_df["y_true"] = day_df["y"].values
                day_df["model_name"] = config_name
                preds.append(day_df)
            except Exception as e:
                logger.debug(f"    Day {day}: {e}")
                continue

        if not preds:
            return

        full = pd.concat(preds, ignore_index=True)

        # Search/confirm split
        search_mask = full["target_day"].isin(search_days)
        confirm_mask = full["target_day"].isin(confirm_days)
        s_smape = smape_floor50_score(
            full.loc[search_mask, "y_true"].values,
            full.loc[search_mask, "y_pred"].values,
        ) if search_mask.sum() >= 10 else None
        c_smape = smape_floor50_score(
            full.loc[confirm_mask, "y_true"].values,
            full.loc[confirm_mask, "y_pred"].values,
        ) if confirm_mask.sum() >= 10 else None
        f_smape = smape_floor50_score(full["y_true"].values, full["y_pred"].values)

        all_results[config_name] = (full, s_smape, c_smape)

        log_entry = {
            "config": config_name, "window": window,
            "objective": params.get("objective", "rmse"),
            "num_leaves": params.get("num_leaves"),
            "min_data_in_leaf": params.get("min_data_in_leaf"),
            "lambda_l1": params.get("lambda_l1"),
            "lambda_l2": params.get("lambda_l2"),
            "learning_rate": params.get("learning_rate"),
            "feature_fraction": params.get("feature_fraction"),
            "bagging_fraction": params.get("bagging_fraction"),
            "bagging_freq": params.get("bagging_freq"),
            "search_smape": round(s_smape, 4) if s_smape else None,
            "confirm_smape": round(c_smape, 4) if c_smape else None,
            "full_smape": round(f_smape, 4),
        }
        config_log.append(log_entry)

        s_str = f"{s_smape:.4f}%" if s_smape else "N/A"
        c_str = f"{c_smape:.4f}%" if c_smape else "N/A"
        logger.info(f"  {config_name}: full={f_smape:.4f}% search={s_str} confirm={c_str}")

    # ════════════════════════════════════════
    # Step 1: Confirm 90d high_leaf_regularized
    # ════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Confirm 90d high_leaf_regularized")
    logger.info("=" * 60)

    base_params = {
        "boosting_type": "gbdt", "num_leaves": 127, "learning_rate": 0.02,
        "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 3,
        "lambda_l1": 2.0, "lambda_l2": 2.0, "min_data_in_leaf": 50,
        "objective": "rmse", "metric": "rmse", "verbosity": -1,
    }
    evaluate_config(base_params.copy(), 90, "lightgbm_90d_high_leaf_confirm")

    # ════════════════════════════════════════
    # Step 2: Window search
    # ════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Window search")
    logger.info("=" * 60)

    for w in WINDOWS:
        evaluate_config(base_params.copy(), w, f"lightgbm_{w}d_high_leaf")
        if "lightgbm_90d_high_leaf_confirm" not in all_results:
            break  # if basic confirm fails, abort all

    # ════════════════════════════════════════
    # Step 3: Random search (30 trials)
    # ════════════════════════════════════════
    logger.info("\n" + "=" * 60)
    logger.info(f"STEP 3: Random search ({args.n_trials} trials)")
    logger.info("=" * 60)

    for trial in range(args.n_trials):
        params = base_params.copy()
        window = random.choice(WINDOWS)
        obj = random.choice(OBJECTIVES)
        params["objective"] = obj
        params["metric"] = "rmse" if obj == "rmse" else ("mae" if obj == "mae" else "huber")

        for k, vals in PARAM_GRID.items():
            params[k] = random.choice(vals)

        name = f"trial_{trial+1:02d}_w{window}_nl{params['num_leaves']}_lr{params['learning_rate']}"
        logger.info(f"  Trial {trial+1}: w={window}d nl={params['num_leaves']} lr={params['learning_rate']} "
                    f"obj={obj} l1={params['lambda_l1']} l2={params['lambda_l2']}")
        evaluate_config(params, window, name)

    # ── Save ──
    for name, (full, _, _) in all_results.items():
        path = output_root / "predictions" / f"{name}_dayahead.csv"
        full.to_csv(str(path), index=False, encoding="utf-8-sig")

    config_df = pd.DataFrame(config_log)
    if len(config_df) > 0:
        config_df = config_df.sort_values("full_smape")
        config_df.to_csv(str(output_root / "metrics" / "config_search_results.csv"),
                         index=False, encoding="utf-8-sig")

    # Summary
    summary_rows = []
    for name, (full, s_s, c_s) in all_results.items():
        m = compute_all_metrics(full["y_true"].values, full["y_pred"].values)
        m["model_name"] = name
        m["search_smape"] = s_s
        m["confirm_smape"] = c_s
        summary_rows.append(m)
    summary_df = pd.DataFrame(summary_rows).sort_values("sMAPE_floor50")
    summary_df.to_csv(str(output_root / "metrics" / "summary.csv"), index=False, encoding="utf-8-sig")

    # Hour/Period metrics for best model
    if len(summary_df) > 0:
        best_name = summary_df.iloc[0]["model_name"]
        best_full = all_results[best_name][0]
        hour_rows = []
        for h, grp in best_full.groupby("hour_business"):
            m = compute_all_metrics(grp["y_true"].values, grp["y_pred"].values)
            m["hour_business"] = h
            hour_rows.append(m)
        pd.DataFrame(hour_rows).to_csv(str(output_root / "metrics" / "hour_metrics.csv"),
                                        index=False, encoding="utf-8-sig")

        period_rows = []
        for p, grp in best_full.groupby("period"):
            m = compute_all_metrics(grp["y_true"].values, grp["y_pred"].values)
            m["period"] = p
            period_rows.append(m)
        pd.DataFrame(period_rows).to_csv(str(output_root / "metrics" / "period_metrics.csv"),
                                          index=False, encoding="utf-8-sig")

    # ── Report ──
    lines = []
    def w(s=""):
        lines.append(s)

    # ... (report generation)
    w(f"# LightGBM Stage-2 Day-Ahead Report")
    w(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w(f"**Search window**: Feb 1-20 ({len(search_days)} days)")
    w(f"**Confirm window**: Feb 21-Mar 2 ({len(confirm_days)} days)")
    w()

    if "lightgbm_90d_high_leaf_confirm" in all_results:
        _, s_s, c_s = all_results["lightgbm_90d_high_leaf_confirm"]
        w("## 1. Confirmation Run (90d high_leaf_regularized)")
        w(f"- Full sMAPE: {summary_df[summary_df['model_name']=='lightgbm_90d_high_leaf_confirm']['sMAPE_floor50'].iloc[0]:.4f}%")
        w(f"- Search: {s_s:.4f}%" if s_s else "- Search: N/A")
        w(f"- Confirm: {c_s:.4f}%" if c_s else "- Confirm: N/A")
        w()

    w("## 2. Top 5 Configurations")
    w("| Config | Full sMAPE | Search | Confirm | Window | Params |")
    w("|---|---|---|---|---|---|")
    for _, r in config_df.head(5).iterrows():
        w(f"| {r['config']} | {r['full_smape']:.2f}% | {r['search_smape'] or 'N/A'}% | {r['confirm_smape'] or 'N/A'}% | {r['window']}d | nl={r['num_leaves']} lr={r['learning_rate']} |")
    w()

    best_row = summary_df.iloc[0] if len(summary_df) > 0 else None
    if best_row is not None:
        best_s = best_row["sMAPE_floor50"]
        w("## 3. Target Check")
        w(f"- **Best**: {best_row['model_name']} ({best_s:.2f}%)")
        w(f"- Below 12.47% (spike)? {'✅' if best_s < 12.47 else '❌'}")
        w(f"- Below 12%? {'✅' if best_s < 12 else '❌'}")
        w(f"- Below 11.5%? {'✅' if best_s < 11.5 else '❌'}")
        w(f"- Below 11%? {'✅' if best_s < 11 else '❌'}")
        w()

    w("## 4. Recommendations")
    if best_row and best_row["sMAPE_floor50"] < 12:
        w("✅ LightGBM has broken 12%.")
    if best_row and best_row["sMAPE_floor50"] < 11.91:
        w("✅ Improved over 11.91% baseline.")
    w()

    report = "\n".join(lines)
    report_path = output_root / "reports" / "dayahead_lgbm_stage2_report.md"
    with open(str(report_path), "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report: {report_path}")

    print("\n" + "=" * 70)
    print("LIGHTGBM STAGE-2 SUMMARY")
    print("=" * 70)
    print(f"{'Config':45s} {'Full':>8s} {'Search':>8s} {'Confirm':>8s}")
    print("-" * 70)
    for _, r in config_df.head(10).iterrows():
        beats = "✅" if r["full_smape"] < 12 else "❌" if r["full_smape"] else ""
        print(f"{r['config'][:45]:45s} {r['full_smape']:7.2f}% "
              f"{str(r['search_smape'] or 'N/A')[:7]:>8s} "
              f"{str(r['confirm_smape'] or 'N/A')[:7]:>8s} {beats}")
    print("-" * 70)
    print(f"{'CatBoost baseline':45s} {'12.58':>7}%")
    print(f"{'Spike residual':45s} {'12.47':>7}%")
    if best_row:
        print(f"{'Best LGBM':45s} {best_row['sMAPE_floor50']:7.2f}%")
    print("=" * 70)

    manifest = {
        "n_trials": args.n_trials,
        "best_config": config_df.iloc[0]["config"] if len(config_df) > 0 else None,
        "best_full_smape": config_df.iloc[0]["full_smape"] if len(config_df) > 0 else None,
        "best_confirm_smape": config_df.iloc[0].get("confirm_smape") if len(config_df) > 0 else None,
        "confirm_window": f"{confirm_days[0]} → {confirm_days[-1]}" if confirm_days else "N/A",
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    with open(str(output_root / "debug" / "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
