"""
run_lgbm_focused.py — Focused LightGBM search with per-day save.

Only runs mae/rmse objectives, windows 75-150d.
Saves predictions per config to avoid total data loss.
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


def build_features(raw):
    from src.common.feature_builder import build_features
    df = build_features(raw)
    adjusted = df["ds"] - pd.Timedelta(seconds=1)
    df["hour_business"] = adjusted.dt.hour + 1
    df["period"] = np.select(
        [df["hour_business"].between(1, 8), df["hour_business"].between(9, 16)],
        ["1_8", "9_16"], default="17_24")
    df["target_day"] = df["ds"].dt.date.astype(str)
    sf = pd.Timestamp("2026-02-17")
    ds = df["ds"]
    df["is_spring_festival_window"] = ((ds >= sf - timedelta(days=10)) & (ds <= sf + timedelta(days=10))).astype(int)
    for wd, suf in [(7,"7d"),(14,"14d")]:
        wh = wd*24
        sh = df.groupby("hour_business")["y"].transform(lambda x: x.shift(24))
        df[f"same_hour_mean_{suf}"] = sh.rolling(wh, min_periods=1).mean()
    df["price_momentum_24_168"] = ((df["y"].shift(24) - df["y"].shift(168)) / df["y"].shift(168).abs().replace(0, 1e-8))
    return df.ffill().fillna(0).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", default="outputs/dayahead_lgbm_stage2_30d")
    p.add_argument("--n-trials", type=int, default=15)
    args = p.parse_args()
    out = Path(args.output_root)
    for s in ["predictions","metrics","reports","debug"]: (out/s).mkdir(parents=True,exist_ok=True)

    from src.common.repo_paths import get_data_path
    raw = load_data(get_data_path(), "dayahead")
    df = build_features(raw)
    df = df[df["ds"] >= "2025-11-01"].reset_index(drop=True)

    all_days = sorted(df[(df["ds"]>="2026-02-01")&(df["ds"]<="2026-03-02 23:00")]["target_day"].unique())
    search_days = [d for d in all_days if d <= "2026-02-20"]
    confirm_days = [d for d in all_days if d > "2026-02-20"]

    exclude = {"ds","y","target_day","business_day","hour_business","period","date_only","lag_24h","lag_168h"}
    feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype in (np.float64,np.int64,np.float32,np.int32)]

    import lightgbm as lgb

    configs = [
        # Confirm
        {"name":"lgbm_90d_default","window":90,"params":{"boosting_type":"gbdt","num_leaves":127,"learning_rate":0.02,"feature_fraction":0.7,"bagging_fraction":0.7,"bagging_freq":3,"lambda_l1":2.0,"lambda_l2":2.0,"min_data_in_leaf":50,"objective":"rmse","metric":"rmse","verbosity":-1}},
        # Windows
        {"name":"lgbm_75d_default","window":75,"params":{"boosting_type":"gbdt","num_leaves":127,"learning_rate":0.02,"feature_fraction":0.7,"bagging_fraction":0.7,"bagging_freq":3,"lambda_l1":2.0,"lambda_l2":2.0,"min_data_in_leaf":50,"objective":"rmse","metric":"rmse","verbosity":-1}},
        {"name":"lgbm_120d_default","window":120,"params":{"boosting_type":"gbdt","num_leaves":127,"learning_rate":0.02,"feature_fraction":0.7,"bagging_fraction":0.7,"bagging_freq":3,"lambda_l1":2.0,"lambda_l2":2.0,"min_data_in_leaf":50,"objective":"rmse","metric":"rmse","verbosity":-1}},
        {"name":"lgbm_150d_default","window":150,"params":{"boosting_type":"gbdt","num_leaves":127,"learning_rate":0.02,"feature_fraction":0.7,"bagging_fraction":0.7,"bagging_freq":3,"lambda_l1":2.0,"lambda_l2":2.0,"min_data_in_leaf":50,"objective":"rmse","metric":"rmse","verbosity":-1}},
    ]

    # Generate random trials (mae/rmse only, 120-150d)
    WINDOWS = [120, 150]
    for t in range(args.n_trials):
        cfg = {
            "boosting_type": "gbdt",
            "num_leaves": random.choice([127, 191, 255]),
            "learning_rate": random.choice([0.015, 0.02, 0.03]),
            "feature_fraction": random.choice([0.7, 0.85, 0.95]),
            "bagging_fraction": random.choice([0.7, 0.85, 0.95]),
            "bagging_freq": random.choice([1, 5]),
            "lambda_l1": random.choice([0.0, 0.1, 0.5, 1.0]),
            "lambda_l2": random.choice([1.0, 2.0, 5.0]),
            "min_data_in_leaf": random.choice([30, 50, 80]),
            "objective": random.choice(["rmse", "mae"]),
            "verbosity": -1,
        }
        cfg["metric"] = cfg["objective"]
        w = random.choice(WINDOWS)
        configs.append({"name": f"trial_{t+1}_w{w}_nl{cfg['num_leaves']}_lr{cfg['learning_rate']}",
                        "window": w, "params": cfg})

    results = []
    config_log = []

    for cfg in configs:
        name = cfg["name"]
        window = cfg["window"]
        params = cfg["params"]
        logger.info(f"\n{'='*50}")
        logger.info(f"Config: {name}")

        preds = []
        failed = False

        for day in all_days:
            try:
                target_dt = pd.Timestamp(day)
                train_df = df[(df["target_day"]<day)&(df["ds"]>=target_dt-timedelta(days=window))].copy()
                if len(train_df) < 100:
                    continue
                val_df = df[df["ds"].between(target_dt-timedelta(days=30), target_dt-timedelta(hours=1))].copy()

                X_tr, y_tr = train_df[feat_cols].values.astype(float), train_df["y"].values.astype(float)
                if len(val_df) >= 50:
                    X_val, y_val = val_df[feat_cols].values.astype(float), val_df["y"].values.astype(float)
                    model = lgb.train(params, lgb.Dataset(X_tr, y_tr), num_boost_round=2000,
                                      valid_sets=[lgb.Dataset(X_val, y_val)], valid_names=["eval"],
                                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                else:
                    model = lgb.train(params, lgb.Dataset(X_tr, y_tr), num_boost_round=2000,
                                      callbacks=[lgb.log_evaluation(0)])

                day_df = df[df["target_day"]==day].copy()
                day_df["y_pred"] = model.predict(day_df[feat_cols].values.astype(float))
                day_df["y_true"] = day_df["y"].values
                day_df["model_name"] = name
                preds.append(day_df)
            except Exception as e:
                logger.warning(f"  Day {day}: {e}")
                continue

        if not preds:
            logger.warning(f"  {name}: no predictions, skipping")
            continue

        full = pd.concat(preds, ignore_index=True)
        # Save per-config immediately
        path = out / "predictions" / f"{name}_dayahead.csv"
        full.to_csv(str(path), index=False, encoding="utf-8-sig")
        logger.info(f"  Saved {path} ({len(full)} rows)")

        # Metrics
        full_smape = smape_floor50(full["y_true"].values, full["y_pred"].values)
        search_smape = smape_floor50(full[full["target_day"].isin(search_days)]["y_true"].values,
                                      full[full["target_day"].isin(search_days)]["y_pred"].values) if len(full[full["target_day"].isin(search_days)]) >= 5 else None
        confirm_smape = smape_floor50(full[full["target_day"].isin(confirm_days)]["y_true"].values,
                                       full[full["target_day"].isin(confirm_days)]["y_pred"].values) if len(full[full["target_day"].isin(confirm_days)]) >= 5 else None

        config_log.append({"config":name,"window":window,"full_smape":round(full_smape,4),
                           "search_smape":round(search_smape,4) if search_smape else None,
                           "confirm_smape":round(confirm_smape,4) if confirm_smape else None,
                           "objective":params["objective"],"num_leaves":params["num_leaves"],
                           "lr":params["learning_rate"],"l1":params["lambda_l1"],"l2":params["lambda_l2"]})

        logger.info(f"  full={full_smape:.4f}% search={search_smape or 'N/A'} confirm={confirm_smape or 'N/A'}")

    # Save log
    config_df = pd.DataFrame(config_log).sort_values("full_smape")
    config_df.to_csv(str(out/"metrics"/"config_search_results.csv"), index=False, encoding="utf-8-sig")

    # Summary
    print("\n"+"="*70)
    print("LIGHTGBM FOCUSED SEARCH SUMMARY")
    print("="*70)
    for _, r in config_df.iterrows():
        beats="✅" if r["full_smape"]<12 else "❌"
        print(f"{r['config'][:45]:45s} {r['full_smape']:7.2f}% {r.get('search_smape','N/A') or 'N/A':>8s} {r.get('confirm_smape','N/A') or 'N/A':>8s} {beats}")
    print("-"*70)
    print(f"{'CatBoost baseline':45s} 12.58%")
    print(f"{'Spike residual':45s} 12.47%")
    if len(config_df)>0:
        best=config_df.iloc[0]
        print(f"{'Best LGBM':45s} {best['full_smape']:7.2f}%")
    print("="*70)

    json.dump({"n_configs":len(config_df),"best":config_df.iloc[0]["config"] if len(config_df)>0 else None,
               "best_smape":config_df.iloc[0]["full_smape"] if len(config_df)>0 else None,
               "completed_at":datetime.now().strftime("%Y-%m-%d %H:%M")},
              open(str(out/"debug"/"run_manifest.json"),"w"),ensure_ascii=False,indent=2)

if __name__=="__main__":
    main()
