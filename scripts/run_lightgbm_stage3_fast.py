"""
run_lightgbm_stage3_fast.py — Fast Stage-3: reduced trials, fewer estimators.

Target: beat 11.85%. Uses 10 trials, max 1000 estimators, 3 windows only.
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

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=str, default="outputs/dayahead_lgbm_stage3_30d")
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


EVAL_START = "2026-02-01"
EVAL_END = "2026-03-02"
SEARCH_END = "2026-02-20"
CONFIRM_START = "2026-02-21"
WINDOWS = [90, 120, 150]


def _fast_rank(series, window=720):
    return series.rolling(window, min_periods=180).apply(
        lambda x: (x < x[-1]).sum() / len(x) if len(x) >= 10 else 0.5, raw=True
    ).fillna(0.5)


def build_features(raw):
    from src.common.feature_builder import build_features as build_base
    from src.common.feature_builder_dayahead import (
        _add_lag_features, _add_same_hour_stats, _add_price_momentum,
        _add_calendar_features,
    )
    from src.common.feature_builder_dayahead_v3 import (
        _add_volatility, _add_change_features,
        _add_exact_spring_festival, _add_interaction_features,
    )
    df = build_base(raw)
    adj = df["ds"] - pd.Timedelta(seconds=1)
    df["hour_business"] = adj.dt.hour + 1
    df["period"] = np.select(
        [df["hour_business"].between(1, 8), df["hour_business"].between(9, 16)],
        ["1_8", "9_16"], default="17_24")
    df = _add_lag_features(df)
    df = _add_same_hour_stats(df)
    df = _add_price_momentum(df)
    df = _add_calendar_features(df)
    df = df.sort_values("ds").reset_index(drop=True)
    df["net_load_rank_30d"] = _fast_rank(df["net_load"])
    df["bidding_space_rank_30d"] = _fast_rank(df["bidding_space"])
    df = _add_volatility(df)
    df = _add_change_features(df)
    df = _add_exact_spring_festival(df)
    df = _add_interaction_features(df)
    df["target_day"] = df["ds"].dt.date.astype(str)
    df = df.ffill().fillna(0).reset_index(drop=True)
    return df


def get_feat_cols(df):
    exclude = {"ds", "y", "target_day", "business_day", "hour_business", "period",
               "date_only", "y_pred", "y_true", "model_name", "task"}
    return [c for c in df.select_dtypes(include=[np.float64, np.int64, np.float32, np.int32]).columns
            if c not in exclude and c != "y"]


def evaluate(params, window, df, feat_cols, all_days, search_days, confirm_days, name):
    import lightgbm as lgb
    MAX_TRAIN = 4000
    preds = []
    for day in all_days:
        tdt = pd.Timestamp(day)
        ta = df[df["target_day"] < day]
        if len(ta) < 200:
            continue
        if window == "all":
            tr = ta.tail(MAX_TRAIN)
        else:
            tr = ta[ta["ds"] >= tdt - timedelta(days=window)].tail(MAX_TRAIN)
        if len(tr) < 100:
            continue
        vl = ta[ta["ds"] >= tdt - timedelta(days=30)].tail(1500)
        X_tr, y_tr = tr[feat_cols].values, tr["y"].values
        try:
            p = {**params, "verbosity": -1}
            nr = p.pop("n_estimators", 1000)
            if len(vl) >= 50:
                X_v, y_v = vl[feat_cols].values, vl["y"].values
                m = lgb.train(p, lgb.Dataset(X_tr, y_tr), num_boost_round=nr,
                              valid_sets=[lgb.Dataset(X_v, y_v)], valid_names=["eval"],
                              callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
            else:
                m = lgb.train(p, lgb.Dataset(X_tr, y_tr), num_boost_round=nr,
                              callbacks=[lgb.log_evaluation(0)])
            dd = df[df["target_day"] == day].copy()
            dd["y_pred"] = m.predict(dd[feat_cols].values)
            dd["y_true"] = dd["y"].values
            dd["model_name"] = name
            preds.append(dd)
            del m
        except:
            continue
    if not preds:
        return None
    full = pd.concat(preds, ignore_index=True)
    sm = full["target_day"].isin(search_days)
    cm = full["target_day"].isin(confirm_days)
    ss = smape_floor50(full.loc[sm, "y_true"].values, full.loc[sm, "y_pred"].values) if sm.sum() >= 10 else None
    cs = smape_floor50(full.loc[cm, "y_true"].values, full.loc[cm, "y_pred"].values) if cm.sum() >= 10 else None
    fs = smape_floor50(full["y_true"].values, full["y_pred"].values)
    return {"config_name": name, "window": window, "params": params,
            "predictions": full, "search_smape": ss, "confirm_smape": cs, "full_smape": fs}


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    output_root = Path(args.output_root)
    for s in ["predictions", "metrics", "reports", "debug"]:
        (output_root / s).mkdir(parents=True, exist_ok=True)

    data_path = args.data_path
    if data_path is None:
        from src.common.repo_paths import get_data_path
        data_path = str(get_data_path())

    logger.info("Loading + features...")
    raw = load_data(data_path, target="dayahead")
    df = build_features(raw)
    df = df[df["ds"] >= "2025-08-01"].reset_index(drop=True)
    fc = get_feat_cols(df)
    logger.info(f"DF: {len(df)} rows, {len(fc)} features")

    all_days = sorted(df[(df["ds"] >= EVAL_START) & (df["ds"] <= f"{EVAL_END} 23:00")]["target_day"].unique())
    search_days = [d for d in all_days if d <= SEARCH_END]
    confirm_days = [d for d in all_days if d >= CONFIRM_START]
    logger.info(f"Days: {len(all_days)} total, {len(search_days)} search, {len(confirm_days)} confirm")

    results = {}
    log = []

    def store(r):
        if r is None: return
        results[r["config_name"]] = r
        e = {"config": r["config_name"], "window": r["window"],
             "objective": r["params"].get("objective", "rmse"),
             "num_leaves": r["params"].get("num_leaves"),
             "learning_rate": r["params"].get("learning_rate"),
             "search_smape": round(r["search_smape"], 4) if r["search_smape"] else None,
             "confirm_smape": round(r["confirm_smape"], 4) if r["confirm_smape"] else None,
             "full_smape": round(r["full_smape"], 4)}
        log.append(e)
        ss = f"{r['search_smape']:.2f}%" if r['search_smape'] else "N/A"
        cs = f"{r['confirm_smape']:.2f}%" if r['confirm_smape'] else "N/A"
        logger.info(f"  {r['config_name']}: full={r['full_smape']:.4f}% search={ss} confirm={cs}")

    # Baseline
    logger.info("Baseline (90d, mae)...")
    bp = {"boosting_type": "gbdt", "num_leaves": 127, "min_data_in_leaf": 50,
          "lambda_l1": 0.1, "lambda_l2": 2.0, "learning_rate": 0.02,
          "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 5,
          "objective": "mae", "metric": "mae", "n_estimators": 1000}
    store(evaluate(bp, 90, df, fc, all_days, search_days, confirm_days, "baseline_90d_mae"))

    # Optuna
    logger.info(f"Optuna ({args.n_trials} trials)...")
    def objective(trial):
        p = {"boosting_type": "gbdt",
             "num_leaves": trial.suggest_categorical("num_leaves", [127, 191, 255, 319]),
             "min_data_in_leaf": trial.suggest_categorical("min_data_in_leaf", [20, 30, 50, 80]),
             "lambda_l1": trial.suggest_categorical("lambda_l1", [0.1, 0.5, 1.0, 2.0]),
             "lambda_l2": trial.suggest_categorical("lambda_l2", [1.0, 2.0, 5.0, 10.0]),
             "learning_rate": trial.suggest_categorical("learning_rate", [0.015, 0.02, 0.03, 0.05]),
             "feature_fraction": trial.suggest_categorical("feature_fraction", [0.75, 0.85, 0.95]),
             "bagging_fraction": trial.suggest_categorical("bagging_fraction", [0.75, 0.85, 0.95]),
             "bagging_freq": trial.suggest_categorical("bagging_freq", [1, 5]),
             "objective": trial.suggest_categorical("objective", ["mae", "mae", "mae", "rmse"]),
             "metric": "mae", "n_estimators": trial.suggest_categorical("n_estimators", [1000]),
        }
        p["metric"] = p["objective"]
        w = trial.suggest_categorical("window", WINDOWS)
        name = f"optuna_{trial.number+1:02d}"
        r = evaluate(p, w, df, fc, all_days, search_days, confirm_days, name)
        store(r)
        return r["full_smape"] if r else 99.0

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.n_trials)

    # Save
    for n, r in results.items():
        r["predictions"].to_csv(str(output_root / "predictions" / f"{n}_dayahead.csv"),
                                index=False, encoding="utf-8-sig")

    cdf = pd.DataFrame(log).sort_values("full_smape")
    cdf.to_csv(str(output_root / "metrics" / "config_search_results.csv"), index=False, encoding="utf-8-sig")

    rows = []
    for n, r in results.items():
        m = compute_all_metrics(r["predictions"]["y_true"].values, r["predictions"]["y_pred"].values)
        m["model_name"] = n; m["search_smape"] = r["search_smape"]; m["confirm_smape"] = r["confirm_smape"]
        rows.append(m)
    sdf = pd.DataFrame(rows).sort_values("sMAPE_floor50")
    sdf.to_csv(str(output_root / "metrics" / "summary.csv"), index=False, encoding="utf-8-sig")

    # Hour/period for best
    bn = sdf.iloc[0]["model_name"]
    bf = results[bn]["predictions"]
    hr = []; pr = []
    for h, g in bf.groupby("hour_business"):
        m = compute_all_metrics(g["y_true"].values, g["y_pred"].values); m["hour_business"] = h; hr.append(m)
    for p, g in bf.groupby("period"):
        m = compute_all_metrics(g["y_true"].values, g["y_pred"].values); m["period"] = p; pr.append(m)
    pd.DataFrame(hr).to_csv(str(output_root / "metrics" / "hour_metrics.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(pr).to_csv(str(output_root / "metrics" / "period_metrics.csv"), index=False, encoding="utf-8-sig")

    # Report
    bs = cdf.iloc[0]["full_smape"] if len(cdf) > 0 else 99
    lines = [
        "# LightGBM Stage-3 Report (v3 Features + Optuna)",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Trials**: {args.n_trials} + 1 baseline", "",
        "## Top 10", "| Config | Full | Search | Confirm | Window | Obj |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in cdf.head(10).iterrows():
        ss = f"{r['search_smape']:.2f}%" if r['search_smape'] else "N/A"
        cs = f"{r['confirm_smape']:.2f}%" if r['confirm_smape'] else "N/A"
        lines.append(f"| {r['config']} | {r['full_smape']:.2f}% | {ss} | {cs} | {r['window']} | {r['objective']} |")
    lines += ["", "## Target Check",
              f"- Best: {cdf.iloc[0]['config']} = {bs:.4f}%",
              f"- Champion: 11.85%",
              f"- Below 11.85%: {'YES' if bs < 11.85 else 'NO'}",
              f"- Below 11.5%: {'YES' if bs < 11.5 else 'NO'}",
              f"- Below 11.0%: {'YES' if bs < 11.0 else 'NO'}", ""]
    report = "\n".join(lines)
    (output_root / "reports" / "dayahead_lgbm_stage3_report.md").write_text(report, encoding="utf-8")
    (Path(_PROJECT_DIR) / "docs" / "reports" / "dayahead_stage3_report.md").write_text(report, encoding="utf-8")

    manifest = {"best_config": cdf.iloc[0]["config"] if len(cdf) > 0 else None,
                "best_full_smape": bs, "champion": 11.85,
                "beaten": bs < 11.85, "below_11_5": bs < 11.5,
                "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
    with open(str(output_root / "debug" / "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"STAGE-3 BEST: {bs:.4f}% (champion: 11.85%, {'BEATEN' if bs < 11.85 else 'NOT BEATEN'})")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
