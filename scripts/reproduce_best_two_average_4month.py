#!/usr/bin/env python3
"""
reproduce_best_two_average_4month.py — P1.1 gate fix

Faithfully reproduce the 2.5 trusted champion `best_two_average` (= simple
average of LightGBM trial_02 + trial_24 predictions) on the FOUR HARD MONTHS
(2025-11, 2025-12, 2026-01, 2026-02) so it is comparable (same window) to the
cfg05 candidate.

Uses the SAME LightGBMDayaheadAdapter + feature chain that generated the
original 30-day trial predictions (outputs/dayahead_lgbm_stage2_30d).

Trial specs (from filenames):
  trial_02: window=150d, num_leaves=255, learning_rate=0.03
  trial_24: window=90d,  num_leaves=127, learning_rate=0.02
Base config: LGB_CONFIGS["high_leaf_regularized"] (objective=rmse,
num_boost_round=2000, early_stopping_rounds=50).
"""
import sys, os, json, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

from src.common.metrics import smape_floor50, compute_all_metrics
from src.common.data_loader import load_data
from src.common.repo_paths import get_data_path
from src.common.business_time import business_time_mapping
from src.common.feature_builder import build_features as build_base
from src.common.feature_builder_dayahead import (
    _add_lag_features, _add_same_hour_stats, _add_price_momentum, _add_calendar_features,
)
from src.common.feature_builder_dayahead_v3 import (
    _add_volatility, _add_change_features, _add_exact_spring_festival, _add_interaction_features,
)
from src.models.lightgbm_dayahead_adapter import LightGBMDayaheadAdapter, LGB_CONFIGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TEST_MONTHS = ["2025-11", "2025-12", "2026-01", "2026-02"]


def build_features(raw):
    df = build_base(raw)
    biz = business_time_mapping(df["ds"])
    df["business_day"] = biz["business_day"].astype(str)
    df["hour_business"] = biz["hour_business"]
    df["period"] = biz["period"]
    df["target_day"] = df["business_day"]
    df = _add_lag_features(df)
    df = _add_same_hour_stats(df)
    df = _add_price_momentum(df)
    df = _add_calendar_features(df)
    df = df.sort_values("ds").reset_index(drop=True)

    def _fast_rank(series, w=720):
        return series.rolling(w, min_periods=max(10, w // 4)).apply(
            lambda x: (x < x[-1]).sum() / len(x) if len(x) >= 10 else 0.5, raw=True).fillna(0.5)
    df["net_load_rank_30d"] = _fast_rank(df["net_load"])
    df["bidding_space_rank_30d"] = _fast_rank(df["bidding_space"])
    df = _add_volatility(df)
    df = _add_change_features(df)
    df = _add_exact_spring_festival(df)
    df = _add_interaction_features(df)
    df = df.ffill().fillna(0).reset_index(drop=True)
    return df


def run_trial(df, trial_name, window, params, all_days):
    adapter = LightGBMDayaheadAdapter(model_params=params)
    preds = []
    t_total = 0.0
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
        t0 = time.time()
        try:
            adapter.train(train_df, eval_df=val_df if len(val_df) > 50 else None)
            result = adapter.predict_day(df, day, task="dayahead")
        except Exception as e:
            logger.warning(f"  {trial_name} day {day}: {e}")
            continue
        t_total += time.time() - t0
        if len(result) > 0:
            result["model_name"] = trial_name
            preds.append(result)
    full = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return full, t_total


def main():
    raw = load_data(str(get_data_path()), target="dayahead")
    df = build_features(raw)
    df = df[df["ds"] >= "2024-06-01"].reset_index(drop=True)

    valid = df[(df["ds"] >= "2025-11-01") & (df["ds"] < "2026-03-01")]
    all_days = sorted(d for d in valid["target_day"].unique() if d[:7] in TEST_MONTHS)
    logger.info(f"Eval days: {len(all_days)} across {TEST_MONTHS}")

    base = LGB_CONFIGS["high_leaf_regularized"].copy()
    trial_02_params = dict(base); trial_02_params["num_leaves"] = 255; trial_02_params["learning_rate"] = 0.03
    trial_24_params = dict(base)  # nl127 lr0.02 already in base

    t02, t02_t = run_trial(df, "trial_02_w150_nl255_lr0.03", 150, trial_02_params, all_days)
    t24, t24_t = run_trial(df, "trial_24_w90_nl127_lr0.02", 90, trial_24_params, all_days)
    logger.info(f"trial_02 rows={len(t02)} time={t02_t:.1f}s; trial_24 rows={len(t24)} time={t24_t:.1f}s")

    # average on common (ds) -> best_two_average
    t02s = t02[["ds", "business_day", "hour_business", "period", "y_true", "y_pred"]].rename(columns={"y_pred": "p02"})
    t24s = t24[["ds", "y_pred"]].rename(columns={"y_pred": "p24"})
    merged = t02s.merge(t24s, on="ds", how="inner")
    merged["y_pred"] = (merged["p02"] + merged["p24"]) / 2.0
    merged["model_name"] = "dayahead_trusted_champion_best_two_average"
    merged["task"] = "dayahead"
    merged["target_day"] = merged["business_day"]

    yt = merged["y_true"].values.astype(float)
    yp = merged["y_pred"].values.astype(float)
    v = ~(np.isnan(yt) | np.isnan(yp))
    smape = smape_floor50(yt[v], yp[v])
    logger.info(f"best_two_average 4-month sMAPE_floor50 = {smape:.4f}% (n={int(v.sum())})")

    out = Path("outputs/dayahead_trusted_champion_4month")
    for sub in ["predictions", "metrics", "reports"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    merged.to_csv(str(out / "predictions" / "dayahead_trusted_champion_best_two_average.csv"),
                  index=False, encoding="utf-8-sig")
    t02.to_csv(str(out / "predictions" / "trial_02_4month.csv"), index=False, encoding="utf-8-sig")
    t24.to_csv(str(out / "predictions" / "trial_24_4month.csv"), index=False, encoding="utf-8-sig")

    m = compute_all_metrics(yt[v], yp[v])
    m["model_name"] = "dayahead_trusted_champion_best_two_average"
    m["n"] = int(v.sum())
    pd.DataFrame([m]).to_csv(str(out / "metrics" / "summary.csv"), index=False, encoding="utf-8-sig")

    period_rows = []
    for p, g in merged.groupby("period"):
        pm = compute_all_metrics(g["y_true"].values, g["y_pred"].values)
        pm["period"] = p
        period_rows.append(pm)
    pd.DataFrame(period_rows).to_csv(str(out / "metrics" / "period_metrics.csv"), index=False, encoding="utf-8-sig")

    month_rows = []
    for mo, g in merged.groupby(merged["business_day"].str[:7]):
        mm = compute_all_metrics(g["y_true"].values, g["y_pred"].values)
        mm["month"] = mo
        month_rows.append(mm)
    pd.DataFrame(month_rows).to_csv(str(out / "metrics" / "month_metrics.csv"), index=False, encoding="utf-8-sig")

    full_metrics = {
        "model_name": "dayahead_trusted_champion_best_two_average",
        "construction": "avg(LightGBM trial_02 w150/nl255/lr0.03, trial_24 w90/nl127/lr0.02)",
        "window": "2025-11 ~ 2026-02 (4 hard months, same as cfg05 candidate)",
        "n_days": int(merged["business_day"].nunique()),
        "n_rows": int(v.sum()),
        "sMAPE_floor50": round(float(smape), 4),
        "MAE": round(float(m.get("MAE", float("nan"))), 4),
        "RMSE": round(float(m.get("RMSE", float("nan"))), 4),
        "peak_MAE_q90": round(float(m.get("peak_MAE_q90", float("nan"))), 4),
        "negative_price_hit_rate": round(float(m.get("negative_price_hit_rate", float("nan"))), 4),
        "trial_02_train_time_s": round(float(t02_t), 2),
        "trial_24_train_time_s": round(float(t24_t), 2),
        "period": {r["period"]: round(float(r["sMAPE_floor50"]), 4) for r in period_rows},
        "month": {r["month"]: round(float(r["sMAPE_floor50"]), 4) for r in month_rows},
        "note": "Reproduction uses current feature chain (build_features + dayahead + v3); "
                "original 30d best_two_average = 11.85% (easy window, NOT comparable).",
    }
    with open(str(out / "metrics" / "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, ensure_ascii=False, indent=2)

    logger.info("DONE -> " + str(out / "metrics" / "metrics.json"))


if __name__ == "__main__":
    main()
