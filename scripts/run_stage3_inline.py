"""
run_stage3_inline.py — Quick Stage-3 with 5 hand-picked configs.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from datetime import timedelta
from src.common.metrics import smape_floor50, compute_all_metrics
from src.common.data_loader import load_data
import lightgbm as lgb

from src.common.feature_builder import build_features as build_base
from src.common.feature_builder_dayahead import (
    _add_lag_features, _add_same_hour_stats, _add_price_momentum, _add_calendar_features)
from src.common.feature_builder_dayahead_v3 import (
    _add_volatility, _add_change_features, _add_exact_spring_festival, _add_interaction_features)

print("Loading + building features...")
raw = load_data("D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_model2.0_exp/data/shandong_pmos_hourly.csv", target="dayahead")
df = build_base(raw)
adj = df["ds"] - pd.Timedelta(seconds=1)
df["hour_business"] = adj.dt.hour + 1
df["period"] = np.select([df["hour_business"].between(1,8), df["hour_business"].between(9,16)],
                          ["1_8","9_16"], default="17_24")
df = _add_lag_features(df)
df = _add_same_hour_stats(df)
df = _add_price_momentum(df)
df = _add_calendar_features(df)
df = df.sort_values("ds").reset_index(drop=True)
df["net_load_rank_30d"] = df["net_load"].rolling(720, min_periods=180).apply(
    lambda x: (x < x[-1]).sum() / len(x) if len(x) >= 10 else 0.5, raw=True).fillna(0.5)
df["bidding_space_rank_30d"] = df["bidding_space"].rolling(720, min_periods=180).apply(
    lambda x: (x < x[-1]).sum() / len(x) if len(x) >= 10 else 0.5, raw=True).fillna(0.5)
df = _add_volatility(df)
df = _add_change_features(df)
df = _add_exact_spring_festival(df)
df = _add_interaction_features(df)
df["target_day"] = df["ds"].dt.date.astype(str)
df = df.ffill().fillna(0)
df = df[df["ds"] >= "2025-08-01"].reset_index(drop=True)

exclude = {"ds","y","target_day","business_day","hour_business","period",
           "date_only","y_pred","y_true","model_name","task"}
fc = [c for c in df.select_dtypes(include=[np.float64,np.int64,np.float32,np.int32]).columns
      if c not in exclude and c != "y"]
print(f"DF: {len(df)} rows, {len(fc)} features")

all_days = sorted(df[(df["ds"] >= "2026-02-01") & (df["ds"] <= "2026-03-02 23:00")]["target_day"].unique())
search_days = [d for d in all_days if d <= "2026-02-20"]
confirm_days = [d for d in all_days if d >= "2026-02-21"]
print(f"Days: {len(all_days)} total, {len(search_days)} search, {len(confirm_days)} confirm")


def run_eval(params, window, name):
    preds = []
    for day in all_days:
        tdt = pd.Timestamp(day)
        ta = df[df["target_day"] < day]
        if len(ta) < 200:
            continue
        if window == "all":
            tr = ta.tail(4000)
        else:
            tr = ta[ta["ds"] >= tdt - timedelta(days=window)].tail(4000)
        if len(tr) < 100:
            continue
        vl = ta[ta["ds"] >= tdt - timedelta(days=30)].tail(1500)
        X_tr, y_tr = tr[fc].values, tr["y"].values
        p = {k: v for k, v in params.items() if k != "n_estimators"}
        p["verbosity"] = -1
        nr = params.get("n_estimators", 1000)
        try:
            if len(vl) >= 50:
                X_v, y_v = vl[fc].values, vl["y"].values
                m = lgb.train(p, lgb.Dataset(X_tr, y_tr), num_boost_round=nr,
                              valid_sets=[lgb.Dataset(X_v, y_v)], valid_names=["eval"],
                              callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
            else:
                m = lgb.train(p, lgb.Dataset(X_tr, y_tr), num_boost_round=nr,
                              callbacks=[lgb.log_evaluation(0)])
            dd = df[df["target_day"] == day].copy()
            dd["y_pred"] = m.predict(dd[fc].values)
            dd["y_true"] = dd["y"].values
            dd["model_name"] = name
            preds.append(dd)
            del m
        except Exception:
            continue
    if not preds:
        return None
    full = pd.concat(preds, ignore_index=True)
    sm = full["target_day"].isin(search_days)
    cm = full["target_day"].isin(confirm_days)
    ss = smape_floor50(full.loc[sm, "y_true"].values, full.loc[sm, "y_pred"].values) if sm.sum() >= 10 else None
    cs = smape_floor50(full.loc[cm, "y_true"].values, full.loc[cm, "y_pred"].values) if cm.sum() >= 10 else None
    fs = smape_floor50(full["y_true"].values, full["y_pred"].values)
    return {"name": name, "window": window, "full": fs, "search": ss, "confirm": cs, "predictions": full}


# 5 configs covering different hyperparameter regions
configs = [
    ("cfg01_90d_mae_nl127",
     {"boosting_type": "gbdt", "num_leaves": 127, "min_data_in_leaf": 50,
      "lambda_l1": 0.1, "lambda_l2": 2.0, "learning_rate": 0.02,
      "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 5,
      "objective": "mae", "metric": "mae", "n_estimators": 1000}, 90),

    ("cfg02_120d_mae_nl191",
     {"boosting_type": "gbdt", "num_leaves": 191, "min_data_in_leaf": 30,
      "lambda_l1": 0.5, "lambda_l2": 5.0, "learning_rate": 0.02,
      "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 5,
      "objective": "mae", "metric": "mae", "n_estimators": 1000}, 120),

    ("cfg03_90d_mae_nl255",
     {"boosting_type": "gbdt", "num_leaves": 255, "min_data_in_leaf": 30,
      "lambda_l1": 0.1, "lambda_l2": 2.0, "learning_rate": 0.03,
      "feature_fraction": 0.95, "bagging_fraction": 0.85, "bagging_freq": 5,
      "objective": "mae", "metric": "mae", "n_estimators": 1000}, 90),

    ("cfg04_150d_mae_nl127",
     {"boosting_type": "gbdt", "num_leaves": 127, "min_data_in_leaf": 50,
      "lambda_l1": 1.0, "lambda_l2": 2.0, "learning_rate": 0.015,
      "feature_fraction": 0.85, "bagging_fraction": 0.75, "bagging_freq": 1,
      "objective": "mae", "metric": "mae", "n_estimators": 1000}, 150),

    ("cfg05_120d_rmse_nl255",
     {"boosting_type": "gbdt", "num_leaves": 255, "min_data_in_leaf": 20,
      "lambda_l1": 0.5, "lambda_l2": 10.0, "learning_rate": 0.03,
      "feature_fraction": 0.75, "bagging_fraction": 0.95, "bagging_freq": 5,
      "objective": "rmse", "metric": "rmse", "n_estimators": 1000}, 120),
]

all_results = []
for name, params, window in configs:
    t0 = time.time()
    r = run_eval(params, window, name)
    elapsed = time.time() - t0
    if r:
        ss = f"{r['search']:.2f}%" if r["search"] else "N/A"
        cs = f"{r['confirm']:.2f}%" if r["confirm"] else "N/A"
        print(f"{name}: full={r['full']:.4f}% search={ss} confirm={cs} ({elapsed:.0f}s)")
        all_results.append(r)
    else:
        print(f"{name}: FAILED ({elapsed:.0f}s)")

# Save
output_root = "outputs/dayahead_lgbm_stage3_30d"
for sub in ["predictions", "metrics", "debug"]:
    os.makedirs(f"{output_root}/{sub}", exist_ok=True)

for r in all_results:
    r["predictions"].to_csv(f"{output_root}/predictions/{r['name']}_dayahead.csv",
                            index=False, encoding="utf-8-sig")

rows = []
for r in all_results:
    m = compute_all_metrics(r["predictions"]["y_true"].values, r["predictions"]["y_pred"].values)
    m["model_name"] = r["name"]
    m["search_smape"] = r["search"]
    m["confirm_smape"] = r["confirm"]
    rows.append(m)
sdf = pd.DataFrame(rows).sort_values("sMAPE_floor50")
sdf.to_csv(f"{output_root}/metrics/summary.csv", index=False, encoding="utf-8-sig")

# Hour/period for best
bn = sdf.iloc[0]["model_name"]
bf = [r for r in all_results if r["name"] == bn][0]["predictions"]
hr = []
for h, g in bf.groupby("hour_business"):
    m = compute_all_metrics(g["y_true"].values, g["y_pred"].values)
    m["hour_business"] = h
    hr.append(m)
pd.DataFrame(hr).to_csv(f"{output_root}/metrics/hour_metrics.csv", index=False, encoding="utf-8-sig")
pr = []
for p, g in bf.groupby("period"):
    m = compute_all_metrics(g["y_true"].values, g["y_pred"].values)
    m["period"] = p
    pr.append(m)
pd.DataFrame(pr).to_csv(f"{output_root}/metrics/period_metrics.csv", index=False, encoding="utf-8-sig")

best = sdf.iloc[0]
print(f"\n{'='*60}")
print(f"BEST: {best['model_name']} = {best['sMAPE_floor50']:.4f}%")
print(f"Champion: 11.85%")
print(f"Beaten: {'YES' if best['sMAPE_floor50'] < 11.85 else 'NO'}")
print(f"Below 11.5%: {'YES' if best['sMAPE_floor50'] < 11.5 else 'NO'}")
print(f"{'='*60}")

manifest = {
    "best_config": best["model_name"],
    "best_smape": float(best["sMAPE_floor50"]),
    "champion": 11.85,
    "beaten": bool(best["sMAPE_floor50"] < 11.85),
    "below_11_5": bool(best["sMAPE_floor50"] < 11.5),
    "completed_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
}
with open(f"{output_root}/debug/run_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print("DONE")
