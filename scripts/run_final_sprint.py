#!/usr/bin/env python3
"""
run_final_sprint.py — Day-ahead final sprint.

Three sequential tasks:
  A: LightGBM micro-search (8 configs) on corrected business-day mapping
  B: Safe fusion final        (if A doesn't reach 11.5)
  C: XGBoost sentinel mini    (if A+B don't reach 11.5)

Usage:
    python scripts/run_final_sprint.py
"""
import sys, os, json, logging, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from src.common.metrics import smape_floor50, compute_all_metrics
from src.common.data_loader import load_data
from src.common.repo_paths import get_data_path
from src.common.business_time import business_time_mapping

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_EVAL_START = "2026-02-01"
_EVAL_END = "2026-03-02"
_SEARCH_END = "2026-02-20"
_CONFIRM_START = "2026-02-21"
_MAX_TRAIN_ROWS = 5000


# ── Shared feature builder (business-day-correct) ──
def build_v3_features(raw):
    from src.common.feature_builder import build_features as build_base
    from src.common.feature_builder_dayahead import (
        _add_lag_features, _add_same_hour_stats, _add_price_momentum, _add_calendar_features,
    )
    from src.common.feature_builder_dayahead_v3 import (
        _add_volatility, _add_change_features, _add_exact_spring_festival, _add_interaction_features,
    )
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


def get_feature_cols(df):
    exclude = {"ds", "y", "target_day", "business_day", "hour_business", "period",
               "date_only", "y_pred", "y_true", "model_name", "task", "lag_48h_raw", "lag_168h_raw"}
    numeric = df.select_dtypes(include=[np.float64, np.int64, np.float32, np.int32, np.int8, bool]).columns
    return [c for c in numeric if c not in exclude and c != "y"]


def train_and_predict(params, window, df, feat_cols, all_days, max_rows=_MAX_TRAIN_ROWS):
    """Rolling LightGBM train/predict. Returns (config_name, DataFrame)."""
    import lightgbm as lgb
    config_name = params.pop("_name", "unnamed")
    preds = []
    for day in all_days:
        target_dt = pd.Timestamp(day)
        train_all = df[df["target_day"] < day]
        if len(train_all) < 200:
            continue
        train_df = train_all.tail(max_rows) if window == "all" else \
            train_all[train_all["ds"] >= (target_dt - timedelta(days=window))].copy()
        if len(train_df) > max_rows:
            train_df = train_df.tail(max_rows)
        if len(train_df) < 100:
            continue

        val_df = train_all[train_all["ds"].between(
            target_dt - timedelta(days=30), target_dt - timedelta(hours=1))]
        if len(val_df) > 2000:
            val_df = val_df.tail(2000)

        X_tr = train_df[feat_cols].values.astype(float)
        y_tr = train_df["y"].values.astype(float)
        lgb_params = {k: v for k, v in params.items() if k != "_name"}
        lgb_params["verbosity"] = -1
        lgb_params["metric"] = lgb_params.get("objective", "rmse")

        try:
            if len(val_df) >= 50:
                X_val = val_df[feat_cols].values.astype(float)
                y_val = val_df["y"].values.astype(float)
                model = lgb.train(lgb_params, lgb.Dataset(X_tr, y_tr),
                                  num_boost_round=params.get("n_estimators", 2000),
                                  valid_sets=[lgb.Dataset(X_val, y_val)], valid_names=["eval"],
                                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            else:
                model = lgb.train(lgb_params, lgb.Dataset(X_tr, y_tr),
                                  num_boost_round=params.get("n_estimators", 2000),
                                  callbacks=[lgb.log_evaluation(0)])
            day_df = df[df["target_day"] == day].copy()
            day_df["y_pred"] = model.predict(day_df[feat_cols].values.astype(float))
            day_df["y_true"] = day_df["y"].values
            day_df["model_name"] = config_name
            preds.append(day_df)
            del model; gc.collect()
        except Exception as e:
            logger.debug(f"  {config_name} day {day}: {e}")
            continue
    return config_name, pd.concat(preds, ignore_index=True) if preds else None


# ═══════════════════════════════════════════════════════════
#  TASK A: LightGBM micro-search (8 configs)
# ═══════════════════════════════════════════════════════════
MICRO_CONFIGS = [
    dict(_name="cfg01", boosting_type="gbdt", objective="mae", num_leaves=127,
         min_data_in_leaf=50, learning_rate=0.02, lambda_l1=0.1, lambda_l2=2.0,
         feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5, n_estimators=2000),
    dict(_name="cfg02", boosting_type="gbdt", objective="mae", num_leaves=191,
         min_data_in_leaf=50, learning_rate=0.02, lambda_l1=0.5, lambda_l2=2.0,
         feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5, n_estimators=2000),
    dict(_name="cfg03", boosting_type="gbdt", objective="mae", num_leaves=255,
         min_data_in_leaf=30, learning_rate=0.03, lambda_l1=1.0, lambda_l2=2.0,
         feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1, n_estimators=2000),
    dict(_name="cfg04", boosting_type="gbdt", objective="mae", num_leaves=127,
         min_data_in_leaf=80, learning_rate=0.02, lambda_l1=0.5, lambda_l2=5.0,
         feature_fraction=0.95, bagging_fraction=0.85, bagging_freq=5, n_estimators=2000),
    dict(_name="cfg05", boosting_type="gbdt", objective="mae", num_leaves=191,
         min_data_in_leaf=30, learning_rate=0.015, lambda_l1=0.1, lambda_l2=5.0,
         feature_fraction=0.85, bagging_fraction=0.95, bagging_freq=5, n_estimators=2000),
    dict(_name="cfg06", boosting_type="gbdt", objective="mae", num_leaves=191,
         min_data_in_leaf=50, learning_rate=0.02, lambda_l1=1.0, lambda_l2=5.0,
         feature_fraction=0.75, bagging_fraction=0.85, bagging_freq=1, n_estimators=2000),
    dict(_name="cfg07", boosting_type="gbdt", objective="mae", num_leaves=127,
         min_data_in_leaf=80, learning_rate=0.02, lambda_l1=1.0, lambda_l2=10.0,
         feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5, n_estimators=2000),
    dict(_name="cfg08", boosting_type="gbdt", objective="rmse", num_leaves=127,
         min_data_in_leaf=50, learning_rate=0.02, lambda_l1=0.1, lambda_l2=2.0,
         feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5, n_estimators=2000),
]

MICRO_WINDOWS = [90, 120, 150, 120, 90, 150, "all", 90]


def task_a_lgbm_microsearch(df, feat_cols, all_days, search_days, confirm_days):
    logger.info("\n" + "=" * 65)
    logger.info("TASK A: LightGBM micro-search (8 configs)")
    logger.info("=" * 65)
    out = Path("outputs/dayahead_lgbm_microsearch_30d")
    for d in ["predictions", "metrics", "reports", "debug"]:
        (out / d).mkdir(parents=True, exist_ok=True)

    results = {}
    config_log = []
    for params, w in zip(MICRO_CONFIGS, MICRO_WINDOWS):
        name = params["_name"]
        logger.info(f"  Running {name} (window={w})...")
        cname, pred_df = train_and_predict(dict(params), w, df, feat_cols, all_days)
        if pred_df is None or len(pred_df) < 100:
            logger.warning(f"  {name}: failed")
            continue
        results[name] = pred_df
        pred_df.to_csv(str(out / "predictions" / f"{name}_dayahead.csv"), index=False, encoding="utf-8-sig")

        # Metrics
        full = pred_df
        f_smape = smape_floor50(full["y_true"].values, full["y_pred"].values)
        s_valid = full["target_day"].isin(search_days)
        c_valid = full["target_day"].isin(confirm_days)
        s_smape = smape_floor50(full.loc[s_valid, "y_true"].values, full.loc[s_valid, "y_pred"].values) if s_valid.sum() >= 10 else None
        c_smape = smape_floor50(full.loc[c_valid, "y_true"].values, full.loc[c_valid, "y_pred"].values) if c_valid.sum() >= 10 else None

        config_log.append(dict(config=name, window=w, objective=params["objective"],
                               num_leaves=params["num_leaves"], full_smape=round(f_smape, 4),
                               search_smape=round(s_smape, 4) if s_smape else None,
                               confirm_smape=round(c_smape, 4) if c_smape else None))
        logger.info(f"    {name}: full={f_smape:.4f}% search={s_smape} confirm={c_smape}")

    # Save config log
    cdf = pd.DataFrame(config_log).sort_values("full_smape")
    cdf.to_csv(str(out / "metrics" / "config_search_results.csv"), index=False, encoding="utf-8-sig")

    if len(config_log) == 0:
        logger.error("No micro-search results!")
        return None

    best_name = cdf.iloc[0]["config"]
    best_smape = cdf.iloc[0]["full_smape"]
    best_pred = results[best_name]
    logger.info(f"  Micro-search best: {best_name} = {best_smape:.4f}%")

    # Full summary
    summary_rows = []
    for name, pred_df in results.items():
        m = compute_all_metrics(pred_df["y_true"].values, pred_df["y_pred"].values)
        m["model_name"] = name
        m["task"] = "dayahead"
        summary_rows.append(m)
    sdf = pd.DataFrame(summary_rows).sort_values("sMAPE_floor50")
    sdf.to_csv(str(out / "metrics" / "summary.csv"), index=False, encoding="utf-8-sig")

    # Hour/period for best
    for h, grp in best_pred.groupby("hour_business"):
        pass
    pd.DataFrame([{"hour_business": int(h)} | compute_all_metrics(grp["y_true"].values, grp["y_pred"].values)
                  for h, grp in best_pred.groupby("hour_business")]).to_csv(
        str(out / "metrics" / "hour_metrics.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([{"period": p} | compute_all_metrics(grp["y_true"].values, grp["y_pred"].values)
                  for p, grp in best_pred.groupby("period")]).to_csv(
        str(out / "metrics" / "period_metrics.csv"), index=False, encoding="utf-8-sig")

    # Quick report snippet
    lines = [f"# LightGBM Micro-Search Report",
             f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             f"> Business-day corrected mapping",
             "", "## Ranking", "",
             "| Config | Full sMAPE | Search | Confirm | Window | Obj | nl |",
             "|---|---|---|---|---|---|---|"]
    for _, r in cdf.iterrows():
        s = f"{r['search_smape']:.2f}%" if r['search_smape'] else "N/A"
        c = f"{r['confirm_smape']:.2f}%" if r['confirm_smape'] else "N/A"
        lines.append(f"| {r['config']} | {r['full_smape']:.2f}% | {s} | {c} | {r['window']} | {r['objective']} | {r['num_leaves']} |")
    lines.append("")
    lines.append(f"**Best**: {best_name} = {best_smape:.4f}%")
    lines.append(f"Below 11.85%? {'✅' if best_smape < 11.85 else '❌'}")
    lines.append(f"Below 11.5%? {'✅' if best_smape < 11.5 else '❌'}")
    lines.append(f"Below 11.0%? {'✅' if best_smape < 11.0 else '❌'}")
    (out / "reports" / "lgbm_microsearch_report.md").write_text("\n".join(lines), encoding="utf-8")

    return out, best_name, best_smape, results


# ═══════════════════════════════════════════════════════════
#  TASK B: Safe fusion final
# ═══════════════════════════════════════════════════════════
def task_b_safe_fusion(micro_best_pred=None, micro_smape=None):
    logger.info("\n" + "=" * 65)
    logger.info("TASK B: Safe fusion final")
    logger.info("=" * 65)
    out = Path("outputs/dayahead_final_fusion_30d")
    for d in ["fusion", "metrics", "reports"]:
        (out / d).mkdir(parents=True, exist_ok=True)

    # ── Load all candidate predictions (aligned on ds) ──
    def load_pred(path, name):
        df = pd.read_csv(path, encoding="utf-8-sig")
        yp = df["y_pred_cb"].values if "y_pred_cb" in df.columns else df["y_pred"].values
        return df[["ds", "y_true"]].assign(y_pred=yp, model_name=name)

    candidates = {
        "best_two_average_champion": "outputs/dayahead_lgbm_freeze_30d/predictions/best_two_average_dayahead.csv",
        "catboost_sota": "outputs/dayahead_30d_core/predictions/catboost_sota_dayahead.csv",
        "catboost_spike_residual": "outputs/dayahead_corrections_30d/predictions/catboost_spike_residual_corrected_dayahead.csv",
        "tabpfn_ts_sota": "outputs/dayahead_30d_core/predictions/tabpfn_ts_sota_dayahead.csv",
        "stage3_baseline": "outputs/dayahead_lgbm_stage3_business_fixed_30d/predictions/stage3_baseline_90d_mae_dayahead.csv",
    }
    if micro_best_pred is not None:
        candidates["micro_best"] = (micro_best_pred, "inline")

    loaded = {}
    for name, path in candidates.items():
        try:
            if isinstance(path, tuple):
                df = path[0]
            else:
                df = load_pred(path, name)
            loaded[name] = df[["ds", "y_pred", "y_true"]].copy()
        except Exception as e:
            logger.warning(f"  Could not load {name}: {e}")

    if len(loaded) < 2:
        logger.error("Need at least 2 models for fusion")
        return None

    # Merge on ds
    merged = None
    for name, df in loaded.items():
        tmp = df[["ds", "y_pred", "y_true"]].rename(columns={"y_pred": name})
        if merged is None:
            merged = tmp
        else:
            merged = merged.merge(tmp, on=["ds", "y_true"], how="inner")
    logger.info(f"  Merged {len(merged)} rows from {len(loaded)} models")

    # Split into search/confirm
    search_mask = merged["ds"].between(f"{_EVAL_START} 01:00:00", f"{_SEARCH_END} 23:00:00")
    confirm_mask = merged["ds"].between(f"{_CONFIRM_START} 01:00:00", f"{_EVAL_END} 23:00:00")

    # ── Fusion methods (all weights from search window only) ──
    y_true = merged["y_true"].values
    model_cols = [c for c in merged.columns if c not in ["ds", "y_true"]]

    fusion_results = {}

    # 1. Simple average of top2 by search sMAPE
    search_perf = {c: smape_floor50(merged.loc[search_mask, "y_true"].values,
                                     merged.loc[search_mask, c].values) for c in model_cols}
    top2 = sorted(search_perf, key=search_perf.get)[:2]
    fusion_results["avg_top2"] = merged[top2].mean(axis=1).values

    # 2. Median of top3
    top3 = sorted(search_perf, key=search_perf.get)[:3]
    fusion_results["median_top3"] = merged[top3].median(axis=1).values

    # 3. Inverse search-smape weight
    inv_w = {c: 1.0 / max(s, 0.01) for c, s in search_perf.items()}
    total_w = sum(inv_w.values())
    fusion_results["inv_smape_weight"] = sum(
        merged[c].values * inv_w[c] / total_w for c in model_cols)

    # 4. Winner by hour (based on search window only)
    merged_hour = merged.copy()
    merged_hour["hour"] = pd.to_datetime(merged_hour["ds"]).dt.hour
    # Winner per hour: best model on search window per hour
    hour_winners = {}
    for h in range(0, 24):
        hm = merged_hour["hour"] == h
        hm_search = hm & search_mask
        if hm_search.sum() < 5:
            hour_winners[h] = model_cols[0]
            continue
        best = min(model_cols, key=lambda c: smape_floor50(
            merged_hour.loc[hm_search, "y_true"].values,
            merged_hour.loc[hm_search, c].values))
        hour_winners[h] = best
    fusion_results["winner_by_hour"] = np.array([
        merged_hour.loc[i, hour_winners[merged_hour.loc[i, "hour"]]]
        for i in merged_hour.index])

    # 5. Winner by period (based on search window only)
    from src.common.business_time import infer_period
    merged_hour["period"] = merged_hour["hour"].apply(lambda h: infer_period(h + 1 if h < 23 else 24))
    period_winners = {}
    for p in ["1_8", "9_16", "17_24"]:
        pm = merged_hour["period"] == p
        pm_search = pm & search_mask
        if pm_search.sum() < 10:
            period_winners[p] = model_cols[0]
            continue
        best = min(model_cols, key=lambda c: smape_floor50(
            merged_hour.loc[pm_search, "y_true"].values,
            merged_hour.loc[pm_search, c].values))
        period_winners[p] = best
    fusion_results["winner_by_period"] = np.array([
        merged_hour.loc[i, period_winners[merged_hour.loc[i, "period"]]]
        for i in merged_hour.index])

    # ── Evaluate ──
    rows = []
    # Individual models
    for c in model_cols:
        for mask, label in [(slice(None), "full"), (search_mask, "search"), (confirm_mask, "confirm")]:
            yt = y_true[mask]
            yp = merged[c].values[mask]
            if len(yt) < 5:
                continue
            rows.append({"model_name": c, "split": label, "sMAPE_floor50": round(smape_floor50(yt, yp), 4)})
    # Fusion methods
    for fname, fpred in fusion_results.items():
        for mask, label in [(slice(None), "full"), (search_mask, "search"), (confirm_mask, "confirm")]:
            yt = y_true[mask]
            yp = fpred[mask]
            if len(yt) < 5:
                continue
            rows.append({"model_name": f"fused_{fname}", "split": label, "sMAPE_floor50": round(smape_floor50(yt, yp), 4)})

    rdf = pd.DataFrame(rows)
    rdf.to_csv(str(out / "metrics" / "summary.csv"), index=False, encoding="utf-8-sig")
    full_fusion = rdf[rdf["split"] == "full"].sort_values("sMAPE_floor50")

    # ── Save fusion predictions ──
    for fname, fpred in fusion_results.items():
        fout = merged[["ds", "y_true"]].copy()
        fout["y_pred"] = fpred
        fout["model_name"] = f"fused_{fname}"
        fout.to_csv(str(out / "fusion" / f"fused_{fname}_dayahead.csv"), index=False, encoding="utf-8-sig")

    # ── Report ──
    best_fusion = full_fusion.iloc[0]
    lines = [f"# Day-Ahead Final Fusion Report",
             f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             f"> Candidate models: {list(loaded.keys())}",
             "", "## Fusion Ranking (Full 30d)", "",
             "| Model | sMAPE_floor50 |",
             "|---|---|"]
    for _, r in full_fusion.iterrows():
        lines.append(f"| {r['model_name']} | {r['sMAPE_floor50']:.4f}% |")
    lines.append("")
    bf = best_fusion["sMAPE_floor50"]
    lines.append(f"**Best**: {best_fusion['model_name']} = {bf:.4f}%")
    lines.append(f"Below 11.85%? {'✅' if bf < 11.85 else '❌'}")
    lines.append(f"Below 11.5%? {'✅' if bf < 11.5 else '❌'}")
    lines.append(f"Below 11.0%? {'✅' if bf < 11.0 else '❌'}")
    (out / "reports" / "dayahead_final_fusion_report.md").write_text("\n".join(lines), encoding="utf-8")

    logger.info(f"Fusion best: {best_fusion['model_name']} = {bf:.4f}%")
    return out, best_fusion["model_name"], bf


# ═══════════════════════════════════════════════════════════
#  TASK C: XGBoost sentinel mini
# ═══════════════════════════════════════════════════════════
def task_c_xgboost_mini(df, feat_cols, all_days, search_days, confirm_days):
    logger.info("\n" + "=" * 65)
    logger.info("TASK C: XGBoost sentinel mini (4 configs)")
    logger.info("=" * 65)
    out = Path("outputs/dayahead_xgboost_sentinel_mini_30d")
    for d in ["predictions", "metrics", "reports"]:
        (out / d).mkdir(parents=True, exist_ok=True)
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error

    XGB_CONFIGS = [
        dict(_name="xgb01", objective="reg:absoluteerror", max_depth=6, learning_rate=0.03,
             subsample=0.85, colsample_bytree=0.85, reg_lambda=5, reg_alpha=0.5,
             min_child_weight=5, n_estimators=1500),
        dict(_name="xgb02", objective="reg:absoluteerror", max_depth=6, learning_rate=0.03,
             subsample=0.85, colsample_bytree=0.85, reg_lambda=5, reg_alpha=0.5,
             min_child_weight=10, n_estimators=1500),
        dict(_name="xgb03", objective="reg:pseudohubererror", max_depth=6, learning_rate=0.03,
             subsample=0.85, colsample_bytree=0.85, reg_lambda=5, reg_alpha=0.5,
             min_child_weight=10, n_estimators=1500),
        dict(_name="xgb04", objective="reg:squarederror", max_depth=4, learning_rate=0.03,
             subsample=0.85, colsample_bytree=0.85, reg_lambda=10, reg_alpha=1.0,
             min_child_weight=10, n_estimators=1500),
    ]
    XGB_WINDOWS = [90, 120, 150, 90]

    results = {}
    config_log = []
    for params, w in zip(XGB_CONFIGS, XGB_WINDOWS):
        name = params.pop("_name")
        logger.info(f"  Running {name} (window={w})...")
        preds = []
        for day in all_days:
            target_dt = pd.Timestamp(day)
            train_all = df[df["target_day"] < day]
            if len(train_all) < 200:
                continue
            if w == "all":
                train_df = train_all.tail(_MAX_TRAIN_ROWS)
            else:
                train_df = train_all[train_all["ds"] >= (target_dt - timedelta(days=w))].tail(_MAX_TRAIN_ROWS)
            if len(train_df) < 100:
                continue

            X_tr = train_df[feat_cols].values.astype(float)
            y_tr = train_df["y"].values.astype(float)
            try:
                model = xgb.XGBRegressor(**params, verbosity=0, n_jobs=1, random_state=42)
                model.fit(X_tr, y_tr)
                day_df = df[df["target_day"] == day].copy()
                day_df["y_pred"] = model.predict(day_df[feat_cols].values.astype(float))
                day_df["y_true"] = day_df["y"].values
                day_df["model_name"] = name
                preds.append(day_df)
            except Exception as e:
                logger.warning(f"    {name} day {day}: {e}")
                continue
        params["_name"] = name

        if not preds:
            continue
        full = pd.concat(preds, ignore_index=True)
        results[name] = full
        full.to_csv(str(out / "predictions" / f"{name}_dayahead.csv"), index=False, encoding="utf-8-sig")

        f_smape = smape_floor50(full["y_true"].values, full["y_pred"].values)
        s_valid = full["target_day"].isin(search_days)
        c_valid = full["target_day"].isin(confirm_days)
        s_smape = smape_floor50(full.loc[s_valid, "y_true"].values, full.loc[s_valid, "y_pred"].values) if s_valid.sum() >= 10 else None
        c_smape = smape_floor50(full.loc[c_valid, "y_true"].values, full.loc[c_valid, "y_pred"].values) if c_valid.sum() >= 10 else None
        config_log.append(dict(config=name, window=w, objective=params.get("objective", "?"),
                               full_smape=round(f_smape, 4),
                               search_smape=round(s_smape, 4) if s_smape else None,
                               confirm_smape=round(c_smape, 4) if c_smape else None))
        logger.info(f"    {name}: full={f_smape:.4f}%")

    if not config_log:
        logger.warning("No XGBoost results")
        return None

    cdf = pd.DataFrame(config_log).sort_values("full_smape")
    cdf.to_csv(str(out / "metrics" / "summary.csv"), index=False, encoding="utf-8-sig")
    best_smape = cdf.iloc[0]["full_smape"]
    best_name = cdf.iloc[0]["config"]
    logger.info(f"XGBoost best: {best_name} = {best_smape:.4f}%")

    lines = [f"# XGBoost Sentinel Mini Report",
             f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             "", "## Ranking", "",
             "| Config | Full sMAPE | Search | Confirm |",
             "|---|---|---|---|"]
    for _, r in cdf.iterrows():
        s = f"{r['search_smape']:.2f}%" if r['search_smape'] else "N/A"
        c = f"{r['confirm_smape']:.2f}%" if r['confirm_smape'] else "N/A"
        lines.append(f"| {r['config']} | {r['full_smape']:.2f}% | {s} | {c} |")
    lines.append(f"\n**Best XGBoost**: {best_name} = {best_smape:.4f}%")
    (out / "reports" / "xgboost_sentinel_report.md").write_text("\n".join(lines), encoding="utf-8")
    return out, best_name, best_smape


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════
def main():
    # ── Load data once ──
    logger.info("Loading data and building features...")
    raw = load_data(str(get_data_path()), target="dayahead")
    df = build_v3_features(raw)
    df = df[df["ds"] >= "2025-08-01"].reset_index(drop=True)
    feat_cols = get_feature_cols(df)
    logger.info(f"Feature DF: {len(df)} rows, {len(feat_cols)} features")

    # All evaluation days (correct business-day mapping)
    all_days = sorted(df[
        (df["ds"] >= f"{_EVAL_START} 01:00:00") & (df["ds"] <= f"{_EVAL_END} 23:00:00")
    ]["target_day"].unique())
    # Filter to only Feb 1 - Mar 2
    all_days = [d for d in all_days if d >= _EVAL_START and d <= _EVAL_END]
    search_days = [d for d in all_days if d <= _SEARCH_END]
    confirm_days = [d for d in all_days if d >= _CONFIRM_START]
    logger.info(f"Evaluation: {len(all_days)} days ({all_days[0]} -> {all_days[-1]})")

    # ── A: LGBM micro-search ──
    a_result = task_a_lgbm_microsearch(df, feat_cols, all_days, search_days, confirm_days)
    if a_result is None:
        logger.error("Task A failed — aborting")
        return
    a_out, a_best_name, a_best_smape, a_all_preds = a_result
    a_below_115 = a_best_smape < 11.5
    a_below_1185 = a_best_smape < 11.85

    # ── Decide: skip B if A already <= 11.5? No — user says proceed to B regardless ──
    # "如果任意配置 full_30d <= 11.5，停止后续大搜索，进入 safe fusion final"
    # So we always run B after A.

    # ── B: Safe fusion ──
    micro_best_pred = a_all_preds.get(a_best_name) if a_all_preds else None
    b_result = task_b_safe_fusion(micro_best_pred=micro_best_pred, micro_smape=a_best_smape)
    b_below_115 = False
    b_below_1185 = False
    b_best_smape = 99.0
    b_best_name = "none"
    if b_result is not None:
        b_out, b_best_name, b_best_smape = b_result
        b_below_115 = b_best_smape < 11.5
        b_below_1185 = b_best_smape < 11.85

    # ── C: XGBoost mini (only if A+B haven't reached 11.5) ──
    c_result = None
    if not (a_below_115 or b_below_115):
        logger.info("A+B did not reach 11.5 — running XGBoost sentinel")
        c_result = task_c_xgboost_mini(df, feat_cols, all_days, search_days, confirm_days)
    else:
        logger.info("Already below 11.5 — skipping XGBoost")

    c_best_smape = 99.0
    c_best_name = "none"
    c_below_115 = False
    if c_result is not None:
        c_out, c_best_name, c_best_smape = c_result
        c_below_115 = c_best_smape < 11.5

    # ── Determine champion ──
    overall_best = min(
        ("best_two_average", 11.85),
        (f"micro_{a_best_name}", a_best_smape) if a_best_smape < 11.85 else None,
        (f"fusion_{b_best_name}", b_best_smape) if b_best_smape < 11.85 else None,
        (f"xgb_{c_best_name}", c_best_smape) if c_best_smape < 11.85 else None,
        key=lambda x: x[1] if x else 99.0,
    )
    # Actually just pick the minimum sMAPE from all runs
    all_results_list = [("best_two_average (champion)", 11.85)]
    if a_result:
        all_results_list.append((f"micro_{a_best_name}", a_best_smape))
    if b_result:
        all_results_list.append((f"fusion_{b_best_name}", b_best_smape))
    if c_result:
        all_results_list.append((f"xgb_{c_best_name}", c_best_smape))

    all_results_list.sort(key=lambda x: x[1])

    # ── Final report ──
    from pathlib import Path as _PP
    docs = _PP("docs/reports")
    docs.mkdir(parents=True, exist_ok=True)

    best_name, best_smape = all_results_list[0]

    lines = []
    lines.append("# Day-Ahead Final Sprint Report")
    lines.append(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## 1. Current Trusted Champion")
    lines.append("")
    lines.append("**best_two_average = 11.85%**")
    lines.append("- LightGBM trial_02 + trial_24 pure prediction average")
    lines.append("- 720 rows, hours 1-24, business-day correct")
    lines.append("- No y_true leakage")
    lines.append("")
    lines.append("## 2. Invalid Result")
    lines.append("")
    lines.append("- lgbm_spike_residual_corrected (11.27%): **INVALIDATED** — target leakage in prediction features")
    lines.append("")
    lines.append("## 3. Stage3 Business-Day Fix")
    lines.append("")
    lines.append("- Old Stage3 (11.64%): invalid — natural-day grouping error")
    lines.append("- Fixed Stage3 (business_time_mapping): 11.86% — did NOT beat champion")
    lines.append("")

    lines.append("## 4. LightGBM Micro-Search (Task A)")
    lines.append("")
    lines.append(f"**Best**: {a_best_name} = {a_best_smape:.4f}%")
    lines.append(f"Below 11.85%? {'✅' if a_below_1185 else '❌'}")
    lines.append(f"Below 11.5%? {'✅' if a_below_115 else '❌'}")
    lines.append("")

    lines.append("## 5. Safe Fusion Final (Task B)")
    lines.append("")
    lines.append(f"**Best**: {b_best_name} = {b_best_smape:.4f}%")
    lines.append(f"Below 11.85%? {'✅' if b_below_1185 else '❌'}")
    lines.append(f"Below 11.5%? {'✅' if b_below_115 else '❌'}")
    lines.append("")

    lines.append("## 6. XGBoost Sentinel Mini (Task C)")
    if c_result:
        lines.append(f"**Best**: {c_best_name} = {c_best_smape:.4f}%")
        lines.append(f"Below 11.5%? {'✅' if c_below_115 else '❌'}")
    else:
        lines.append("Skipped (A+B already reached 11.5 target)")
    lines.append("")

    lines.append("## 7. Final Ranking")
    lines.append("")
    lines.append("| Rank | Model | sMAPE |")
    lines.append("|:----:|------|:-----:|")
    for i, (n, s) in enumerate(all_results_list, 1):
        lines.append(f"| {i} | {n} | {s:.4f}% |")
    lines.append("")

    # Decision
    if best_smape <= 11.5:
        lines.append("## Decision: New trusted champion candidate found!")
        lines.append(f"**{best_name} = {best_smape:.4f}%** — below 11.5% target.")
        champion_status = "NEW"
    elif best_smape < 11.85:
        lines.append(f"## Decision: Improved but not enough")
        lines.append(f"**{best_name} = {best_smape:.4f}%** — below 11.85% but not below 11.5%.")
        champion_status = "IMPROVED"
    else:
        lines.append("## Decision: No further improvement")
        lines.append("Keep best_two_average 11.85% as trusted champion.")
        champion_status = "UNCHANGED"

    lines.append("")
    lines.append("## 8. Recommendations")
    lines.append("")
    lines.append(f"- {'AutoGluon/N-BEATSx not recommended for this sprint.' if champion_status != 'NEW' else 'Freeze new champion. Sprint complete.'}")
    lines.append("- If further improvement needed: AutoGluon light preset or N-BEATSx with exogenous variables")
    lines.append("- These require longer timelines and are not suitable for current sprint.")
    lines.append("")

    report = "\n".join(lines)
    (docs / "dayahead_final_sprint_report.md").write_text(report, encoding="utf-8")
    print()
    print("=" * 65)
    print(report)
    print("=" * 65)


if __name__ == "__main__":
    main()
