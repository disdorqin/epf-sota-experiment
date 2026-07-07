"""
run_dayahead_p1_walkforward.py — P1 Day-ahead 统一 walk-forward 对比框架

在 epf-sota-experiment 中，基于 2.5 日前数据处理/业务时间/指标口径，
系统复现并对比日前电价预测候选模型，并与 2.5 lightgbm 基线严格比较。

设计要点（务必遵守红线）：
- 预测 D+1 日前电价，仅使用 <= D 日可见数据；禁止任何 D+1 / 目标日 actual 进特征。
- hour_business 1..24；period 1_8 / 9_16 / 17_24（与 2.5 一致）。
- 主指标 sMAPE_floor50（floor 50），与 2.5 fusion/metrics.py 一致。
- walk-forward 采用「按月重训」：每月用历史(<= 月初)训练，预测当月所有 24h 完整日；
  训练数据严格早于目标日 -> 无 leakage。
- 输出：predictions csv / metrics json / comparison report md，格式对齐 3.0 candidate contract。

用法：
  python scripts/run_dayahead_p1_walkforward.py \
      --data-path "..."/shandong_pmos_hourly.csv \
      --test-months 2025-01,2025-03,2025-06,2025-09,2025-12,2026-01,2026-02,2026-03,2026-04,2026-05,2026-06 \
      --models baseline_lgbm25,lightgbm_variant,catboost,xgboost,ensemble \
      --train-window-months 18 \
      --output-root outputs/p1_dayahead/run_YYYYMMDD_HHMM \
      --run-id p1_001
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from src.common.data_loader import load_data
from src.common.metrics import smape_floor50, mae, rmse, compute_all_metrics
from src.common.business_time import infer_period, business_time_mapping

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "hour", "month", "day_of_week", "is_weekend", "hour_sin", "hour_cos",
    "lag_price_target", "price_rolling_mean_24h",
    "load", "wind", "solar", "interconnect",
    "bidding_space", "space_ratio", "net_load", "solar_ratio", "net_load_sq",
    "wind_ratio", "renew_penetration", "ramp_load", "ramp_solar",
    "prev_day_avg", "prev_day_max", "prev_day_min",
]
VALLEY = list(range(1, 9))      # 1_8
SOLAR = list(range(9, 17))       # 9_16
PEAK = list(range(17, 25))       # 17_24


# ─────────────────────────── 特征工程（对齐 2.5，leakage-free） ───────────────────────────
def build_features_25(df: pd.DataFrame) -> pd.DataFrame:
    """复刻 2.5 lightGBM/train_da_fix.py::feature_engineering（日前版）。
    所有滞后/滚动/昨日统计均引用历史 y，目标日 actual 不会泄漏。"""
    df = df.copy()
    adjusted = df["ds"] - pd.Timedelta(seconds=1)  # 1秒偏移法
    df["hour"] = adjusted.dt.hour + 1
    df["month"] = adjusted.dt.month
    df["day_of_week"] = adjusted.dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * (df["hour"] - 1) / 23)
    df["hour_cos"] = np.cos(2 * np.pi * (df["hour"] - 1) / 23)

    df["lag_24h"] = df["y"].shift(24)
    df["lag_168h"] = df["y"].shift(168)
    df["lag_price_target"] = np.where(df["day_of_week"] == 0, df["lag_168h"], df["lag_24h"])
    df["price_rolling_mean_24h"] = df["y"].shift(24).rolling(24).mean()
    df["lag_price_target"] = df["lag_price_target"].ffill().fillna(0)
    df["price_rolling_mean_24h"] = df["price_rolling_mean_24h"].ffill().fillna(0)

    safe_load = df["load"].replace(0, 1)
    df["net_load"] = df["load"] - df["wind"] - df["solar"]
    df["solar_ratio"] = df["solar"] / safe_load
    df["net_load_sq"] = (df["net_load"] / 1000) ** 2
    df["bidding_space"] = df["net_load"] - df["interconnect"]
    df["space_ratio"] = df["bidding_space"] / safe_load
    df["wind_ratio"] = df["wind"] / safe_load
    df["renew_penetration"] = (df["wind"] + df["solar"]) / safe_load
    df["ramp_load"] = df["load"].diff().fillna(0)
    df["ramp_solar"] = df["solar"].diff().fillna(0)

    df["date_only"] = adjusted.dt.date
    daily_stats = (
        df.groupby("date_only")["y"]
        .agg(prev_day_avg="mean", prev_day_max="max", prev_day_min="min")
        .shift(1)
        .reset_index()
    )
    df = df.merge(daily_stats, on="date_only", how="left")
    df = df.drop(columns=["date_only", "lag_24h", "lag_168h"], errors="ignore")

    df["business_day"] = adjusted.dt.date
    df["hour_business"] = df["hour"]
    df["period"] = df["hour_business"].map(infer_period)
    df["target_day"] = df["business_day"].astype(str)
    df = df.sort_values("ds").reset_index(drop=True)
    return df


# ─────────────────────────── 富特征（对齐 cfg05：2.5 base + dayahead lag/stats/momentum/calendar/volatility/interaction） ───────────────────────────
def build_features_rich(raw: pd.DataFrame) -> pd.DataFrame:
    """复刻 run_champion_cfg05.build_features 的富特征集（不含日期过滤），用于 cfg05 / 富特征候选的公平对比。"""
    try:
        from src.common.feature_builder import build_features as build_base
        from src.common.feature_builder_dayahead import (
            _add_lag_features, _add_same_hour_stats, _add_price_momentum, _add_calendar_features,
        )
        from src.common.feature_builder_dayahead_v3 import (
            _add_volatility, _add_change_features, _add_exact_spring_festival, _add_interaction_features,
        )
    except Exception as e:
        logger.warning(f"rich features unavailable ({e}); falling back to build_features_25")
        return build_features_25(raw)
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


def get_rich_feature_cols(df):
    exclude = {"ds", "y", "target_day", "business_day", "hour_business", "period",
               "date_only", "y_pred", "y_true", "model_name", "task", "lag_48h_raw", "lag_168h_raw"}
    numeric = df.select_dtypes(include=[np.float64, np.int64, np.float32, np.int32, np.int8, bool]).columns
    return [c for c in numeric if c not in exclude and c != "y"]


def _prep_X(df, cols=FEATURE_COLS):
    avail = [c for c in cols if c in df.columns]
    X = df[avail].copy()
    for c in ["hour", "month", "day_of_week", "is_weekend"]:
        if c in X.columns:
            X[c] = X[c].astype(int)
    return X.fillna(0)


# ─────────────────────────── 模型适配器（统一接口） ───────────────────────────
class BaseAdapter:
    name = "base"
    version = "v1"
    source_repo = "epf-sota-experiment"

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None):
        raise NotImplementedError

    def predict(self, target_df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def train_predict_month(self, feat_df: pd.DataFrame, month_start: pd.Timestamp,
                             window_months: int, month_days: list[str]):
        """按月重训：train on ds < month_start (windowed)，predict 指定 business_day 列表。"""
        t0 = time.time()
        cutoff = month_start
        train_start = cutoff - pd.DateOffset(months=window_months)
        train_mask = (feat_df["ds"] >= train_start) & (feat_df["ds"] < cutoff)
        train_df = feat_df[train_mask].dropna(subset=FEATURE_COLS + ["y"]).copy()
        if len(train_df) < 2000:
            # 兜底：用全部历史
            train_df = feat_df[feat_df["ds"] < cutoff].dropna(subset=FEATURE_COLS + ["y"]).copy()
        val_df = None
        try:
            self.fit(train_df, val_df)
        except Exception as e:
            logger.error(f"  [{self.name}] fit failed: {e}")
            return {}, time.time() - t0

        preds = {}
        for bd in month_days:
            tday = pd.Timestamp(bd)
            day_mask = feat_df["business_day"] == tday.date()
            day_df = feat_df[day_mask].copy()
            if len(day_df) == 0:
                continue
            try:
                p = self.predict(day_df)
                preds[bd] = float(np.nanmean(p)) if len(p) == 0 else p
            except Exception as e:
                logger.error(f"  [{self.name}] predict {bd} failed: {e}")
                preds[bd] = None
        return preds, time.time() - t0


# ─────────────────────────── GPU 优先 / CPU 回退 ───────────────────────────
def _detect_gpu():
    """探测各库 GPU 可用性（优先 GPU，不可用则回退 CPU）。"""
    cap = {"lightgbm": False, "catboost": False, "xgboost": False}
    try:
        import numpy as np, lightgbm as lgb
        X = np.random.randn(100, 4); y = np.random.randn(100)
        lgb.LGBMRegressor(objective="regression", n_estimators=5, num_leaves=7,
                          device="gpu", gpu_device_id=0, verbose=-1).fit(X, y)
        cap["lightgbm"] = True
    except Exception:
        cap["lightgbm"] = False
    try:
        import numpy as np
        from catboost import CatBoostRegressor
        X = np.random.randn(100, 4); y = np.random.randn(100)
        CatBoostRegressor(iterations=5, depth=3, task_type="GPU", devices="0",
                          verbose=False).fit(X, y)
        cap["catboost"] = True
    except Exception:
        cap["catboost"] = False
    # 本环境 xgboost 构建无 GPU 支持（tree_method 仅含 hist/approx/exact），固定 CPU
    cap["xgboost"] = False
    return cap

GPU = _detect_gpu()
logger.info(f"GPU detection: lightgbm={GPU['lightgbm']} catboost={GPU['catboost']} "
            f"xgboost={GPU['xgboost']} (GPU 优先，CPU 回退)")

def _lgbm_device():
    """LightGBM 设备参数（sklearn API 与 lgb.train 通用）。"""
    if GPU["lightgbm"]:
        # 注意：lightgbm GPU 路径不要设 num_threads=1（已知会与 GPU histogram
        # 工作线程死锁导致训练卡死 0% util）。交由 lightgbm 自行管理 GPU 线程。
        return {"device": "gpu", "gpu_device_id": 0}
    return {"num_threads": 4}

def _catboost_device():
    """CatBoost 设备参数。"""
    if GPU["catboost"]:
        return {"task_type": "GPU", "devices": "0"}
    return {}


class BaselineLGBM25(BaseAdapter):
    """忠实复刻 2.5 日前 ThreeStageLGBM（valley/solar/peak + solar 负价分类器）。"""
    name = "baseline_lgbm25"
    version = "v25"
    source_repo = "electricity_forecast_model2.5"

    def fit(self, train_df, val_df=None):
        import lightgbm as lgb
        y_clip = train_df["y"].clip(lower=-100, upper=train_df["y"].quantile(0.995))
        self.models = {}
        common = dict(verbose=-1, random_state=42, **_lgbm_device())
        vr = lgb.LGBMRegressor(objective="regression", n_estimators=2000, learning_rate=0.05,
                               num_leaves=31, **common)
        vr.fit(train_df[train_df.hour_business.isin(VALLEY)][FEATURE_COLS], y_clip[train_df.hour_business.isin(VALLEY)])
        self.models["valley"] = vr
        sr = lgb.LGBMRegressor(objective="regression", n_estimators=3000, learning_rate=0.03,
                               num_leaves=63, **common)
        sr.fit(train_df[train_df.hour_business.isin(SOLAR)][FEATURE_COLS], y_clip[train_df.hour_business.isin(SOLAR)])
        self.models["solar"] = sr
        sc = lgb.LGBMClassifier(objective="binary", n_estimators=1000, learning_rate=0.05,
                                class_weight="balanced", **common)
        sc.fit(train_df[train_df.hour_business.isin(SOLAR)][FEATURE_COLS],
               (train_df[train_df.hour_business.isin(SOLAR)]["y"] < 0).astype(int))
        self.models["solar_clf"] = sc
        pr = lgb.LGBMRegressor(objective="regression", n_estimators=3000, learning_rate=0.03,
                               num_leaves=40, **common)
        pr.fit(train_df[train_df.hour_business.isin(PEAK)][FEATURE_COLS], y_clip[train_df.hour_business.isin(PEAK)])
        self.models["peak"] = pr

    def predict(self, target_df):
        Xv = _prep_X(target_df[target_df.hour_business.isin(VALLEY)])
        Xs = _prep_X(target_df[target_df.hour_business.isin(SOLAR)])
        Xp = _prep_X(target_df[target_df.hour_business.isin(PEAK)])
        pred = np.zeros(len(target_df))
        pred[target_df.hour_business.isin(VALLEY).values] = self.models["valley"].predict(Xv)
        sp = self.models["solar"].predict(Xs)
        neg = self.models["solar_clf"].predict_proba(Xs)[:, 1]
        sp[(neg > 0.7) & (sp > 0)] -= 80  # 日前负价修正（阈值0.7，幅度-80）
        pred[target_df.hour_business.isin(SOLAR).values] = sp
        pred[target_df.hour_business.isin(PEAK).values] = self.models["peak"].predict(Xp)
        return np.clip(pred, -80, None)


class LightGBMVariant(BaseAdapter):
    """LightGBM 变体（cfg05 风格：90d 窗口 + MAE objective + 三段）。"""
    name = "lightgbm_variant"
    version = "v_cfg05"
    source_repo = "epf-sota-experiment"

    def fit(self, train_df, val_df=None):
        import lightgbm as lgb
        self.models = {}
        common = dict(verbose=-1, random_state=42, **_lgbm_device(),
                      objective="regression_l1", n_estimators=1500,
                      learning_rate=0.03, num_leaves=63,
                      early_stopping_rounds=80)
        for seg, hours in [("valley", VALLEY), ("solar", SOLAR), ("peak", PEAK)]:
            sub = train_df[train_df.hour_business.isin(hours)].sort_values("ds")
            y = sub["y"].clip(lower=-100, upper=sub["y"].quantile(0.995))
            n = len(sub)
            split = max(1, int(n * 0.85))  # 末段 15% 作为验证（时间序列不受未来泄漏）
            X_tr, y_tr = sub[FEATURE_COLS].iloc[:split], y.iloc[:split]
            X_val, y_val = sub[FEATURE_COLS].iloc[split:], y.iloc[split:]
            m = lgb.LGBMRegressor(**common)
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="l1")
            self.models[seg] = m

    def predict(self, target_df):
        pred = np.zeros(len(target_df))
        for seg, hours in [("valley", VALLEY), ("solar", SOLAR), ("peak", PEAK)]:
            mask = target_df.hour_business.isin(hours).values
            pred[mask] = self.models[seg].predict(_prep_X(target_df[mask]))
        return np.clip(pred, -80, None)


class CatBoostCandidate(BaseAdapter):
    name = "catboost"
    version = "v1"
    source_repo = "epf-sota-experiment"

    def fit(self, train_df, val_df=None):
        from catboost import CatBoostRegressor
        self.models = {}
        for seg, hours in [("valley", VALLEY), ("solar", SOLAR), ("peak", PEAK)]:
            sub = train_df[train_df.hour_business.isin(hours)]
            m = CatBoostRegressor(iterations=1500, learning_rate=0.03, depth=8,
                                   loss_function="MAE", l2_leaf_reg=5.0,
                                   verbose=False, random_state=42, **_catboost_device())
            m.fit(sub[FEATURE_COLS], sub["y"].clip(lower=-100, upper=sub["y"].quantile(0.995)))
            self.models[seg] = m

    def predict(self, target_df):
        pred = np.zeros(len(target_df))
        for seg, hours in [("valley", VALLEY), ("solar", SOLAR), ("peak", PEAK)]:
            mask = target_df.hour_business.isin(hours).values
            pred[mask] = self.models[seg].predict(_prep_X(target_df[mask]))
        return np.clip(pred, -80, None)


class XGBoostCandidate(BaseAdapter):
    name = "xgboost"
    version = "v1"
    source_repo = "epf-sota-experiment"

    def fit(self, train_df, val_df=None):
        import xgboost as xgb
        self.models = {}
        params = dict(objective="reg:absoluteerror", max_depth=8, eta=0.05,
                      subsample=0.9, colsample_bytree=0.9, alpha=1.0,
                      min_child_weight=5, verbosity=0, **{"lambda": 1.0})
        for seg, hours in [("valley", VALLEY), ("solar", SOLAR), ("peak", PEAK)]:
            sub = train_df[train_df.hour_business.isin(hours)].sort_values("ds")
            y = sub["y"].clip(lower=-100, upper=sub["y"].quantile(0.995)).values
            n = len(sub)
            split = max(1, int(n * 0.85))  # 末段 15% 作为验证（时间序列不受未来泄漏）
            dtr = xgb.DMatrix(_prep_X(sub.iloc[:split]).values, label=y[:split])
            dval = xgb.DMatrix(_prep_X(sub.iloc[split:]).values, label=y[split:])
            m = xgb.train(params, dtr, num_boost_round=1500,
                          evals=[(dtr, "train"), (dval, "eval")],
                          verbose_eval=False,
                          callbacks=[xgb.callback.EarlyStopping(80, False)])
            self.models[seg] = m

    def predict(self, target_df):
        import xgboost as xgb
        pred = np.zeros(len(target_df))
        for seg, hours in [("valley", VALLEY), ("solar", SOLAR), ("peak", PEAK)]:
            mask = target_df.hour_business.isin(hours).values
            dte = xgb.DMatrix(_prep_X(target_df[mask]).values)
            pred[mask] = self.models[seg].predict(dte)
        return np.clip(pred, -80, None)


# ─────────────────────────── 富特征候选（对齐 cfg05 方法论：rich features + 90d 滚动窗口，隔离模型族效应） ───────────────────────────
RICH_WINDOW_DAYS = 90
RICH_MAX_TRAIN = 5000
RICH_VAL_DAYS = 30
OVERRIDE_ROUNDS = None  # 若设置(int)，覆盖各 rich 模型的 n_estimators/num_boost_round，用于快速对比实验

CFG05_PARAMS = dict(
    boosting_type="gbdt", objective="mae", num_leaves=191,
    min_data_in_leaf=30, learning_rate=0.015, lambda_l1=0.1, lambda_l2=5.0,
    feature_fraction=0.85, bagging_fraction=0.95, bagging_freq=5, n_estimators=2000,
)


class RichGBMBase(BaseAdapter):
    """按 business_day 滚动训练（90d 窗口，<=5000 行，MAE 目标），与 cfg05 同口径；
    仅替换模型族（LightGBM/CatBoost/XGBoost）以隔离模型族效应。"""
    name = "rich_gbm"
    version = "v_rich"
    source_repo = "epf-sota-experiment"
    feat_cols = None
    predict_kind = "default"  # "default" -> model.predict(np_array); "xgb" -> wrap DMatrix

    def build_model(self, train_df, val_df):
        raise NotImplementedError

    def _predict(self, model, X):
        if self.predict_kind == "xgb":
            import xgboost as xgb
            return model.predict(xgb.DMatrix(X))
        return model.predict(X)

    def train_predict_month(self, feat_df, month_start, window_months, month_days):
        t0 = time.time()
        if self.feat_cols is None:
            self.feat_cols = get_rich_feature_cols(feat_df)
        feat_df = feat_df.sort_values("ds")
        preds = {}
        for bd in month_days:
            target_dt = pd.Timestamp(bd)
            hist = feat_df[feat_df["business_day"].astype(str) < bd]
            if len(hist) < 200:
                continue
            train_df = hist[hist["ds"] >= (target_dt - pd.Timedelta(days=RICH_WINDOW_DAYS))]
            if len(train_df) > RICH_MAX_TRAIN:
                train_df = train_df.tail(RICH_MAX_TRAIN)
            if len(train_df) < 100:
                train_df = hist.tail(RICH_MAX_TRAIN)
            if len(train_df) < 100:
                continue
            val_df = hist[hist["ds"] >= (target_dt - pd.Timedelta(days=RICH_VAL_DAYS))]
            if len(val_df) > 2000:
                val_df = val_df.tail(2000)
            try:
                model = self.build_model(train_df, val_df)
            except Exception as e:
                logger.error(f"  [{self.name}] train {bd} failed: {e}")
                continue
            day_df = feat_df[feat_df["business_day"].astype(str) == bd]
            if len(day_df) == 0:
                continue
            try:
                p = self._predict(model, day_df[self.feat_cols].values.astype(float))
                preds[bd] = np.clip(np.asarray(p, float), -80, None)
            except Exception as e:
                logger.error(f"  [{self.name}] predict {bd} failed: {e}")
                preds[bd] = None
        return preds, time.time() - t0


class Cfg05Champion(RichGBMBase):
    """cfg05 冠军基准（rich features + 90d, MAE, num_leaves=191）。"""
    name = "cfg05"
    version = "v_cfg05"
    source_repo = "epf-sota-experiment"

    def build_model(self, train_df, val_df):
        import lightgbm as lgb
        X = train_df[self.feat_cols].values.astype(float)
        y = train_df["y"].values.astype(float)
        p = dict(CFG05_PARAMS); p["verbosity"] = -1; p["metric"] = "mae"
        p.update(_lgbm_device())
        if len(val_df) >= 50:
            Xv = val_df[self.feat_cols].values.astype(float)
            yv = val_df["y"].values.astype(float)
            return lgb.train(p, lgb.Dataset(X, y), num_boost_round=OVERRIDE_ROUNDS or CFG05_PARAMS["n_estimators"],
                             valid_sets=[lgb.Dataset(Xv, yv)], valid_names=["eval"],
                             callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        return lgb.train(p, lgb.Dataset(X, y), num_boost_round=OVERRIDE_ROUNDS or CFG05_PARAMS["n_estimators"],
                         callbacks=[lgb.log_evaluation(0)])


class CatBoostRich(RichGBMBase):
    """CatBoost on rich features + 90d window（与 cfg05 同口径，隔离模型族）。"""
    name = "catboost_rich"
    version = "v_rich"
    source_repo = "epf-sota-experiment"

    def build_model(self, train_df, val_df):
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=OVERRIDE_ROUNDS or 2000, learning_rate=0.03, depth=8,
                               loss_function="MAE", l2_leaf_reg=5.0,
                               early_stopping_rounds=50, verbose=False, random_state=42,
                               **_catboost_device())
        if len(val_df) >= 50:
            m.fit(train_df[self.feat_cols], train_df["y"].values.astype(float),
                  eval_set=(val_df[self.feat_cols], val_df["y"].values.astype(float)))
        else:
            m.fit(train_df[self.feat_cols], train_df["y"].values.astype(float))
        return m


class XGBoostRich(RichGBMBase):
    """XGBoost on rich features + 90d window（与 cfg05 同口径，隔离模型族）。"""
    name = "xgboost_rich"
    version = "v_rich"
    source_repo = "epf-sota-experiment"
    predict_kind = "xgb"

    def build_model(self, train_df, val_df):
        import xgboost as xgb
        p = dict(objective="reg:absoluteerror", max_depth=8, eta=0.03,
                 subsample=0.9, colsample_bytree=0.9, alpha=1.0,
                 min_child_weight=5, verbosity=0, **{"lambda": 1.0})
        X = train_df[self.feat_cols].values.astype(float)
        y = train_df["y"].values.astype(float)
        dtr = xgb.DMatrix(X, label=y)
        if len(val_df) >= 50:
            dv = xgb.DMatrix(val_df[self.feat_cols].values.astype(float),
                             label=val_df["y"].values.astype(float))
            return xgb.train(p, dtr, num_boost_round=OVERRIDE_ROUNDS or 1500,
                             evals=[(dtr, "train"), (dv, "eval")], verbose_eval=False,
                             callbacks=[xgb.callback.EarlyStopping(rounds=50)])
        return xgb.train(p, dtr, num_boost_round=OVERRIDE_ROUNDS or 1500, verbose_eval=False)


def build_adapter(model_id: str):
    if model_id == "baseline_lgbm25":
        return BaselineLGBM25()
    if model_id == "lightgbm_variant":
        return LightGBMVariant()
    if model_id == "catboost":
        return CatBoostCandidate()
    if model_id == "xgboost":
        return XGBoostCandidate()
    if model_id == "cfg05":
        return Cfg05Champion()
    if model_id == "catboost_rich":
        return CatBoostRich()
    if model_id == "xgboost_rich":
        return XGBoostRich()
    raise ValueError(f"unknown model {model_id}")


# ─────────────────────────── 指标 ───────────────────────────
def spike_metrics(y_true, y_pred):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    valid = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[valid], yp[valid]
    if len(yt) == 0:
        return {}
    thr = np.quantile(yt, 0.9)
    spike = yt >= thr
    normal = ~spike
    out = {
        "MAE": mae(yt, yp), "RMSE": rmse(yt, yp),
        "sMAPE_floor50": smape_floor50(yt, yp),
        "spike_MAE": mae(yt[spike], yp[spike]) if spike.sum() else float("nan"),
        "spike_RMSE": rmse(yt[spike], yp[spike]) if spike.sum() else float("nan"),
        "spike_sMAPE_floor50": smape_floor50(yt[spike], yp[spike]) if spike.sum() else float("nan"),
        "normal_MAE": mae(yt[normal], yp[normal]) if normal.sum() else float("nan"),
        "normal_sMAPE_floor50": smape_floor50(yt[normal], yp[normal]) if normal.sum() else float("nan"),
        "normal_degradation_vs_overall_MAE": (mae(yt[normal], yp[normal]) - mae(yt, yp)) if normal.sum() else float("nan"),
    }
    return out


def period_metrics(y_true, y_pred, period_arr):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    res = {}
    for p in ["1_8", "9_16", "17_24"]:
        m = period_arr == p
        if m.sum() == 0:
            continue
        res[f"period_{p}_sMAPE_floor50"] = smape_floor50(yt[m], yp[m])
        res[f"period_{p}_MAE"] = mae(yt[m], yp[m])
    return res


def df_to_md(df):
    """dependency-free markdown table (avoids tabulate)."""
    if len(df) == 0:
        return "(empty)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                v = f"{v:.4g}"
            elif v is None:
                v = ""
            vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


# 模型元信息（版本 + 来源，用于输出 schema 与最终报告）
VERSION_MAP = {"baseline_lgbm25": "v25", "lightgbm_variant": "v_cfg05",
               "catboost": "v1", "xgboost": "v1", "ensemble": "v1",
               "cfg05": "v_cfg05", "catboost_rich": "v_rich",
               "xgboost_rich": "v_rich", "ensemble_rich": "v_rich"}
SOURCE_MAP = {"baseline_lgbm25": "electricity_forecast_model2.5"}
RICH_MODELS = {"cfg05", "catboost_rich", "xgboost_rich"}


def _model_long_df(mid, preds, fdf):
    """把单模型预测组装成长表（含 y_true），用于 checkpoint 与最终汇总。"""
    rows = []
    for bd, p in preds.items():
        g = fdf[fdf["target_day"] == bd].sort_values("ds")
        for i, (_, r) in enumerate(g.iterrows()):
            rows.append({
                "business_day": bd, "ds": r["ds"], "hour_business": int(r["hour_business"]),
                "period": r["period"], "y_true": r["y"], "y_pred": p[i] if i < len(p) else np.nan,
                "model_name": mid,
            })
    df = pd.DataFrame(rows)
    df["run_id"] = None  # 占位，main 中填充
    df["model_version"] = VERSION_MAP.get(mid, "v1")
    df["source_repo"] = SOURCE_MAP.get(mid, "epf-sota-experiment")
    return df[["business_day", "ds", "hour_business", "period", "y_pred",
               "model_name", "model_version", "source_repo", "run_id", "y_true"]]


def _load_preds_from_ckpt(ckpt_csv):
    df = pd.read_csv(ckpt_csv)
    preds = {}
    for bd, g in df.groupby("business_day"):
        preds[bd] = g.sort_values("ds")["y_pred"].values.astype(float)
    return preds


# ─────────────────────────── 汇总与最终化 ───────────────────────────
def _aggregate_and_write(long_df, timing, skip_info, args, out_root):
    """从长表计算 all/period/month/spike 指标并写出 CSV + JSON + 报告。"""
    overall, per_period, per_month = [], [], []
    for mid, gp in long_df.groupby("model_name"):
        yt = gp["y_true"].values.astype(float); yp = gp["y_pred"].values.astype(float)
        valid = ~(np.isnan(yt) | np.isnan(yp))
        m = compute_all_metrics(yt[valid], yp[valid])
        m.update({"model_name": mid, "n_days": int(gp["business_day"].nunique()),
                  "NaN_count": int(np.isnan(yp).sum()),
                  "train_infer_time_s": timing.get(mid, 0.0)})
        overall.append(m)
        per_period.append({"model_name": mid, **period_metrics(yt, yp, gp["period"].values)})
        for mo, g2 in gp.groupby(gp["business_day"].str[:7]):
            y2 = g2["y_true"].values.astype(float); p2 = g2["y_pred"].values.astype(float)
            v2 = ~(np.isnan(y2) | np.isnan(p2))
            per_month.append({"model_name": mid, "month": mo,
                              "sMAPE_floor50": smape_floor50(y2[v2], p2[v2]),
                              "MAE": mae(y2[v2], p2[v2])})
    overall_df = pd.DataFrame(overall)
    per_period_df = pd.DataFrame(per_period)
    per_month_df = pd.DataFrame(per_month)

    spike_rows = []
    for mid, gp in long_df.groupby("model_name"):
        sm = spike_metrics(gp["y_true"].values.astype(float), gp["y_pred"].values.astype(float))
        sm["model_name"] = mid
        spike_rows.append(sm)
    spike_df = pd.DataFrame(spike_rows)

    (out_root / "metrics").mkdir(parents=True, exist_ok=True)
    overall_df.to_csv(out_root / "metrics" / "overall_metrics.csv", index=False, encoding="utf-8-sig")
    per_period_df.to_csv(out_root / "metrics" / "period_metrics.csv", index=False, encoding="utf-8-sig")
    per_month_df.to_csv(out_root / "metrics" / "month_metrics.csv", index=False, encoding="utf-8-sig")
    spike_df.to_csv(out_root / "metrics" / "spike_metrics.csv", index=False, encoding="utf-8-sig")

    metrics_json = {
        "run_id": args.run_id, "test_months": args.test_months_list,
        "overall": overall_df.to_dict(orient="records"),
        "period": per_period_df.to_dict(orient="records"),
        "spike": spike_df.to_dict(orient="records"),
        "timing": timing, "skipped": skip_info,
        "month": per_month_df.to_dict(orient="records"),
    }
    with open(out_root / "metrics" / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, ensure_ascii=False, indent=2, default=str)

    rep = ["# P1 Day-ahead Walk-forward Comparison", "",
           f"run_id={args.run_id}  test_months={args.test_months_list}",
           f"train_window_months={args.train_window_months}", "",
           "## Overall (multi-month)", df_to_md(overall_df),
           "", "## Period sMAPE_floor50", df_to_md(per_period_df),
           "", "## Spike / Normal", df_to_md(spike_df),
           "", "## Skipped", json.dumps(skip_info, ensure_ascii=False)]
    (out_root / "reports").mkdir(parents=True, exist_ok=True)
    (out_root / "reports" / "comparison_report.md").write_text("\n".join(rep), encoding="utf-8")

    logger.info("DONE. overall:\n" + overall_df.to_string())
    if skip_info:
        logger.warning("SKIPPED: " + json.dumps(skip_info, ensure_ascii=False))


def _finalize(output_root, run_id):
    """读取所有 per-model checkpoint CSV，合并成 combined metrics.json + 报告。"""
    out_root = Path(output_root)
    pred_dir = out_root / "predictions"
    ckpts = [p for p in pred_dir.glob("*.csv") if p.name != "all_predictions.csv"]
    if not ckpts:
        logger.error("FINALIZE: no per-model checkpoints found in %s", pred_dir)
        return
    parts = []
    for c in ckpts:
        df = pd.read_csv(c)
        df["run_id"] = run_id
        parts.append(df)
    long_df = pd.concat(parts, ignore_index=True)
    long_df.to_csv(pred_dir / "all_predictions.csv", index=False, encoding="utf-8-sig")

    class _A:
        pass
    _A.test_months_list = sorted(long_df["business_day"].str[:7].unique().tolist())
    _A.train_window_months = "n/a"
    _A.run_id = run_id
    _aggregate_and_write(long_df, {}, {}, _A, out_root)
    logger.info(f"FINALIZE done: {len(ckpts)} models merged -> {out_root/'metrics'/'metrics.json'}")


# ─────────────────────────── 主流程 ───────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--test-months", required=True,
                    help="Comma list of YYYY-MM to evaluate (each full month).")
    ap.add_argument("--models", default="baseline_lgbm25,cfg05,catboost_rich,xgboost_rich,ensemble_rich")
    ap.add_argument("--train-window-months", type=int, default=18)
    ap.add_argument("--rich-window-days", type=int, default=90,
                    help="Rolling window (days) for rich-frame candidates (cfg05/xgboost_rich/...). Overrides module default RICH_WINDOW_DAYS.")
    ap.add_argument("--num-boost-round", type=int, default=None,
                    help="Quick-experiment override: if set, replaces each rich model's n_estimators/num_boost_round (e.g. 600). Default None = model's own config.")
    ap.add_argument("--output-root", default="outputs/p1_dayahead/run_p1")
    ap.add_argument("--run-id", default="p1_auto")
    ap.add_argument("--allow-skip", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Ignore existing per-model checkpoints and retrain everything.")
    ap.add_argument("--finalize", action="store_true",
                    help="Skip training: read all per-model checkpoints and merge into combined metrics.json + report.")
    ap.add_argument("--cpu-only", action="store_true",
                    help="Force CPU for ALL models (disable GPU), e.g. when lightgbm GPU hangs.")
    args = ap.parse_args()

    if args.cpu_only:
        global GPU
        GPU = {"lightgbm": False, "catboost": False, "xgboost": False}
        logger.info("cpu-only mode: GPU disabled by --cpu-only")

    global RICH_WINDOW_DAYS, OVERRIDE_ROUNDS
    RICH_WINDOW_DAYS = args.rich_window_days
    OVERRIDE_ROUNDS = args.num_boost_round
    logger.info(f"[DIAG] OVERRIDE_ROUNDS={OVERRIDE_ROUNDS!r} (from --num-boost-round={args.num_boost_round!r})")

    if args.finalize:
        _finalize(args.output_root, args.run_id)
        return

    if args.data_path is None:
        from src.common.repo_paths import get_data_path
        args.data_path = str(get_data_path())

    out_root = Path(args.output_root)
    for sub in ["predictions", "metrics", "reports", "debug"]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    months = [m.strip() for m in args.test_months.split(",") if m.strip()]
    args.test_months_list = months
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    ensemble_ids = [m for m in model_ids if m in ("ensemble", "ensemble_rich")]
    candidate_ids = [m for m in model_ids if m not in ensemble_ids]

    logger.info(f"data={args.data_path} months={months} models={model_ids}")

    raw = load_data(args.data_path, target="dayahead")
    feat = build_features_25(raw)
    logger.info(f"feature df (24f): {len(feat)} rows")
    feat_rich = build_features_rich(raw)
    logger.info(f"feature df (rich): {len(feat_rich)} rows, {len(get_rich_feature_cols(feat_rich))} features")

    # 模型 -> 特征帧路由（24f 对齐 2.5；rich 对齐 cfg05 方法论）
    def feat_for(mid):
        return feat_rich if mid in RICH_MODELS else feat

    # 每个 business_day 是否 24h 完整且有 actual
    day_groups = feat.groupby("target_day")
    valid_days = {}
    for bd, g in day_groups:
        yv = g["y"].notna().sum()
        if len(g) == 24 and yv == 24:
            valid_days[bd] = g["ds"].min()
    # 按月归集
    month_days = {m: [] for m in months}
    for bd in valid_days:
        if bd[:7] in month_days:
            month_days[bd[:7]].append(bd)
    for m in months:
        logger.info(f"month {m}: {len(month_days[m])} valid days")

    # 运行候选模型（每个模型训练完立即 checkpoint，支持断点续跑）
    all_preds = {}  # model_id -> {business_day: np.array(24)}
    timing = {}
    skip_info = {}
    for mid in candidate_ids:
        ckpt = out_root / "predictions" / f"{mid}.csv"
        if ckpt.exists() and not args.force:
            logger.info(f"\n=== model {mid} === (checkpoint exists, resume)")
            all_preds[mid] = _load_preds_from_ckpt(ckpt)
            timing[mid] = 0.0
            continue
        try:
            adapter = build_adapter(mid)
        except Exception as e:
            msg = f"adapter init failed: {e}"
            logger.warning(f"{mid}: {msg}")
            skip_info[mid] = msg
            if args.allow_skip:
                continue
            else:
                raise
        logger.info(f"\n=== model {mid} ===")
        preds = {}
        t_total = 0.0
        for m in months:
            if not month_days[m]:
                continue
            mstart = pd.Timestamp(m + "-01")
            pm, t = adapter.train_predict_month(feat_for(mid), mstart, args.train_window_months, month_days[m])
            t_total += t
            for bd, p in pm.items():
                if p is not None:
                    preds[bd] = np.asarray(p, float)
        all_preds[mid] = preds
        timing[mid] = round(t_total, 2)
        # 立即 checkpoint（断点续跑，避免长跑中途崩溃丢全部结果）
        mp = _model_long_df(mid, preds, feat_for(mid))
        mp["run_id"] = args.run_id
        mp.to_csv(ckpt, index=False, encoding="utf-8-sig")
        logger.info(f"{mid} done: {len(preds)} days, train+infer {t_total:.1f}s -> checkpoint saved")
        del adapter  # 释放显存/内存，降低长跑 OOM 风险

    # ensemble / ensemble_rich = 候选均值（同样 checkpoint）
    if ensemble_ids and all_preds:
        ens_specs = {
            "ensemble": [mid for mid in ["baseline_lgbm25", "catboost", "xgboost"] if mid in all_preds and all_preds[mid]],
            "ensemble_rich": [mid for mid in ["cfg05", "catboost_rich", "xgboost_rich"] if mid in all_preds and all_preds[mid]],
        }
        for ens_id, src in ens_specs.items():
            if ens_id not in ensemble_ids or not src:
                continue
            ckpt = out_root / "predictions" / f"{ens_id}.csv"
            if ckpt.exists() and not args.force:
                logger.info(f"\n=== {ens_id} === (checkpoint exists, resume)")
                all_preds[ens_id] = _load_preds_from_ckpt(ckpt)
                timing[ens_id] = 0.0
                continue
            common_days = set.intersection(*[set(all_preds[mm].keys()) for mm in src]) if src else set()
            ens = {bd: np.stack([all_preds[mm][bd] for mm in src], axis=0).mean(axis=0) for bd in common_days}
            all_preds[ens_id] = ens
            timing[ens_id] = 0.0
            mp = _model_long_df(ens_id, ens, feat_for(src[0]))
            mp["run_id"] = args.run_id
            mp.to_csv(ckpt, index=False, encoding="utf-8-sig")
            logger.info(f"{ens_id} done: {len(ens)} days (from {src}) -> checkpoint saved")

    # 组装长表 + 计算指标
    rows = []
    for mid, preds in all_preds.items():
        fdf = feat_for(mid)
        for bd, p in preds.items():
            g = fdf[fdf["target_day"] == bd].sort_values("ds")
            for i, (_, r) in enumerate(g.iterrows()):
                rows.append({
                    "business_day": bd, "ds": r["ds"], "hour_business": int(r["hour_business"]),
                    "period": r["period"], "y_true": r["y"], "y_pred": p[i] if i < len(p) else np.nan,
                    "model_name": mid,
                })
    long_df = pd.DataFrame(rows)
    if len(long_df) == 0:
        # 所有模型都失败/被跳过：写空的合规长表，避免 KeyError 崩溃
        logger.error("No predictions produced for any model (all failed/skipped). "
                     "Writing empty outputs; check per-model errors above.")
        empty_cols = ["business_day", "ds", "hour_business", "period", "y_pred",
                      "model_name", "model_version", "source_repo", "run_id", "y_true"]
        long_df = pd.DataFrame(columns=empty_cols)
        long_df["run_id"] = args.run_id
        long_df.to_csv(out_root / "predictions" / "all_predictions.csv", index=False, encoding="utf-8-sig")
        _aggregate_and_write(long_df, timing, skip_info, args, out_root)
        return
    long_df["run_id"] = args.run_id
    long_df["model_version"] = long_df["model_name"].map(VERSION_MAP).fillna("v1")
    long_df["source_repo"] = np.where(long_df["model_name"] == "baseline_lgbm25",
                                      "electricity_forecast_model2.5", "epf-sota-experiment")
    # 统一输出 schema
    long_df = long_df[["business_day", "ds", "hour_business", "period", "y_pred",
                       "model_name", "model_version", "source_repo", "run_id", "y_true"]]
    long_df.to_csv(out_root / "predictions" / "all_predictions.csv", index=False, encoding="utf-8-sig")

    _aggregate_and_write(long_df, timing, skip_info, args, out_root)


if __name__ == "__main__":
    main()
