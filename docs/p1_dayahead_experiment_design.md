# P1 Day-ahead 实验设计文档（Phase B — 统一 walk-forward 框架）

> 本文件定义 P1 日前模型开放探索的**统一评估协议**，保证所有候选模型在「同一数据、同一口径、同一指标」下可比。所有结论均可由 `scripts/run_dayahead_p1_walkforward.py` 复现。

## 1. 目标与对照
- **目标**：在 2.5 稳定工程经验基础上，系统复现/比较/筛选更强的日前电价预测候选模型。
- **对照双列（诚实口径，不混淆）**：
  1. `baseline_lgbm25` — 忠实复现 2.5 日前 ThreeStageLGBM（24 特征，三段 + 光伏负价分类）。本框架在 2026-02 = **17.51%**。
  2. `cfg05` — prior 冻结的 trusted champion（rich 特征 + 90d 窗口, MAE 目标）。本框架复现 = **11.67%**（2026-02）。
- 候选模型若显著优于 cfg05（或在 cfg05 不可用时优于 baseline_lgbm25），进入 Phase E 调优并晋级 Phase F candidate package。

## 2. 评估协议（walk-forward）
- **滚动训练**：对每个待预测 business_day `bd`，训练集 = `{d : d.ds < bd, d 落在 [bd-窗口, bd) }`。
  - 24f 帧：`train_window_months`（默认 18）按月回看。
  - rich 帧：`RICH_WINDOW_DAYS=90` 天，最多 `RICH_MAX_TRAIN=5000` 行（兜底取最近 5000）。
- **验证集**：训练集末尾 `RICH_VAL_DAYS`（默认 ~14 天）用于 early-stopping；不足则无早停。
- **预测**：仅用 `bd` 当日业务特征（D-1 截止，无未来信息）。
- **每月独立重训**，绝不跨月泄漏。

## 3. 特征帧（双轨路由）
| 帧 | 候选模型 | 特征来源 | 列数 | 窗口 |
|---|---|---|---|---|
| **24f** | baseline_lgbm25, lightgbm_variant, catboost, xgboost, ensemble | `build_features_25`（对齐 2.5 特征工程，无泄漏 lags/shifts） | ~24 | 18 月 |
| **rich** | cfg05, catboost_rich, xgboost_rich, ensemble_rich | `build_features_rich`（feature_builder + _v3：lag/同小时/momentum/calendar/volatility/交互，~40+ 列） | ~40+ | 90 天 |

- rich 帧与 cfg05 同口径，目的是**隔离「模型族」效应**：LightGBM/CatBoost/XGBoost 在完全相同特征+窗口下对比，差异只来自模型族。

## 4. 模型注册表（本引擎）
| model_id | 类 | 族 | 帧 | 备注 |
|---|---|---|---|---|
| baseline_lgbm25 | BaselineLGBM25 | LightGBM | 24f | 忠实 2.5 三段 + 光伏负价分类(阈值0.7, -80 校正) |
| lightgbm_variant | LightGBMVariant | LightGBM | 24f | cfg05 风格(regression_l1, ~90d) |
| catboost | CatBoostCandidate | CatBoost | 24f | 三段 GBM |
| xgboost | XGBoostCandidate | XGBoost | 24f | 三段 GBM |
| cfg05 | Cfg05Champion | LightGBM | rich | champion 基准 |
| catboost_rich | CatBoostRich | CatBoost | rich | 同 cfg05 口径 |
| xgboost_rich | XGBoostRich | XGBoost | rich | 同 cfg05 口径（`predict_kind="xgb"` 包装 DMatrix） |
| ensemble | — | 均值 | 24f | baseline_lgbm25/catboost/xgboost 逐日均值 |
| ensemble_rich | — | 均值 | rich | cfg05/catboost_rich/xgboost_rich 逐日均值 |

## 5. 指标（强制 floor50 口径）
- 主指标 `sMAPE_floor50`：true/pred 均 floor 到 50 后算 sMAPE×100（与 2.5 `fusion/metrics.py` 完全一致）。
- 辅助：MAE, RMSE, peak_MAE_q90（true top-decile 小时 MAE）, negative_price_hit_rate。
- 分段：`period_1_8 / period_9_16 / period_17_24` 的 sMAPE_floor50 与 MAE。
- 稳健性：逐月 sMAPE_floor50、spike/normal 子集误差、NaN 计数、缺失天数、训练+推理耗时。
- **硬门槛**：24 行/天完整、NaN=0、无 target leakage。

## 6. 泄漏防护（红线）
- 特征仅用 D-1 及之前可见量；`y_true` / D+1 日前·实时 actual 严禁进特征。
- business_day 映射：`hour_business=(ds-1s).hour+1`，物理 00:00 → 上一业务日 hour 24。
- 训练集严格 `ds < bd`；验证/预测只取 `bd` 当日。

## 7. 测试月份与数据可用性
- 计划：2025(01,03,06,09,12) + 2026(01–06)。
- 数据截止 2026-06-09 → 2026-06 仅可测 6/1–6/8（部分月，需标注「部分」）。
- 某月数据不足（<200 训练行 / 无完整 24h 日）自动跳过并在报告 `skipped` 字段说明，**不编造**。

## 8. 复现命令
```bash
PY="D:/computer_download/environment/conda/epf-2/python.exe"
export HF_ENDPOINT="https://hf-mirror.com"
"$PY" scripts/run_dayahead_p1_walkforward.py \
  --test-months 2026-02 \
  --models baseline_lgbm25,cfg05,catboost_rich,xgboost_rich,ensemble_rich \
  --train-window-months 18 \
  --output-root outputs/p1_dayahead/run_rich_pilot2 \
  --run-id p1_rich_pilot2 --allow-skip
```
输出：`predictions/all_predictions.csv`, `metrics/{overall,period,month,spike}_metrics.csv` + `metrics.json`, `reports/comparison_report.md`。

## 9. 已知限制
- Chronos（zero-shot 基础模型）未安装 → 标记 SKIPPED 候选，待后装再补；不影响「模型族 + 树模型」主结论。
- timesfm / timemixer（2.5 另两个日前模型）需各自依赖，本 P1 以 2.5 lightgbm 忠实复现为 baseline，另两个作为后续可选对照。
- 大文件不入库；`.gitignore` 已覆盖 outputs/。
