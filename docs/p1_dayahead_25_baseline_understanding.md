# P1 Day-ahead：2.5 日前链路理解文档（Baseline Understanding）

> 目标：弄清楚 2.5 日前 `lightgbm` 是如何训练/预测/输出/评估的，以及哪些逻辑应复用、哪些可替换。
> 只读来源：`electricity_forecast_model2.5`（本地 `其他资料/electricity_forecast_model2.5`）。**未修改 2.5 仓**。

## 1. 2.5 日前模型入口
- 训练核心：`lightGBM/train_da_fix.py`
  - `LGBMPowerPredictor`：日前主训练类（特征工程 + 动态窗口寻优）。
  - `ThreeStageLGBM`：三段式封装（valley / solar / peak 三段回归 + solar 负电价分类器）。
- 推理：`lightGBM/infer_da_fix.py`（`predict` 同样按 hour 分三段，solar 段做负电价修正）。
- 正式链路（pipeline）：`pipelines/ledger_predict.py → ledger_weight → ledger_fuse → ledger_classifier → final_outputs → postflight`；`pipelines/prediction_ledger.py` 管理预测台账与输出 schema。
- 指标：`fusion/metrics.py` 的 `smape_floor50`（见第 4 节，已与 epf-sota 的 `src/common/metrics.py` 逐字节一致）。

## 2. 数据处理方式（对齐要点）
来源 `LGBMPowerPredictor.load_and_process_data` / `feature_engineering`：
- 读取：`时刻`(→`ds`, GBK)、`日前电价`(→`y`, 目标)、`直调负荷预测值/风电/光伏/联络线受电`(→ load/wind/solar/interconnect，ffill)。
- **业务时间 1 秒偏移法**（与 epf-sota `business_time.py` 一致）：`adjusted = ds - 1s`；`hour = adjusted.dt.hour + 1`（物理 00:00 → 上一业务日 24 点）；`period`= valley(1-8)/solar(9-16)/peak(17-24) ⇔ 1_8/9_16/17_24。
- 特征（日前版 `features_list`）：
  - 时间：`hour, month, day_of_week, is_weekend, hour_sin, hour_cos`
  - 滞后：`lag_price_target`（周一用 `lag_168h` 上周同期，其余 `lag_24h` 24h 前）、`price_rolling_mean_24h`
  - 物理：`load, wind, solar, interconnect, bidding_space=net_load-interconnect, space_ratio, net_load=load-wind-solar, solar_ratio, net_load_sq, wind_ratio, renew_penetration, ramp_load, ramp_solar`
  - 昨日统计：`prev_day_avg/max/min`（按业务日 groupby 后 `.shift(1)`，保证只用 D-1 全天）
- 标签处理：训练时 `y_clipped = y.clip(-100, q995)`；预测值 clip 到 `>=-80`。

## 3. 输出格式
- 2.5 正式交付：`outputs/runs/YYYY-MM-DD/final/submission_ready.csv`，标准列 `business_day, ds, hour_business, period, dayahead_price, realtime_price`。
- epf-sota 预测行统一 schema（沿用）：`business_day, ds, hour_business, period, y_pred, model_name, model_version, source_repo, run_id`；每个 business_day 必须 24 行、hour_business 1..24、y_pred 无 NaN。
- 24 小时完整性 + NaN guard 是硬门槛（2.5 的 `check_target_day_nan_regression` / `verify_*_pipeline` 即做此校验）。

## 4. 指标口径（已核对一致）
`fusion/metrics.py::smape_floor50` 与 `epf-sota/src/common/metrics.py` **完全相同**：
```
true_clip = clip(y_true, min=50); pred_clip = clip(y_pred, min=50)
sMAPE = mean(|pred_clip - true_clip| / ((|true_clip|+|pred_clip|)/2)) * 100
```
→ 跨仓对比口径一致，可直接比较。附加指标 MAE/RMSE 亦一致；尖峰时段取 true 值 top-decile（q90）。

## 5. 可复用模块（直接对齐，不重造轮子）
- **业务时间/period 映射**：epf-sota `src/common/business_time.py` 已实现且与 2.5 一致 → 复用。
- **指标**：`src/common/metrics.py` 与 2.5 `fusion/metrics.py` 一致 → 复用。
- **数据读取**：`src/common/data_loader.py`（`load_data` GBK、列映射、日前/实时目标）已复刻 `train_fa_fix.load_and_process_data` → 复用。
- **特征工程思路**：滞后（24/168h）、滚动均值、昨日统计、net_load/bidding_space 等，epf-sota `feature_builder_dayahead*.py` 已对齐 → 复用/扩展。
- **period/时段三段结构**：2.5 的 valley/solar/peak 直接对应 1_8/9_16/17_24 → 候选模型可做 period-specific。
- **负电价处理**：solar 段分类器 + 修正（阈值 0.7，幅度 -80），可参考用于 spike/负价 robust。

## 6. 不建议改动的模块
- 业务时间映射（1 秒偏移）与 period 划分（1_8/9_16/17_24）——改了就无法与 2.5 对齐。
- `smape_floor50` 口径（floor 50）——改了指标不可比。
- 24 行/天、hour 1..24、无 NaN 的完整性约束。
- 训练只能用“目标日之前可见数据”的边界（walk-forward）——这是 leakage 红线。

## 7. 可替换 / 可扩展模块
- **模型主体**：2.5 单 LightGBM（三段）→ 替换为候选 zoo（CatBoost / XGBoost / LightGBM variant / Chronos / ensemble）。
- **特征集**：可在 2.5 基础上增加 calendar（holiday/month/season）、lag 变体、rolling median、last-day-same-hour 等（需 ablation）。
- **训练窗口**：2.5 动态窗口（12 月步进 6）；候选可用固定 90d（cfg05 已证更优）或自适应。
- **损失/鲁棒**：候选可引入 MAE objective、robust clipping、outlier handling、postprocess smoothing（不抹尖峰）。
- **融合**：2.5 用 Ledger 自适应权重；候选阶段可做 simple stacking / blending。

## 8. Leakage 防护（红线，来自黑名单教训）
- 2.5 特征全部为 D-1 或更早（lag_24h / lag_168h / rolling on shifted y / prev_day 经 shift(1)）→ 天然无泄漏。
- **禁止**：把 `y_true`、D+1 actual、任何目标日未来 actual 或由其派生的特征放入训练。
  - 反例：被黑名单的 `lgbm_spike_residual_1127`(11.27%) 正是因为 y_true 进特征导致 leakage，成绩作废。
- 评估（metrics）只能用训练时可见窗口之外的目标日 actual；walk-forward 严格按 D-1 截止构造特征。

## 9. 与 epf-sota 现状的关系
- epf-sota 已沉淀 cfg05（LightGBM, window=90d, objective=mae）= 11.48% day-ahead sMAPE，优于 2.5 ~12 baseline；模型动物园含多个 LightGBM/CatBoost 变体。
- 本次任务在 epf-sota 基础上：补齐 2.5 单模型 baseline（lightgbm，必要时 timesfm/timemixer）、跨用户指定多月（2025×5 + 2026×6）严格对比、复现/新增候选、ablation、产出 3.0 candidate package。
