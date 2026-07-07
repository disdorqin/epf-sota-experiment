# P1 Day-ahead 经验与复盘文档（Experience & Retrospective）

> 用途：跨对话压缩存活的“技术介绍 + 复盘”文档。每次重要进展/转折都在此追加，不要在最终报告里重复底层细节，这里保留可复用的经验。

## 0. 任务定位
在 2.5 稳定工程经验（7 模型 Ledger 自适应融合 + 极端价分类校正）基础上，系统探索、复现、比较、筛选更强的**日前电价预测**候选模型，目标优于 2.5 日前基线（sMAPE_floor50 ≈ 12）。工作集中在 `epf-sota-experiment`（本地 `models`），最终合格 candidate 以 package 形式沉淀到 3.0。

## 1. 业务与口径（必须对齐 2.5）
- 预测目标：D+1 **日前电价**（`日前电价`）。允许用 D 日完整日前电价；**禁止** D+1 日前/实时 actual，禁止任何目标日未来 actual 派生特征。
- `hour_business` = 1..24；物理 00:00 → 上一业务日 hour 24（即 `(ds-1s).hour+1`）。
- `period`：`1_8` / `9_16` / `17_24`。
- 主指标 `sMAPE_floor50`：true/pred 均 floor 到 50，再算 sMAPE×100。
- 交付 schema：`business_day, ds, hour_business, period, dayahead_price, realtime_price`（当前 P1 只预测日前，realtime 可留空/占位）。
- 尖峰时段：取 true 值 top-decile 小时（q90）作为 spike hours 子集。
- 24 小时完整性、无 NaN、无 target leakage 是硬门槛。

## 2. 环境（重要，避免重复踩坑）
- conda `epf-2`（`D:/computer_download/environment/conda/epf-2`，Py 3.11.14）。运行：`"<env>/python.exe" script.py`。
- 已装：lightgbm 4.6 / catboost 1.2.10 / xgboost 3.2 / torch 2.5.1。**未装 chronos_forecasting**（安装重，先记录，后尝试；Chronos-Bolt-Small 可用，Chronos-2 gated）。
- 数据 CSV 是 **GBK 编码**，`data_loader.load_data` 已处理；直接用 `encoding='gbk'` 读。
- 路径含中文/空格：脚本用 `os.path.abspath(__file__)`，不用 `Path.resolve()`（Git Bash 下 segfault）。
- 脚本调用优先 `--data-path`，其次 `configs/paths.yaml`，当前指向 `electricity_forecast_model2.0_exp/data/...csv`。

## 3. 已验证结论（续跑起点 + 2026-07-06 校正）
- **cfg05（LightGBM, window=90d, objective=mae）= 11.48%**（prior 30天）。**2026-07-06 本框架复现 = 11.67%（2026-02, 28天）** → 框架接线正确，可作为 trusted champion / 标杆。
- **重要校正**：用户引用的“2.5 基线 sMAPE_floor50 ≈ 12”实际指的是 cfg05 量级，**并非忠实复现的 2.5 ThreeStageLGBM**。本框架忠实复现 2.5 日前 ThreeStageLGBM（24 特征, 三段 + 光伏负价分类）在 2026-02 = **17.51%**（MAE 51.85）。在 2025–2026 硬月份上整体约 21.87%（11个月）。
  - 含义：若以“忠实 2.5 基线”为对照，cfg05(11.67%) 已大幅优于它；若以“用户口中的 ~12”为对照，则 cfg05 即该标杆本身。
  - **诚实口径**：Phase D 同时报告「忠实 2.5 基线 (baseline_lgbm25, 17.51%)」与「cfg05 冠军 (11.67%)」两列，不混淆、不编造。
- 模型动物园有效模型（prior）：cfg05(11.48)/best_two_average(11.85)/stage3_business_fixed(11.86)/catboost_spike_residual(12.47)/catboost_sota(12.58)。
- 黑名单（教训）：lgbm_spike_residual_1127(11.27%) 因 **y_true 进特征导致 leakage**；stage3_old(11.64%) 自然日映射错；lightgbm_90d_orig(11.97%) 评估不完整(690 rows, 缺 hour24)。
- → **永远不要把 y_true / D+1 actual 放进特征**；必须 24 行/天；business_day 映射要正确。

## 4. 关键技术决策记录（随进展追加）
- **统一 walk-forward 引擎**：`scripts/run_dayahead_p1_walkforward.py`（2026-07-06 新建并修复）。
  - 双特征帧路由：`RICH_MODELS={cfg05, catboost_rich, xgboost_rich}` 用 rich 特征（feature_builder + _v3, ~40+ 列, 90d 窗口）；其余（baseline_lgbm25/lightgbm_variant/catboost/xgboost）用 24 特征帧（对齐 2.5）。
  - 模型族：`BaselineLGBM25`(忠实 2.5 三段) / `LightGBMVariant`(cfg05 风格 regression_l1) / `CatBoostCandidate` / `XGBoostCandidate`（24f 候选）/ `Cfg05Champion` / `CatBoostRich` / `XGBoostRich`（rich 候选，隔离模型族）/ `ensemble`(24f 均值) / `ensemble_rich`(rich 均值)。
  - `RichGBMBase.train_predict_month` 按 business_day 滚动；`predict_kind` 区分 default(xgb 需 DMatrix 包装)。
  - 指标：overall / period(1_8/9_16/17_24) / month / spike / normal + NaN / 缺失天数 / 时序。无 tabulate 依赖（自带 `df_to_md`）。
  - CLI：`--test-months`, `--models`, `--train-window-months`, `--output-root`, `--run-id`, `--allow-skip`。
- **2026-02 单月 pilot（run_rich_pilot2, 进行中）**：预期 baseline_lgbm25≈17.5, cfg05≈11.7；待 catboost_rich/xgboost_rich/ensemble_rich 结果。
- (待填) 各候选跨多月结果（11 个月全量 sweep）
- (待填) ablation 结论
- (待填) 3.0 candidate package 与 promotion 决策

## 5. 坑位速查
- sMAPE 必须 floor50 才与 2.5 口径一致；裸 sMAPE 不能用于对比。
- 评估只能用“训练时可见”的数据：walk-forward 严格按 D-1 截止构造特征。
- 大文件（data/models/outputs）禁止提交 Git；`.gitignore` 已覆盖。
- 2.5 仓只读；epf-sota-experiment 可写，但 3.0 正式链路不碰。
