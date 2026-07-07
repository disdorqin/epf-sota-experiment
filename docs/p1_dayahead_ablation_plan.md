# P1 Day-ahead Ablation 计划（Phase E）

> 目的：在 Phase C/D 找到的有效候选（cfg05 / xgboost_rich / catboost_rich）上做受控消融，定位「什么在起作用」，并产出可晋级 3.0 的最终 candidate 配置。
> 所有 ablation 复用 `scripts/run_dayahead_p1_walkforward.py`，通过新增 `--models` 变体或临时参数实现；结果写入 `outputs/p1_dayahead/ablation_<name>/`。

## 1. 已确认的宏观结论（来自 2026-02 pilot）
- **特征丰富度 > 模型族**：rich 帧(90d, ~40+特征) 下三族聚在 11.7–13%；24f 帧下忠实 2.5 基线 17.51%。→ 特征工程是主增益来源。
- 族内排序（rich 帧, 2026-02）：cfg05(11.67) < ensemble_rich(11.74) < xgboost_rich(12.03) < catboost_rich(13.01) < baseline(17.51)。

## 2. Ablation 维度（按性价比排序）
1. **窗口长度**（rich 帧，cfg05 配置）：90d vs 60d vs 120d vs 180d。验证 90d 是否最优。
2. **目标函数**（rich 帧，LightGBM）：MAE vs MSE vs Huber（delta=1）vs Quantile(0.5)。
3. **模型族稳健性**（rich 帧，90d，MAE）：LightGBM vs CatBoost vs XGBoost 全 11 月复跑，确认 xgboost_rich 是否稳定次优。
4. **负价处理**：cfg05 是否需补光伏负价分类（参考 baseline 的 0.7 阈值 -80 校正）；rich 候选当前无负价分类，看尖峰/负价段误差。
5. **集成策略**：cfg05 + xgboost_rich 等权均值 vs 按 period 加权 vs Ledger 动态权重（2.5 既有机制）。

## 3. 执行策略（省算力的关键）
- 先用 **3 个代表月**（2026-02 硬月、2025-09 过渡月、2026-05 夏月）做全维度 ablation，快速锁定最优配置。
- 对 ablation 胜出配置，再跑 **全 11 月** 确认泛化，作为晋级候选。
- 不重复跑已知结论（特征帧差异已定）。

## 4. 晋级 3.0 的门槛（promotion gate）
- 全 11 月 sMAPE_floor50 显著 ≤ cfg05 冠军（或在 cfg05 不可用时 ≤ 忠实 2.5 基线）。
- 无 NaN、无泄漏、24 行/天完整、无「仅好月份」偏差。
- 尖峰段(peak_q90)与负价段误差可接受（不显著劣于 cfg05）。
- 训练+推理耗时在 3.0 服务预算内（rich 候选逐日重训，单月 < 10 min 级别可接受）。

## 5. 预期产物
- `outputs/p1_dayahead/ablation_*/metrics/metrics.json` 每组。
- `docs/p1_dayahead_ablation_report.md`：逐维度结论 + 最终推荐配置。
- 晋级候选清单 → Phase F `exports/efm3_candidates/dayahead/<run_id>/`。
