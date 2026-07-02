# SOTA 模型 7 天赛马结果报告

**评估周期**: 2026-02-01 → 2026-02-07 (7天)
**数据口径**: 山东 PMOS 小时级电价
**评估任务**: Day-ahead (日前) + Realtime (实时)
**输出维度**: 每模型每任务 168 行 (24 小时 × 7 天)

---

## 1. CatBoost 7天结果

| 指标 | Day-ahead | Realtime |
|------|-----------|----------|
| avg_MAE | 43.76 | 104.74 |
| avg_RMSE | 53.43 | 140.47 |
| **avg_sMAPE** | **16.78%** | **38.27%** |
| avg_peak_MAE_q90 | 58.23 | 114.23 |
| avg_negative_hit_rate | 68.75% | 15.08% |

> 传统 gradient boosting 方法的强基线，dayahead 精度在所有模型中排名第二，realtime 排名第二。

---

## 2. TabPFN 7天结果

| 指标 | Day-ahead | Realtime |
|------|-----------|----------|
| avg_MAE | 40.25 | 100.00 |
| avg_RMSE | 48.60 | 131.48 |
| **avg_sMAPE** | **15.36%** | **36.24%** |
| avg_peak_MAE_q90 | 55.36 | 105.69 |
| avg_negative_hit_rate | 62.50% | 18.57% |

> TabPFN-TS (3个月训练窗口, 2208行/天, 10K max-train-rows) 表现极为出色。
> 训练耗时: ~2min 45s/天 (CPU), 单任务 7 天共 ~19分钟。

---

## 3. Chronos-Bolt 7天结果

| 指标 | Day-ahead | Realtime |
|------|-----------|----------|
| avg_MAE | 106.07 | 134.99 |
| avg_RMSE | 132.54 | 177.44 |
| **avg_sMAPE** | **43.22%** | **46.31%** |
| avg_peak_MAE_q90 | 101.25 | 107.10 |
| avg_negative_hit_rate | 25.00% | 4.17% |

> Chronos-Bolt 作为零样本方法表现较差，sMAPE 明显高于基于训练的方法。
> 零样本泛化能力在该电价数据集上不理想。

---

## 4. TiRex 7天结果

| 指标 | Day-ahead | Realtime |
|------|-----------|----------|
| avg_MAE | 101.71 | 138.70 |
| avg_RMSE | 128.33 | 182.94 |
| **avg_sMAPE** | **42.14%** | **47.21%** |
| avg_peak_MAE_q90 | 72.86 | 105.31 |
| avg_negative_hit_rate | 0.00% | 0.00% |

> TiRex 零样本预测在 day-ahead 和 realtime 上均表现不佳，与 Chronos-Bolt 水平相当。
> GPU 推理速度极快 (1分钟完成 14 个预测日)。
> quantile 输出已正确解析为 (horizon, num_quantiles) 格式 ✅

---

## 5. TabPFN vs Chronos-Bolt 同口径对比

| 对比维度 | TabPFN | Chronos-Bolt | TabPFN 优势 |
|----------|--------|--------------|-------------|
| Day-ahead sMAPE | **15.36%** | 43.22% | **↓64.5%** |
| Day-ahead MAE | **40.25** | 106.07 | **↓62.1%** |
| Realtime sMAPE | **36.24%** | 46.31% | **↓21.7%** |
| Realtime MAE | **100.00** | 134.99 | **↓25.9%** |

**结论**: TabPFN 在同一 7 天口径下显著优于 Chronos-Bolt:
- Day-ahead sMAPE 低 64.5%
- Realtime sMAPE 低 21.7%
- 两任务四项指标全胜，优势极为明显。

---

## 6. 全模型排名 (sMAPE)

| 排名 | 模型 | Day-ahead sMAPE | Realtime sMAPE | 加权平均 |
|------|------|-----------------|----------------|----------|
| 🥇 | **TabPFN-TS** | **15.36%** | **36.24%** | **25.80%** |
| 🥈 | **CatBoost** | **16.78%** | **38.27%** | **27.52%** |
| 🥉 | TiRex | 42.14% | 47.21% | 44.68% |
| 4 | Chronos-Bolt | 43.22% | 46.31% | 44.77% |

---

## 7. 融合候选池推荐

基于 7 天赛马结果，推荐以下模型进入融合候选池:

| 优先级 | 模型 | 理由 |
|--------|------|------|
| **✅ 必选** | **TabPFN-TS** | 双任务 sMAPE 最低，dayahead 仅 15.36%，realtime 36.24%，与 CatBoost 互补性强 |
| **✅ 必选** | **CatBoost** | 传统方法强基线，16.78% dayahead sMAPE，与 TabPFN 形成树模型 + 预训练时序模型的双引擎 |
| ❌ 暂不推荐 | Chronos-Bolt | sMAPE 43-46%，远高于 TabPFN，不适合融合 |
| ❌ 暂不推荐 | TiRex | sMAPE 42-47%，与 Chronos-Bolt 同档，不适合融合 |

**推荐融合策略**: TabPFN + CatBoost 双模型加权融合，待积累 30 天 ledger 后学习动态融合权重。

---

## 8. 修复验证状态

| 检查项 | 状态 | 说明 |
|--------|------|------|
| Smoke check | ✅ 25/25 | 全部通过 |
| TiRex quantile 解析 | ✅ | (24,9) → horizon×quantiles, p50=q_np[:,4] |
| TabPFN 跨 task 隔离 | ✅ | realtime y_true 正确来自实时电价 |
| TabPFN feature_df_task | ✅ | dayahead=39168行target=dayahead, realtime=39168行target=realtime |
