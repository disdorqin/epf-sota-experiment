# Day-Ahead Model Selection — Corrected Report

> 生成时间: 2026-07-03
> 数据窗口: 2026-02-01 ~ 2026-03-02 (30天)
> 口径: 所有对比基于 **30 天 sMAPE_floor50**

---

## 1. 口径修正说明

**此前报告存在以下口径错误：**

| 错误 | 正确 |
|---|---|
| `catboost_dayahead_tuned 13.89%` 与 `catboost_sota 7天 16.78%` 对比 | 必须与 **30 天 `catboost_sota 12.58%`** 对比 |
| "tuned 优于 baseline" 结论 | tuned **未优于** 30 天 CatBoost baseline |

**⚠️ 所有报告中的 7 天 baseline 引用已全部修正为 30 天 baseline。**

---

## 2. 最终排名 (30天 day-ahead)

| 排名 | 模型 | sMAPE | vs CatBoost | 超越? |
|---|---|---|---|---|
| 🥇 | **catboost_sota** | **12.58%** | — | — |
| 2 | fused_winner_by_hour | 12.91% | +0.33pp | ❌ |
| 3 | fused_inverse_smape_period | 12.94% | +0.36pp | ❌ |
| 4 | fused_simple_average | 13.01% | +0.43pp | ❌ |
| 5 | fused_winner_by_period | 13.10% | +0.52pp | ❌ |
| 6 | tabpfn_ts_sota | 13.64% | +1.06pp | ❌ |
| 7 | catboost_dayahead_tuned | 13.89% | +1.31pp | ❌ |
| 8 | catboost_period_specialist | 14.62% | +2.04pp | ❌ |
| 9 | ridge_stacking_fusion | 14.76% | +2.19pp | ❌ |

**结论：没有任何模型或融合方法超过 `catboost_sota` 的 12.58%。**

---

## 3. 各模型结论

### 3.1 TabPFN-ts-sota
- sMAPE: 13.64%
- **未超过 CatBoost** (+1.06pp)
- 在 30 天尺度上 TabPFN 落后于 CatBoost

### 3.2 catboost_dayahead_tuned (Optuna)
- sMAPE: 13.89%
- **未超过 CatBoost baseline** (+1.31pp)
- Optuna 自动调参反而比默认参数更差
- 原因推测：默认参数经过人工验证，Optuna 的 30 trials 不够充分

### 3.3 catboost_period_specialist
- sMAPE: 14.62%
- **未超过 CatBoost** (+2.04pp)
- 分 period 训练未带来增益，反而因训练数据变少而退化

### 3.4 融合方法
- 最佳融合: winner_by_hour (12.91%)
- **所有融合均未超过 CatBoost 单模型**
- ridge stacking 最差 (14.76%，+2.19pp)
- 普通融合方向已到瓶颈

---

## 4. 目标达成情况

| 目标 | 当前最接近 | 差距 |
|---|---|---|
| sMAPE < 12% | catboost_sota 12.58% | ❌ +0.58pp |
| sMAPE < 10% | catboost_sota 12.58% | ❌ +2.58pp |
| sMAPE < 8% | catboost_sota 12.58% | ❌ +4.58pp |

---

## 5. 最终建议

**下一阶段不再扩大普通模型赛马，而是围绕 `catboost_sota` 进行 residual/spike correction：**

1. **selected-hour residual correction** — 针对 hour 17 (21.49%) / hour 5 (19.80%) / hour 10 (19.31%) 等高误差小时做残差修正
2. **spike correction** — 降低高价位 spike 误差 (peak_MAE_q90 = 55.60)
3. **春节窗口修正** — 处理春节前后极端模式
4. **hour 11/12/13 专项修正** — 这些小时已接近 8%，可进一步压分

> 当前最优模型 `catboost_sota` 12.58% 距 8% 目标还有 4.58pp，
> 需要 residual/spike 层面的定向优化，而非模型堆叠。
