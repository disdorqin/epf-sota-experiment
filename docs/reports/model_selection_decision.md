# Model Selection Decision — Phase 2 SOTA Race Conclusion

**Date**: 2026-07-02
**Evaluation Period**: 2026-02-01 → 2026-02-07 (7 days)
**Tasks**: Day-ahead + Realtime

---

## Entering Fusion Candidate Pool

| Priority | Model | Day-ahead sMAPE | Realtime sMAPE | Reason |
|----------|-------|----------------:|---------------:|--------|
| **✅ Selected** | **TabPFN-TS** | **15.36%** | **36.24%** | Best-in-class on both tasks; strong complementarity with CatBoost |
| **✅ Selected** | **CatBoost** | **16.78%** | **38.27%** | Strong gradient-boosting baseline; proven on structured tabular data |

## Paused (Not Entering Fusion at This Stage)

| Model | Day-ahead sMAPE | Realtime sMAPE | Status |
|-------|----------------:|---------------:|--------|
| Chronos-Bolt | 43.22% | 46.31% | **Paused** — sMAPE > 40%, weak zero-shot |
| TiRex | 42.14% | 47.21% | **Paused** — sMAPE > 40%, weak zero-shot |

### Retention Policy
- Chronos-Bolt and TiRex are **NOT deleted** from the codebase.
- They remain available as **weak zero-shot baselines** for ablation studies and future reference.
- Code and configs are kept intact; only inference priority is lowered.

## Rationale
1. **Top-2 gap is large**: The sMAPE gap between TabPFN/CatBoost (~16-17%) and TiRex/Chronos (~42-47%) exceeds 25 percentage points.
2. **Zero-shot methods underperform**: Both Chronos-Bolt and TiRex are zero-shot (no fine-tuning). They do not benefit from the training data and produce weaker results.
3. **No value in weak fusion**: Adding a 40%+ model into fusion would degrade overall performance even with low weights.
4. **Complementarity potential**: TabPFN (pre-trained transformer) and CatBoost (gradient boosting) represent fundamentally different model families, making their ensemble potentially more robust.

## Recommended Next Steps
1. Run **30-day walk-forward** for CatBoost + TabPFN-TS on both tasks.
2. Build fusion prototype with 3 methods: simple average, inverse-sMAPE weight, period-best.
3. Analyze complementarity: error correlation, regime-specific strengths, disagreement patterns.
4. If fusion outperforms both singles → integrate into production fusion pipeline.
