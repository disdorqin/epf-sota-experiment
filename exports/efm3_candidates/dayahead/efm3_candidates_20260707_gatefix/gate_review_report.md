# P1.1 Dayahead Gate Fix Report

Generated: 2026-07-07T13:11:21

Scope: fix the P1 candidate-package gate gaps + same-window champion retest. No new model search.


## §1 Candidate Package Gate (6 fixes)

| # | Gate | Status | Note |
|---|---|---|---|
| ① | Gating files present (metrics/manifest/promotion) | PASS | regenerated in this package |
| ② | Report naming/paths follow 3.0 contract | PASS | FINAL_REPORT→gate_review_report; predictions/metrics/manifest/promotion/comparison/ablation/config_snapshot all present |
| ③ | Same-window retest of 2.5 trusted champion | PASS | best_two_average reproduced = 15.04% on four hard months |
| ④ | Unified comparison window = four hard months | PASS | 2025-11/12/2026-01/02 for every model; easy-window excluded |
| ⑤ | CPU-only hardened | PASS | engine default now CPU; --gpu to opt-in; daemon gpu_disabled=true |
| ⑥ | Negative/spike/period review | PASS | see §7 |

## §2 Same-window Baseline Table (2025-11~2026-02)

| Baseline | sMAPE_floor50 (%) | Usable? | Note |
|---|---|---|---|
| faithful 2.5 ThreeStageLGBM | 21.87 | YES (reference) | same four hard months (established P1 value) |
| faithful 2.5 engine baseline_lgbm25 (this run) | 22.84 | YES (proxy) | same four hard months; lighter single-stage faithful proxy, 41.8s |
| **trusted champion best_two_average (reproduced)** | **15.04** | **YES (baseline)** | same four hard months, n=2760 |
| old cfg05 prior 11.48 | 11.48 | NO | different/easier window, not reproduced on four hard months |
| lgbm_spike_residual 11.27 | 11.27 | NO | INVALIDATED (data leakage) |

## §3 Candidate Metrics Table

| Model | Overall | 1_8 | 9_16 | 17_24 | Spike | Normal | neg_hit(%) |
|---|---|---|---|---|---|---|---|
| cfg05 | 14.68 | 13.91 | 16.01 | 14.12 | 13.51 | 14.81 | 72.39 |
| cfg05_180d | 14.25 | 13.97 | 15.33 | 13.45 | 13.64 | 14.32 | 77.11 |
| xgboost_rich | 14.70 | 13.29 | 16.62 | 14.19 | 12.93 | 14.90 | 71.89 |
| ensemble_rich | 14.54 | 13.44 | 16.14 | 14.05 | 13.15 | 14.70 | 72.39 |
| baseline_lgbm25 (faithful proxy, this run) | 22.84 | 26.44 | 22.92 | 19.16 | - | - | - |
| **trusted champion (same-window)** | **15.04** | 14.75 | 16.20 | 14.05 | - | - | 71.97 |

> Note: faithful 2.5 reference in §2 = established ThreeStageLGBM 21.87% (four hard months). Engine baseline_lgbm25 re-run here = 22.84% (lighter single-stage faithful proxy, 41.8s). Both confirm rich >> faithful; the 1pp gap is immaterial to the conclusion.

## §4 CPU-only Reproducibility

| Item | Value |
|---|---|
| GPU disabled | TRUE (daemon gpu_disabled=true; engine --cpu-only) |
| Engine GPU default | FIXED: default CPU; `--gpu` opt-in only (was GPU-preferred → hazard) |
| Training time (cfg05 90d) | 505.3s |
| Training time (cfg05 180d) | 1841.4s |
| Training time (xgboost_rich) | 1140.0s |
| Training time (ensemble_rich) | 0.0s |
| Training time (baseline_lgbm25) | 41.8s |
| Inference | walk-forward, D-1 only features, no cross-month leakage |
| Daemon status | cpu-only, watchdog, GPU_DISABLED fallback |

## §5 Promotion Decision

**P1_1_RECOMMENDATION: SHADOW**

| Model | Decision | beats trusted champ? |
|---|---|---|
| cfg05 | shadow | YES |
| cfg05_180d | shadow | YES |
| xgboost_rich | shadow | YES |
| ensemble_rich | shadow | YES |

Lead shadow model: **cfg05** (sMAPE=14.68%, beats same-window trusted champion 15.04%).

## §6 Final Verdict

**P1_1_RESULT: PASS**

- All 6 gate fixes applied.
- Same-window validation: cfg05 beats trusted champion on four hard months.
- No champion promotion (forbidden); shadow only.


## §7 Negative-price / Spike / Period Review

| Model | neg_hit(%) | spike_sMAPE | normal_sMAPE | 17_24 vs tc |
|---|---|---|---|---|
| cfg05 | 72.39 | 13.51 | 14.81 | +0.07 |
| cfg05_180d | 77.11 | 13.64 | 14.32 | -0.60 |
| xgboost_rich | 71.89 | 12.93 | 14.90 | +0.14 |
| ensemble_rich | 72.39 | 13.15 | 14.70 | +0.00 |

Negative-price hit-rate ~72%% (comparable to faithful 2.5). Spike error lower than overall (rich features help spikes). 17_24 within +0.5pp of trusted champion → period not worsened.

## §8 Window Unification Confirmation

Every model in §3 is evaluated on the identical window: **2025-11, 2025-12, 2026-01, 2026-02** (n=120 days). No easy-window single-month number is compared against four-hard-month numbers.

## §9 3.0 Contract Compliance (naming/paths)

| Expected file | Present |
|---|---|
| predictions.csv | YES |
| metrics.json | YES |
| manifest.json | YES |
| promotion_decision.json | YES |
| comparison_report.md | YES |
| ablation_report.md | YES |
| config_snapshot.yaml | YES |
| gate_review_report.md | YES (this report) |

## §10 Honest Comparison Statement

- cfg05 (rich, 90d) = 14.68% on four hard months **honestly beats** faithful 2.5 ThreeStageLGBM = 21.87% on the SAME four hard months → rich feature frame is confirmed better.
- cfg05 = 14.68% **beats the same-window trusted champion** best_two_average = 15.04% (reproduced on the same four hard months). The earlier 11.85% figure was on an easier single month (Feb1–Mar2) and is NOT comparable.
- We do NOT claim cfg05 replaces the 3.0 production dayahead model. It is promoted only to **shadow** (forbidden: champion).
- lgbm_spike_residual 11.27%% is INVALIDATED (leakage) and old cfg05 11.48%% is not reproduced on four hard months — neither is used as a baseline.
