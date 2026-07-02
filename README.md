# SOTA 单模型实验区

**外部独立实验工程**，用于在电力现货价格预测项目中，复现并评估 SOTA 单模型（CatBoost、Chronos）作为原 LightGBM、TimesFM 的候选替换。

## 目录结构

```
models/
├── README.md
├── requirements_sota.txt
├── configs/
│   ├── paths.yaml               # 源仓库和数据路径
│   ├── experiment.yaml          # 实验配置
│   ├── model_catboost.yaml      # CatBoost 超参
│   └── model_chronos.yaml       # Chronos 模型配置
├── src/
│   ├── common/
│   │   ├── data_loader.py       # 数据加载（中文列名兼容）
│   │   ├── business_time.py     # 业务小时规则（1-24点, 00:00→前日24:00）
│   │   ├── feature_builder.py   # 特征工程（复现原LightGBM 21维特征）
│   │   ├── metrics.py           # 评估指标（sMAPE_floor50等）
│   │   ├── output_schema.py     # long table输出格式
│   │   ├── split_utils.py       # 时间序训练/测试切分
│   │   └── repo_paths.py        # 路径解析（pathlib安全处理中文）
│   ├── models/
│   │   ├── catboost_adapter.py  # CatBoostRegressor 适配器
│   │   └── chronos_adapter.py   # Chronos-2/Bolt 零样本适配器
│   ├── experiments/
│   └── reports/
│       └── build_report.py      # 对比报告生成
├── scripts/
│   ├── run_catboost_single_day.py
│   ├── run_chronos_single_day.py
│   ├── run_sota_walkforward.py
│   └── compare_sota_vs_original.py
├── outputs/
│   ├── debug/
│   │   └── source_repo_scan.json
│   └── .gitkeep
└── tests/
```

## 安装依赖

```bash
# 推荐在虚拟环境中安装
python -m venv sota_env
sota_env\Scripts\activate

# 核心依赖
pip install -r requirements_sota.txt

# 如果 Chronos-2 不可用，安装 Chronos-Bolt 作为 fallback
pip install chronos-bolt
```

## 使用方式

### CatBoost 单日预测

```bash
python scripts/run_catboost_single_day.py ^
    --target-date 2026-02-15 ^
    --task dayahead
```

### Chronos 单日预测

```bash
python scripts/run_chronos_single_day.py ^
    --target-date 2026-02-15 ^
    --task dayahead
```

### Walk-forward 评估

```bash
python scripts/run_sota_walkforward.py ^
    --start 2026-02-01 ^
    --end 2026-02-03 ^
    --target both ^
    --models catboost_sota,chronos2_zero_shot ^
    --output-root outputs/sota_walkforward
```

### 生成对比报告

```bash
python scripts/compare_sota_vs_original.py ^
    --walkforward-dir outputs/sota_walkforward ^
    --start 2026-02-01 ^
    --end 2026-02-03 ^
    --models catboost_sota,chronos2_zero_shot
```

## 输出说明

- `outputs/sota_walkforward/predictions/` — 每日预测 CSV
- `outputs/sota_walkforward/metrics/` — 指标汇总
- `outputs/sota_walkforward/debug/` — 运行日志、配置
- `outputs/sota_walkforward/reports/sota_comparison_report.md` — 对比报告

## 关键设计

1. **不修改原仓库** — 所有代码在独立目录，不污染主线
2. **业务小时规则** — hour_business=24 对应下一自然日 00:00
3. **sMAPE_floor50** — 严格复现原仓库公式
4. **Chronos fallback** — Chronos-2 失败自动降级到 Chronos-Bolt
5. **CatBoost 类别特征** — hour_business/period/day_of_week/month/is_weekend
