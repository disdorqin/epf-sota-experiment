# Project Memory — SOTA Experiment Zone

## 关键决策记录

### 环境配置
- **conda 环境**: `epf-2` at `D:\computer_download\environment\conda\epf-2`
- **Python**: 3.11.14
- **运行方式**: `HF_ENDPOINT="https://huggingface.co" conda run -n epf-2 python <script>`
- **GitHub**: `https://github.com/disdorqin/epf-sota-experiment` (public)

### CatBoost 适配器
- 使用 `CatBoostRegressor` 直接 fit/predict（避免 Pool 对象在 Windows 上的 segfault）
- 类别特征通过 `cat_features` 参数和 `astype(str)` 转换
- 默认参数：depth=8, lr=0.03, iterations=1500, l2_leaf_reg=5.0
- 已验证：35K 训练样本，24行输出，hour 24 = 下一日 00:00

### Chronos 适配器
- 安装: `pip install chronos-forecasting` （非 `chronos` 包！）
- 已知 bug：包的 `__init__.py` 有时安装不完整，需手动创建
- Chronos-2 (amazon/chronos-2-small): 在当前 HF 镜像站不可用（gated access）
- Chronos-Bolt-Small (amazon/chronos-bolt-small): 可用，44M params
- API: `ChronosBoltPipeline.predict(inputs, prediction_length=24)` → `(1, 9, 24)`
- 返回 9 个 quantiles [0.1-0.9]，p50 (index 4) 作为 y_pred
- 已通过真实数据验证

### 路径处理
- 所有脚本使用 `os.path.abspath(__file__)` 确保含中文字符路径稳定
- 不使用 `Path(__file__).resolve()` 避免 Git Bash 下的 segfault
- 推荐始终用绝对路径调用 Python 和脚本

## 2026-07-02 修复汇总
- 修复了所有脚本缺少 `import os` 的问题（4个脚本）
- 默认路径改为可配置，优先 `--data-path` 参数，其次 yaml，最后 fallback 到 `_exp` 路径
- 输出 schema 补齐 `business_day` 列
- 新增 `bidding_space_raw` 列的读取，特征工程中优先使用真实竞价空间
- Chronos `predict_context()` 支持多种输出 shape（quantile/sample/direct），修复 tensor→numpy 的 `detach().cpu()`
- 新增 `scripts/smoke_check.py`（25 项检查全部通过）
- walk-forward 脚本修复 metrics 空值处理
- 所有验收命令均通过
