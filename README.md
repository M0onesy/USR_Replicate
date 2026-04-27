# Understanding Systematic Risk 高频复现项目

本仓库用于复现 Pelger (2020)《Understanding Systematic Risk: A High-Frequency Approach》的核心高频因子流程，并将数据口径替换为中国 A 股 5 分钟 K 线。

当前工作流已经明确拆成两步：

1. `Code/preprocess_cn_data.py`：唯一接触原始 K 线的预处理脚本。它读取 `Data\kline_Data\EXTRA_STOCK_A\<symbol>\data.bz2`，匹配 `Data\fact_Data\backward_factor.csv`，生成后复权收益数组和快速 IO 面板。
2. `Code/allcode_Need.py`：论文复现核心脚本。它只读取 `Data\proc_Data\pelger_cn_adjusted`，运行跳跃分解、PCA、因子数估计、Sharpe、滚动稳定性和年度 99% 覆盖样本稳健性分析。

## 当前数据状态

- 原始 K 线目录：`Data\kline_Data\EXTRA_STOCK_A\`
- 当前本地原始股票目录数：`5799`
- 每只股票目录下应包含：`data.bz2`
- 复权因子：`Data\fact_Data\backward_factor.csv`
- 预处理产物：`Data\proc_Data\pelger_cn_adjusted\`
- 复现结果：`Result\pelger_cn_adjusted\`

严格平衡样本数量、年度 99% 覆盖样本数量、实际样本区间等不再写死在 README 中，统一以预处理生成的 `manifest.json`、`metadata\universe_summary.json` 和 `metadata\universe.csv` 为准。

## 目录职责

| 路径 | 职责 |
| --- | --- |
| `Code\preprocess_cn_data.py` | 原始 K 线读取、清洗、后复权、收益构造、样本筛选、快速面板输出 |
| `Code\allcode_Need.py` | 读取预处理面板并运行论文复现核心统计流程 |
| `Code\getApidb.py` | 外部数据源下载辅助脚本，不参与默认复现流程 |
| `Data\kline_Data\EXTRA_STOCK_A\` | 原始 5 分钟 K 线归档，每只股票一个 `data.bz2` |
| `Data\fact_Data\` | 本地复权因子等事实数据，默认使用 `backward_factor.csv` |
| `Data\proc_Data\pelger_cn_adjusted\` | 预处理后的收益数组、样本元数据和面板文件 |
| `Result\pelger_cn_adjusted\` | 论文复现输出表格、图和诊断文件 |

## 1. 生成后复权预处理数据

正式预处理：

```bash
python Code/preprocess_cn_data.py --raw-root Data\kline_Data\EXTRA_STOCK_A --factor-path Data\fact_Data\backward_factor.csv --proc-root Data\proc_Data\pelger_cn_adjusted --workers 8 --panel-workers 4
```

小样本 smoke run：

```bash
python Code/preprocess_cn_data.py --years 2016 --max-stocks 10 --workers 2 --panel-workers 2 --refresh
```

`--workers` 控制逐股票读取、清洗、复权和收益构造的多进程并行度；`--panel-workers` 控制面板拼接时读取逐股票中间数组的线程数。默认值为 `min(cpu_count - 1, 8)`，不会自动开满所有核心，避免 `.bz2` 解压、磁盘 IO 和 BLAS 线程互相抢资源。需要排查问题时可使用 `--workers 1 --panel-workers 1` 回到完全串行。

预处理脚本会逐股票读取 pandas-bz2 格式的 `data.bz2`，校验 48 根 5 分钟 bar、OHLC 正值和重复时间戳，按交易日匹配后复权因子，并使用：

```text
adjusted_ohlc = raw_ohlc * backward_factor
```

随后构造：

- `R_intra`：日内收益，默认 `log(close/open)`。
- `R_night`：隔夜收益，前一有效交易日最后一根 close 到下一有效交易日第一根 open。
- `R_daily`：日收益，满足 `sum(R_intra by day) + R_night == R_daily`。
- `strict_balanced`：严格平衡主样本面板。
- `near_balanced_99`：年度 99% 覆盖非平衡稳健性面板。

## 2. 使用预处理数据复现论文

正式复现：

```bash
python Code/allcode_Need.py --proc-root Data\proc_Data\pelger_cn_adjusted --output-root Result\pelger_cn_adjusted --workers 8
```

小样本 smoke run：

```bash
python Code/allcode_Need.py --proc-root Data\proc_Data\pelger_cn_adjusted --output-root Result\pelger_cn_adjusted_smoke --years 2016 --max-stocks 10 --workers 2 --no-robustness
```

复现脚本不再读取、扫描或解压原始 K 线文件。如果 `proc-root` 下缺少 `manifest.json` 或面板文件，脚本会提示先运行 `Code\preprocess_cn_data.py`。

复现阶段的 `--workers` 主要用于滚动 PCA 和年度 99% 覆盖样本稳健性分析。主 PCA 仍交给 NumPy/SciPy 的线性代数库处理，避免外层并行和 BLAS 内部线程过度叠加。

默认会生成论文编号 `Figure_01` 到 `Figure_15` 的 PNG。不要在正式复现时加 `--no-plots`；该参数只用于纯表格调试，会让所有图片导出状态记为 `skipped`。

## 预处理产物结构

`Data\proc_Data\pelger_cn_adjusted\` 的主要文件包括：

- `manifest.json`：预处理版本、输入路径、原始股票数量、处理股票数量、复权因子签名、样本摘要和面板清单。
- `metadata\universe.csv`：逐股票覆盖率、有效交易日、缺失交易日和异常隔夜收益诊断。
- `metadata\universe_summary.json`：样本区间、交易日历、严格平衡和 99% 覆盖样本摘要。
- `metadata\corp_action_risk_after_adjustment.csv`：复权前后异常隔夜收益计数。
- `symbol_returns\<symbol>.npz`：逐股票日内、隔夜和日度收益中间数组。
- `panels\strict_balanced\`：严格平衡样本面板。
- `panels\near_balanced_99\`：年度 99% 覆盖稳健性面板。

主面板优先使用 `.npy + .json` 结构保存，方便快速 IO 和后续扩展；CSV 只用于人工可读的摘要表和诊断表。

## 结果目录结构

`Result\pelger_cn_adjusted\` 按类型拆分输出：

- `tables\Table_01_sample_summary.csv`：样本摘要。
- `tables\Table_02_jump_stats.csv`：跳跃分解统计。
- `tables\Table_03_factor_counts.csv`：高频、连续、跳跃因子数。
- `tables\Table_04_sharpes.csv`：日内、隔夜、日度 Sharpe。
- `tables\Table_05_rolling_gc.csv`：滚动 generalized correlation。
- `tables\Table_06_rolling_explained_variation.csv`：滚动解释度。
- `tables\Table_07_robustness_yearly_gc.csv`：年度 99% 覆盖样本稳健性。
- `tables\Table_08_paper_style_yearly_jump_stats.csv`：更接近论文 Table I 的年度多阈值跳跃统计和解释度。
- `tables\Table_09_paper_style_factor_count_diagnostics.csv`：扰动特征值比率诊断数据，对应论文 Figures 1-2 的可支持部分。
- `tables\Table_10_paper_style_factor_sharpes.csv`：连续 PCA 因子的日内、隔夜、日度 Sharpe，对应论文 Table V 的可支持部分。
- `tables\Table_11_continuous_pca_weights.csv`：连续 PCA 因子权重。
- `tables\Table_12_proxy_factor_weights.csv`：proxy factor 权重。
- `tables\Table_13_monthly_pca_weights.csv`：月频 PCA 权重。
- `tables\Table_14_factor_return_summary.csv`：连续 PCA 因子的日内、隔夜、日度均值。
- `figures\Figure_01_perturbed_er_balanced.png` 到 `figures\Figure_15_asset_pricing_size_value_portfolios.png`：按论文编号导出的 15 张图。
- `diagnostics\plot_export_status.csv`：每张图的生成状态，区分真实生成、占位、跳过和错误。
- `diagnostics\replication_coverage_report.csv`：论文表图覆盖说明，区分已实现、中国市场适配、部分实现和需要外部数据的内容。
- `diagnostics\`：样本清单、主摘要、复权后异常隔夜收益诊断等辅助文件。

## 常用检查

```bash
python -m py_compile Code/preprocess_cn_data.py Code/allcode_Need.py
```

```bash
rg "from allcode_Need import" Code/preprocess_cn_data.py
```

上面第二条命令正常情况下不应返回内容，表示预处理脚本没有反向依赖复现脚本。

## 注意事项

- 默认复权口径是后复权因子 `backward_factor.csv`。
- `adj_factor.csv` 当前只作为后续校验扩展，不参与默认流程。
- 当前仅依赖 A 股 5 分钟 K 线和后复权因子，无法一比一复现需要行业因子、Fama-French-Carhart 因子或测试资产组合的全部论文表图；Figure 14 和 Figure 15 在未提供外部测试资产数据时会生成明确占位图，覆盖情况以 `diagnostics\replication_coverage_report.csv` 和 `diagnostics\plot_export_status.csv` 为准。
- `Data\fact_Data\`、`Data\proc_Data\` 和 `Result\` 属于本地大数据或生成结果，已加入 `.gitignore`。
- 替换原始 K 线或复权因子后，请用 `--refresh` 重新运行预处理。
