# Understanding Systematic Risk 高频复现项目

本仓库用于复现 Pelger (2020)《Understanding Systematic Risk: A High-Frequency Approach》的核心高频方法，并将原论文的美国 TAQ 数据口径适配到中国 A 股 5 分钟后复权数据。

当前工作流已经固定为两阶段：

1. 先运行 `Code/preprocess_cn_data.py`
   从 `Data/kline_Data/EXTRA_STOCK_A` 读取原始 `.bz2` K 线，用 `Data/fact_Data/backward_factor.csv` 做后复权，生成可快速读取的面板与元数据到 `Data/proc_Data/pelger_cn_adjusted`。
2. 再运行 `Code/allcode_Need.py`
   只读取 `Data/proc_Data/pelger_cn_adjusted`，执行论文主干复现，输出表格、图片和诊断文件到 `Result/pelger_cn_adjusted`。

## 当前目录关系

- `Data/kline_Data/EXTRA_STOCK_A`
  原始 A 股 5 分钟 K 线目录。当前本地原始股票目录数是 `5799`。
- `Data/fact_Data/backward_factor.csv`
  后复权因子。默认复权公式为 `adjusted_ohlc = raw_ohlc * backward_factor`。
- `Data/proc_Data/pelger_cn_adjusted`
  预处理产物，包含面板、逐股票缓存和元数据。
- `Result/pelger_cn_adjusted`
  论文复现输出目录，包含 `tables/`、`figures/`、`diagnostics/`。

说明：

- 原始目录数是 `5799`，但预处理后 `metadata/universe.pkl` 中的“可处理股票数”可能更少。
- 常见原因包括：原始文件损坏、交易日结构不合法、缺少复权因子、预处理规则剔除坏日后无法满足面板要求。
- 严格平衡面板数量请以 `Data/proc_Data/pelger_cn_adjusted/manifest.json` 和 `metadata/universe_summary.json` 为准，不在 README 里写死。

## 面板口径

`Data/proc_Data/pelger_cn_adjusted/panels/strict_balanced` 当前只保留严格平衡口径。

- `full`
  全区间严格平衡面板。股票必须在完整样本区间内零缺失。
- `year_YYYY`
  年度严格平衡面板。股票只要求在该年内部零缺失，不要求跨年都在样本中。

面板中的核心数组定义如下：

- `R_daily`
  日度总对数收益，前一交易日收盘到当日收盘，含隔夜与日内。
- `R_intra`
  日内对数收益，当日开盘到当日收盘，形状为 `(D, N)`。
- `R_night`
  隔夜对数收益，前一交易日收盘到当日开盘，形状为 `(D, N)`。
- `R_5min_full`
  高频主序列，5 分钟连续收盘接续对数收益，形状为 `(D*48, N)`。
- `day_ids`
  高频行所属交易日索引，只和 `R_5min_full` 对应。

论文主流程现在统一以 `R_5min_full` 作为 jump decomposition、PCA、rolling analysis 的高频输入，不再把 `R_intra` 当作高频矩阵使用。

## 运行命令

预处理：

```bash
python Code/preprocess_cn_data.py --raw-root Data\kline_Data\EXTRA_STOCK_A --factor-path Data\fact_Data\backward_factor.csv --proc-root Data\proc_Data\pelger_cn_adjusted
```

论文复现：

```bash
python Code/allcode_Need.py --proc-root Data\proc_Data\pelger_cn_adjusted --output-root Result\pelger_cn_adjusted
```

小样本 smoke run：

```bash
python Code/preprocess_cn_data.py --years 2016 --max-stocks 10 --refresh
python Code/allcode_Need.py --proc-root Data\proc_Data\pelger_cn_adjusted --output-root Result\pelger_cn_adjusted_smoke --years 2016 --max-stocks 10 --workers 2
```

说明：

- `--return-mode` 现在只是兼容参数，不再改变面板收益定义。
- 正式跑图时不要加 `--no-plots`，否则不会导出 PNG。
- 在 PyCharm 里直接运行 `Code/allcode_Need.py` 也可以，只要脚本参数指向现有 `proc_Data` 即可。

## 论文编号输出

`Code/allcode_Need.py` 现在优先按论文编号输出 canonical 文件，并保留旧编号别名供兼容脚本继续使用。

canonical 表格：

- `tables/Table_I_summary_statistics_for_continuous_and_jump_returns.csv`
- `tables/Table_II_balanced_and_unbalanced_panel_results.csv`
- `tables/Table_III_generalized_correlations_with_industry_and_ffc_factors.csv`
- `tables/Table_IV_time_variation_decomposition.csv`
- `tables/Table_V_intraday_overnight_daily_sharpe_ratios.csv`

canonical 图片：

- `figures/Figure_1_*.png` 到 `figures/Figure_15_*.png`

兼容别名：

- `Table_08`、`Table_09`、`Table_10`
- `Figure_01` 到 `Figure_15`

## 当前复现覆盖范围

当前目标是“论文结构、编号、输出完整性尽量对齐”，不是强行让中国 A 股数值与原始美股 TAQ 逐项完全一致。

已能真实生成的内容：

- `Table I`
  balanced / unbalanced 两块的连续收益与跳跃收益统计。
- `Table II`
  balanced vs unbalanced 的 factor-space generalized correlation。
- `Table IV`
  rolling generalized correlation 与 explained variation 的时间变化汇总。
- `Table V`
  continuous PCA 因子的 intraday / overnight / daily Sharpe。
- `Figure 1` 到 `Figure 13`
  当前数据可支持的真实图。

需要外部数据的内容：

- `Table III`
  需要行业组合收益和 Fama-French-Carhart 因子。
- `Figure 14`
  需要行业组合收益。
- `Figure 15`
  需要 size/value 测试资产组合。

这些缺口不会再静默降级，而会在下列文件中明确标出：

- `diagnostics/replication_coverage_report.csv`
- `diagnostics/plot_export_status.csv`

## 常用辅助脚本

- `Code/export_panel_csv.py`
  可把 `strict_balanced/full` 或 `strict_balanced/year_YYYY` 中的 `R_daily`、`R_intra`、`R_night`、`R_5min_full` 导出成可读 CSV。

## 结果目录说明

- `tables/`
  论文编号表格、兼容表格别名和附加诊断表。
- `figures/`
  论文编号图片、兼容图片别名和外部数据占位图。
- `diagnostics/`
  `universe_scan.csv`、`main_summary.json`、`replication_coverage_report.csv`、`plot_export_status.csv` 等诊断文件。

## 当前状态提示

- `preprocess_cn_data.py` 是唯一读取原始 `.bz2` 的脚本。
- `allcode_Need.py` 不再读取原始 K 线，只消费 `proc_Data`。
- 如果替换了原始数据或复权因子，请重新运行预处理，必要时加 `--refresh`。
- 如果 `allcode_Need.py` 报找不到 `manifest.json` 或面板结构过旧，请先重新运行预处理。
