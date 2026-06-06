# Data 目录说明

`Data/` 只存放原始数据、外部补充数据、预处理产物和运行期缓存，不存放最终正式论文图表。

## 总体结构

```text
Data/
├─ kline_Data/
│  └─ EXTRA_STOCK_A/
├─ fact_Data/
│  └─ backward_factor.csv
├─ external_Data/
│  └─ pelger_tail/
└─ proc_Data/
   ├─ pelger_cn_adjusted/
   └─ mom_5min/
```

## 上游输入层

### `kline_Data/EXTRA_STOCK_A`

- 每只股票一个子目录，内部包含 `data.bz2`
- 这是项目的原始 5 分钟 K 线输入
- 直接消费者：
  - `Code/preprocess_cn_data.py`
  - `Code/build_mom_5min.py`

### `fact_Data/backward_factor.csv`

- 后复权因子宽表
- 直接消费者：
  - `Code/preprocess_cn_data.py`
  - `Code/build_mom_5min.py`

当前使用口径：

```text
adjusted_ohlc = raw_ohlc * backward_factor
```

## 外部补充数据层：`external_Data/pelger_tail`

这是论文后半段 `paper_tail` 使用的外部支持数据，不应再依赖临时“补充数据及代码”目录。

当前关键子目录：

- `factors/ff3/`
- `factors/ff5/`
- `factors/rf/shibor_对数收益率.csv`
- `industry/stock_full_info_std_industry_final.csv`
- `industry/行业映射表_终版.csv`
- `industry/行业映射表_终版.md`
- `size_value/raw/`
- `size_value/reference/`

这些文件由 `Code/main.py` 间接通过 `paper_tail` 刷新层消费。

## 预处理主产物：`proc_Data/pelger_cn_adjusted`

这是主复现链路的数据根目录。

### 目录职责

- `manifest.json`
  - 预处理版本、输入签名、参数、样本统计、面板输出清单
- `metadata/`
  - 样本宇宙、年份覆盖和风险诊断
- `symbol_returns/`
  - 逐股票收益缓存
- `panels/`
  - 主 PCA / rolling / paper 读取的面板文件
- `runtime/`
  - 断点续跑 checkpoint、进度日志、资源计划、运行期诊断
- `paper_tail/`
  - 后半段资产、FFC、行业组合、校验文件和诊断

### `panels/` 下的两套主样本

- `panels/strict_balanced/`
  - 严格平衡面板
- `panels/paper_lenient/`
  - 论文宽松平衡面板

每套面板都应包含：

- `full/`
- `full.json`
- `year_YYYY/`
- `year_YYYY.json`

数组目录中的主要文件：

- `R_daily.npy`
- `R_intra.npy`
- `R_night.npy`
- `R_5min_full.npy`
- `day_ids.npy`

### 数组口径

- `R_daily`
  - 日度总对数收益，前一交易日收盘到当日收盘
- `R_intra`
  - 日内对数收益，当日开盘到当日收盘
- `R_night`
  - 隔夜对数收益，前一交易日收盘到当日开盘
- `R_5min_full`
  - 高频主序列，形状为 `(D*48, N)` 的 5 分钟连续收盘接续对数收益
- `day_ids`
  - `R_5min_full` 行到交易日的映射索引

### 与主流程的关系

- `Code/preprocess_cn_data.py`
  - 负责生成整个 `proc_Data/pelger_cn_adjusted`
- `Code/main.py`
  - 只读取这里的预处理产物，不回退读取原始 `.bz2`
- `Code/export_panel_csv.py`
  - 读取 `strict_balanced` 面板并导出人工检查用 CSV

## 运行期目录：`proc_Data/pelger_cn_adjusted/runtime`

这是运行过程中的状态目录，不是最终论文结果目录。

主要内容：

- `checkpoints/run_state.json`
- `checkpoints/rolling/chunk_XXXXX.npz`
- `checkpoints/paper/year_YYYY/`
- `diagnostics/progress.jsonl`
- `diagnostics/resource_plan.json`
- `diagnostics/stage_timings.json`
- `diagnostics/replication_coverage_report.csv`
- `diagnostics/plot_export_status.csv`

## 后半段缓存：`proc_Data/pelger_cn_adjusted/paper_tail`

这是 `paper_tail` 的规范化缓存目录。

主要内容：

- `manifest.json`
- `assets/`
- `factors/`
- `tables/`
- `figures/`
- `validation/`
- `diagnostics/`

常见关键文件：

- `factors/ffc_external_daily.csv`
- `factors/ffc_segmented_returns.csv`
- `assets/industry_portfolios.csv`
- `assets/size_value_portfolios.csv`
- `validation/ffc_daily_validation_summary.json`
- `validation/size_value_daily_parity_summary.json`
- `diagnostics/factor_matrix_diagnostics.json`

## MOM 因子产物：`proc_Data/mom_5min`

这是 `Code/build_mom_5min.py` 的输出目录。

主要文件：

- `mom_factor_5min.csv`
- `mom_factor_5min.pkl`
- `mom_factor_5min.parquet`
- `metadata.json`

结果表当前核心列包括：

- `kline_time`
- `MOM`
- `winner_ret`
- `loser_ret`
- `n_stocks`
- `n_winners`
- `n_losers`

## 使用时的判断原则

- 看样本规模、覆盖年份、面板是否齐全：
  - 先看 `proc_Data/pelger_cn_adjusted/manifest.json`
  - 再看 `proc_Data/pelger_cn_adjusted/metadata/universe_summary.json`
- 看 MOM 是否重建完成：
  - 看 `proc_Data/mom_5min/metadata.json`
- 看运行过程是否正常：
  - 看 `proc_Data/pelger_cn_adjusted/runtime/diagnostics/`
- 看后半段外部数据适配是否通过：
  - 看 `proc_Data/pelger_cn_adjusted/paper_tail/validation/`
