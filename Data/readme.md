# Data 目录说明

`Data/` 只存放原始数据、复权因子和下游产物，不存放论文主复现代码。

当前数据流分成两段：

1. `Code/preprocess_cn_data.py`
   从原始 `.bz2` 生成 `proc_Data/pelger_cn_adjusted`
2. `Code/build_mom_5min.py`
   从原始 `.bz2` 直接生成 `proc_Data/mom_5min`

## 目录总览

```text
Data/
├─ kline_Data/
│  └─ EXTRA_STOCK_A/
├─ fact_Data/
│  └─ backward_factor.csv
└─ proc_Data/
   ├─ pelger_cn_adjusted/
   └─ mom_5min/
```

## 上游：原始 5 分钟 K 线

- 目录：`kline_Data/EXTRA_STOCK_A`
- 结构：每只股票一个子目录，内部包含 `data.bz2`
- 主要消费者：
  - `Code/preprocess_cn_data.py`
  - `Code/build_mom_5min.py`

这里是项目的原始输入层。`Code/allcode_Need.py` 不会直接读取这里的 `.bz2`。

如果需要重新抓取原始数据，可使用 `Code/getApidb.py` 将 API 返回结果导出成这类目录结构。

## 复权因子

- 文件：`fact_Data/backward_factor.csv`
- 主要消费者：
  - `Code/preprocess_cn_data.py`
  - `Code/build_mom_5min.py`

当前默认且唯一使用的是后复权口径：

```text
adjusted_ohlc = raw_ohlc * backward_factor
```

如果你替换了 `backward_factor.csv`，应重新运行依赖它的下游脚本。

## 下游一：`proc_Data/pelger_cn_adjusted`

这是论文主复现链路使用的预处理产物根目录。

### 主要内容

- `manifest.json`
  - 预处理版本、输入签名、参数、样本统计、面板输出清单
- `metadata/`
  - `universe.pkl`
  - `universe.csv`
  - `universe_summary.json`
  - `corp_action_risk_after_adjustment.csv`
- `symbol_returns/`
  - 逐股票收益缓存，供后续非平衡样本构建和调试使用
- `panels/strict_balanced/`
  - 严格平衡面板

`manifest.json` 当前会记录一组稳定的统计字段，例如：

- `raw_symbol_count`
- `processed_symbol_count`
- `failed_symbol_count`
- `strict_balanced_symbols`
- `strict_balanced_symbols_by_year`
- `panel_return_scheme`

这些字段比 README 中手写数字更可靠，后续如需确认当前样本规模，应以 `manifest.json` 为准。

### 严格平衡面板结构

当前仅保留 `strict_balanced` 口径。

- `panels/strict_balanced/full`
  - 全样本区间严格平衡面板数组目录
- `panels/strict_balanced/full.json`
  - `full` 面板元数据
- `panels/strict_balanced/year_YYYY`
  - 单年严格平衡面板数组目录
- `panels/strict_balanced/year_YYYY.json`
  - 单年面板元数据

数组目录中主要文件包括：

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
  - 5 分钟连续收盘接续对数收益，形状为 `(D*48, N)`，是高频主流程输入
- `day_ids`
  - 高频行到交易日的映射索引，只和 `R_5min_full` 配套

当前 `Code/allcode_Need.py` 的 jump decomposition、PCA 和 rolling analysis 都统一使用 `R_5min_full` 作为高频输入。

### 与脚本的关系

- `Code/preprocess_cn_data.py`
  - 负责生成整个 `proc_Data/pelger_cn_adjusted`
- `Code/allcode_Need.py`
  - 只读取这个目录，不回退读取原始 `.bz2`
- `Code/export_panel_csv.py`
  - 从 `panels/strict_balanced/full` 或 `year_YYYY` 导出可读 CSV

## 下游二：`proc_Data/mom_5min`

这是 `Code/build_mom_5min.py` 的输出目录。

### 主要内容

- `mom_factor_5min.csv`
  - 市场级 5 分钟 MOM 因子时序表
- `mom_factor_5min.pkl`
  - 同一结果的 Pickle 版本
- `mom_factor_5min.parquet`
  - 同一结果的 Parquet 版本
- `metadata.json`
  - 因子构建参数、年份覆盖范围、并行策略、处理统计

结果表当前包含的核心列为：

- `kline_time`
- `MOM`
- `winner_ret`
- `loser_ret`
- `n_stocks`
- `n_winners`
- `n_losers`

`metadata.json` 当前会记录例如：

- `raw_root`
- `factor_path`
- `proc_root`
- `lookback_bars`
- `skip_bars`
- `winner_pct`
- `loser_pct`
- `min_stocks`
- `years`
- `workers`
- `parallel_strategy`
- `raw_symbol_files`
- `processed_symbols`
- `skipped_symbols`

如果环境缺少 Parquet 引擎，脚本会跳过 `mom_factor_5min.parquet` 并打印 warning；其余文件仍会正常写出。

### 与脚本的关系

- `Code/build_mom_5min.py`
  - 直接从 `Data/kline_Data/EXTRA_STOCK_A` 和 `Data/fact_Data/backward_factor.csv` 生成该目录
- `Code/allcode_Need.py`
  - 不消费这个目录

## 推荐顺序

### 论文复现链路

1. 准备原始 `data.bz2`
2. 运行 `Code/preprocess_cn_data.py`
3. 运行 `Code/allcode_Need.py`

### 动量因子链路

1. 准备原始 `data.bz2`
2. 确认 `backward_factor.csv`
3. 运行 `Code/build_mom_5min.py`

## 常见提示

- `proc_Data/pelger_cn_adjusted` 和 `proc_Data/mom_5min` 是两套独立产物，不应混用
- 替换原始 K 线或复权因子后，应重跑对应下游脚本
- 是否成功处理了多少股票、覆盖了哪些年份，应优先查看：
  - `proc_Data/pelger_cn_adjusted/manifest.json`
  - `proc_Data/mom_5min/metadata.json`
