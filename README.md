# Understanding Systematic Risk 高频复现项目

本仓库围绕 Pelger (2020)《Understanding Systematic Risk: A High-Frequency Approach》在中国 A 股 5 分钟数据上的适配与复现，当前有两条主要工作流：

1. `Code/preprocess_cn_data.py -> Code/allcode_Need.py`
   先把原始 5 分钟 K 线预处理成后复权面板，再运行论文主复现链路并导出表格、图和诊断文件。
2. `Code/build_mom_5min.py`
   直接从原始 5 分钟 K 线构建市场级 5 分钟动量因子时间序列。

项目默认目录与当前代码入口已经稳定收口到：

- 原始 K 线：`Data/kline_Data/EXTRA_STOCK_A`
- 后复权因子：`Data/fact_Data/backward_factor.csv`
- 高频预处理产物：`Data/proc_Data/pelger_cn_adjusted`
- 5 分钟动量因子产物：`Data/proc_Data/mom_5min`
- 论文复现输出：`Result/pelger_cn_adjusted`

## 仓库结构

```text
Reposit/
├─ Code/
│  ├─ preprocess_cn_data.py
│  ├─ allcode_Need.py
│  ├─ build_mom_5min.py
│  ├─ export_panel_csv.py
│  └─ getApidb.py
├─ Data/
│  ├─ kline_Data/
│  ├─ fact_Data/
│  └─ proc_Data/
├─ Result/
└─ requirements.txt
```

- `Code/`：主脚本入口与辅助工具
- `Data/`：原始数据、复权因子、预处理产物、动量因子产物
- `Result/`：论文复现结果、运行中 checkpoint、进度和诊断文件

`requirements.txt` 只包含公开运行依赖，不包含 `Code/getApidb.py` 使用的 proprietary SDK。

## 推荐工作流

### 1. 准备原始 5 分钟数据

如果你已经有 `Data/kline_Data/EXTRA_STOCK_A/<symbol>/data.bz2`，可以跳过这一步。

上游抓取脚本是 `Code/getApidb.py`：

- 输入：AmazingData / `api_AmazingData_professional` 环境
- 输出：用户指定 `--output-root` 下的原始 `data.bz2` 目录树
- 是否直接读原始 `.bz2`：否，它负责生成原始 `.bz2`
- 典型用途：从 API 批量导出原始 5 分钟 K 线

### 2. 预处理原始数据

`Code/preprocess_cn_data.py` 是唯一直接读取原始 `.bz2` 的预处理脚本。

- 输入：
  - `Data/kline_Data/EXTRA_STOCK_A`
  - `Data/fact_Data/backward_factor.csv`
- 输出：
  - `Data/proc_Data/pelger_cn_adjusted/manifest.json`
  - `Data/proc_Data/pelger_cn_adjusted/metadata/`
  - `Data/proc_Data/pelger_cn_adjusted/symbol_returns/`
  - `Data/proc_Data/pelger_cn_adjusted/panels/strict_balanced/`
- 是否直接读原始 `.bz2`：是
- 典型用途：后复权、清洗、逐股票收益缓存、严格平衡面板构建

常用命令：

```bash
python Code/preprocess_cn_data.py --raw-root Data\kline_Data\EXTRA_STOCK_A --factor-path Data\fact_Data\backward_factor.csv --proc-root Data\proc_Data\pelger_cn_adjusted --workers 8 --panel-workers 8
```

小样本或局部重建示例：

```bash
python Code/preprocess_cn_data.py --years 2016 --max-stocks 10 --refresh
```

说明：

- `--return-mode` 当前仅保留兼容，不改变面板收益定义
- `--refresh` 会忽略已有 `manifest.json` 并重建预处理产物
- `--compress-symbol-returns` 更省磁盘，但逐股票 IO 会更慢

### 3. 运行论文主复现链路

`Code/allcode_Need.py` 只消费 `proc_Data`，不会回退读取原始 `.bz2`。

- 输入：
  - `Data/proc_Data/pelger_cn_adjusted`
- 输出：
  - `Result/pelger_cn_adjusted/tables/`
  - `Result/pelger_cn_adjusted/figures/`
  - `Result/pelger_cn_adjusted/diagnostics/`
  - `Result/pelger_cn_adjusted/checkpoints/`
- 是否直接读原始 `.bz2`：否
- 典型用途：论文主流程、滚动 PCA、年度附表、图表和诊断导出

常用命令：

```bash
python Code/allcode_Need.py --proc-root Data\proc_Data\pelger_cn_adjusted --output-root Result\pelger_cn_adjusted --workers 8 --paper-workers 4 --rolling-workers 8 --memory-budget-gb 20 --progress-interval-sec 10
```

小样本 smoke run：

```bash
python Code/allcode_Need.py --proc-root Data\proc_Data\pelger_cn_adjusted --output-root Result\pelger_cn_adjusted_smoke --years 2016 --max-stocks 10 --workers 2 --paper-workers 1 --rolling-workers 2 --no-plots
```

说明：

- `--no-plots` 会跳过 PNG 输出，但表格和诊断仍会生成
- `--return-mode` 当前是兼容参数，主复现实际口径由预处理阶段固定
- 如果替换了原始数据或复权因子，应先重跑 `preprocess_cn_data.py`

### 4. 构建 5 分钟动量因子

`Code/build_mom_5min.py` 独立于论文主复现链路，直接从原始 5 分钟 K 线构建市场级 MOM 时序。

- 输入：
  - `Data/kline_Data/EXTRA_STOCK_A`
  - `Data/fact_Data/backward_factor.csv`
- 输出：
  - `Data/proc_Data/mom_5min/mom_factor_5min.csv`
  - `Data/proc_Data/mom_5min/mom_factor_5min.pkl`
  - `Data/proc_Data/mom_5min/mom_factor_5min.parquet`
  - `Data/proc_Data/mom_5min/metadata.json`
- 是否直接读原始 `.bz2`：是
- 典型用途：按 5 分钟频率生成市场级动量因子序列

常用命令：

```bash
python Code/build_mom_5min.py --raw-root Data\kline_Data\EXTRA_STOCK_A --factor-path Data\fact_Data\backward_factor.csv --proc-root Data\proc_Data\mom_5min --lookback-bars 48 --skip-bars 1 --winner-pct 0.3 --loser-pct 0.3 --min-stocks 5 --workers 8
```

说明：

- 输出是一张市场级 MOM 时间序列表，不是股票级长表
- `metadata.json` 会记录参数、年份范围、并行策略、处理与跳过的股票数
- 如果环境缺少 `pyarrow` 或 `fastparquet`，脚本会跳过 parquet 并打印 warning，但不会影响 `csv`、`pkl` 和 `metadata.json`

### 5. 导出可读 CSV 面板

`Code/export_panel_csv.py` 用于把严格平衡面板导出成可读 CSV。

- 输入：
  - `Data/proc_Data/pelger_cn_adjusted/panels/strict_balanced/full`
  - 或 `Data/proc_Data/pelger_cn_adjusted/panels/strict_balanced/year_YYYY`
- 输出：
  - `Code/<panel_name>/R_daily.csv`
  - `Code/<panel_name>/R_intra.csv`
  - `Code/<panel_name>/R_night.csv`
  - `Code/<panel_name>/R_5min_full.csv`
  - `Code/<panel_name>/day_ids_map.csv`
- 是否直接读原始 `.bz2`：否
- 典型用途：人工抽查、外部分析、快速浏览面板内容

常用命令：

```bash
python Code/export_panel_csv.py --panel-name full
python Code/export_panel_csv.py --panel-name year_2024
```

## 脚本总表

| 脚本 | 直接读取原始 `.bz2` | 主要输入 | 主要输出 | 典型用途 |
| --- | --- | --- | --- | --- |
| `Code/getApidb.py` | 否 | AmazingData API | 用户指定 `--output-root` 下的原始 `data.bz2` | 抓取原始 5 分钟 K 线 |
| `Code/preprocess_cn_data.py` | 是 | `Data/kline_Data/EXTRA_STOCK_A`、`backward_factor.csv` | `Data/proc_Data/pelger_cn_adjusted` | 后复权、清洗、严格平衡面板 |
| `Code/allcode_Need.py` | 否 | `Data/proc_Data/pelger_cn_adjusted` | `Result/pelger_cn_adjusted` | 论文主复现、滚动分析、图表导出 |
| `Code/build_mom_5min.py` | 是 | `Data/kline_Data/EXTRA_STOCK_A`、`backward_factor.csv` | `Data/proc_Data/mom_5min` | 5 分钟 MOM 因子构建 |
| `Code/export_panel_csv.py` | 否 | `panels/strict_balanced/<panel>` | `Code/<panel_name>/` CSV 文件 | 面板可视化与人工抽查 |

## 面板与数据口径

当前论文主复现使用 `Data/proc_Data/pelger_cn_adjusted/panels/strict_balanced/` 下的严格平衡面板。

- `full`
  - 全样本区间严格平衡面板
- `year_YYYY`
  - 单年严格平衡面板

核心数组口径：

- `R_daily`
  - 日度总对数收益，前一交易日收盘到当日收盘
- `R_intra`
  - 日内对数收益，当日开盘到当日收盘
- `R_night`
  - 隔夜对数收益，前一交易日收盘到当日开盘
- `R_5min_full`
  - 高频主序列，5 分钟连续收盘接续对数收益，形状为 `(D*48, N)`
- `day_ids`
  - 高频行对应的交易日索引，只和 `R_5min_full` 配套

当前代码里，jump decomposition、PCA、rolling analysis 的高频输入统一使用 `R_5min_full`。

## 续跑与稳态执行

`Code/allcode_Need.py` 已改成默认自动续跑模式：

- 同一个 `--output-root` 下再次启动时，会自动检查是否存在兼容 checkpoint
- 兼容时会跳过已完成的 `rolling` 块和 `paper` 年份，只补未完成部分
- `--restart` 会清空当前输出目录下兼容 checkpoint 与未完成导出，然后从头重跑

当前续跑粒度：

- `rolling`
  - 固定窗口块 checkpoint，落盘到 `checkpoints/rolling/chunk_XXXXX.npz`
- `paper`
  - 按年份 checkpoint，落盘到 `checkpoints/paper/year_YYYY/`

运行中重要文件：

- `Result/pelger_cn_adjusted/checkpoints/run_state.json`
  - 当前运行状态、已完成 rolling 块、已完成 paper 年份、是否已完成 export
- `Result/pelger_cn_adjusted/diagnostics/progress.jsonl`
  - 控制台同步的结构化进度日志
- `Result/pelger_cn_adjusted/diagnostics/resource_plan.json`
  - 自适应资源计划、paper 年度内存估算与 worker 配置
- `Result/pelger_cn_adjusted/diagnostics/stage_timings.json`
  - 各阶段耗时

## 结果目录说明

`Result/pelger_cn_adjusted` 在运行中和运行完成后可能呈现不同形态。

### 运行中

最先出现的通常是：

- `checkpoints/`
- `diagnostics/`

其中：

- `checkpoints/` 保存断点续跑所需中间结果
- `diagnostics/` 保存进度、资源计划、阶段耗时等运行状态文件

### 完整成功后

除了上面的运行期文件，还会导出：

- `tables/`
  - 论文主表、兼容别名表和附加诊断表
- `figures/`
  - 论文图、兼容别名图、外部数据缺失时的占位图
- `diagnostics/`
  - `universe_scan.csv`、`main_summary.json`、`replication_coverage_report.csv`、`plot_export_status.csv` 等

不要把 `Result/pelger_cn_adjusted` 理解成“始终已经有 tables/figures”；在长任务运行中，目录可能只有 checkpoint 和 diagnostics。

## 论文输出与覆盖范围

当前目标是尽量对齐论文的结构、编号、输出完整性，而不是让中国 A 股结果逐项等于原始美股 TAQ 数值。

当前仓库可直接生成的主要内容包括：

- `Table I`
  - continuous / jump 统计
- `Table II`
  - balanced vs unbalanced 因子空间 generalized correlation
- `Table IV`
  - rolling generalized correlation 与 explained variation 汇总
- `Table V`
  - continuous PCA 因子的 intraday / overnight / daily Sharpe
- `Figure 1` 到 `Figure 13`
  - 当前数据可直接支持的图形

仍依赖外部测试资产或因子数据的内容：

- `Table III`
  - 行业组合收益与 Fama-French-Carhart 因子
- `Figure 14`
  - 行业组合测试资产
- `Figure 15`
  - size/value 测试资产

这些缺口会在下列文件中明确标注：

- `diagnostics/replication_coverage_report.csv`
- `diagnostics/plot_export_status.csv`

## Data 目录补充说明

`Data/` 下当前同时维护两类下游产物：

- `Data/proc_Data/pelger_cn_adjusted`
  - 高频论文复现链路使用的后复权面板和元数据
- `Data/proc_Data/mom_5min`
  - 5 分钟 MOM 因子输出目录

详细的数据目录契约见 [Data/readme.md](Data/readme.md)。

## 常见提示

- `preprocess_cn_data.py` 是唯一会直接读取原始 `.bz2` 的预处理脚本
- `allcode_Need.py` 只消费 `proc_Data`，不会回退到原始 K 线
- 替换了原始 K 线或 `backward_factor.csv` 后，应重新运行预处理，必要时加 `--refresh`
- 如果 `allcode_Need.py` 报缺少 `manifest.json` 或面板结构不兼容，应先重新运行预处理
- `build_mom_5min.py` 和 `allcode_Need.py` 彼此独立，不共享同一个输出目录
