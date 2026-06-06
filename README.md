# Understanding Systematic Risk 高频复现项目

当前仓库围绕 Pelger (2020)《Understanding Systematic Risk: A High-Frequency Approach》在中国 A 股 5 分钟数据上的适配复现展开。现在的真实主入口已经收口为 `Code/main.py + Code/core/config.py`，不再使用旧的 `allcode_Need.py` 单文件入口。

## 当前真实结构

```text
Reposit/
├─ Code/
│  ├─ main.py
│  ├─ preprocess_cn_data.py
│  ├─ build_mom_5min.py
│  ├─ export_panel_csv.py
│  ├─ core/
│  ├─ figcode/
│  └─ tablecode/
├─ Data/
│  ├─ kline_Data/
│  ├─ fact_Data/
│  ├─ external_Data/
│  └─ proc_Data/
├─ Result/
└─ requirements.txt
```

- `Code/`：运行入口、复现引擎、图表与表格脚本。
- `Data/`：原始数据、补充外部数据、预处理产物、运行期缓存与诊断。
- `Result/`：最终正式论文输出。

## 两条工作流

### 1. 高频主复现链路

```text
Data/kline_Data/EXTRA_STOCK_A
        + Data/fact_Data/backward_factor.csv
        -> Code/preprocess_cn_data.py
        -> Data/proc_Data/pelger_cn_adjusted
        -> Code/main.py
        -> Result/pelger_cn_adjusted
```

### 2. 5 分钟 MOM 因子链路

```text
Data/kline_Data/EXTRA_STOCK_A
        + Data/fact_Data/backward_factor.csv
        -> Code/build_mom_5min.py
        -> Data/proc_Data/mom_5min
```

## 从零完整生成全部结果

### 1. 准备输入数据

至少需要：

- `Data/kline_Data/EXTRA_STOCK_A/<symbol>/data.bz2`
- `Data/fact_Data/backward_factor.csv`
- `Data/external_Data/pelger_tail/**`

`requirements.txt` 只包含公开依赖，不包含 `Code/getApidb.py` 所需的专有 SDK。

### 2. 重建 5 分钟 MOM

```bash
python Code/build_mom_5min.py --raw-root Data\kline_Data\EXTRA_STOCK_A --factor-path Data\fact_Data\backward_factor.csv --proc-root Data\proc_Data\mom_5min --lookback-bars 48 --skip-bars 1 --winner-pct 0.3 --loser-pct 0.3 --min-stocks 5 --workers 8
```

产物目录：

- `Data/proc_Data/mom_5min/mom_factor_5min.csv`
- `Data/proc_Data/mom_5min/mom_factor_5min.pkl`
- `Data/proc_Data/mom_5min/mom_factor_5min.parquet`
- `Data/proc_Data/mom_5min/metadata.json`

### 3. 重建高频预处理产物

```bash
python Code/preprocess_cn_data.py --raw-root Data\kline_Data\EXTRA_STOCK_A --factor-path Data\fact_Data\backward_factor.csv --proc-root Data\proc_Data\pelger_cn_adjusted --refresh --workers 8 --panel-workers 8
```

这一步完成后必须同时出现：

- `Data/proc_Data/pelger_cn_adjusted/panels/strict_balanced/`
- `Data/proc_Data/pelger_cn_adjusted/panels/paper_lenient/`

并且以下文件中要出现 `balanced_paper_*` 相关字段：

- `Data/proc_Data/pelger_cn_adjusted/manifest.json`
- `Data/proc_Data/pelger_cn_adjusted/metadata/universe_summary.json`

### 4. 切到正式全量 profile

打开 `Code/core/config.py`，把：

```python
ACTIVE_MAIN_PROFILE = "final_paper_export"
```

这个 profile 的默认正式语义是：

- `task_selectors=("all",)`：只导出正式 Figure 1–15 与 Table I–V
- `rebuild_result=True`
- `restart=True`
- `balanced_mode=PAPER_LENIENT_SAMPLE`
- `strict_final_export=True`

如果只想复用现有结果快速重导正式论文输出，用：

```python
ACTIVE_MAIN_PROFILE = "reuse_export_smoke"
```

如果只想跑附加诊断表，用：

```python
ACTIVE_MAIN_PROFILE = "diagnostics_only"
```

### 5. 运行主复现

从仓库根目录运行：

```bash
python Code/main.py
```

或进入 `Code/` 后运行：

```bash
python main.py
```

`main.py` 不再接受 `--only`、`--restart`、`--help` 之类命令行参数。运行控制统一来自 `Code/core/config.py`。

## 运行期目录与正式输出目录

### 运行期目录

运行中间产物全部写到 `Data/proc_Data/pelger_cn_adjusted/` 下：

- `runtime/checkpoints/`
- `runtime/diagnostics/`
- `paper_tail/`

其中：

- `runtime/checkpoints/`：断点续跑状态、rolling 块、paper 年度 checkpoint。
- `runtime/diagnostics/`：`progress.jsonl`、`resource_plan.json`、`stage_timings.json`、样本摘要、滚动诊断、覆盖报告等。
- `paper_tail/`：后半段补充数据规范化产物、校验结果和中间诊断。

### 正式输出目录

正式论文结果只看：

- `Result/pelger_cn_adjusted/figures/`
- `Result/pelger_cn_adjusted/tables/`

目标状态：

- `figures/` 只保留 Figure 1–15 的正式文件。
- `tables/` 只保留 Table I–V 的正式文件。

附加诊断表、滚动底座表、权重表、因子收益摘要不再作为正式论文输出写入 `Result/tables/`。

## 主要脚本说明

| 脚本 | 是否直接读取原始 `.bz2` | 主要输入 | 主要输出 | 典型用途 |
| --- | --- | --- | --- | --- |
| `Code/preprocess_cn_data.py` | 是 | `Data/kline_Data/EXTRA_STOCK_A`、`backward_factor.csv` | `Data/proc_Data/pelger_cn_adjusted` | 预处理、逐股收益缓存、`strict_balanced` 与 `paper_lenient` 面板 |
| `Code/main.py` | 否 | `Data/proc_Data/pelger_cn_adjusted`、`paper_tail` 外部数据 | `Result/pelger_cn_adjusted` 与 `runtime/` | 论文主复现、正式图表输出 |
| `Code/build_mom_5min.py` | 是 | `Data/kline_Data/EXTRA_STOCK_A`、`backward_factor.csv` | `Data/proc_Data/mom_5min` | 5 分钟 MOM 因子构建 |
| `Code/export_panel_csv.py` | 否 | `panels/strict_balanced/<panel>` | `Code/<panel_name>/` CSV | 面板人工抽查 |
| `Code/getApidb.py` | 否 | AmazingData API | 原始 `data.bz2` 目录树 | 上游数据抓取 |

## 验收重点

- 预处理后必须同时具备 `strict_balanced` 与 `paper_lenient`。
- `runtime/checkpoints/run_state.json`、`runtime/diagnostics/progress.jsonl`、`resource_plan.json`、`stage_timings.json` 必须正常生成。
- `plot_export_status.csv` 中 Figure 1–15 应全部为 `generated`。
- `replication_coverage_report.csv` 不应再保留旧占位语义。
- `Result/pelger_cn_adjusted/figures` 不应再混有旧别名图。
- `Result/pelger_cn_adjusted/tables` 不应再混有诊断底座表或兼容别名表。

## 补充说明

- 当前最可信的运行真源是：
  - `Code/main.py`
  - `Code/core/config.py`
  - `Code/preprocess_cn_data.py`
- 如果旧文档、旧截图、旧日志与当前行为冲突，应以当前代码为准。
- `Data/` 目录契约详见 [Data/readme.md](Data/readme.md)。
