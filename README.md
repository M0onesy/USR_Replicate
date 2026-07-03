# A 股高频系统性风险复现项目

本项目基于 Pelger 高频 PCA 系统性风险框架，使用中国 A 股 5 分钟数据做适配复现。当前仓库采用“交稿版精简口径”：重点保证方法链路清楚、核心图表可复现、目录结构易维护，而不是逐图逐表 1:1 复刻美股原文数值。

当前唯一顶层运行入口是：

```bash
python Code/main.py
```

无参数运行时读取 `Code/config.yaml`；命令行显式参数优先于 YAML。旧的 `export_submission_*_fast.py`、顶层 `preprocess_cn_data.py`、顶层 `build_mom_5min.py` 等入口已移除，相关能力由 `main.py` 统一调度。

## 交稿结果范围

正文默认保留 9 张图：

- `fig1`: Figure 1，跨年股票池变化、年内平衡的年度高频因子个数。
- `fig2`: Figure 2，全样本固定股票交集平衡面板的年度高频因子个数。
- `fig4`: Figure 4，连续 PCA 因子组合权重与行业结构。
- `fig7`: Figure 7，局部窗口因子权重与全样本权重的广义相关性。
- `fig10`: Figure 10，因子结构时间变化分解。
- `fig12`: Figure 12，日内、隔夜、日频期望收益关系。
- `fig13`: Figure 13，连续 PCA、年度对齐 PCA、FFC 因子累计收益。
- `fig14`: Figure 14，行业组合资产定价。
- `fig15`: Figure 15，规模-价值 2x3 组合资产定价。

正文默认保留 4 张表：

- `table_i`: Table I，连续收益与跳跃收益描述统计。
- `table_ii`: Table II，平衡/非平衡面板 PCA 结果。
- `table_iii`: Table III，PCA 因子与行业、FFC 因子的经济含义对照。
- `table_v`: Table V，不同因子组的日内、隔夜、日频 Sharpe ratio 对照。

Figure 3/5/6/8/9/11、Table IV 和旧诊断表任务已从交稿版注册表中删除，不再作为默认可运行任务维护。

## 快速运行

在仓库根目录运行一键交稿流程：

```bash
python Code/main.py
```

只生成图：

```bash
python Code/main.py --stages figures
```

只生成表：

```bash
python Code/main.py --stages tables
```

生成指定图或表：

```bash
python Code/main.py --figures fig1,fig2,fig4
python Code/main.py --tables table_i,table_ii
```

查看所有可运行任务：

```bash
python Code/main.py --list
```

执行数据预处理步骤：

```bash
python Code/main.py --stages data --data-steps preprocess_panels
python Code/main.py --stages data --data-steps mom_5min --workers 1
python Code/main.py --all
```

指定配置文件或输出目录：

```bash
python Code/main.py --config Code/config.yaml
python Code/main.py --stages figures --figures fig13 --output-root Result
```

## 配置规则

`Code/config.yaml` 是用户日常修改的运行编排配置，主要控制：

- `stages`: 运行 `data`、`figures`、`tables` 中哪些阶段。
- `figures`: 生成哪些交稿图。
- `tables`: 生成哪些交稿表。
- `data_steps`: 执行哪些数据步骤。
- `paths`: 处理后数据、输出目录、外部数据、RF 文件路径。
- `run`: fail-fast、是否刷新 paper_tail、worker、内存预算等运行参数。
- `data`: 预处理和 MOM 因子的步骤参数。

核心 PCA 和缓存签名参数仍由 `Code/prepareCore/config.py` 的 `RunConfig` 做内部校验；一般不需要直接改 Python 配置。

## 目录结构

```text
Reposit/
├─ Code/
│  ├─ main.py
│  ├─ config.yaml
│  ├─ prepareCore/
│  ├─ dataPrepare/
│  ├─ figCode/
│  ├─ tableCode/
│  └─ devTools/
├─ Data/
│  ├─ kline_Data/
│  ├─ fact_Data/
│  ├─ external_Data/
│  └─ proc_Data/
├─ Result/
│  ├─ figures/
│  └─ tables/
├─ requirements.txt
└─ README.md
```

`Code/prepareCore/` 存放核心数据结构、PCA 引擎、缓存、paper_tail 资产逻辑、任务注册表和轻量交稿结果构造。`main.py` 中保留了 `core -> prepareCore` 的临时兼容别名，仅用于降低旧 pickle/缓存反序列化风险；新代码应统一使用 `prepareCore`。

`Code/dataPrepare/` 存放数据准备步骤：

- `step0_get_apidb.py`: 外部 API 原始数据抓取工具。
- `step1_preprocess_panels.py`: 从 raw `.bz2` 和复权因子生成 `strict_balanced` / `paper_lenient` 面板。
- `step2_build_mom_5min.py`: 构建 5 分钟 MOM 因子。

`Code/figCode/` 存放交稿版保留图的渲染代码。共享绘图 helper 放在 `_weights.py`、`_timevar.py` 等内部模块中。

`Code/tableCode/` 存放交稿版保留表的导出代码。每张核心表对应一个文件：`table_i.py`、`table_ii.py`、`table_iii.py`、`table_v.py`。

`Code/devTools/` 存放开发调试工具，不属于交稿主流程。

历史 `docs/` 目录不作为当前交付必备目录；若以后恢复历史修复记录，以当前 README 和当前代码为准。

## 数据流与样本口径

原始数据入口：

```text
Data/kline_Data/EXTRA_STOCK_A/<symbol>/data.bz2
Data/fact_Data/backward_factor.csv
Data/external_Data/pelger_tail/**
```

预处理产物：

```text
Data/proc_Data/pelger_cn_adjusted/panels/strict_balanced/
Data/proc_Data/pelger_cn_adjusted/panels/paper_lenient/
Data/proc_Data/pelger_cn_adjusted/paper_tail/
Data/proc_Data/mom_5min/mom_factor_5min.csv
```

交稿轻量链路读取已有 `proc_Data`：

- Figure 1 读取 `strict_balanced/year_YYYY`，每年内部平衡、跨年股票池可变。
- Figure 2 读取 `strict_balanced/full`，全样本固定股票交集，逐年切片后 N 恒定。
- Figure 4/7/10/13/14/15 使用 strict 主样本连续 PCA。
- Figure 12 复用已有 `paper_tail` 测试资产数据，只走当前渲染层。
- Table I/II/III/V 使用 `tableCode` 中的交稿版 fast 构造逻辑。

## 无风险利率

当前唯一默认 RF 文件位置是：

```text
Data/external_Data/pelger_tail/factors/rf/risk_free.csv
```

程序不再依赖仓库根目录下的 `无风险利率/` 文件夹。若需要临时指定其他 RF 文件，可使用：

```bash
python Code/main.py --rf-file path/to/risk_free.csv
```

## 输出位置

最终图：

```text
Result/figures/
```

最终表：

```text
Result/tables/
```

轻量运行诊断：

```text
Data/proc_Data/pelger_cn_adjusted/runtime/submission_fast/diagnostics/
```

运行成功后重点检查：

- `export_summary.json` 中 `failures` 为空。
- `Result/figures/` 中有 9 张交稿图。
- `Result/tables/` 中有 Table I、II、III、V。
- 控制台不应进入完整 `paper_tables` 重跑。

## devTools 调试工具

推荐在交稿前运行：

```bash
python Code/devTools/smoke_imports.py
python Code/devTools/inspect_config.py
python Code/devTools/check_project_structure.py
python Code/devTools/check_submission_outputs.py
```

工具用途：

- `smoke_imports.py`: 检查 9 图 4 表注册任务是否都能导入。
- `inspect_config.py`: 打印 YAML 与 CLI 合并后的实际运行配置。
- `check_project_structure.py`: 检查当前目录结构、RF 文件和旧 `Code/core` 目录是否符合新架构。
- `check_submission_outputs.py`: 检查 `Result/figures` 与 `Result/tables` 中的交稿产物是否齐全且非空。
- `export_panel_csv.py`: 面板抽样导出工具，仅用于调试和数据交接。

## 依赖环境

安装公开依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 不包含 `step0_get_apidb.py` 可能需要的专有 SDK/API 依赖；如需抓取原始数据，应按数据服务方说明单独配置。

## 最终提交检查清单

- `python -m py_compile` 覆盖 `Code/**/*.py` 后无语法错误。
- `python Code/main.py --list` 能列出 9 个 figure 任务、4 个 table 任务和 3 个 data step。
- `python Code/main.py --no-fail-fast` 能生成 9 图 4 表，且 `failures=[]`。
- `python Code/devTools/check_submission_outputs.py` 全部为 `[OK]`。
- `Code/` 顶层只有 `main.py`、`config.yaml` 和子目录。
- `Data/external_Data/pelger_tail/factors/rf/risk_free.csv` 存在。
- README、正文和图表 caption 中的样本口径一致。
- 大体量原始 `Data/` 是否提交，按课程或仓库管理要求单独决定。
