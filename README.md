# A 股高频系统性风险复现项目

本项目基于 Pelger (2020) 高频 PCA 系统性风险框架，使用中国 A 股 5 分钟数据做适配复现。当前版本采用“交稿版精简口径”：重点保证方法链路清楚、核心图表可复现、目录结构易读易维护，而不是逐图逐表 1:1 复刻美股原文数值。

当前仓库唯一顶层运行入口是：

```bash
python Code/main.py
```

无参数运行时读取 `Code/config.yaml`；命令行显式参数优先于 YAML。旧的 `export_submission_*_fast.py`、顶层 `preprocess_cn_data.py`、顶层 `build_mom_5min.py` 等入口已移除，相关能力由 `main.py` 统一调度。

## 交稿版结果范围

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

指定配置文件：

```bash
python Code/main.py --config Code/config.yaml
```

## 配置规则

`Code/config.yaml` 是用户日常修改的配置文件，主要控制：

- `stages`: 运行 `data`、`figures`、`tables` 中哪些阶段。
- `figures`: 要生成哪些交稿图。
- `tables`: 要生成哪些交稿表。
- `data_steps`: 要执行哪些数据步骤。
- `paths`: 处理后数据、输出目录、外部数据、RF 文件路径。
- `run`: fail-fast、是否刷新 paper_tail、worker、内存预算等运行编排参数。
- `data`: 预处理和 MOM 因子的步骤参数。

命令行显式传入的参数覆盖 YAML。例如：

```bash
python Code/main.py --stages figures --figures fig13 --output-root Result/pelger_cn_adjusted
```

核心 PCA 和缓存签名参数仍由 `Code/core/config.py` 的 `RunConfig` 做内部校验；一般不需要直接改 Python 配置。

## 目录结构

```text
Reposit/
├─ Code/
│  ├─ main.py
│  ├─ config.yaml
│  ├─ core/
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
│  └─ pelger_cn_adjusted/
├─ docs/
├─ requirements.txt
└─ README.md
```

`Code/core/` 存放核心数据结构、PCA 引擎、缓存、paper_tail 资产逻辑、任务注册表和轻量交稿结果构造。

`Code/dataPrepare/` 存放数据准备步骤：

- `step_00_get_apidb.py`: 外部 API 原始数据抓取工具。
- `step_01_preprocess_panels.py`: 从 raw `.bz2` 和复权因子生成 `strict_balanced` / `paper_lenient` 面板。
- `step_02_build_mom_5min.py`: 构建 5 分钟 MOM 因子。

`Code/figCode/` 存放交稿版保留图的渲染代码。共享绘图 helper 放在 `_weights.py`、`_timevar.py` 等内部模块中。

`Code/tableCode/` 存放交稿版保留表的导出代码。每张核心表对应一个文件：`table_i.py`、`table_ii.py`、`table_iii.py`、`table_v.py`。

`Code/devTools/` 存放开发调试工具，例如面板 CSV 导出工具。这里的脚本不属于交稿主流程。

## 数据流

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
Result/pelger_cn_adjusted/figures/
```

最终表：

```text
Result/pelger_cn_adjusted/tables/
```

轻量运行诊断：

```text
Data/proc_Data/pelger_cn_adjusted/runtime/submission_fast/diagnostics/
```

运行成功后重点检查：

- `export_summary.json` 中 `failures` 为空。
- `figures/` 中有 9 张交稿图。
- `tables/` 中有 Table I、II、III、V。
- 控制台不应出现完整 `paper_tables` 重跑。

## 依赖环境

安装公开依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 不包含 `step_00_get_apidb.py` 可能需要的专有 SDK/API 依赖；如需抓取原始数据，应按数据服务方说明单独配置。

## 提交检查清单

- `python Code/main.py --list` 能列出 9 个 figure 任务、4 个 table 任务和 3 个 data step。
- `python Code/main.py --stages figures` 能生成 9 张图。
- `python Code/main.py --stages tables` 能生成 4 张表。
- `Code/` 顶层只有 `main.py`、`config.yaml` 和子目录。
- `Data/external_Data/pelger_tail/factors/rf/risk_free.csv` 存在。
- README、正文和图表 caption 中的样本口径一致。
- 大体量原始 `Data/` 是否提交，按课程或仓库管理要求单独决定。

