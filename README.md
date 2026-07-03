# A 股高频系统性风险复现项目

本项目基于 Pelger (2020) 高频 PCA 系统性风险研究框架，使用中国 A 股 5 分钟数据做适配复现。当前版本的目标不是逐图逐表 1:1 复刻美股原文数值，而是在原文方法论下完成一套可解释、可运行、可交稿的 A 股实证：提取高频系统性因子，区分连续与跳跃成分，考察因子结构稳定性，并用行业组合、规模-价值组合等测试资产展示定价含义。

当前 README 以“交稿版精简链路”为准。如果旧 `docs/`、旧输出文件或历史说明与本文冲突，以当前代码和本 README 为准。

## 当前交稿版范围

正文默认保留 9 张图：

- `Figure 1`: 跨年股票池变化、年内平衡的年度高频因子个数。
- `Figure 2`: 全样本固定股票交集平衡面板的年度高频因子个数。
- `Figure 4`: 连续 PCA 因子组合权重与行业结构。
- `Figure 7`: 局部窗口因子权重与全样本权重的广义相关性。
- `Figure 10`: 因子结构时间变化分解。
- `Figure 12`: 日内、隔夜、日频期望收益关系。
- `Figure 13`: 连续 PCA、年度对齐 PCA、FFC 因子累计收益。
- `Figure 14`: 行业组合资产定价。
- `Figure 15`: 规模-价值 2x3 组合资产定价。

正文默认保留 4 张表：

- `Table I`: 连续收益与跳跃收益的描述统计，是必须完善的核心表。
- `Table II`: 平衡/非平衡面板 PCA 结果，是必须完善的核心表。
- `Table III`: PCA 因子与行业、FFC 因子的经济含义对照。
- `Table V`: 不同因子组的日内、隔夜、日频 Sharpe ratio 对照。

`Figure 3, 5, 6, 8, 9, 11` 和 `Table IV` 不进入当前交稿版核心正文。它们可以作为后续扩展或附录思路保留，但不作为明天交稿的必要产物。

## 快速运行

请在仓库根目录运行命令。交稿版图表不要默认走 `Code/main.py`，而是使用轻量入口：

```bash
python Code/export_submission_core_fast.py
```

生成核心表格：

```bash
python Code/export_submission_table_i_fast.py
python Code/export_submission_table_ii_fast.py
python Code/export_submission_table_iii_fast.py
python Code/export_submission_table_v_fast.py
```

可选地把输出写到自定义目录：

```bash
python Code/export_submission_core_fast.py --output-root Result/pelger_cn_adjusted
python Code/export_submission_table_i_fast.py --output-root Result/pelger_cn_adjusted
```

当前图表入口默认严格失败；如果某张图失败，应先查看诊断文件，而不是让脚本静默降级成错误口径。

## 输出位置

最终图表输出：

```text
Result/pelger_cn_adjusted/figures/
```

最终表格输出：

```text
Result/pelger_cn_adjusted/tables/
```

轻量交稿链路诊断：

```text
Data/proc_Data/pelger_cn_adjusted/runtime/submission_fast/diagnostics/
```

图表运行成功时，重点检查：

- `Data/proc_Data/pelger_cn_adjusted/runtime/submission_fast/diagnostics/export_summary.json` 中 `failures` 为空。
- `Result/pelger_cn_adjusted/figures/` 中有 9 张交稿图。
- `Result/pelger_cn_adjusted/tables/` 中有 Table I、II、III、V 对应 CSV。

当前已知交稿图文件名如下：

```text
Figure_1_number_of_hf_factors_unbalanced.png
Figure_2_number_of_hf_factors_balanced.png
Figure_4_continuous_pca_factor_portfolio_weights.png
Figure_7_locally_estimated_continuous_factors.png
Figure_10_factor_structure_time_variation_decomposition.png
Figure_12_expected_intraday_and_overnight_returns.png
Figure_13_cumulative_factor_returns.png
Figure_14_asset_pricing_of_industry_portfolios.png
Figure_15_asset_pricing_of_size_and_value_sorted_portfolios.png
```

当前已知交稿表文件名如下：

```text
Table_I_summary_statistics_for_continuous_and_jump_returns_aligned.csv
Table_II_balanced_and_unbalanced_panel_results_paper_style_fixed_k.csv
Table_III_generalized_correlations_with_industry_and_ffc_factors.csv
Table_V_intraday_overnight_daily_sharpe_ratios.csv
```

## 目录结构

```text
Reposit/
├─ Code/
│  ├─ export_submission_core_fast.py
│  ├─ export_submission_table_i_fast.py
│  ├─ export_submission_table_ii_fast.py
│  ├─ export_submission_table_iii_fast.py
│  ├─ export_submission_table_v_fast.py
│  ├─ main.py
│  ├─ preprocess_cn_data.py
│  ├─ build_mom_5min.py
│  ├─ core/
│  ├─ figcode/
│  └─ tablecode/
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

`Code/` 存放所有运行入口、核心引擎、图表脚本和表格脚本。交稿版默认入口是 `export_submission_core_fast.py` 和四个 `export_submission_table_*_fast.py`。

`Data/` 存放原始行情、复权因子、外部补充数据、预处理面板、运行期缓存和轻量链路诊断。真实大数据不建议直接提交到普通代码仓库。

`Result/` 存放最终论文图表输出。交稿时主要检查 `Result/pelger_cn_adjusted/figures/` 和 `Result/pelger_cn_adjusted/tables/`。

`docs/` 存放历史修复记录、审计说明和写作辅助材料。由于项目经过多轮重构，历史文档可能保留旧口径；发生冲突时，以当前代码和本 README 为准。

如果仓库中存在 `data_small/`，它应被理解为结构化小样本，用于 I/O smoke test、代码演示和数据交接；正式实证仍应替换为完整 `Data/`。

## 数据流

原始数据入口：

```text
Data/kline_Data/EXTRA_STOCK_A/<symbol>/data.bz2
Data/fact_Data/backward_factor.csv
Data/external_Data/pelger_tail/**
```

预处理链路：

```text
raw 5-minute bars
  + backward_factor.csv
  -> Code/preprocess_cn_data.py
  -> Data/proc_Data/pelger_cn_adjusted/
```

预处理后最重要的面板目录：

```text
Data/proc_Data/pelger_cn_adjusted/panels/strict_balanced/
Data/proc_Data/pelger_cn_adjusted/panels/paper_lenient/
Data/proc_Data/pelger_cn_adjusted/paper_tail/
```

交稿轻量链路不会从原始行情重新构造所有中间产物。它读取已有 `proc_Data`：

- 从 `strict_balanced/full` 构造主样本 PCA。
- 从 `strict_balanced/year_YYYY` 构造 Figure 1 年度面板诊断。
- 从 `strict_balanced/full` 的固定股票交集按年份切片构造 Figure 2 诊断。
- 从已有 `paper_tail` 复用行业组合、规模-价值组合等测试资产原材料。
- 只在 `runtime/submission_fast/diagnostics/` 写轻量中间结果，不触发完整 `paper_tables` 重跑。

## 样本口径

`strict_balanced` 是当前交稿图表的主口径。它强调干净、可解释、快速可复现，适合明天交稿的核心实证。

`paper_lenient` 是更接近原论文“尽量保留样本”的完整链路口径，适合后续全量重建和稳健性扩展。当前交稿图表不默认以它作为主输入。

Figure 1 的定义：

```text
读取 strict_balanced/year_YYYY
每一年内部先构造平衡面板
不同年份允许股票池 N 变化
```

这张图里的“非平衡”指跨年股票池变化，不指年内允许缺失。

Figure 2 的定义：

```text
读取 strict_balanced/full
取全样本固定股票交集
再按年份切片逐年做 PCA
所有年份股票数 N 必须相同
```

Figure 4、7、10、13、14、15 使用轻量 strict 主样本 PCA 作为连续因子基准。Figure 12 复用已有 `paper_tail` 描述性资产散点数据，只使用当前新版渲染层。Figure 13、14、15 会基于 strict 连续 PCA 重建交稿版中间数据，不继续沿用旧 `paper_lenient` 定价结果。

## 主要代码文件

| 文件 | 职责 |
| --- | --- |
| `Code/export_submission_core_fast.py` | 交稿 9 张图的轻量导出入口；只读已有 `proc_Data`，不跑完整 `paper_tables`。 |
| `Code/export_submission_table_i_fast.py` | 导出 Table I，服务连续/跳跃收益描述统计。 |
| `Code/export_submission_table_ii_fast.py` | 导出 Table II，服务平衡/非平衡 PCA 核心结果对比。 |
| `Code/export_submission_table_iii_fast.py` | 导出 Table III，服务 PCA 因子经济含义解释。 |
| `Code/export_submission_table_v_fast.py` | 导出 Table V，服务因子收益 Sharpe ratio 对照。 |
| `Code/preprocess_cn_data.py` | 从原始 `.bz2` 行情和复权因子生成 `strict_balanced` / `paper_lenient` 面板。 |
| `Code/build_mom_5min.py` | 构建 5 分钟 MOM 因子，供扩展分析使用。 |
| `Code/main.py` | 完整框架/历史全链路入口；配置来自 `Code/core/config.py`，当前不作为交稿轻量图表默认入口。 |
| `Code/core/config.py` | 路径、profile、样本模式、输出根目录和运行参数的配置真源。 |
| `Code/core/engine.py` | 高频面板、PCA、连续/跳跃分解、因子个数诊断和完整 replication 引擎。 |
| `Code/core/submission_fast.py` | 交稿轻量链路核心：strict 最小 PCA、Figure 1/2 诊断、Table fast 构造、paper_tail 资产复用。 |
| `Code/core/paper_tail.py` | Figure 12-15、Table III/V 相关测试资产、FFC、行业组合、规模-价值组合逻辑。 |
| `Code/figcode/` | 单图渲染层，尽量只处理图形表达，不改变底层统计口径。 |
| `Code/tablecode/` | 单表导出层，尽量只处理表格组织和字段排版。 |
| `Code/core/pipeline_cache.py` | ReplicationResult 缓存识别与复用逻辑，主要服务完整链路。 |
| `Code/core/logging_utils.py` | 控制台日志格式与运行提示。 |

## 从零重建流程

交稿阶段通常不需要从零重建。只有在原始行情、复权因子或预处理口径变化时，才建议执行本节。

安装依赖：

```bash
pip install -r requirements.txt
```

重建 5 分钟 MOM：

```bash
python Code/build_mom_5min.py --raw-root Data\kline_Data\EXTRA_STOCK_A --factor-path Data\fact_Data\backward_factor.csv --proc-root Data\proc_Data\mom_5min --lookback-bars 48 --skip-bars 1 --winner-pct 0.3 --loser-pct 0.3 --min-stocks 5 --workers 8
```

重建高频预处理面板：

```bash
python Code/preprocess_cn_data.py --raw-root Data\kline_Data\EXTRA_STOCK_A --factor-path Data\fact_Data\backward_factor.csv --proc-root Data\proc_Data\pelger_cn_adjusted --refresh --workers 8 --panel-workers 8
```

预处理完成后应至少看到：

```text
Data/proc_Data/pelger_cn_adjusted/panels/strict_balanced/
Data/proc_Data/pelger_cn_adjusted/panels/paper_lenient/
Data/proc_Data/pelger_cn_adjusted/manifest.json
```

完整 `main.py` 链路只作为扩展和全量重建入口：

```bash
python Code/main.py
```

注意：`main.py` 不接受 `--only`、`--restart` 等 CLI 参数，具体行为由 `Code/core/config.py` 中的 profile 决定。不要在交稿图表修正阶段随意切到 full rebuild profile，否则可能触发长时间完整重跑。

## 常见问题

### 为什么交稿图表不推荐运行 `Code/main.py`？

`main.py` 是完整框架入口，服务长期全链路复现、缓存复用和完整 paper_tail 生成。它功能更全，但也更重，容易进入 `paper_tables` 等长耗时阶段。交稿版只需要基于已有 `proc_Data` 生成核心图表，因此使用 `Code/export_submission_core_fast.py` 更安全。

### 为什么 Figure 1 和 Figure 2 的平衡定义要拆开？

原论文关注平衡/非平衡样本下因子数的稳健性。A 股数据里如果直接把“年内缺失”和“跨年股票池变化”混在一起，图的解释会变脏。当前口径把它拆清楚：

- Figure 1 表示跨年非平衡，但每年内部平衡。
- Figure 2 表示全样本固定股票交集，所有年份 N 恒定。

### 为什么 Figure 12 可以复用 `paper_tail`，但 Figure 13/14/15 要重建 strict 输入？

Figure 12 主要是描述性资产散点和收益关系展示，复用已有测试资产原材料即可。Figure 13/14/15 涉及连续 PCA 因子收益、资产定价预测和 Sharpe ratio 对照，必须和当前交稿主 PCA 口径一致，所以需要用 strict 连续 PCA 重建中间数据。

### 为什么 Table IV 可删减？

原论文 Table IV 更偏稳健性和扩展验证。当前课程交稿最需要支撑的方法链条是：样本描述与收益分解、平衡/非平衡 PCA 核心结果、经济含义解释、因子收益表现。Table I、II、III、V 已覆盖这条主线。Table IV 可以后续补成附录，但不是当前核心结论的必要支柱。

### 如果缺少 `proc_Data` 怎么办？

先运行 `Code/preprocess_cn_data.py` 生成面板。交稿轻量入口不会自动从原始行情补齐缺失面板，因为那样容易把一次图表导出变成不可控的全量重跑。

## 提交检查清单

交稿前建议逐项检查：

- `python Code/export_submission_core_fast.py` 已成功运行。
- `python Code/export_submission_table_i_fast.py` 已成功运行。
- `python Code/export_submission_table_ii_fast.py` 已成功运行。
- `python Code/export_submission_table_iii_fast.py` 已成功运行。
- `python Code/export_submission_table_v_fast.py` 已成功运行。
- `Result/pelger_cn_adjusted/figures/` 中有 9 张交稿图。
- `Result/pelger_cn_adjusted/tables/` 中有 Table I、II、III、V。
- `Data/proc_Data/pelger_cn_adjusted/runtime/submission_fast/diagnostics/export_summary.json` 中 `failures` 为空。
- README、正文和图表标题中的样本口径一致。
- 不把 `main.py` 描述成交稿轻量图表默认入口。
- 大体量原始 `Data/` 是否提交，应按课程或仓库管理要求单独决定。

