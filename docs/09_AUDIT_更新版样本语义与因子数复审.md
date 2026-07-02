·  # 更新版 Pelger 复刻代码深度复审

本报告只做只读复核，不修改任何代码、配置或运行结果。结论以当前仓库 `D:\Reposit\Reposit` 的最新版主入口和运行期诊断文件为准。

## 最新主路径真实语义

### 1. 最新主入口默认已经不是旧的 `strict_balanced`

- 当前主 profile 是 `final_paper_export_resume`，见 [Code/core/config.py](D:\Reposit\Reposit\Code\core\config.py:266)。
- 主路径正式 profile 已切到 `paper_lenient`，见 [Code/core/config.py](D:\Reposit\Reposit\Code\core\config.py:192) 和 [Code/core/config.py](D:\Reposit\Reposit\Code\core\config.py:199)。
- 但几个辅助 profile 仍然默认 `strict_balanced`：
  - `diagnostics_only` 见 [Code/core/config.py](D:\Reposit\Reposit\Code\core\config.py:203)
  - `figures_only` 见 [Code/core/config.py](D:\Reposit\Reposit\Code\core\config.py:209)
  - `tables_only` 见 [Code/core/config.py](D:\Reposit\Reposit\Code\core\config.py:215)

这意味着一句“balanced panel”在当前仓库里已经不再只有一种含义。后续所有审计、画图和表格解释，都必须先问清楚到底说的是哪一种 balanced。

### 2. 当前仓库里同时存在四类样本对象

| 术语 | 真实对象 | 典型规模 | 代码/文件证据 | 主要用途 |
| --- | --- | --- | --- | --- |
| `strict_balanced/full` | 全样本严格交集面板 | 115 只 | `strict_balanced_symbols_full=115`，见 `Data/proc_Data/pelger_cn_adjusted/metadata/universe_summary.json` | 旧语义、辅助 profile、某些历史检查 |
| `paper_lenient/full` | 全样本近似平衡主样本 | 391 只 | `balanced_paper_symbols_full=391`；`main_summary.json` 里 `sample_mode=paper_lenient`、`n_symbols_selected=391` | 最新主流程全样本 pipeline |
| `paper_lenient/year_YYYY` | 逐年宽松平衡面板 | 1827 到 5004 只 | `panels/paper_lenient/year_2013...year_2025`，以及 Table II 年度 `balanced_n_symbols` | Figure 1/2、Table II 的 balanced-year |
| `unbalanced_yearly/year_YYYY` | 逐年非平衡面板 | 2255 到 5465 只 | 年度分析中由 `_build_unbalanced_year_5min_panel(...)` 动态构造 | Figure 1、Table II 的 unbalanced-year |

相关代码入口：

- 全样本加载：`load_proc_hf_panel(... sample_mode=balanced_mode)`，见 [Code/core/engine.py](D:\Reposit\Reposit\Code\core\engine.py:5264)。
- 逐年 balanced-year 加载：`load_proc_5min_panel(... sample_mode=sample_mode, years=[year])`，见 [Code/core/engine.py](D:\Reposit\Reposit\Code\core\engine.py:3643)。
- 逐年 unbalanced-year 构造：`_build_unbalanced_year_5min_panel(...)`，见 [Code/core/engine.py](D:\Reposit\Reposit\Code\core\engine.py:3655)。

### 3. 论文里的 “balanced panel” 和当前主流程里的 “full-sample balanced” 不是一回事

原论文的 Figure 1/2 和 Table II 是逐年比较 balanced 与 unbalanced 的年度面板，不是拿一个全样本固定交集去代表所有图表。原文 PDF 明确说：

- balanced panel 是“全时段交集样本”的概念；
- Figure 1/2 展示的是逐年扰动特征值比；
- Table II 展示的是 balanced / unbalanced 年度因子的广义相关。

当前仓库的对应实现也确实是“逐年逻辑”，不是“拿主 pipeline 的全样本 K 直接作图/作表”：

- Figure 1/2 的数据源来自 `paper_factor_count_diagnostics.csv`，见 [Code/core/engine.py](D:\Reposit\Reposit\Code\core\engine.py:4858) 和 [Code/core/engine.py](D:\Reposit\Reposit\Code\core\engine.py:4859)。
- Table II 的 `balanced_n_symbols` / `unbalanced_n_symbols` 来自年度 `balanced_year.N` / `unbalanced_year.N`，见 [Code/core/engine.py](D:\Reposit\Reposit\Code\core\engine.py:3509) 到 [Code/core/engine.py](D:\Reposit\Reposit\Code\core\engine.py:3512)，以及 [Code/core/engine.py](D:\Reposit\Reposit\Code\core\engine.py:3862) 到 [Code/core/engine.py](D:\Reposit\Reposit\Code\core\engine.py:3865)。

## 因子数真相表

### 1. 最新主流程下，“平衡面板只能提取 1 个系统性因子”不是普遍真相

当前运行期诊断文件 `Data/proc_Data/pelger_cn_adjusted/runtime/diagnostics/main_summary.json` 显示：

- `sample_mode = "paper_lenient"`
- `n_symbols_selected = 391`
- `K_hf_hat = 2`
- `K_cont_hat = 2`
- `K_jump_hat = 2`
- `display_cont_factor_count = 4`
- `g_fn = "median_sqrtN"`

这说明最新版主路径的全样本主样本并没有“只剩 1 个系统性因子”。

`factor_counts_summary.csv` 也一致写明：

- `K_hf_hat=2`
- `K_cont_hat=2`
- `K_jump_hat=2`
- `scope_note=full_sample_pipeline_only`
- `figure_1_2_source=yearly_paper_factor_counts`

### 2. `strict_balanced/full` 也不是“天然只剩 1 个”

本轮只读复核已经确认：

- `strict_balanced/full`：`K_hf=2, K_cont=2, K_jump=1`
- `paper_lenient/full`：`K_hf=2, K_cont=2, K_jump=2`

所以“balanced 面板只能提 1 个因子”至多是某个特定对象、某个特定年份、某个特定收益分量里的局部现象，不是最新版主路径的全局事实。

### 3. 年度 balanced continuous `K_hat` 并不恒等于 1

`paper_factor_count_diagnostics.csv` 的 `Balanced panel + continuous` 年度结果如下：

| year | K_hat |
| --- | ---: |
| 2013 | 3 |
| 2014 | 3 |
| 2015 | 5 |
| 2016 | 3 |
| 2017 | 1 |
| 2018 | 3 |
| 2019 | 3 |
| 2020 | 4 |
| 2021 | 5 |
| 2022 | 5 |
| 2023 | 8 |
| 2024 | 6 |
| 2025 | 6 |

结论很直接：

- 13 个年份里只有 2017 年这一个 balanced continuous 年度点是 `K_hat=1`。
- 所以“平衡面板还是只能提取到 1 个因子”如果来自 Figure 2 或年度诊断，基本属于误读。

### 4. `pca_factors(..., K=1)` 不是这里的 bug

年度因子数诊断里确实写了：

- `hf: pca_factors(..., K=1).eigvals`
- `continuous: pca_factors(..., K=1).eigvals`
- `jump: pca_factors(..., K=1).eigvals`

见 [Code/core/engine.py](D:\Reposit\Reposit\Code\core\engine.py:4320) 到 [Code/core/engine.py](D:\Reposit\Reposit\Code\core\engine.py:4325)。

但 `pca_factors()` 的实现是：

- 先算完整特征值谱 `eigvals`
- 再用 `K` 只截取 `Lambda` 和 `F`

见 [Code/core/engine.py](D:\Reposit\Reposit\Code\core\engine.py:2140) 到 [Code/core/engine.py](D:\Reposit\Reposit\Code\core\engine.py:2148)。

所以这里的 `K=1` 不会把 `eigvals` 截断成只有一个特征值。它不是“因子数全变成 1”的根因。

### 5. 统计上的 `K` 和展示层固定输出 4 个连续因子不是一回事

`main_summary.json` 已经把这两件事分开写了：

- `K_cont_hat` 是统计估计的因子数
- `display_cont_factor_count` 是展示层输出的连续 PCA 因子个数

相关代码见 [Code/core/engine.py](D:\Reposit\Reposit\Code\core\engine.py:4950) 到 [Code/core/engine.py](D:\Reposit\Reposit\Code\core\engine.py:4962)。

因此：

- `display_cont_factor_count=4` 不等于“统计上估出来 4 个因子”
- `K_cont_hat=2` 也不等于“后面所有图只能画 2 条因子”

## 源头-中间诊断-最终图表映射表

| 源头对象 | 中间诊断文件 | 最终图/表 | 当前真实语义 | 关键证据 |
| --- | --- | --- | --- | --- |
| `paper_lenient/full` 全样本主样本 | `main_summary.json` | 主运行摘要、解释主 pipeline K | 全样本 pipeline，不是 Figure 1/2 数据源 | [Code/core/engine.py](D:\Reposit\Reposit\Code\core\engine.py:4947) |
| `paper_lenient/full` 全样本主样本 | `factor_counts_summary.csv` | 因子数汇总 | 只汇总全样本 `pipeline.K_*_hat` | [Code/core/engine.py](D:\Reposit\Reposit\Code\core\engine.py:5006) |
| `paper_lenient/year_YYYY` + `unbalanced_yearly/year_YYYY` | `paper_factor_count_diagnostics.csv` | Figure 1、Figure 2 | 年度 paper-style 因子数诊断 | [Code/core/engine.py](D:\Reposit\Reposit\Code\core\engine.py:5050) |
| 年度 `unbalanced_yearly` | `paper_factor_count_diagnostics.csv` 里 `panel_block=Unbalanced panel` | Figure 1 | 年度非平衡面板 HF/continuous/jump 的 `K_hat` | [Code/figcode/figure_01.py](D:\Reposit\Reposit\Code\figcode\figure_01.py:32) |
| 年度 `paper_lenient/year_YYYY` | `paper_factor_count_diagnostics.csv` 里 `panel_block=Balanced panel` | Figure 2 | 年度平衡面板 HF/continuous/jump 的 `K_hat` | [Code/figcode/figure_02.py](D:\Reposit\Reposit\Code\figcode\figure_02.py:26) |
| 年度 `paper_lenient/year_YYYY` + `unbalanced_yearly/year_YYYY` | `Table_II_balanced_and_unbalanced_panel_results.csv` | Table II | 年度 balanced vs unbalanced 的广义相关对比 | [Code/core/engine.py](D:\Reposit\Reposit\Code\core\engine.py:3509) |

Figure 2 的脚本里还直接写了提醒：

- “不要与全样本 `pipeline.K_hf_hat` 混淆”

见 [Code/figcode/figure_02.py](D:\Reposit\Reposit\Code\figcode\figure_02.py:48)。

这句话本身已经说明：Figure 2 和 `main_summary.json` 里的全样本 K 不是一个对象。

## 最容易犯错的细节清单

### A. 已基本修复的问题

1. 主路径 `g_fn` 已切到 `median_sqrtN`
   - 这和论文扰动特征值比的推荐设定一致，见原文关于 `sqrt(N) * median eigenvalue` 的描述。
2. 主路径 balanced 语义已切到 `paper_lenient`
   - 不再是旧版默认的 `strict_balanced`。
3. `paper_fidelity.py` 已经把多个论文保真项接上
   - full-market size/value
   - clean segmented FFC
   - risk-free split
   - stock-level `carhart_daily`
4. Figure 14/15 的 Panel B 已实现
   - 见 [Code/core/paper_tail.py](D:\Reposit\Reposit\Code\core\paper_tail.py:2129) 到 [Code/core/paper_tail.py](D:\Reposit\Reposit\Code\core\paper_tail.py:2165)。
5. 因子符号定向逻辑已经存在
   - 见 [Code/core/engine.py](D:\Reposit\Reposit\Code\core\engine.py:4550) 和 [Code/core/engine.py](D:\Reposit\Reposit\Code\core\engine.py:4551)。

### B. 仍高风险、会影响经济含义的问题

1. 行业因子如果没有冻结，仍会走占位选择逻辑
   - 冻结模式见 [Code/core/paper_tail.py](D:\Reposit\Reposit\Code\core\paper_tail.py:900)
   - 当前 fallback 明确写着 `PLACEHOLDER`，见 [Code/core/paper_tail.py](D:\Reposit\Reposit\Code\core\paper_tail.py:945)
   - 这会让“行业因子是什么”仍然带有样本内反推色彩，经济解释风险很高。

2. `size_value_start = 2014-07-01` 仍然截断了 size/value 测试资产样本窗
   - 配置见 [Code/core/config.py](D:\Reposit\Reposit\Code\core\config.py:60)
   - 真正执行裁剪见 [Code/core/paper_tail.py](D:\Reposit\Reposit\Code\core\paper_tail.py:1801) 到 [Code/core/paper_tail.py](D:\Reposit\Reposit\Code\core\paper_tail.py:1804)
   - 这会直接影响 Figure 15、相关 alpha、以及所有基于 size/value 资产的定价检验样本长度。

3. Carhart MOM 仍然是近似实现，不是原论文那种数据库直供月频因子
   - 当前默认 `ffc_mom_mode = "carhart_daily"`，见 [Code/core/config.py](D:\Reposit\Reposit\Code\core\config.py:57)
   - 构造逻辑见 [Code/core/paper_tail.py](D:\Reposit\Reposit\Code\core\paper_tail.py:1821) 到 [Code/core/paper_tail.py](D:\Reposit\Reposit\Code\core\paper_tail.py:1844)
   - 这不一定是“错”，但会让和原论文、美股数据库版本的 FFC 结果存在结构性偏差。

4. 辅助 profile 和正式 profile 的样本语义不一致
   - `main.py` 会直接读取当前激活 profile，见 [Code/main.py](D:\Reposit\Reposit\Code\main.py:211) 到 [Code/main.py](D:\Reposit\Reposit\Code\main.py:222)
   - 如果有人切到 `figures_only` / `tables_only` / `diagnostics_only`，很可能重新回到 `strict_balanced`
   - 这会制造“同一仓库、同一脚本名、不同语义输出”的系统性混淆。

### C. 主要是误读风险，不一定是真 bug

1. 把 `strict_balanced/full`、`paper_lenient/full`、`paper_lenient/year_YYYY` 混成一个“balanced”
2. 把全样本 `pipeline.K_*_hat` 和 Figure 1/2 的逐年 `K_hat` 混为一谈
3. 把 `continuous`、`hf`、`jump` 三类收益分量混着看
4. 把 `display_cont_factor_count=4` 误解成“统计上估出来 4 个因子”
5. 看到旧报错文案“当前数据政策仅保留 strict_balanced 面板”就误以为主流程还在用旧样本
   - 这条报错文案确实过时，见 [Code/core/engine.py](D:\Reposit\Reposit\Code\core\engine.py:1501) 和 [Code/core/engine.py](D:\Reposit\Reposit\Code\core\engine.py:1611)
6. 看到旧绝对路径或旧结果目录字符串，就误判当前运行逻辑仍然指向旧语义

## 仍需后续修复的高风险问题

### 1. 行业因子定义尚未完全“论文化”

这是目前最值得优先修的风险点之一。原因不是“代码跑不通”，而是：

- 原论文要求行业组合/行业因子应当有清晰、事前固定的经济定义；
- 当前 fallback 仍然允许按 PCA 权重集中度反推行业；
- 这样做容易把统计结构误当成经济行业结构。

如果后面只允许优先修两件事，这件应排在第一梯队。

### 2. 样本语义混淆仍然会持续制造假结论

当前仓库最大的问题之一不是“数学公式写错”，而是：

- 主路径是 `paper_lenient/full`
- Figure 2 看的是 `paper_lenient/year_YYYY`
- Table II 看的是 `paper_lenient/year_YYYY` vs `unbalanced_yearly/year_YYYY`
- 辅助 profile 又可能回到 `strict_balanced`

如果不先把这些术语和默认入口统一，人很容易对着不同对象得出互相冲突的“审计结论”。

### 3. size/value 与 FFC 的 A 股替代实现仍需明确“近似边界”

当前 A 股复刻里：

- size/value 资产窗口从 2014-07-01 才开始
- MOM 采用日频重建近似
- 与美股原论文数据库因子不完全同口径

这些都不意味着实现无效，但它们必须在最后成文解释里被明确标注成“口径差异”，否则很容易把“中美市场差异”和“实现偏误”混在一起。

## 不要再混用的术语表

| 术语 | 本仓库里应当如何理解 |
| --- | --- |
| full-sample balanced | 指一个全样本固定股票集合上的 full panel，不自动等于论文 Figure 1/2 的年度 balanced panel |
| yearly balanced | 指 `year_YYYY` 的年度平衡面板，是 Figure 2 和 Table II 的 balanced 语义 |
| strict balanced | 指 2013-2025 全时段严格交集样本，当前全样本只有 115 只 |
| paper lenient | 指当前正式主路径使用的宽松平衡语义；全样本 391 只，逐年还能扩展到更大年度 balanced 面板 |
| display factors | 为展示和作图保留的因子数，不等于统计估计的 `K` |
| statistical K | 扰动特征值比或相关统计准则估出来的因子数，和展示层输出条数不是一回事 |

## 最终审计结论

### 1. 最新版主流程下，平衡面板是否真的“只能提取 1 个系统性因子”

不是。

最新版主流程的主样本是 `paper_lenient/full`，其运行期诊断明确给出：

- `K_hf_hat=2`
- `K_cont_hat=2`
- `K_jump_hat=2`

而且年度 balanced continuous 的 `K_hat` 也只有 2017 年这一个点等于 1，不存在“整个平衡面板始终只提到 1 个”的现象。

### 2. 如果不是，哪个文件/图/表最容易让人误判成“只有 1 个”

最容易误判的对象有三个：

1. `paper_factor_count_diagnostics.csv`
   - 因为这是年度表，里面某一年某个分量确实可能出现 `K_hat=1`
   - 例如 2017 年 `Balanced panel / continuous / K_hat = 1`

2. Figure 2
   - 因为它用的是年度 balanced 行，而不是全样本主 pipeline 的 `K_hf_hat`
   - 如果只盯住某一年，或者把 balanced-year 当成 full-sample balanced，就很容易误判

3. 辅助 profile 重新跑出的结果
   - 因为 `figures_only` / `tables_only` / `diagnostics_only` 仍默认 `strict_balanced`
   - 这样用户可能以为自己还在看“最新版主路径”，实际却切回了旧语义

### 3. 当前仓库里还剩下哪些真正会导致论文经济含义偏掉的系统性风险点

最核心的三类是：

1. 行业因子未冻结时仍带占位选择逻辑
2. size/value 测试资产样本从 2014-07-01 起步，存在窗截断
3. Carhart MOM 与 FFC 的 A 股替代实现是近似口径，不是原论文数据库同口径复刻

相比之下，“平衡面板只能提 1 个因子”更像是样本语义混淆和文件误读，不是当前主流程最本质的问题。

## 给后续修改的建议顺序

如果后续真的要改代码，建议优先顺序如下：

1. 先统一样本语义和默认 profile 文案
2. 再冻结行业因子定义
3. 再审视 size/value 与 MOM/FFC 的口径边界
4. 最后才处理图表文案、README 和错误提示的过时措辞

否则很容易出现“图修漂亮了，但经济含义还是漂着”的情况。
