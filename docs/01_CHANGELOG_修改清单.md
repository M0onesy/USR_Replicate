# 修改清单（CHANGELOG）—— 逐文件 · 逐条（P/N 编号）

> 标注 [复用] = 复用路径即可生效；[完整] = 需重跑预处理/全量 PCA 才生效。

## core/config.py
- **新增视图层论文保真开关**（均**不进 `cache_signature`**，故不影响重结果缓存命中）：
  `paper_faithful_signs`(P4)、`industry_factors_frozen`(P5/D1)、`ffc_mom_mode`(P6)、
  `size_value_full_market`(P9)、`annualization_days`(N6)、`size_value_start`(P10)、
  `industry_info_filename`(新映射)。
- **新增 `export_fidelity_env()`**：把上述开关写入 env，供 engine/paper_tail 读取；
  在 `pipeline_cache.build_result` 与 `_refresh_result` 入口自动调用。
- `g_fn` 字段加注释说明“复用=median_N / 论文=median_sqrtN（需重跑）”。**默认仍 median_N 以保证复用**（P1 属完整路径）。

## core/engine.py
- **新增 `orient_pca_result()`**（P4）：确定性符号定向——因子1=正等权市场（平均载荷为正），
  因子2..K=绝对值最大载荷为正。原地翻 Lambda 与 F。
- **新增 `_paper_faithful_signs_enabled()`**：env `PELGER_PAPER_FAITHFUL_SIGNS`（默认开）。
- **`refresh_replication_result_views()` 注入符号定向** [复用]：在重算 `pca_cont_display`
  之后、计算 `W_display`/display 因子收益之前，对 `pca_cont` 与 `pca_cont_display` 定向。
  ⇒ 复用既有 pickle 时无需重跑 PCA，即可修好 Table_11/12/14、Figure 3/4/13、Table III/V 的符号。
- （注）display_k 本就固定=4（`DISPLAY_CONTINUOUS_FACTOR_COUNT`），即 paper_tail 一直用 4 因子；
  P3 的“估1画4”静默矛盾在完整路径下通过 g_fn 修复后自然消解（K̂→3~4）。

## preprocess_cn_data.py  [完整]
- **N1 纯盘中 5min**：`full_5min[0]` 改为 `log(close[0]/open[0])`（当日开盘→首收），
  隔夜单独存于 `overnight`；`sum(48根)=intraday`（不再含隔夜）。env `PELGER_INTRADAY_ONLY_5MIN`（默认 1）。
- **P2 论文宽松平衡面板**：新增 `is_balanced_paper`（每年覆盖率 ≥ 阈值 + 13 年都在），
  近似论文“缺失阈值 + 插值”口径；`_select_sample_rows(years=None)` 在
  `PELGER_BALANCED_MODE=paper_lenient` 时用它。env `PELGER_BALANCED_MIN_COVERAGE`（默认 0.96）。
- 口径切换会 bump `_effective_return_scheme()`，避免新旧 proc_data 混用。
- ⚠️ 完整路径：切换 N1/P2 会改变 proc_data 与逐年面板 → 重结果缓存失效 → 需 `restart=True` 重跑（含 ~33h 逐年分析）。
- ⚠️ 已知未尽：`is_balanced_paper` 选出“可用宇宙”后，**面板数组构建器仍需对缺失日插值**（增量为0）才能真正喂 PCA；该插值步骤见 SPEC 的 P2 小节（本包未改面板写出逻辑，标注为待补）。

## core/paper_tail.py  [复用]
- 顶部新增 **env 驱动的论文保真选项读取器**（`_industry_info_filename / _industry_frozen /
  _annualization_days / _ffc_mom_mode / _size_value_full_market / _size_value_start`）。
- **PAPER_TAIL_VERSION 3→4、ALGORITHM_VERSION→paper_faithful_v2**：触发尾部缓存重建。
- **新映射接入**：`_discovered_paths["industry_info"]` 用 `_industry_info_filename()`。
- **P5/D1 行业冻结**：`_select_industries_from_pca` 优先用 `PELGER_INDUSTRY_FROZEN` 给的桶，
  否则退回“按集中度自动挑选”的**占位**逻辑并在 selection_rule 标注 PLACEHOLDER。
- **N7 动态标签**：GROUP_TITLES 用 `"{n_industry} Industry Portfolios"` 动态填充（新映射 11 桶）。
- **N4 Figure 12 轴**：散点改 x=盘中、y=隔夜（论文方向）。
- **P13a Figure 12 n_obs**：industry / size-value 用真实有效观测数（按 daily 段非缺失计）。
- **P11 Figure 13 归一化**：三段统一用该因子**日频 std**；前导缺失不当 0（对齐首个有效日再累计）。
- **N6 年化**：tangency 夏普用 `_annualization_days()`（论文 252，可设 243）。
- **N10 Table V**：下半部补 Market/Size/Value/Momentum 个体夏普（取自修好的分段 FFC）。
- **P12 Figure 14/15**：渲染改为 Panel A（散点）+ **Panel B（各资产 alpha 柱状）**，3 段 × 2 模型。
- **P9 / P6 / P7 接入 `_build_payload`**（带 try/except 回退旧实现）：
  - P9：`build_full_market_size_value`（全市场 2×3）替换平衡子集 2×3；按 `size_value_start` 裁剪。
  - P6+P7+P8：`build_full_market_momentum` + `build_ffc_segmented_clean`（股票级分段 FFC，rf 拆分，
    daily=intra+night，删残差强制）。
  - **N8** 校验重定位：size_value/ffc 校验摘要加 `note`，说明已是“差异报告”而非正确性证明。
- **N9 Table III 补全**（best-effort、逐项 try/except）：上半部加 **HF PCA / PCA Proxy** 行；
  下半部加 **ω HF/Jump/Overnight/Daily/Week/Month 的载荷 GC**（对应频率 PCA 载荷 vs 连续 display 载荷）。
  输出新增 `panel` 列区分 top_returns_gc / bottom_loadings_gc。

## core/paper_fidelity.py（新增模块）  [复用]
- `build_full_market_size_value`（P9）、`build_full_market_momentum`（P6）、
  `build_ffc_segmented_clean`（P7）、`split_daily_rf`（P8）。
- 复用 `_build_full_market_assets` 同款“遍历 symbol_returns 累计三段组合收益”的稳健模式。

## core/pipeline_cache.py
- `build_result` 与 `_refresh_result` 入口调用 `cfg.export_fidelity_env()`。

## 未在本包改动、保持原状（核对后确认非 bug）
- GC 公式（`(G'G)⁻¹(G'F)(F'F)⁻¹(F'G)` 特征值，与论文标准典则相关一致）。
- TOD 阈值（CN 用 1/48 ↔ 论文 1/77，口径一致）；个体夏普归一化（footnote 33，mean/std×√ann 正确）。
- proxy 比例（1/0.15/0.11/0.11）；FF3=P9714 综合 A 股；rf=隔夜 shibor 折日。

## 图表 1:1 复刻补丁（figcode/ 与 engine 出图层）[复用]
- **figcode/figure_01.py**：`_plot_er_panel` 加 critical value 判别线 `axhline(1+gamma)`（gamma 取自诊断表，默认 1.08），
  图例标注；figure_02 复用同函数 → Figure 1/2 均含判别线。
- **core/engine.py（export_all_paper_figures 内联 `_save_er_panel`）**：同样加判别线 → all-in-one 路径的 Fig 1/2 也 1:1。
- **figcode/figure_03.py**：新增 `_load_industry_lookup` / `plot_industry_sorted_heatmap` / `_full_weight_heatmap`；
  Figure 3 从 pipeline 重算全市场 proxy 权重，按行业排序全部股票（行业分隔线 + 行业名），异常回退旧热图。
- **figcode/figure_04.py**：改用 `_full_weight_heatmap(proxy=False)` → 全市场连续 PCA 权重、行业排序、符号已定向。
- **figcode/figure_05.py**：对全部股票重算月频 PCA 载荷 + 符号定向（N12）+ 行业排序热图，异常回退旧表。
- 详见 `06_FIGURE_FIDELITY_逐图核对.md`（逐张 1:1 状态 + 两条出图路径说明）。

## 图表形态对照论文实图的修正（第二轮，已核验 PDF）
- 打开论文 PDF 逐张渲染后发现并修正：
  - **Figure 3/4/5**：由热图改为**按行业着色的柱状图**（每因子一子图，x=股票按行业排序，y=Loadings，行业色例）——`figure_03.plot_industry_sorted_bars`，03/04/05 均改用之。
  - **Figure 1/2**：年份分成竖排子图（每组~4 年），判别线改绿色实线、图例标 `Critical value`，贴合论文版式。
  - **Figure 12**：改 3 面板（All/Industry/Sorted）、空心圆、去色条、加向下趋势虚线。
  - **Figure 13**：转置为 行=因子集、列=频段（Intraday/Overnight/Daily）。
  - **Figure 14/15**：拆为 Panel A（2×3 散点+45°线）+ Panel B（2×3 定价误差柱状），加 Panel 标题。
- 仍未达 1:1（已在 06 文档如实标注）：Figure 8（两月份权重快照）、Figure 10/11（三分量分解）受“复用缓存未存所需中间量”限制；Figure 6/7 因子条数需完整路径。

## 时间变化类图（6–11）改到论文形态（第三轮，新增 figcode/_timevar.py）
- **新增 `figcode/_timevar.py`**：自洽滚动计算（21 日窗口局部连续 PCA / 局部回归载荷），产出论文所需量。
- **Figure 6**：6 面板载荷 GC（连续4/HF4/行业4/FFC4/FF3/市场1）。
- **Figure 7**：前 7 个连续 PCA 因子的权重 GC（单面板 7 线）。
- **Figure 8**：两个月份窗口的 4 因子权重行业着色柱状（4×2），默认取低/高波动两窗口，可传日期。
- **Figure 9**：局部 PCA 解释方差时序。
- **Figure 10**：连续 PCA 因子结构分解 [系统性影响/平均载荷/波动]×[原始/归一化]（3×2，4 线）。
- **Figure 11**：同上但用 4 个 FFC 因子（载荷=局部回归 + FFC 因子方差）。
- 接入：figure_06..figure_11 的 generate() 改为“论文形态优先（_timevar），失败回退旧绘图”（try/except）。
- 依赖输入见 docs/06；缺数据时自动回退，不中断。engine all-in-one 路径未改这些图。

## 逐字核对后的最终修复（第四轮，详见 docs/07）
- **g_fn 默认 → median_sqrtN**（论文 footnote 17；旧 2ae5dce6 缓存不再命中，需重跑）。
- **generalized_correlations** 改对称白化形式 `(G'G)^{-1/2}(G'F)(F'F)^{-1}(F'G)(G'G)^{-1/2}`（新增 `_inv_sqrt_psd`，带回退）。
- **Figure 3/4/5 纵轴** 改特征向量尺度 Λ/√N（量级对齐论文，模式/符号不变）。
- **Figure 6/10 估计量** 改"固定权重 + 局部回归载荷"（论文口径；Fig7/8 仍局部 PCA）；`_daily_factor_sets` 增 HF 因子。
- **P2 插值** 落地：paper_lenient 面板缺失日置 0（零增量）于 `_write_panel`。
- **Figure 13** 补 "PCA (unbalanced)" 行（全市场 pairwise PCA 近似，带回退；非论文逐年旋转口径，已注明）。
- 合成数据自检：median_sqrtN→K̂=3；GC 同span≈1/无关<1；Λ'Λ/N=I；|V|=1。42 个 .py 语法全过。重路径仍需你本机跑通。
