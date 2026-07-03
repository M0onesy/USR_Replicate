# 交稿版实证写作指南：给 Codex

本文档给另一台电脑上的 Codex 使用，目标是让它在撰写实证部分时准确理解当前复刻进度、图表口径和可写结论。当前项目不是逐图逐表复制原论文的美股结果，而是借鉴原论文的高频 PCA 系统性风险分析框架，用 A 股数据完成一版方法论复刻。

## 当前实证主线

我们使用 A 股高频收益数据复刻 Pelger 风格的系统性风险分解框架，核心步骤包括高频收益分解、PCA 因子提取、平衡/非平衡样本比较、因子结构稳定性检验、日内/隔夜收益分析，以及测试资产上的定价检验。

当前交稿主样本为 `strict_balanced/full`，即全样本固定股票交集平衡面板。该主样本包含 `N=115` 只股票，样本交易日数为 `3157`。交稿图表不再走完整 `main.py -> paper_tables` 的慢路径，而是通过轻量入口生成：

```text
Code/export_submission_core_fast.py
```

最终图表位置：

```text
Result/pelger_cn_adjusted/figures/
```

最新中间诊断位置：

```text
Data/proc_Data/pelger_cn_adjusted/runtime/submission_fast/diagnostics/
```

正文只保留以下 9 张图：

```text
Figure 1, Figure 2, Figure 4, Figure 7, Figure 10, Figure 12, Figure 13, Figure 14, Figure 15
```

正文不引用以下图：

```text
Figure 3, Figure 5, Figure 6, Figure 8, Figure 9, Figure 11
```

## 图表数据来源

Figure 1 使用：

```text
figure1_yearwise_balanced_changing_universe_diagnostics.csv
```

Figure 2 使用：

```text
figure2_fixed_intersection_yearly_diagnostics.csv
```

Figure 7 和 Figure 10 使用：

```text
rolling_gc.csv
rolling_explained_variation.csv
```

Figure 12 使用：

```text
figure12_data.csv
```

Figure 13 使用：

```text
figure13_data.csv
figure13_yearly_alignment.json
```

Figure 14 使用：

```text
pricing_industry.csv
```

Figure 15 使用：

```text
pricing_size_value.csv
```

## 各图写作用法

### Figure 1

Figure 1 展示跨年非平衡、逐年平衡面板中的年度高频系统性因子数。这里的“非平衡”不是指年内允许缺失，而是指从全样本视角看股票池每年会变化；每一年内部都先构造完整可用的年内平衡面板，再做 PCA 因子数诊断。

写作时应强调：该图回答的是“如果每年使用当年可用且完整的股票池，A 股每年能提取多少高频系统性因子”。由于 A 股市场扩容，后期可用股票数显著增加，系统性结构也更丰富。

Figure 1 的 HF 因子数为：

```text
2013-2025: 4, 2, 2, 3, 2, 3, 3, 4, 5, 5, 8, 7, 6
```

### Figure 2

Figure 2 展示全样本固定股票交集平衡面板中的年度高频系统性因子数。这里每年使用同一批股票，股票数量固定为 `N=115`。该图比 Figure 1 更保守，因为它只保留 2013-2025 年全样本期间都连续存在且可用的股票。

写作时应强调：Figure 2 是全样本固定交集口径，因此每年 `N` 必须相同。它和 Figure 1 的差异说明样本构造会影响年度因子数判断。

Figure 2 的 HF 因子数为：

```text
2013-2025: 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 4, 5
```

### Figure 4

Figure 4 展示 strict 主样本连续 PCA 因子的组合权重。股票按行业排序，并用颜色表示行业组。该图的核心功能是解释 PCA 因子的经济含义。

写作时可以说：第一连续 PCA 因子的权重较广泛分布在股票上，更接近市场共同成分；后续因子的权重在行业或风格维度上呈现更强分化，说明 PCA 不仅捕捉市场共同波动，也捕捉 A 股内部结构性风险。

不要写成“精确复刻了原论文的美股行业结论”。本项目只能说基于 A 股数据观察到行业聚集和结构性分化。

### Figure 7

Figure 7 展示 21 个交易日局部窗口下，局部连续 PCA 权重与全样本权重之间的广义相关性变化。图中展示前 7 个连续 PCA 因子，是稳健性和结构变化可视化，不是最终主因子数结论。

写作时应强调：GC 越高，表示局部窗口内的因子权重结构越接近全样本基准结构。当前结果显示第一因子高度稳定，而后续因子波动更明显。

主诊断文件 `rolling_gc.csv` 中只有 `gc_1` 和 `gc_2`，对应核心两因子结构。滚动 GC 的核心统计为：

```text
gc_1 mean ≈ 0.971
gc_2 mean ≈ 0.615
```

因此可以写：第一系统性因子具有较强稳定性，第二因子更具有时间变化特征。

### Figure 10

Figure 10 展示连续 PCA 因子结构随时间变化的分解。它用于说明系统性风险结构并非完全固定，而会随市场阶段变化。

写作时可以结合 Figure 7：Figure 7 说明局部因子权重和全样本基准之间的相似度，Figure 10 则进一步拆解结构变化的来源，包括平均载荷、波动率以及二者共同作用。

图中较明显的变化阶段包括 2015 年前后、2020 年附近以及 2024 年附近。写作时不需要过度解释每一个峰值，只需要说明 A 股系统性风险结构具有阶段性变化。

### Figure 12

Figure 12 展示三类资产的平均日内和隔夜超额收益关系：

```text
全市场个股
行业组合
2x3 规模-价值组合
```

诊断数据中，全市场个股平均日内超额收益约为 `0.000850`，平均隔夜超额收益约为 `-0.000893`。行业组合和 2x3 组合也呈现类似方向：日内平均为正，隔夜平均为负。

写作时可以说：A 股收益具有明显的日内/隔夜分化，日内收益和隔夜收益不是同一种风险补偿机制。这为后续 Figure 13 的分频段因子收益分析提供动机。

### Figure 13

Figure 13 展示三类因子集在日内、隔夜、日度频段上的标准化累计收益：

```text
Continuous PCA
Continuous PCA (unbalanced, yearly aligned)
FFC 4-factor
```

第一行是全样本固定交集 strict 主样本上的连续 PCA 因子。第二行是逐年 changing-universe 面板估计出的 PCA 因子，经年度旋转对齐后拼接得到的跨年序列。第三行是 FFC 四因子基准。

Figure 13 第二行必须写清楚：它不是旧的 pairwise PCA 近似，而是逐年估计后做两步对齐：

```text
year-specific balanced changing-universe -> same-year fixed intersection
same-year fixed intersection -> global fixed-intersection baseline
```

`figure13_yearly_alignment.json` 显示 13 个年份全部完成对齐。写作时可说：该结果说明 changing-universe 年度因子序列可以被转换到共同坐标系中，从而形成可比较的跨年因子收益序列。

### Figure 14

Figure 14 展示行业组合上的资产定价结果。图中比较 Continuous PCA 和 FFC 4-factor 对行业组合平均收益的预测表现和定价误差。

写作时应谨慎表述：Continuous PCA 因子是由高频收益统计结构提取出的数据驱动因子，FFC 4-factor 是传统特征因子。图的意义是比较二者在行业测试资产上的解释能力，而不是宣称 PCA 全面优于传统因子。

诊断中行业组合平均绝对 alpha 大致为：

```text
Continuous PCA: daily ≈ 0.000265, intraday ≈ 0.000211, overnight ≈ 0.000238
FFC 4-factor:  daily ≈ 0.000209, intraday ≈ 0.000248, overnight ≈ 0.000222
```

因此正文应使用中性表述：二者在不同频段和资产组上表现各有差异。

### Figure 15

Figure 15 展示 A 股重建的 `2x3` 规模-价值组合资产定价结果，样本从 `2014-07-01` 开始。组合标签为：

```text
BH, BL, BM, SH, SL, SM
```

图中同样比较 Continuous PCA 和 FFC 4-factor 的预测收益与定价误差。

诊断中 2x3 组合平均绝对 alpha 大致为：

```text
Continuous PCA: daily ≈ 0.000147, intraday ≈ 0.000224, overnight ≈ 0.000129
FFC 4-factor:  daily ≈ 0.000464, intraday ≈ 0.000489, overnight ≈ 0.000107
```

可写结论是：在 A 股重建的规模-价值组合上，Continuous PCA 对日度和日内平均收益的定价误差较小，但隔夜维度上 FFC 4-factor 表现也具有竞争力。不要写成单边胜利。

## 当前可以写入正文的关键结论

第一，A 股高频收益中存在明显的系统性因子结构。逐年 changing-universe 面板的因子数在后期显著增加，说明随着市场扩容，系统性结构更复杂。

第二，样本口径会显著影响年度因子数判断。全样本固定交集平衡面板更保守，早期通常只有 1-2 个高频系统性因子；逐年 changing-universe 面板则在后期识别出更多因子。

第三，第一连续 PCA 因子非常稳定，具有市场共同风险特征；第二因子明显更时变，可能体现行业、风格或市场阶段变化。

第四，A 股日内与隔夜收益呈现方向分化。平均日内超额收益为正，平均隔夜超额收益为负，这说明日内和隔夜风险补偿机制不同。

第五，Continuous PCA 因子可以作为 A 股资产定价中的有用统计因子，但不能夸张成完全替代 FFC 4-factor。Figure 14 和 Figure 15 应写成比较性结果，而不是胜负式结论。

## 写作禁忌

不要说“完全复刻原论文结果”。本项目是方法论复刻和 A 股应用。

不要把 Figure 7 的 7 条线解释为最终提取了 7 个核心系统性因子。Figure 7 是稳健性和结构变化展示。

不要把 Figure 13 中展示前 4 个 PCA 因子写成最终主结论为 4 个强系统性因子。前 4 个因子用于和四因子基准保持展示可比性。

不要引用 Figure 3、Figure 5、Figure 6、Figure 8、Figure 9、Figure 11。

不要把 Figure 1 的 changing-universe 年度面板和 Figure 2 的 fixed-intersection 平衡面板混为一谈。

## 建议正文组织

建议按以下顺序写实证部分：

1. 数据和样本构造。
2. 高频 PCA 因子数诊断，对应 Figure 1 和 Figure 2。
3. 连续 PCA 因子的经济解释和结构稳定性，对应 Figure 4、Figure 7、Figure 10。
4. 日内、隔夜、日度收益分解，对应 Figure 12、Figure 13。
5. 行业组合和规模-价值组合资产定价，对应 Figure 14、Figure 15。

## 表格口径提醒

Table I 和 Table II 原论文式复刻已有基础结果，但正文整理时要注意口径：

Table I 可以保留原论文逻辑，即不同跳跃阈值下的跳跃增量比例、跳跃解释二次变异比例、第一跳跃因子解释的跳跃相关性、前四个连续因子解释的连续相关性。

Table II 不应使用当前动态展开到 `gc_8` 的版本作为正文表。更像原论文的正文表应固定报告前 2/3 个连续 PCA 因子和前 2/3 个跳跃 PCA 因子的广义相关性，避免读者误解为我们主张 8 个核心因子。

