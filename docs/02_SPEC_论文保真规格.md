# 论文保真规格（SPEC）—— 每条的论文口径 + 实现 + 开关

> 论文：Pelger (2020) *Understanding Systematic Risk: A High-Frequency Approach*, J. Finance 75(4):2179-2220。
> 下列页/脚注引用均来自该文。env 开关默认值见括号。

## A. 因子估计与因子数

### P1 因子数扰动项（footnote 17）[完整]
- 论文：`g(N,M) = √N · median{λ1,…,λN}`；`K̂(γ)=max{k≤N−1: ERk>1+γ}`；临界值 1.08。
- 实现：`config.g_fn`。复用路径保持 `"median_N"`（命中缓存）；完整路径设 `"median_sqrtN"` 并 `restart=True`。
- 验证目标：连续 K̂ 按论文模式（危机年 4 / 平时 3）、跳跃 K̂≈1。

### P3 估计K vs 展示K（III.B / III.D）
- 论文：因子数逐年估（3~4）；但 Figure 3/4/proxy/Table III/V 用**全 13 年平衡面板的 4 个连续因子**。
- 实现：display_k 固定 4（已在 `refresh_replication_result_views`）；逐年 K̂ 仅作诊断（Table I/II、Fig 1/2）。
- 完整路径下 g_fn 修复后，逐年 K̂ 标注自然变正确。

### P4 符号定向（Figure 4：因子1“只做多、量级相近”）[复用]
- 实现：`engine.orient_pca_result` —— 因子1令平均载荷为正；因子2..K令最大绝对载荷为正。
  注入 `refresh_replication_result_views`，复用时无需重跑 PCA。
- env：`PELGER_PAPER_FAITHFUL_SIGNS=1`。
- 注：因子2..K 的“经济朝向”（与冻结行业同号）可在冻结行业后进一步定向（可选增强）。

## B. 面板与数据

### P2 平衡面板（附录 A）[完整]
- 论文：年度 universe = 通过宽松缺失过滤（某日前10根全缺 / 某日前缺失>50 / 全年缺失>500 才剔除）
  **+ 缺失插值（增量为 0，不计入二次变差）**，再取 13 年都在的交集（美股 ~332 ≈当年 55%）。
  **不是“每个交易日都在”**。
- 旧实现：`is_strict_balanced` 要求每日都在、零缺失 → 仅 115 只（过严，是 115 的根因）。
- 本包实现：`is_balanced_paper`（每年覆盖率 ≥ `PELGER_BALANCED_MIN_COVERAGE`（默认 0.96，≈ ≤500/12000）
  且 13 年都在）；`PELGER_BALANCED_MODE=paper_lenient` 启用。
- **待补（重要）**：要真正用 paper_lenient 面板跑 PCA，面板数组写出处需对缺失交易日插值
  （“无可用价 → 用下一可用价；否则用上一可用价”，使增量为 0）。本包改了**宇宙选择**与口径标记，
  **面板数组插值步骤需你在 `_save_*` / 面板构建处补上**（约 10–20 行：对每只股票按 global_dates 重索引，
  缺失日 forward/back fill 价位 → 增量 0）。SPEC 末尾给出伪码。

### N1 纯盘中 5min（II.A：“Overnight returns are modeled as separate jumps”）[完整]
- 论文：连续/跳跃 PCA 用纯盘中 5min；隔夜是独立增量、归入跳跃；且从 9:35 起（丢首根）。
- 实现：`full_5min[0]=log(close[0]/open[0])`；隔夜单独存 `overnight`；`sum(48)=intraday`。
- env：`PELGER_INTRADAY_ONLY_5MIN=1`。
- 验证：`frac_jump_increments` 应较旧口径下降（旧 1.8% 偏高，含隔夜被误判为跳跃）。

## C. 特征因子（FFC / 2×3 / rf）

### P9 全市场 2×3（附录 A）[复用]
- 论文：6 个 size/value 组合用**全部满足清洗条件的股票**（含非成分股）。
- 实现：`paper_fidelity.build_full_market_size_value`——遍历全市场 `symbol_returns`，
  按 active_sort_year（7-12 月当年 / 1-6 月上年）查 portfolio + June-end float_mv_adj 权重，
  累计 vw/ew 三段；起点按 `PELGER_SIZE_VALUE_START`（默认 2014-07-01）。
- env：`PELGER_SIZE_VALUE_FULL_MARKET=1`。失败自动回退旧平衡子集实现。

### P6 自建 Carhart 12-1 月 MOM（III.A）[复用]
- 论文：高频版 vw 市场 + SMB + HML + **MOM**，全部高频构造；动量按 Fama-French 标准特征。
- 实现：`paper_fidelity.build_full_market_momentum`——日频拼 (T×N) 收益；月度再平衡日用
  过去 [t-252, t-21] 累计收益作信号（跳过最近 ~1 月），截面 30/70 选 winner/loser，
  持有至下个再平衡；输出 MOM 三段（winner−loser，**隔夜≠0**）。
- env：`PELGER_FFC_MOM_MODE=carhart_daily`（否则用旧高频 1 日动量）。

### P7 股票级分段 FFC（式 7-9 + III.A）[复用]
- 论文：FFC 因子本身高频构造，故各段有真实分量；隔夜 MOM 非 0；无残差强制。
- 实现：`paper_fidelity.build_ffc_segmented_clean`——
  MKT_seg = vw全市场该段 − rf_seg；SMB_seg=mean(small)−mean(big)；HML_seg=0.5(SH+BH)−0.5(SL+BL)；
  MOM_seg=winner−loser；**daily=intra+night（不强制等于官方）**。
- 旧实现的“按 |intra|/|night| 分摊残差强制 daily=官方”（paper_tail 旧 995-1014 行）被替换。

### P8 rf 拆分（III.A：日内常数假设）[复用]
- 实现：`paper_fidelity.split_daily_rf`——盘中 rf = 日 rf × (4h/24h)，隔夜 = 日 rf × (20h/24h)。
- 用于分段 FFC；engine 主流程的 rf=0 只影响 main_summary（复用时不重生成），完整路径可一并填充。

### P10 样本窗（受 2012 账面缺失限制）[复用]
- `FS_Combas` Accper 最早 2013-01-01，无 2012 年报 ⇒ size/value 与分段 FFC **起点固定 2014-07-01**。
- env：`PELGER_SIZE_VALUE_START=2014-07-01`。已在 `_build_payload` 对 size_value_assets 裁剪。

## D. 表/图

### P11 Figure 13（caption：normalized by their daily std）[复用] —— 已实现。
### P12 Figure 14/15（Panel A + Panel B pricing errors）[复用] —— 已实现（含 alpha 柱状）。
### N4 Figure 12 轴 / N6 年化 / N7 标签 / P13a n_obs [复用] —— 已实现。
### N10 Table V 下半部 Mkt/SMB/HML/MOM 个体夏普 [复用] —— 已实现。

### N9 Table III 两面板 [复用，best-effort]
- 论文上半部 10 行：HF PCA / PCA Proxy / Industry(M+3) + 3 ablation / Industry(unbalanced) / FFC / FF3 / Market。
- 论文下半部 6 行：ω HF/Jump/Overnight/Daily/Week/Month 的**载荷 GC**。
- 本包已加 HF PCA、PCA Proxy 行 + 下半部各频率载荷 GC（用对应频率 PCA 载荷 vs 连续 display 载荷，
  载荷空间 GC 同 footnote 19）。逐项 try/except；若复用 pickle 缺 R_jump/R_5min_full 数组则跳过该行。
- 仍未做：Industry(unbalanced)、Continuous PCA(unbalanced) 这类需逐年估+旋转对齐的行（属完整路径增强）。

## E. P2 面板插值伪码（待你在面板写出处补）
```text
对每只入选股票 s：
  按 global_dates 重索引其 5min/intraday/overnight/daily：
    若某交易日无价：用“下一可用 5min 价；否则上一可用价”填充价位
    -> 该日各 5min 增量 = 0，intraday/overnight/daily = 0
  这样缺失日不贡献二次变差，符合论文“interpolated -> zero increments”。
  仅当该股满足 is_balanced_paper（覆盖率阈值 + 13 年都在）才纳入平衡面板数组。
```
