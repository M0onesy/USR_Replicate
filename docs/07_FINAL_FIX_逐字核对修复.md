# 逐字核对后的最终修复说明（FINAL FIX）

> 本轮把"对照全篇论文逐字核验"中发现的**全部不一致与缺口**修复完毕。
> 诚实前提仍然成立：**lite 沙箱无数据，完整管线/滚动/非平衡 PCA 等重路径未运行验证**；
> 但下面标【已数值自检】的几项，我用**合成数据**(不依赖缺失文件)实跑验证过，结论附后。

## 一、本轮修复清单（6 项）

### 1. g_fn 默认 → `median_sqrtN`（论文 footnote 17）【已数值自检】
- 论文：扰动项 `g(N,M)=√N·median{λ}`。原默认 `median_N` 算的是 `N·median`，**不符**。
- 改动：`RunConfig.g_fn` 与 `MainLaunchProfile.g_fn` 默认都改为 `"median_sqrtN"`。
- 后果：`cache_signature` 含 `g_fn`，故**不再命中**旧重结果缓存 `replication_result_2ae5dce6a23fd2a2.pkl`——需重跑（你已计划补数据重跑）。若要复用旧缓存做对照，显式传 `g_fn="median_N"`。
- 自检：合成谱(真 K=3) → `K̂=3`，ER 在第 3 个后跌回 ~1。✅

### 2. `generalized_correlations` 改对称白化形式【已数值自检】
- 论文 footnote 19 / Bai-Ng：平方典则相关 = **对称 PSD** 矩阵
  `Mₛ=(G'G)^{-1/2}(G'F)(F'F)^{-1}(F'G)(G'G)^{-1/2}` 的特征值。
- 原实现对**非对称** `(G'G)^{-1}(G'F)(F'F)^{-1}(F'G)` 做 `(M+M')/2` 近似——会偏。
- 改动：新增 `_inv_sqrt_psd`(特征分解逆平方根)，按对称形式取特征值；带 try/except 回退旧式。
- 自检：近似同span → GC≈[1,1,1,1]；无关 → GC<1（[0.23,0.07,0.01]）。✅

### 3. Figure 3/4/5 纵轴改为特征向量尺度 Λ/√N【已数值自检】
- 论文 Figure 3/4/5 的"Loadings/组合权重"量级 ~0.05–0.2，对应**相关矩阵单位特征向量** V=Λ/√N，
  而非 `Λ/σ`（会按 1/σ 放大尺度）。
- 改动：`figure_03._full_weight_heatmap`、`figure_05`、`_timevar.render_fig8` 一律显示 V=Λ/√N；
  Fig 3 proxy 在该尺度上施加稀疏掩码（保留最大权重、其余置 0）。相对模式与符号不变。
- 自检：`Λ'Λ/N=I`、`|V|` 每因子=1、因子1 平均载荷为正。✅

### 4. Figure 6 / 10 估计量修正（论文："固定权重、载荷随时间变化"）
- 论文 Section III.E：Figure 6（载荷 GC）与 Figure 10/11（分解）都是**保持因子权重不变、用局部回归
  估计随时间变化的载荷**；只有 Figure 7/8 才对每个局部窗口**重做 PCA**。
- 原 `_timevar.render_fig6/10` 用的是局部 PCA——口径错。
- 改动：`render_fig6`（6 面板）与 `render_fig10` 均改用 `_local_regression_loadings`（对各模型**固定的全样本日频因子**回归出局部载荷）；`_daily_factor_sets` 增加 HF(连续+跳跃) 因子（R_5min_full 上 K=4 PCA → 投影日频）。Figure 7/8 仍用局部 PCA（正确）。

### 5. P2 平衡面板"缺失日插值→零增量"落地
- 论文宽松面板：缺失日插值（价格沿用上一可得值 → 当日收益记 0），使面板在宽松阈值下仍是完整矩形。
- 改动：`preprocess_cn_data._write_panel` 中，当 `PELGER_BALANCED_MODE=paper_lenient` 且 `allow_missing` 时，
  把对齐后残留的 NaN 在 R_intra/R_night/R_daily/R_5min_full 上置 0；`sample_report` 记 `missing_interpolated_to_zero`。
  strict 口径不受影响。

### 6. Figure 13 补 "PCA (unbalanced)" 行
- 论文 Figure 13 三行：PCA / **PCA-unbalanced** / FFC。原仅 2 行。
- 改动：新增 `_build_unbalanced_pca_segments`——对**全市场** symbol_returns 日频【总】收益做一次全样本
  pairwise PCA（NaN 容忍）得 4 权重，再施加到三段全市场收益，得非平衡 PCA 三段日频因子；
  `_build_figure13_data` 接受 `pca_unbalanced` 并加该行；渲染器按数据中实际存在的因子集出行（缺则回到 2 行）。
- **口径说明（重要）**：这是**近似**——论文用"逐年 PCA + 旋转到平衡全样本因子"的抗幸存者偏差构造，
  本实现用"单次全样本 pairwise PCA"。方向/反转模式一致，但不是逐字相同的估计量。若需严格一致，
  需接入重 yearly 分析里的逐年非平衡 PCA + 旋转产物。

## 二、合成数据自检结果（不依赖缺失文件，已实跑）
```
median_sqrtN  K_hat = 3 (expect 3) | ER[:5]=[1.488 1.578 2.150 1.000 1.001]
GC 同span      ≈ [1, 1, 1, 0.998]   ；GC 无关 ≈ [0.231, 0.074, 0.008]
Lambda'Lambda/N = [1, 1, 1]         ；|V| 每因子 = [1, 1, 1]；factor1 平均载荷=正
RunConfig().g_fn = median_sqrtN
42 个 .py 全部通过 AST 语法检查
```

## 三、仍未运行验证 / 残留口径说明（诚实边界）
- **重路径未跑**：完整 PCA 管线、`_timevar` 滚动(图 6–11)、`paper_fidelity`(P6/P7/P8/P9)、Table III 扩展、
  Figure 13 非平衡 PCA、P2 插值后的面板——这些都**需要你在本机用真实数据跑通**；均带 try/except 回退。
- **Figure 13 非平衡行是近似**（见上 6）。
- **出图走 figcode/registry(task runner) 路径**；engine all-in-one 路径未含图 3–15 的论文形态。
- 复用语义已变：g_fn 默认翻转后，**默认即论文口径**，但旧 `2ae5dce6` 缓存不再命中（需重跑）。

## 四、运行提示
- 完整论文口径一次到位：`PELGER_BALANCED_MODE=paper_lenient`（+ N1 默认开）+ 默认 `g_fn=median_sqrtN`，
  `restart=True` 重跑主管线与逐年分析；随后走 task runner 出全部图表。
- 详细输入依赖见 `06_FIGURE_FIDELITY_逐图核对.md`。
