# 验证清单（VERIFICATION）—— 跑完在你本机核对

> 因 lite 沙箱无数据、本包代码**未经运行**，请逐项核对。✅=期望结果。

## 一、复用路径跑完后核对

### P4 符号
- `tables/Table_11_continuous_pca_weights.csv`：因子 1 的 top 权重应**为正**（不再全负）。✅
- `tables/Table_14_factor_return_summary.csv`：因子 1 `mean_intraday` 应为正、与市场同向。✅
- `Table_12_proxy_factor_weights.csv`：因子 1 权重应 = **+1/N**（不再 −1/N）。✅

### P9 全市场 2×3
- `paper_tail/.../validation/size_value_daily_parity_summary.json`：`size_value_source="full_market"`；
  `max_abs_diff` 现在**非 0**（与平衡子集参考有差异，正常，已标注为差异报告）。✅
- `assets/size_value_portfolios.csv`：日期应从 **2014-07-01** 起。✅

### P6/P7/P8 分段 FFC
- `factors/ffc_segmented_returns.csv`：隔夜 MOM 列**不再恒为 0**。✅
- `diagnostics/factor_matrix_diagnostics.json`：FFC overnight 的 `rank` 应=4、`condition_number` **有限**（不再 ∞）。✅
- `ffc_daily_validation_summary.json`：`ffc_source="stocklevel_carhart_clean"`，`max_abs_diff` 非 0（不再被强制相等）。✅

### Table V（N10 + P13b）
- `Table_V_*.csv`：FFC 盘中 SR 从 **14.52** 回落到合理量级（个位数内）。✅
- 下半部出现 `Market / Size / Value / Momentum` 个体夏普行（特征因子隔夜为正、盘中弱/负）。✅

### Table III（N9 + P13c）
- `Table_III_*.csv`：新增 `panel` 列；上半部含 `HF PCA`、`PCA Proxy` 行；
  下半部含 `omega HF/Jump/Overnight/Daily/Week/Month` 的载荷 GC 行。✅
- FFC 行 `gc_4` 修复 MOM 后应不再恒为 0（数值随数据）。✅

### 图
- `Figure_12`：横轴=盘中、纵轴=隔夜；标题“{N} Industry Portfolios”显示真实桶数（新映射=11）。✅
- `Figure_13`：三段按同一日频 std 归一化；FFC 曲线不再因前导缺失从 0 平台起跳。✅
- `Figure_14/15`：每张含 Panel A（散点）+ Panel B（各资产 alpha 柱状）。✅
- `industry_selection.json`：`selection_rule` 标 PLACEHOLDER（未冻结时）；冻结后标 EX-ANTE FROZEN。✅

### 新映射
- `assets/industry_portfolios.csv`：portfolio 取值应为 11 个中文桶（大金融/科技成长/…）。✅

## 二、完整路径（重跑后）额外核对

### P1 因子数
- `paper_factor_count_diagnostics.csv` / `Figure_01/02`：连续 K̂ 呈“危机年 4 / 平时 3”模式、跳跃 K̂≈1。✅
- `main_summary.json`：K_cont_hat 不再恒=1。✅

### P2 平衡面板
- `metadata/universe_summary.json`：`balanced_paper_symbols_full` 远大于 `strict_balanced_symbols_full`（115）；
  达到当年宇宙的可观比例（目标向论文 ~55% 看齐）。✅
- 确认你已在面板写出处补了缺失日插值（否则 paper_lenient 面板数组不完整）。⚠️

### N1 纯盘中
- 任取一只票的 `symbol_returns/*.npz`：`sum(full_5min_returns[d]) == intraday_returns[d]`（不再等于 daily）。✅
- 第 0 根与 `overnight_returns` 相关性应**大幅下降**（旧≈0.79）。✅
- `frac_jump_increments` 较旧口径下降。✅

### N11/N12
- `Figure_06`（rolling GC）出现 gc_1..gc_4（不再只有 gc_1）。✅
- `Table_13_monthly_pca_weights.csv` 在重定义面板上、符号定向后。✅

## 三、回退自检（确保安全网生效）
- 设 `PELGER_SIZE_VALUE_FULL_MARKET=0` / `PELGER_FFC_MOM_MODE=legacy_hf` 再跑，应回到旧行为且不报错——
  确认 try/except 回退链路正常（日志无 `[paper_fidelity]` 报错；或有报错则按提示修后重跑，仅花分钟级）。
