# 限制与未尽事项（LIMITATIONS）

## 1. 未经运行验证（最重要）
本包在 lite 沙箱（无大数组、无 checkpoint 二进制）中完成，**所有代码均未实际运行**。
均为按论文规格 correct-by-construction，可能存在运行期问题（列名/形状/边界）。
复用路径只花分钟级，发现问题修正后重跑代价小；请务必走 `04_VERIFICATION`。

## 2. 复用路径修不了的（属完整路径，需重跑 ~33h）
- **P1 g_fn**：在 `cache_signature` 内，改了就让 `2ae5dce6` 失效。
- **P2 平衡面板**：核心 4 因子在旧 115 只面板上估的，烤进了 pickle。
- **N1 纯盘中 5min**：5min 序列烤进了 pickle。
- **P3 逐年 K 显示 / N11 rolling K / N12 月度面板**：来自重结果/逐年分析。
⇒ 复用路径下，**核心 4 个连续 PCA 因子仍来自旧面板/旧 5min**；尾部（FFC、2×3、rf、符号、行业、图表、表结构）已按论文修好。

## 3. P2 面板插值尚需补一段（完整路径）
本包加了 `is_balanced_paper`（论文宽松口径的“可用宇宙”）与选择逻辑，但**面板数组写出处对缺失交易日的插值（增量为 0）未实现**。
不补这段，paper_lenient 面板的数组会因个别缺失日不完整。伪码见 `02_SPEC` 的 E 节（约 10–20 行）。

## 4. N9 Table III 为 best-effort
- 已加：HF PCA、PCA Proxy（上半部）+ ω HF/Jump/Overnight/Daily/Week/Month 载荷 GC（下半部）。
- 逐项 try/except：若复用的 pickle 未保存 `R_jump` / `R_5min_full` 等大数组，则相应行**自动跳过**（不崩溃，日志提示）。
- 未实现：Industry(unbalanced)、Continuous PCA(unbalanced) 等需“逐年估 + 旋转对齐全 horizon”的行（Table V 同名行亦未补），属抗幸存者偏差的完整路径增强。

## 5. P6 动量的口径选择
- 采用 Carhart 12-1 月（[t-252, t-21]、跳过最近 ~1 月）、月度再平衡、30/70 winner-loser、默认市值加权。
- 论文是“高频版 FFC 动量”；A 股无官方日频动量因子表，故自建。formation 参数（lookback/skip/quantile）
  可在 `build_full_market_momentum` 调整；与论文“结果不同但构造同源”一致。

## 6. P8 rf 拆分的近似
- 盘中按 4h/24h、隔夜 20h/24h（含周末按日历近似）。论文指出 rf 远小于股票收益，影响可忽略，但口径一致。
- engine 主流程 Sharpe 的 rf 仍为 0（只影响 main_summary，复用时不重生成）；完整路径可在 `run_cn_replication`
  加载 rf 并填充 `panel.rf_intra/rf_night`（本包未改主流程 rf 填充，避免触动签名/重结果）。

## 7. 涨跌停（非 bug，未处理）
A 股 10%/20%/30% 涨跌停会截断高频动态，影响**数值**而非**构造逻辑**。若要标记/特殊处理涨跌停日，
可作为可选 A 股适配增强（不在本包范围）。

## 8. 与既有诊断/校验的关系
- size_value / ffc 的旧“校验”是自证循环（N8）。本包把摘要重定位为“与旧口径的差异报告”，
  full_market / stocklevel 模式下 `max_abs_diff` 非 0 是**预期**，不代表错误。
