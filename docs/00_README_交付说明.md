# 交付说明（README）—— Pelger (2020) A 股复现 · 论文保真修改包

本包是对你仓库 `Code/` 的**深入修改版**，按论文原文口径修复了此前确认的 25 项问题，并**特意围绕“复用 33 小时重结果 `replication_result_2ae5dce6a23fd2a2`”来设计**。

## 一、最重要的两件事

1. **本修改在 lite 沙箱里无法运行/验证**（大数组与 checkpoint 二进制被省略）。所有代码均为“按论文规格 correct-by-construction”，**未经运行验证**。请务必在你本机全量数据上跑 `docs/04_VERIFICATION_验证清单.md` 确认。

2. **复用机制已被我逆向确认**（关键结论，决定整个设计）：
   - 重结果缓存 `replication_result_{hash}.pkl` 的命中键 = `RunConfig.cache_signature()`（含 `g_fn / return_mode / gamma / jump_a / k_max / proc_root / years / max_stocks`），**不含代码哈希**。
   - 命中后会用**当前（新）代码**重新生成 `paper_tail` 等“视图”（分钟级）。
   - ⇒ **黄金路径**：只要 `g_fn` 保持 `median_N`、`return_mode` 保持 `open_close`、proc_root/years 不变，就命中 `2ae5dce6a23fd2a2`，**且可以随意改本包代码**，分钟级拿到“修好整个尾部 + 修好 PCA 符号”的结果。

## 二、两条路径（务必先读 RUN_GUIDE）

| | 复用路径（默认，分钟级） | 完整路径（贴近论文，需重跑 ~33h） |
|---|---|---|
| 触发 | `g_fn="median_N"` + 不改签名 + 删除/不改 paper_tail 缓存 | `g_fn="median_sqrtN"` + 重跑预处理 + `restart=True` |
| 修复了哪些 | P4 符号 / P5 行业冻结 / P6 自建MOM / P7 分段FFC / P8 rf拆分 / P9 全市场2×3 / P10 窗 / P11 Fig13 / P12 Panel B / P13 / N2 N4 N6 N7 N8 N9 N10 | 以上**全部** + P1 g_fn / P2 平衡面板 / N1 纯盘中5min / P3 逐年K显示 / N11 rolling K / N12 月度面板 |
| 核心 4 因子来源 | 旧 115 只 / 含隔夜 5min 面板（**未变**） | 论文宽松平衡面板 / 纯盘中 5min（**已修**） |

> 直白结论：**复用路径能把“构造逻辑错误”几乎全部修好（FFC、2×3、rf、符号、行业、图表、表结构）**，只有“核心 PCA 因子仍来自旧面板/旧5min”这一点修不了——那需要重跑。两者你都能要。

## 三、目录

- `Code/` —— 修改后的源码（在你原 `Code/` 基础上）。新增 `core/paper_fidelity.py`。
- `external_industry_new/` —— 新版 11 桶行业映射（放到 external 数据目录用，见 RUN_GUIDE）。
- `docs/`
  - `01_CHANGELOG_修改清单.md` —— 逐文件、逐条（P/N 编号）改了什么。
  - `02_SPEC_论文保真规格.md` —— 每条的论文原文口径 + 实现要点 + env 开关。
  - `03_RUN_GUIDE_运行指南.md` —— 两条路径的精确命令与复用机制。
  - `04_VERIFICATION_验证清单.md` —— 跑完要核对的 before/after 指标。
  - `05_LIMITATIONS_限制与未尽事项.md` —— 未验证项、完整路径专属项、N9 best-effort 说明。

## 四、一句话上手

复用路径（最省时）：把新映射放好 → 在 `RunConfig` 里设 `industry_info_filename="stock_full_info_std_industry_final.csv"`、`paper_faithful_signs=True`、`size_value_full_market=True`、`ffc_mom_mode="carhart_daily"`、`refresh_paper_tail=True`，**保持 `g_fn="median_N"`**，删掉 `paper_tail/` 视图缓存后重跑 main → 分钟级出图出表。细节见 RUN_GUIDE。
