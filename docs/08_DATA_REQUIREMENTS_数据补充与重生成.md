# 数据补充与重生成 · 完整说明（DATA REQUIREMENTS）

> 本文回答两个问题：**(1) 还需要补充（放置）哪些数据？(2) 需要重新生成哪些数据？**
> 全部基于对代码实际输入/输出路径的核对。诚实前提：lite 沙箱无数据，下述重路径
> 均**未在本环境运行过**；请在你本机（真实数据）执行。

---

## 0. 一页速查

| 类别 | 对象 | 动作 | 触发原因 |
|---|---|---|---|
| 补充 | 11 桶行业映射 CSV | **放置** 到 industry/ | 新映射，图3/4/5/6/8 + 行业因子 |
| 重算·必做 | `checkpoints/replication_result_*.pkl` + `checkpoints/paper/year_YYYY/*.pkl` | **restart 重跑管线+逐年分析** | g_fn 默认 median_N→median_sqrtN，缓存失效 |
| 重算·条件 | `symbol_returns/*.npz` + `panels/strict_balanced/*` | **`preprocess --refresh`** | N1（隔夜=独立跳跃）写在预处理；旧缓存若非 N1 口径需重建 |
| 重算·条件 | `metadata/universe.csv` | 随 `--refresh` 重生成 | 缺 `is_balanced_paper` 列（P2 选样需要） |
| 自动刷新 | `paper_tail/factors/*.csv`、`figures/*_data.csv` | 无需手动 | 管线 refresh 时按新映射 + P5/6/7/8/9 重建 |
| 不动 | 原始 K 线、后复权因子、外部 rf/ff3/ff5/mom/市值/报表 | 复用 | 这些是输入 |

> **结论一句话**：唯一要"补"的新数据是 **11 桶行业映射 CSV**；要"重算"的是 **checkpoints（因 g_fn）**、**symbol_returns+panels+universe（因 N1/P2）**；paper_tail 产物随重跑**自动刷新**。

---

## 1. 数据目录结构（你本机，示例根 `D:\…\Reposit\`）

```
Data/
├─ kline_Data/EXTRA_STOCK_A/<代码>/data.bz2     # 原始 5 分钟 K 线（输入）
│  └─ raw_symbol_dirs.txt
├─ fact_Data/backward_factor.csv                # 后复权因子（输入）
├─ external_Data/pelger_tail/                   # = external_data_root（外部输入）
│  ├─ industry/                                 # ← 放新行业映射 CSV
│  ├─ factors/{rf, ff3, ff5}/…                  # shibor 对数收益、三/五因子（输入）
│  └─ size_value/raw/…                          # 资产负债表 + 日个股回报（输入）
└─ proc_Data/pelger_cn_adjusted/                # = proc_root（预处理产物）
   ├─ symbol_returns/<代码>.npz (+ .json, index) # 逐股票收益（重算对象 B2）
   ├─ panels/strict_balanced/{full,year_YYYY}/   # 面板数组 *.npy（重算对象 B2）
   ├─ paper_tail/factors/*.csv                   # 行业/FFC 因子（自动刷新 B4）
   └─ metadata/{universe.csv, universe_summary.json}  # 宇宙表（重算对象 B3）

Result/pelger_cn_adjusted/
├─ checkpoints/replication_result_*.pkl          # 主重结果（重算对象 B1）
│  └─ paper/year_YYYY/…(*.pkl)                    # 逐年分析（重算对象 B1）
├─ checkpoints/run_state.json
├─ figures/  tables/  diagnostics/
```

规模基准：**5465 只**、**2013-01-04 ~ 2025-12-31**、**3157 个交易日**、**48 根/日**（5 分钟）；
strict 平衡全样本 = **115 只**（`panels/strict_balanced/full/R_5min_full.npy` 形状 [151536,115]）。

---

## 2. 需要补充（放置，非重算）的数据 —— 仅 1 项

### 2.1 新版 11 桶行业映射 CSV
- 文件：`stock_full_info_std_industry_final.csv`（5519 只 / 11 桶：科技成长1078 / 高端制造944 / 周期资源769 / 可选消费621 / 医药515 / 电力新能源452 / 房地产建筑387 / 公用交运377 / 必需消费160 / 大金融120 / 农林牧渔96）。
- 放到：`Data/external_Data/pelger_tail/industry/`。
- 让代码用它：把配置项 `industry_info_filename` 指向该文件名（`figcode/figure_03._load_industry_lookup` 与 paper_tail 行业因子都读 `external_data_root/industry/<industry_info_filename>`）。
- 影响：Figure 3/4/5（行业排序着色柱状）、Figure 6 行业面板、Figure 8、行业因子（Table III）、P5 行业冻结。
- 其余外部输入（rf=shibor 对数收益、ff3、ff5、mom、市值、size/value 报表）**你已有，不用补**。

---

## 3. 需要重新生成的数据（按"为什么失效"分档）

### B1.（必做）重结果 + 逐年分析 checkpoints —— 因 g_fn 默认翻转
- 失效对象：
  - `Result/pelger_cn_adjusted/checkpoints/replication_result_2ae5dce6….pkl`（主管线，旧哈希）；
  - `Result/pelger_cn_adjusted/checkpoints/paper/year_2013 … year_2025/*.pkl`（Table I/II、Fig1/2 的逐年 PCA，约 90 个 pkl）。
- 原因：`g_fn` 默认由 `median_N`(=N·median) 改为论文 footnote 17 的 `median_sqrtN`(=√N·median)，
  `cache_signature` 含 `g_fn` → 旧哈希**不再命中**。
- 做法：以 **`restart=True`** 重跑主管线 + 逐年 paper 分析。逐年 PCA ≈ **33h**；主管线分钟级。
- 仅用现有 panels/symbol_returns，**不重建数据**；完成后 Fig1/2 + Table I/II 即论文口径。
- 若要对照旧缓存：显式传 `g_fn="median_N"`（= N·median，非论文口径）。

### B2.（条件必做）symbol_returns + panels —— 因 N1（隔夜=独立跳跃）
- 失效对象：`proc_Data/pelger_cn_adjusted/symbol_returns/*.npz`（全 5465 只）+ `panels/strict_balanced/{full,year_YYYY}/*.npy`。
- 原因：N1 把 `full_5min[0]` 改成"当日 open→close"（纯盘中），**写在预处理**（`preprocess_cn_data.py:426`）；
  旧缓存若按 `close/prev_close`（含隔夜）生成，需重建。
- **如何判断要不要做**：查面板/股票缓存元数据里的 `panel_return_scheme`——含 `_intraday5min_v2` = 已是 N1 口径（**免重建**）；否则需重建。
- 做法（默认 `PELGER_INTRADAY_ONLY_5MIN=1` 即开）：
  ```bash
  PELGER_INTRADAY_ONLY_5MIN=1 python preprocess_cn_data.py --refresh
  ```
  需原始 K 线齐全（`kline_Data/EXTRA_STOCK_A/<代码>/data.bz2` 全 5465 只）。完成后再叠 B1。

### B3.（要 P2 宽松面板才需要）universe 重生成 —— 因缺 `is_balanced_paper` 列
- 失效对象：`proc_Data/pelger_cn_adjusted/metadata/universe.csv`（+ `universe_summary.json`）。
- 现状：现有 universe 有 `coverage_ratio`、`is_strict_balanced`、逐年 `coverage_YYYY`，**但无 `is_balanced_paper`**。
  预处理 `_summarize_processed_universe`（已含该列计算）重跑即补上。
- 做法：随 B2 的 `--refresh` 一并生成（几乎零额外成本）。

### B4.（自动刷新，无需手动）paper_tail 因子产物
- 对象：`proc_Data/pelger_cn_adjusted/paper_tail/factors/{industry_factor_returns.csv, ffc_segmented_returns.csv, …}` 及 `figures/figureNN_data.csv`。
- 说明：管线每次 refresh（分钟级）按**新行业映射 + P5/P6/P7/P8/P9** 重建——**B1 重跑时自动刷新**；
  你不用单独生成，但其数值/内容**会变**（图表依赖它们）。

---

## 4. 不需要重算的数据
- 原始 5 分钟 K 线 `kline_Data/.../data.bz2`、后复权因子 `fact_Data/backward_factor.csv`；
- 外部原始因子/报表：`factors/{rf,ff3,ff5}`、`size_value/raw/*`、市值原文件。
- 均为**输入**，除非更换数据源否则复用。

---

## 5. 必须诚实告知的前提（否则上面白做）

### 5.1 P2 论文宽松面板（paper_lenient）尚未端到端打通
仅重算数据不够，还差 3 处代码接线：
1. 预处理目前**只写 `strict_balanced` 面板**（`preprocess_cn_data.py` 第 1016–1022 行），**未写 `paper_lenient/full`**——我加的"缺失日插值→零增量"在 `_write_panel` 里，但需以 `allow_missing=True` + paper_lenient 选样调用它才会触发；
2. 引擎加载器 `_proc_panel_paths` / `load_*`（`core/engine.py` 第 1218、1319 行）**限制只读 `strict_balanced`**（非 strict 直接 raise）；
3. 主运行需按 `PELGER_BALANCED_MODE` 选择面板模式。

> **因此：strict_balanced 路径今天就能跑通**（B1[+B2] 即得论文口径的因子数 / N1 / 新行业）。
> 要真正用论文宽松面板（115→更多只、缺失日插值），需补上述 3 处接线——**可按需补全**。

### 5.2 N1 是预处理时写入，不是运行时
所以启用/变更 N1 必须重建 symbol_returns + panels（见 B2），单独重跑管线无效。

### 5.3 Figure 13 "PCA (unbalanced)" 行需全量 symbol_returns
该行用**全市场 pairwise PCA**（近似口径，非论文逐年旋转）；需 `symbol_returns/` 全 5465 只齐全（B2 产出），否则该图自动回退为 2 行。

---

## 6. 推荐执行顺序（strict 口径，今天可跑）

```bash
# ① 放置新行业映射 CSV（见 §2.1），并设 industry_info_filename 指向它

# ② 重建逐股票缓存 + 面板 + 宇宙（B2 + B3）；默认 N1 开
PELGER_INTRADAY_ONLY_5MIN=1 python preprocess_cn_data.py --refresh
#   可选：--workers N --panel-workers N 控制并行；--years 2013 2014 仅试跑

# ③ 以 restart=True、默认 g_fn=median_sqrtN 重跑主管线 + 逐年分析（B1）
#    （在 core/config.py 的 profile 里设 restart=True；逐年 PCA ≈ 33h）
#    重跑会自动刷新 paper_tail 产物（B4）

# ④ 走 task runner 出图表（figcode/registry 路径）
python main.py --list            # 查看可用 figNN / 表 短名
python main.py <profile/tasks>   # 生成图/表
```

> 出图**必须走 figcode/registry（main.py task runner）路径**；engine all-in-one 导出只共享了
> Fig1/2 判别线，图 3–15 的论文形态不在该路径。

---

## 7. 图/表 → 数据依赖对照（出问题时按此排查）

| 产物 | 直接依赖 | 受哪步影响 |
|---|---|---|
| Fig 1/2（因子数 ER+临界线） | 逐年分析 pkl（K̂、ER） | B1（g_fn=median_sqrtN） |
| Fig 3/4/5（行业着色权重柱状） | 新行业映射 + 主重结果（display PCA 权重 Λ/√N） | A + B1 |
| Fig 6（6 面板载荷 GC） | `R_cont`、各模型全样本日频因子（含行业/FFC） | B1（+A 行业面板） |
| Fig 7（7 因子权重 GC）/ Fig 8（两窗口权重） | `R_cont` + `day_ids` + `R_daily` | B1（+B2 的 5min 数据） |
| Fig 9（解释方差）/ Fig 10（PCA 分解） | `R_cont` / 连续 PCA 日频因子 | B1 |
| Fig 11（FFC 分解） | 分段 FFC 日频因子 | B1（+ rf/ff3/mom 外部） |
| Fig 12（盘中×隔夜散点） | paper_tail figure12_data（全市场/行业/size-value） | B1 + A |
| Fig 13（累计因子收益 3 行） | 连续/FFC 因子 + **全市场 symbol_returns**（unbalanced 行） | B1 + **B2 全量** |
| Fig 14/15（定价 Panel A/B） | 行业/size-value 资产 + 因子 | B1 + A |
| Table I/II | 逐年分析 pkl | B1（g_fn） |
| Table III（GC：行业/FFC/HF/proxy + ω 频段） | 新行业映射 + 因子集 | A + B1 |
| Table V（夏普） | 各因子 + rf | B1 + 外部 |

---

## 8. 完成后的自检（建议）
- `metadata/universe.csv` 含 `is_balanced_paper` 列；`universe_summary.json` 的 `total_symbols≈5465`。
- 面板元数据 `panel_return_scheme` 含 `_intraday5min_v2`（确认 N1 生效）。
- `checkpoints/` 下出现**新哈希**的 `replication_result_*.pkl`（非 2ae5dce6）。
- `symbol_returns/` 文件数 ≈ 5465（Fig 13 unbalanced 行才有数据）。
- 出图走 task runner；逐图形态对照 `docs/06_FIGURE_FIDELITY_逐图核对.md`、口径对照 `docs/07_FINAL_FIX_逐字核对修复.md`。

> 再次声明：以上重路径**未在本环境运行验证**（沙箱无数据）；新增/改动模块均带 try/except 回退，
> 请在本机跑通后按 §8 自检与 `docs/04_VERIFICATION` 核对。
