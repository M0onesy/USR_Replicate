# 运行指南（RUN_GUIDE）—— 复用路径 vs 完整路径

## 0. 复用机制（为什么复用路径分钟级就能修好尾部）

`build_result(cfg)`（`core/pipeline_cache.py`）：
1. 算 `_signature_hash(cfg)` = sha1(`cfg.cache_signature()`)，找 `checkpoints/replication_result_{hash}.pkl`。
2. **命中**（hash==`2ae5dce6a23fd2a2`）→ `pickle.load` → `_refresh_result` → `refresh_replication_result_views`
   用**当前代码**重算 display 因子 + 重建 `paper_tail`（分钟级）→ 返回。**不重跑、不查 run_state。**
3. 未命中 → `run_cn_replication`（含 ~33h 逐年分析）。

`cache_signature()` 只含 `g_fn / return_mode / gamma / jump_a / k_max / proc_root / years / max_stocks`，
**不含代码哈希**。⇒ 改本包代码不影响命中；只有改这几个参数或改 proc_root/数据才会让 `2ae5dce6` 失效。

---

## 1. 复用路径（默认，分钟级，修好整个尾部 + 符号）

### 步骤
1. **放新映射**：把 `external_industry_new/stock_full_info_std_industry_final.csv` 拷到
   `…/external_Data/pelger_tail/industry/` 下。
2. **配置 `RunConfig`**（保持签名不变以命中缓存）：
   ```python
   cfg = RunConfig(
       proc_root=...,            # 必须与生成 2ae5dce6 时一致
       g_fn="median_N",          # ⚠️ 保持不变（改了就失效重跑）
       return_mode="open_close", # ⚠️ 保持不变
       gamma=0.08, jump_a=3.0, k_max=10, years=None, max_stocks=None,  # ⚠️ 全部保持
       # —— 以下是论文保真开关（不进签名，安全）——
       industry_info_filename="stock_full_info_std_industry_final.csv",
       paper_faithful_signs=True,         # P4
       size_value_full_market=True,       # P9
       ffc_mom_mode="carhart_daily",      # P6/P7
       annualization_days=252,            # N6（A 股可改 243）
       size_value_start="2014-07-01",     # P10
       industry_factors_frozen=None,      # P5/D1：第一次跑后再填 3 个桶
       refresh_paper_tail=True,
   )
   ```
   （等价地，也可直接设同名环境变量，见 02_SPEC。）
3. **清掉旧的 paper_tail 视图缓存**（让 PAPER_TAIL_VERSION=4 重建）：删除 `…/paper_tail/` 下旧产物
   （或确保 `refresh_paper_tail=True`）。**不要**删 `checkpoints/replication_result_2ae5dce6*.pkl`。
4. **跑** main（你平时出图出表的入口）。命中缓存 → 分钟级生成全部修好后的图表。

### 这一步会修好
P4 符号、P5 行业（占位/或冻结）、P6 自建 MOM、P7 分段 FFC、P8 rf 拆分、P9 全市场 2×3、
P10 窗、P11 Fig13、P12 Panel B、P13、N2、N4、N6、N7、N8、N9、N10。

### D1 冻结行业（第二次跑）
第一次复用跑完后，看 `Figure_04`（连续 PCA 权重，已定向）与 `industry_selection.json` 的占位结果，
选经济上合理的 3 个 `std_industry` 桶，回填：
```python
cfg.industry_factors_frozen = ["大金融", "电力设备与新能源", "科技成长"]  # 示例，按你看到的为准
```
再跑一次（仍复用），行业因子即冻结，Table III / Figure 14 用这组固定桶。

---

## 2. 完整路径（贴近论文，需重跑 ~33h；当你有时间时）

在复用路径基础上额外应用 P1/P2/N1：
```bash
# 预处理：纯盘中 5min + 论文宽松平衡面板
export PELGER_INTRADAY_ONLY_5MIN=1
export PELGER_BALANCED_MODE=paper_lenient
export PELGER_BALANCED_MIN_COVERAGE=0.96
python Code/preprocess_cn_data.py --refresh        # 重建 proc_data（数据口径变了）
```
```python
cfg = replace(cfg, g_fn="median_sqrtN", restart=True)   # P1 + 强制重跑
build_result(cfg)   # 触发 run_cn_replication：主 pipeline 2 秒，但逐年 paper 分析 ~33h
```
- ⚠️ 这会**重跑 33 小时的逐年 PCA**（Table I/II、Figure 1/2、factor counts），因为 N1/P2 改变了逐年面板输入。
- ⚠️ 需先按 02_SPEC「E. 面板插值伪码」在面板写出处补上缺失日插值，paper_lenient 面板才正确。
- 完整路径额外修好：P1 g_fn、P2 平衡面板、N1 纯盘中、P3 逐年 K 显示、N11 rolling K、N12 月度面板。

---

## 3. 常见问题
- **跑完和论文数值不一致？** 允许——A 股与美股结果本就不同（涨跌停、停牌、行业结构）。要核对的是**构造逻辑**与**模式**（见 04_VERIFICATION），不是数值复制。
- **想临时关掉某项？** 用对应 env（如 `PELGER_PAPER_FAITHFUL_SIGNS=0`、`PELGER_SIZE_VALUE_FULL_MARKET=0`、`PELGER_FFC_MOM_MODE=legacy_hf`）。
- **paper_tail 报错？** 复用路径只花分钟级；看日志里 `[paper_fidelity]/[paper_tail]` 的回退提示，定位后重跑即可（不影响 33h 重结果）。


## 【更新】g_fn 默认与缓存（第四轮）
- `g_fn` 默认已改为论文口径 **median_sqrtN**（√N·median）。因 cache_signature 含 g_fn，
  **旧重结果缓存 `2ae5dce6…` 不再命中**——需 `restart=True` 重跑（你已计划补数据重跑）。
- 论文口径一次到位：`PELGER_BALANCED_MODE=paper_lenient`（缺失日自动插值为零增量）+ 默认 g_fn=median_sqrtN +
  N1 隔夜独立（默认开）；重跑后走 task runner 出全部图表。
- 若要对照旧缓存：显式 `g_fn="median_N"`（= N·median，非论文口径）。
