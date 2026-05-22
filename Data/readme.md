# Data 目录说明

`Data/` 只存放原始数据、复权因子和预处理产物，不存放论文复现主代码。

## 上游：原始 K 线

- `kline_Data/EXTRA_STOCK_A`
  原始 A 股 5 分钟 K 线目录，每只股票一个子目录，内部包含 `data.bz2`。
- 只有 `Code/preprocess_cn_data.py` 会直接读取这里的原始 `.bz2` 文件。

## 复权因子

- `fact_Data/backward_factor.csv`
  默认使用的后复权因子。
- 预处理阶段统一按下面的公式构造后复权价格：

```text
adjusted_ohlc = raw_ohlc * backward_factor
```

## 下游：预处理产物

- `proc_Data/pelger_cn_adjusted`
  预处理输出根目录。

其中主要包含：

- `metadata/`
  样本清单、覆盖统计、manifest 和 universe summary。
- `symbol_returns/`
  逐股票收益缓存，供构造 unbalanced 样本和调试使用。
- `panels/strict_balanced/full`
  全区间严格平衡面板。
- `panels/strict_balanced/year_YYYY`
  年度严格平衡面板。

面板中的主要数组口径：

- `R_daily`
  日度总对数收益。
- `R_intra`
  日内日频对数收益。
- `R_night`
  隔夜日频对数收益。
- `R_5min_full`
  5 分钟连续收盘接续对数收益，是论文高频主流程输入。

## 与复现脚本的关系

- `Code/preprocess_cn_data.py`
  负责读取原始 `.bz2`、做后复权、清洗坏日、生成面板和元数据。
- `Code/allcode_Need.py`
  只读取 `proc_Data/pelger_cn_adjusted`，不再回退到原始 K 线。

推荐顺序：

1. 先运行 `Code/preprocess_cn_data.py`
2. 再运行 `Code/allcode_Need.py`

如果你更新了原始 K 线或 `backward_factor.csv`，请重新运行预处理，必要时加 `--refresh` 重建产物。
