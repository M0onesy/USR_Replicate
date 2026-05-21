# Data 目录说明

本目录只保存数据和处理产物，不保存复现代码。

## 上游原始数据

- `kline_Data/EXTRA_STOCK_A`：原始 A 股 5 分钟 K 线。
- 只有 `Code/preprocess_cn_data.py` 会读取这里的 `.bz2` 文件。

## 复权因子

- `fact_Data/backward_factor.csv`：默认后复权因子。
- 预处理时使用 `adjusted_ohlc = raw_ohlc * backward_factor`。

## 预处理产物

- `proc_Data/pelger_cn_adjusted`：预处理输出。
- `R_daily`：日度总收益。
- `R_intra`：当日开盘到收盘的日内收益。
- `R_night`：前一收盘到当日开盘的隔夜收益。
- `R_5min_full`：5 分钟收盘接续收益。
- `Code/allcode_Need.py` 只读取这里的面板和元数据。

推荐顺序：先运行 `Code/preprocess_cn_data.py`，再运行 `Code/allcode_Need.py`。
