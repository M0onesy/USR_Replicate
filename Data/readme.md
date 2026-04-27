# Data 目录说明

本目录只保存本地数据和处理产物，不保存论文复现代码。

## 上游原始数据

- `kline_Data\EXTRA_STOCK_A\`：原始 A 股 5 分钟 K 线目录。
- 当前本地该目录下有 `5799` 个股票目录，每个目录应包含一个 `data.bz2`。
- 原始压缩文件只由 `Code\preprocess_cn_data.py` 读取。

## 复权因子

- `fact_Data\backward_factor.csv`：默认使用的后复权因子。
- 预处理公式为 `adjusted_ohlc = raw_ohlc * backward_factor`。

## 预处理产物

- `proc_Data\pelger_cn_adjusted\`：预处理脚本输出目录。
- 该目录包含 `manifest.json`、`metadata\`、`symbol_returns\` 和 `panels\`。
- `Code\allcode_Need.py` 只读取这里的处理产物，不直接读取原始 K 线。

推荐顺序：先运行 `Code\preprocess_cn_data.py`，再运行 `Code\allcode_Need.py`。
