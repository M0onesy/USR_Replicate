"""
core —— 共享计算与基础设施层

包含：
  engine.py         论文复现的全部"重活"（由 allcode_Need.py 逐字节迁移，数学不变）
  config.py         单次运行参数（RunConfig）
  pipeline_cache.py 一次计算、多处复用的缓存层（Option A 核心）
  logging_utils.py  每脚本日志 + main 后台心跳
  io_utils.py       图表输出目录、权威文件名、滚动结果切片、底层绘图/落表复用
  runner.py         图表脚本统一运行脚手架（支持单独运行与被 main 调用）
  registry.py       所有图/表任务的总目录，供 main 发现与调度
"""
