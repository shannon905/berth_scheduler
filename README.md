# 港口泊位调度优化器

基于整数规划（PuLP）求解港口泊位调度问题，最小化船舶总等待时间。

## 运行方法
1. 安装依赖：pip install pulp flask
2. 运行：python berth_scheduler.py

## API接口
POST /optimize
输入船舶列表和泊位数量，返回最优调度方案。

## 开源协议
MIT License
