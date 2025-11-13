# 🧠 趋势跟踪系统项目蓝图（Crypto Trend-Following System Blueprint）

```
trend_system/
│
├── config/
│   ├── settings.yaml              # 全局配置文件（参数、风险限制、资产池）
│   └── credentials.yaml           # API密钥等（回测可忽略）
│
├── data/
│   ├── btc.csv
│   ├── eth.csv
│   └── ...                        # 历史K线数据（1d或4h）
│
├── core/
│   ├── data_loader.py             # 加载与预处理数据
│   ├── indicators.py              # 技术指标模块（ATR, ADX, Hurst, ER等）
│   ├── trend_detector.py          # 趋势识别模块（trend_score函数）
│   ├── signal_generator.py        # 海龟系统信号生成（突破/均线）
│   ├── position_sizing.py         # 仓位控制（ATR止损、波动率目标）
│   ├── portfolio_allocator.py     # 多币组合优化（风险平价 / 趋势权重）
│   └── risk_manager.py            # 风险约束（回撤、暴露、再平衡）
│
├── optimization/
│   ├── parameter_search.py        # 参数优化（Bayesian / Grid / GA）
│   ├── robustness_tests.py        # 扰动测试、WFA、蒙特卡洛
│   └── asset_selection.py         # 资产池筛选（趋势强度 / 稳健性评分）
│
├── backtest/
│   ├── backtester.py              # 回测引擎（支持多币多参数）
│   ├── metrics.py                 # 夏普率、Calmar、Sortino等
│   ├── performance_report.py      # 生成分析报告（PDF/HTML）
│   └── visualization.py           # 热力图、回撤曲线、信号对比图
│
├── research/
│   ├── trend_analysis.ipynb       # 趋势特征研究（Hurst、ADX、R²分析）
│   ├── parameter_heatmap.ipynb    # 参数敏感性热力图
│   └── asset_compare.ipynb        # 各币趋势稳定性对比
│
├── main.py                        # 主入口，统一调度整个系统
└── README.md                      # 项目说明与研究方法
```

## 🧠 二、建议执行顺序

| 周次 | 任务 | 输出 |
|------|------|------|
| 第1周 | 数据与趋势检测模块 | 各币趋势得分表 |
| 第2周 | 信号与仓位模块 | 单币回测图表 |
| 第3周 | 参数优化模块 | entry/exit热力图 |
| 第4周 | 稳健性测试模块 | 扰动与WFA报告 |
| 第5周 | 多币组合与最终报告 | 组合绩效与监控 |
