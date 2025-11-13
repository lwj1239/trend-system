"""
趋势跟踪系统 - 主入口
统一调度整个系统
"""
import argparse
from pathlib import Path
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from core.data_loader import DataLoader
from core.trend_detector import TrendDetector
from core.signal_generator import SignalGenerator
from core.position_sizing import PositionSizer
from core.portfolio_allocator import PortfolioAllocator
from core.risk_manager import RiskManager
from optimization.parameter_search import ParameterOptimizer
from optimization.robustness_tests import RobustnessTests
from optimization.asset_selection import AssetSelector
from backtest.backtester import Backtester
from backtest.metrics import PerformanceMetrics
from backtest.visualization import Visualizer
from backtest.performance_report import ReportGenerator


def run_trend_analysis():
    """运行趋势分析"""
    print("\n" + "="*60)
    print("趋势分析模式")
    print("="*60)
    
    # 加载数据
    loader = DataLoader()
    data_dict = loader.load_all_assets()
    
    if not data_dict:
        print("❌ 没有可用数据")
        return
    
    # 趋势检测
    detector = TrendDetector()
    trend_scores = detector.get_trending_assets(data_dict)
    
    print("\n趋势得分排名:")
    print(trend_scores.to_string())
    
    # 资产筛选
    selector = AssetSelector()
    ranked = selector.rank_assets(data_dict)
    selector.print_asset_report(ranked)
    
    # 可视化
    vis = Visualizer()
    vis.plot_trend_scores(trend_scores, save_path="reports/figures/trend_scores.png")


def run_single_asset_backtest(symbol: str = "BTC"):
    """运行单资产回测"""
    print("\n" + "="*60)
    print(f"单资产回测模式: {symbol}")
    print("="*60)
    
    # 加载数据
    loader = DataLoader()
    try:
        df = loader.load_single_asset(symbol)
        df = loader.preprocess(df)
    except Exception as e:
        print(f"❌ 加载{symbol}失败: {e}")
        return
    
    # 生成信号
    signal_gen = SignalGenerator()
    signals_df = signal_gen.generate_signals(df)
    signals_df = signal_gen.calculate_position_changes(signals_df)
    
    # 回测
    backtester = Backtester()
    equity_df = backtester.run_backtest(
        {symbol: df},
        {symbol: signals_df}
    )
    
    # 计算指标
    trades_df = backtester.get_trades_df()
    metrics = PerformanceMetrics.calculate_all_metrics(
        equity_df['portfolio_value'],
        trades_df
    )
    
    # 打印报告
    PerformanceMetrics.print_metrics(metrics)
    
    # 可视化
    vis = Visualizer()
    vis.plot_equity_curve(equity_df['portfolio_value'],
                         title=f"{symbol} 权益曲线")
    vis.plot_drawdown(equity_df['portfolio_value'],
                     title=f"{symbol} 回撤曲线")
    
    # 生成报告
    reporter = ReportGenerator()
    reporter.generate_html_report(metrics, equity_df['portfolio_value'], trades_df)


def run_portfolio_backtest():
    """运行多资产组合回测"""
    print("\n" + "="*60)
    print("多资产组合回测模式")
    print("="*60)
    
    # 加载数据
    loader = DataLoader()
    data_dict = loader.load_all_assets()
    
    if not data_dict:
        print("❌ 没有可用数据")
        return
    
    # 对齐数据
    data_dict = loader.align_timestamps(data_dict)
    
    # 趋势检测
    detector = TrendDetector()
    trend_dict = {}
    signals_dict = {}
    
    for symbol, df in data_dict.items():
        # 计算趋势
        trend_df = detector.calculate_trend_score(df)
        trend_dict[symbol] = trend_df['trend_score'].iloc[-1]
        
        # 生成信号
        signal_gen = SignalGenerator()
        signals = signal_gen.generate_signals(df)
        signals = signal_gen.calculate_position_changes(signals)
        signals_dict[symbol] = signals
    
    # 投资组合分配
    allocator = PortfolioAllocator()
    returns_dict = {s: loader.get_returns(df) for s, df in data_dict.items()}
    weights = allocator.allocate_portfolio(trend_dict, returns_dict)
    
    print("\n投资组合权重:")
    for symbol, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {symbol}: {weight:.2%}")
    
    # 回测
    backtester = Backtester()
    equity_df = backtester.run_backtest(data_dict, signals_dict)
    
    # 计算指标
    trades_df = backtester.get_trades_df()
    metrics = PerformanceMetrics.calculate_all_metrics(
        equity_df['portfolio_value'],
        trades_df
    )
    
    # 风险管理报告
    risk_mgr = RiskManager()
    risk_report = risk_mgr.generate_risk_report(
        equity_df['portfolio_value'],
        equity_df['portfolio_value'].pct_change(),
        backtester.positions,
        {s: df['close'].iloc[-1] for s, df in data_dict.items()}
    )
    risk_mgr.print_risk_report(risk_report)
    
    # 打印绩效报告
    PerformanceMetrics.print_metrics(metrics)
    
    # 可视化
    vis = Visualizer()
    vis.plot_equity_curve(equity_df['portfolio_value'])
    vis.plot_drawdown(equity_df['portfolio_value'])
    
    # 生成报告
    reporter = ReportGenerator()
    reporter.generate_html_report(metrics, equity_df['portfolio_value'], trades_df)


def run_parameter_optimization():
    """运行参数优化"""
    print("\n" + "="*60)
    print("参数优化模式")
    print("="*60)
    
    # 加载数据
    loader = DataLoader()
    data_dict = loader.load_all_assets()
    
    if not data_dict:
        print("❌ 没有可用数据")
        return
    
    # 选择优化资产（使用第一个可用资产）
    symbol = list(data_dict.keys())[0]
    df = data_dict[symbol]
    print(f"\n使用资产: {symbol}")
    print(f"数据范围: {df.index[0]} 到 {df.index[-1]}")
    print(f"数据量: {len(df)} 条记录")
    
    # 定义目标函数
    def objective_function(params):
        """
        目标函数：根据参数运行回测并返回夏普比率
        """
        try:
            # 生成信号 - 手动设置参数
            signal_gen = SignalGenerator()
            # 临时修改配置
            signal_gen.signal_config['entry_period'] = int(params.get('entry_period', 55))
            signal_gen.signal_config['exit_period'] = int(params.get('exit_period', 18))
            
            signals_df = signal_gen.generate_signals(df)
            signals_df = signal_gen.calculate_position_changes(signals_df)
            
            # 回测
            backtester = Backtester()
            equity_df = backtester.run_backtest(
                {symbol: df},
                {symbol: signals_df}
            )
            
            # 计算夏普比率
            returns = equity_df['portfolio_value'].pct_change().dropna()
            if len(returns) < 2 or returns.std() == 0:
                return -999
            
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
            return sharpe
        except Exception as e:
            print(f"  参数 {params} 评估失败: {e}")
            return -999
    
    # 初始化优化器
    optimizer = ParameterOptimizer()
    
    # 定义参数空间
    print("\n" + "="*60)
    print("参数优化设置")
    print("="*60)
    print("优化方法: 网格搜索 (快速)")
    print("优化目标: 夏普比率")
    
    # 使用网格搜索（计算量较小）
    param_grid = {
        'entry_period': [20, 30, 40, 55],
        'exit_period': [10, 15, 20]
    }
    
    print("\n参数空间:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\n总组合数: {total_combinations}")
    print("\n开始优化...\n")
    
    # 执行优化
    best_params, best_score, all_results = optimizer.grid_search(
        objective_func=objective_function,
        param_grid=param_grid,
        verbose=True
    )
    
    # 打印结果
    print("\n" + "="*60)
    print("优化结果")
    print("="*60)
    print(f"最优夏普比率: {best_score:.4f}")
    print(f"最优参数:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # 显示前5个结果
    print("\n前5个最佳参数组合:")
    sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"\n{i}. 夏普比率: {result['score']:.4f}")
        print(f"   参数: {result['params']}")
    
    # 使用最优参数运行完整回测
    print("\n" + "="*60)
    print("使用最优参数运行完整回测")
    print("="*60)
    
    signal_gen = SignalGenerator()
    signal_gen.signal_config['entry_period'] = int(best_params['entry_period'])
    signal_gen.signal_config['exit_period'] = int(best_params['exit_period'])
    signals_df = signal_gen.generate_signals(df)
    signals_df = signal_gen.calculate_position_changes(signals_df)
    
    backtester = Backtester()
    equity_df = backtester.run_backtest(
        {symbol: df},
        {symbol: signals_df}
    )
    
    trades_df = backtester.get_trades_df()
    metrics = PerformanceMetrics.calculate_all_metrics(
        equity_df['portfolio_value'],
        trades_df
    )
    
    PerformanceMetrics.print_metrics(metrics)
    
    # 可视化
    vis = Visualizer()
    vis.plot_equity_curve(
        equity_df['portfolio_value'],
        title=f"{symbol} 优化后权益曲线 (Entry={best_params['entry_period']}, Exit={best_params['exit_period']})"
    )
    
    # 创建参数热力图
    if len(param_grid) == 2:
        param_names = list(param_grid.keys())
        heatmap_data = optimizer.create_heatmap_data(
            all_results,
            param_names[0],
            param_names[1]
        )
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0)
        plt.title(f'{symbol} 参数优化热力图 (夏普比率)')
        plt.xlabel(param_names[0])
        plt.ylabel(param_names[1])
        plt.tight_layout()
        plt.savefig('reports/figures/parameter_heatmap.png', dpi=150, bbox_inches='tight')
        print(f"\n参数热力图已保存至: reports/figures/parameter_heatmap.png")
        plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='趋势跟踪量化系统')
    parser.add_argument('mode', choices=['trend', 'single', 'portfolio', 'optimize'],
                       help='运行模式: trend(趋势分析), single(单资产回测), portfolio(组合回测), optimize(参数优化)')
    parser.add_argument('--symbol', type=str, default='BTC',
                       help='单资产回测时的资产代码')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 趋势跟踪量化系统")
    print("="*60)
    
    if args.mode == 'trend':
        run_trend_analysis()
    elif args.mode == 'single':
        run_single_asset_backtest(args.symbol)
    elif args.mode == 'portfolio':
        run_portfolio_backtest()
    elif args.mode == 'optimize':
        run_parameter_optimization()
    
    print("\n✅ 运行完成！")


if __name__ == "__main__":
    # 如果没有命令行参数，显示菜单
    if len(sys.argv) == 1:
        print("\n" + "="*60)
        print("🚀 趋势跟踪量化系统")
        print("="*60)
        print("\n请选择运行模式:")
        print("  1. 趋势分析 (python main.py trend)")
        print("  2. 单资产回测 (python main.py single --symbol BTC)")
        print("  3. 多资产组合回测 (python main.py portfolio)")
        print("  4. 参数优化 (python main.py optimize)")
        print("\n示例: python main.py single --symbol BTC")
        print("="*60 + "\n")
    else:
        main()
