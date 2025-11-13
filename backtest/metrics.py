"""
绩效指标计算模块
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple


class PerformanceMetrics:
    """绩效指标计算器"""
    
    @staticmethod
    def calculate_returns(equity_curve: pd.Series) -> pd.Series:
        """计算收益率"""
        return equity_curve.pct_change().dropna()
    
    @staticmethod
    def calculate_total_return(equity_curve: pd.Series) -> float:
        """计算总收益率"""
        return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    
    @staticmethod
    def calculate_annualized_return(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
        """计算年化收益率"""
        total_return = PerformanceMetrics.calculate_total_return(equity_curve)
        n_periods = len(equity_curve)
        years = n_periods / periods_per_year
        
        if years > 0:
            annualized = (1 + total_return) ** (1 / years) - 1
            return annualized
        return 0
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
        """计算年化波动率"""
        return returns.std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """计算夏普比率"""
        excess_returns = returns - (risk_free_rate / periods_per_year)
        
        if returns.std() == 0:
            return 0
        
        sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(periods_per_year)
        return sharpe
    
    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """计算索提诺比率（只考虑下行波动）"""
        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        
        sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(periods_per_year)
        return sortino
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        计算最大回撤
        
        Returns:
            (最大回撤, 开始时间, 结束时间)
        """
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_end = drawdown.idxmin()
        
        # 找到回撤开始点
        max_dd_start = equity_curve[:max_dd_end].idxmax()
        
        return abs(max_dd), max_dd_start, max_dd_end
    
    @staticmethod
    def calculate_calmar_ratio(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
        """计算卡玛比率（年化收益/最大回撤）"""
        ann_return = PerformanceMetrics.calculate_annualized_return(equity_curve, periods_per_year)
        max_dd, _, _ = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        
        if max_dd == 0:
            return 0
        
        return ann_return / max_dd
    
    @staticmethod
    def calculate_win_rate(trades_df: pd.DataFrame) -> float:
        """计算胜率"""
        if len(trades_df) == 0 or 'pnl' not in trades_df.columns:
            return 0
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        return len(winning_trades) / len(trades_df)
    
    @staticmethod
    def calculate_profit_factor(trades_df: pd.DataFrame) -> float:
        """计算盈利因子（总盈利/总亏损）"""
        if len(trades_df) == 0 or 'pnl' not in trades_df.columns:
            return 0
        
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        
        if gross_loss == 0:
            return 0
        
        return gross_profit / gross_loss
    
    @staticmethod
    def calculate_all_metrics(
        equity_curve: pd.Series,
        trades_df: pd.DataFrame = None,
        periods_per_year: int = 252
    ) -> Dict:
        """
        计算所有绩效指标
        
        Args:
            equity_curve: 权益曲线
            trades_df: 交易记录
            periods_per_year: 每年周期数
            
        Returns:
            指标字典
        """
        returns = PerformanceMetrics.calculate_returns(equity_curve)
        
        metrics = {
            'total_return': PerformanceMetrics.calculate_total_return(equity_curve),
            'annualized_return': PerformanceMetrics.calculate_annualized_return(equity_curve, periods_per_year),
            'volatility': PerformanceMetrics.calculate_volatility(returns, periods_per_year),
            'sharpe_ratio': PerformanceMetrics.calculate_sharpe_ratio(returns, periods_per_year=periods_per_year),
            'sortino_ratio': PerformanceMetrics.calculate_sortino_ratio(returns, periods_per_year=periods_per_year),
            'calmar_ratio': PerformanceMetrics.calculate_calmar_ratio(equity_curve, periods_per_year),
        }
        
        max_dd, dd_start, dd_end = PerformanceMetrics.calculate_max_drawdown(equity_curve)
        metrics['max_drawdown'] = max_dd
        metrics['max_dd_start'] = dd_start
        metrics['max_dd_end'] = dd_end
        
        if trades_df is not None and len(trades_df) > 0:
            metrics['num_trades'] = len(trades_df)
            metrics['win_rate'] = PerformanceMetrics.calculate_win_rate(trades_df)
            metrics['profit_factor'] = PerformanceMetrics.calculate_profit_factor(trades_df)
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict):
        """打印指标报告"""
        print("\n" + "="*60)
        print("绩效指标报告")
        print("="*60)
        print(f"总收益率:        {metrics['total_return']:>10.2%}")
        print(f"年化收益率:      {metrics['annualized_return']:>10.2%}")
        print(f"年化波动率:      {metrics['volatility']:>10.2%}")
        print(f"夏普比率:        {metrics['sharpe_ratio']:>10.2f}")
        print(f"索提诺比率:      {metrics['sortino_ratio']:>10.2f}")
        print(f"卡玛比率:        {metrics['calmar_ratio']:>10.2f}")
        print(f"最大回撤:        {metrics['max_drawdown']:>10.2%}")
        
        if 'num_trades' in metrics:
            print(f"\n交易次数:        {metrics['num_trades']:>10}")
            print(f"胜率:            {metrics['win_rate']:>10.2%}")
            print(f"盈利因子:        {metrics['profit_factor']:>10.2f}")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    print("绩效指标模块加载成功")
    print("支持指标: Sharpe, Sortino, Calmar, 最大回撤等")
