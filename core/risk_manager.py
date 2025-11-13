"""
风险管理模块
监控和控制投资组合风险
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import yaml


class RiskManager:
    """风险管理器"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """初始化风险管理器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.risk_config = self.config['risk_management']
        self.initial_capital = self.config['backtest']['initial_capital']
    
    def calculate_portfolio_exposure(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float]
    ) -> float:
        """
        计算投资组合总暴露
        
        Args:
            positions: 各资产仓位字典（合约数）
            prices: 各资产价格字典
            
        Returns:
            总暴露比例
        """
        total_value = sum(
            abs(positions.get(symbol, 0) * prices.get(symbol, 0))
            for symbol in set(positions.keys()) | set(prices.keys())
        )
        
        exposure_ratio = total_value / self.initial_capital
        
        return exposure_ratio
    
    def check_max_exposure(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float]
    ) -> Tuple[bool, float]:
        """
        检查是否超过最大暴露限制
        
        Args:
            positions: 仓位字典
            prices: 价格字典
            
        Returns:
            (是否超限, 当前暴露比例)
        """
        current_exposure = self.calculate_portfolio_exposure(positions, prices)
        max_exposure = self.risk_config['max_exposure']
        
        return current_exposure > max_exposure, current_exposure
    
    def calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """
        计算回撤序列
        
        Args:
            equity_curve: 权益曲线
            
        Returns:
            回撤序列
        """
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        
        return drawdown
    
    def check_max_drawdown(self, equity_curve: pd.Series) -> Tuple[bool, float]:
        """
        检查是否超过最大回撤限制
        
        Args:
            equity_curve: 权益曲线
            
        Returns:
            (是否超限, 当前最大回撤)
        """
        drawdown = self.calculate_drawdown(equity_curve)
        current_max_dd = abs(drawdown.min())
        max_dd_limit = self.risk_config['max_drawdown']
        
        return current_max_dd > max_dd_limit, current_max_dd
    
    def check_position_limit(
        self,
        symbol: str,
        position_value: float,
        total_capital: float
    ) -> Tuple[bool, float]:
        """
        检查单个资产仓位是否超限
        
        Args:
            symbol: 资产代码
            position_value: 仓位价值
            total_capital: 总资金
            
        Returns:
            (是否超限, 当前仓位比例)
        """
        position_ratio = abs(position_value) / total_capital
        max_position = self.config['position_sizing']['max_position']
        
        return position_ratio > max_position, position_ratio
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        window: int = 252
    ) -> float:
        """
        计算风险价值（Value at Risk）
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
            window: 计算窗口
            
        Returns:
            VaR值
        """
        recent_returns = returns.tail(window)
        var = recent_returns.quantile(1 - confidence_level)
        
        return abs(var)
    
    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        window: int = 252
    ) -> float:
        """
        计算条件风险价值（Conditional VaR / Expected Shortfall）
        
        Args:
            returns: 收益率序列
            confidence_level: 置信水平
            window: 计算窗口
            
        Returns:
            CVaR值
        """
        recent_returns = returns.tail(window)
        var = recent_returns.quantile(1 - confidence_level)
        cvar = recent_returns[recent_returns <= var].mean()
        
        return abs(cvar)
    
    def risk_adjusted_position(
        self,
        target_position: float,
        current_dd: float,
        current_volatility: float
    ) -> float:
        """
        根据风险状况调整仓位
        
        Args:
            target_position: 目标仓位
            current_dd: 当前回撤
            current_volatility: 当前波动率
            
        Returns:
            调整后的仓位
        """
        # 回撤调整因子
        max_dd = self.risk_config['max_drawdown']
        dd_factor = 1 - (current_dd / max_dd) if current_dd < max_dd else 0
        
        # 波动率调整因子
        target_vol = self.config['position_sizing']['volatility_target']
        vol_factor = target_vol / current_volatility if current_volatility > 0 else 1
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        
        # 综合调整
        adjusted_position = target_position * dd_factor * vol_factor
        
        return adjusted_position
    
    def should_halt_trading(
        self,
        equity_curve: pd.Series,
        current_positions: Dict[str, float]
    ) -> Tuple[bool, str]:
        """
        判断是否应该停止交易
        
        Args:
            equity_curve: 权益曲线
            current_positions: 当前仓位
            
        Returns:
            (是否停止, 原因)
        """
        # 检查回撤
        is_dd_exceeded, current_dd = self.check_max_drawdown(equity_curve)
        if is_dd_exceeded:
            return True, f"超过最大回撤限制: {current_dd:.2%}"
        
        # 检查资金损失
        current_capital = equity_curve.iloc[-1]
        loss_ratio = (self.initial_capital - current_capital) / self.initial_capital
        
        if loss_ratio > 0.5:  # 损失超过50%
            return True, f"账户损失过大: {loss_ratio:.2%}"
        
        return False, ""
    
    def generate_risk_report(
        self,
        equity_curve: pd.Series,
        returns: pd.Series,
        positions: Dict[str, float],
        prices: Dict[str, float]
    ) -> Dict:
        """
        生成风险报告
        
        Args:
            equity_curve: 权益曲线
            returns: 收益率序列
            positions: 仓位字典
            prices: 价格字典
            
        Returns:
            风险报告字典
        """
        drawdown = self.calculate_drawdown(equity_curve)
        current_dd = abs(drawdown.iloc[-1])
        max_dd = abs(drawdown.min())
        
        exposure = self.calculate_portfolio_exposure(positions, prices)
        
        var_95 = self.calculate_var(returns, 0.95)
        cvar_95 = self.calculate_cvar(returns, 0.95)
        
        volatility = returns.tail(60).std() * np.sqrt(252)
        
        report = {
            'current_drawdown': current_dd,
            'max_drawdown': max_dd,
            'total_exposure': exposure,
            'volatility_annual': volatility,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'num_positions': len([p for p in positions.values() if p != 0]),
            'max_exposure_limit': self.risk_config['max_exposure'],
            'max_drawdown_limit': self.risk_config['max_drawdown']
        }
        
        return report
    
    def print_risk_report(self, report: Dict):
        """打印风险报告"""
        print("\n" + "="*50)
        print("风险报告")
        print("="*50)
        print(f"当前回撤: {report['current_drawdown']:.2%}")
        print(f"最大回撤: {report['max_drawdown']:.2%} (限制: {report['max_drawdown_limit']:.2%})")
        print(f"总暴露: {report['total_exposure']:.2%} (限制: {report['max_exposure_limit']:.2%})")
        print(f"年化波动率: {report['volatility_annual']:.2%}")
        print(f"VaR(95%): {report['var_95']:.2%}")
        print(f"CVaR(95%): {report['cvar_95']:.2%}")
        print(f"持仓数量: {report['num_positions']}")
        print("="*50 + "\n")


if __name__ == "__main__":
    # 测试代码
    print("风险管理模块加载成功")
    print("功能: 回撤监控、暴露控制、VaR计算")
