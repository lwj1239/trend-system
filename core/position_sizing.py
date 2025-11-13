"""
仓位管理模块
基于ATR的动态仓位控制
"""
import pandas as pd
import numpy as np
from typing import Dict
from .indicators import TechnicalIndicators
import yaml


class PositionSizer:
    """仓位管理器"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """初始化仓位管理器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.position_config = self.config['position_sizing']
        self.risk_config = self.config['risk_management']
        self.backtest_config = self.config['backtest']
        
        self.indicators = TechnicalIndicators()
        self.initial_capital = self.backtest_config['initial_capital']
    
    def calculate_position_size(
        self, 
        df: pd.DataFrame, 
        capital: float = None,
        risk_per_trade: float = None
    ) -> pd.Series:
        """
        基于ATR计算仓位大小
        
        Args:
            df: 包含价格和ATR的DataFrame
            capital: 账户资金
            risk_per_trade: 单次交易风险比例
            
        Returns:
            仓位大小序列（合约数量）
        """
        if capital is None:
            capital = self.initial_capital
        
        if risk_per_trade is None:
            risk_per_trade = self.position_config['risk_per_trade']
        
        # 计算ATR
        if 'atr' not in df.columns:
            atr = self.indicators.calculate_atr(
                df, period=self.config['trend_detection']['atr_period']
            )
        else:
            atr = df['atr']
        
        # 每次交易的风险金额
        risk_amount = capital * risk_per_trade
        
        # ATR止损倍数
        atr_multiplier = self.position_config['atr_multiplier']
        
        # 仓位大小 = 风险金额 / (ATR * 止损倍数)
        position_size = risk_amount / (atr * atr_multiplier)
        
        # 根据最大仓位限制调整
        max_position_value = capital * self.position_config['max_position']
        max_contracts = max_position_value / df['close']
        
        position_size = np.minimum(position_size, max_contracts)
        
        return position_size
    
    def calculate_volatility_position(
        self,
        df: pd.DataFrame,
        capital: float = None
    ) -> pd.Series:
        """
        基于波动率目标的仓位管理
        
        Args:
            df: 包含价格数据的DataFrame
            capital: 账户资金
            
        Returns:
            仓位大小序列
        """
        if capital is None:
            capital = self.initial_capital
        
        # 计算历史波动率（年化）
        returns = df['close'].pct_change()
        realized_vol = returns.rolling(window=20).std() * np.sqrt(252)
        
        # 波动率目标
        target_vol = self.position_config['volatility_target']
        
        # 仓位 = 目标波动率 / 实际波动率
        vol_scalar = target_vol / realized_vol
        vol_scalar = np.clip(vol_scalar, 0.2, 2.0)  # 限制在0.2-2倍之间
        
        # 基础仓位（账户的一定比例）
        base_position = capital / df['close']
        
        # 调整后的仓位
        position_size = base_position * vol_scalar
        
        # 应用最大仓位限制
        max_position_value = capital * self.position_config['max_position']
        max_contracts = max_position_value / df['close']
        
        position_size = np.minimum(position_size, max_contracts)
        
        return position_size
    
    def calculate_stop_loss(self, df: pd.DataFrame, entry_price: float, direction: int) -> float:
        """
        计算止损价格
        
        Args:
            df: 包含ATR的DataFrame
            entry_price: 入场价格
            direction: 方向（1=多，-1=空）
            
        Returns:
            止损价格
        """
        if 'atr' not in df.columns:
            atr = self.indicators.calculate_atr(df).iloc[-1]
        else:
            atr = df['atr'].iloc[-1]
        
        atr_multiplier = self.risk_config['stop_loss_atr']
        
        if direction > 0:  # 做多
            stop_loss = entry_price - (atr * atr_multiplier)
        else:  # 做空
            stop_loss = entry_price + (atr * atr_multiplier)
        
        return stop_loss
    
    def calculate_take_profit(self, df: pd.DataFrame, entry_price: float, direction: int) -> float:
        """
        计算止盈价格
        
        Args:
            df: 包含ATR的DataFrame
            entry_price: 入场价格
            direction: 方向（1=多，-1=空）
            
        Returns:
            止盈价格
        """
        if 'atr' not in df.columns:
            atr = self.indicators.calculate_atr(df).iloc[-1]
        else:
            atr = df['atr'].iloc[-1]
        
        atr_multiplier = self.risk_config['take_profit_atr']
        
        if direction > 0:  # 做多
            take_profit = entry_price + (atr * atr_multiplier)
        else:  # 做空
            take_profit = entry_price - (atr * atr_multiplier)
        
        return take_profit
    
    def update_trailing_stop(
        self,
        current_price: float,
        entry_price: float,
        current_stop: float,
        direction: int,
        atr: float
    ) -> float:
        """
        更新跟踪止损
        
        Args:
            current_price: 当前价格
            entry_price: 入场价格
            current_stop: 当前止损价
            direction: 方向（1=多，-1=空）
            atr: 当前ATR
            
        Returns:
            新的止损价格
        """
        if not self.risk_config['trailing_stop']:
            return current_stop
        
        atr_multiplier = self.risk_config['stop_loss_atr']
        
        if direction > 0:  # 做多
            new_stop = current_price - (atr * atr_multiplier)
            return max(new_stop, current_stop)  # 只能上移
        else:  # 做空
            new_stop = current_price + (atr * atr_multiplier)
            return min(new_stop, current_stop)  # 只能下移
    
    def apply_risk_limits(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        total_capital: float
    ) -> Dict[str, float]:
        """
        应用风险限制
        
        Args:
            positions: 各资产仓位字典
            prices: 各资产价格字典
            total_capital: 总资金
            
        Returns:
            调整后的仓位字典
        """
        # 计算总暴露
        total_exposure = sum(
            abs(pos * prices[symbol]) 
            for symbol, pos in positions.items()
        )
        
        max_exposure = total_capital * self.risk_config['max_exposure']
        
        # 如果超过最大暴露，等比例缩减
        if total_exposure > max_exposure:
            scale_factor = max_exposure / total_exposure
            positions = {
                symbol: pos * scale_factor 
                for symbol, pos in positions.items()
            }
        
        return positions


if __name__ == "__main__":
    # 测试代码
    print("仓位管理模块加载成功")
    print("功能: ATR仓位计算、波动率目标、止损止盈")
