"""
信号生成模块
基于海龟交易系统的突破和均线交叉策略
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from .indicators import TechnicalIndicators
import yaml


class SignalGenerator:
    """交易信号生成器"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """初始化信号生成器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.signal_config = self.config['signal']
        self.indicators = TechnicalIndicators()
    
    def generate_breakout_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成突破信号（海龟系统1号）
        
        Args:
            df: 包含OHLCV的DataFrame
            
        Returns:
            包含信号的DataFrame
        """
        result = df.copy()
        
        entry_period = self.signal_config['entry_period']
        exit_period = self.signal_config['exit_period']
        
        # 唐奇安通道（Donchian Channel）
        result['entry_high'] = result['high'].rolling(window=entry_period).max()
        result['entry_low'] = result['low'].rolling(window=entry_period).min()
        result['exit_high'] = result['high'].rolling(window=exit_period).max()
        result['exit_low'] = result['low'].rolling(window=exit_period).min()
        
        # 生成入场信号
        result['long_entry'] = (result['close'] > result['entry_high'].shift(1))
        result['short_entry'] = (result['close'] < result['entry_low'].shift(1))
        
        # 生成出场信号
        result['long_exit'] = (result['close'] < result['exit_low'].shift(1))
        result['short_exit'] = (result['close'] > result['exit_high'].shift(1))
        
        # 综合信号（1=做多，-1=做空，0=无仓位）
        result['signal'] = 0
        result.loc[result['long_entry'], 'signal'] = 1
        result.loc[result['short_entry'], 'signal'] = -1
        result.loc[result['long_exit'], 'signal'] = 0
        result.loc[result['short_exit'], 'signal'] = 0
        
        return result
    
    def generate_ma_cross_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成均线交叉信号
        
        Args:
            df: 包含OHLCV的DataFrame
            
        Returns:
            包含信号的DataFrame
        """
        result = df.copy()
        
        fast_period = self.signal_config['fast_ma']
        slow_period = self.signal_config['slow_ma']
        
        # 计算快慢均线
        fast_ma, slow_ma = self.indicators.calculate_moving_averages(
            df, fast=fast_period, slow=slow_period
        )
        result['fast_ma'] = fast_ma
        result['slow_ma'] = slow_ma
        
        # 均线交叉信号
        result['ma_cross'] = 0
        
        # 金叉：快线上穿慢线
        golden_cross = (
            (result['fast_ma'] > result['slow_ma']) & 
            (result['fast_ma'].shift(1) <= result['slow_ma'].shift(1))
        )
        result.loc[golden_cross, 'ma_cross'] = 1
        
        # 死叉：快线下穿慢线
        death_cross = (
            (result['fast_ma'] < result['slow_ma']) & 
            (result['fast_ma'].shift(1) >= result['slow_ma'].shift(1))
        )
        result.loc[death_cross, 'ma_cross'] = -1
        
        # 持续信号
        result['signal'] = 0
        result.loc[result['fast_ma'] > result['slow_ma'], 'signal'] = 1
        result.loc[result['fast_ma'] < result['slow_ma'], 'signal'] = -1
        
        return result
    
    def generate_signals(self, df: pd.DataFrame, signal_type: str = None) -> pd.DataFrame:
        """
        生成交易信号（根据配置选择策略）
        
        Args:
            df: 包含OHLCV的DataFrame
            signal_type: 信号类型（breakout或ma_cross），None则使用配置
            
        Returns:
            包含信号的DataFrame
        """
        if signal_type is None:
            signal_type = self.signal_config['signal_type']
        
        if signal_type == 'breakout':
            return self.generate_breakout_signals(df)
        elif signal_type == 'ma_cross':
            return self.generate_ma_cross_signals(df)
        else:
            raise ValueError(f"未知的信号类型: {signal_type}")
    
    def add_filters(self, df: pd.DataFrame, trend_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        添加信号过滤器（仅在趋势状态下交易）
        
        Args:
            df: 包含信号的DataFrame
            trend_data: 包含趋势指标的DataFrame
            
        Returns:
            过滤后的信号DataFrame
        """
        result = df.copy()
        
        if trend_data is not None:
            # 仅在强趋势时交易
            strong_trend = trend_data['trend_state'].isin(['strong_up', 'strong_down'])
            result.loc[~strong_trend, 'signal'] = 0
        
        return result
    
    def calculate_position_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算仓位变化（用于回测）
        
        Args:
            df: 包含信号的DataFrame
            
        Returns:
            包含仓位变化的DataFrame
        """
        result = df.copy()
        
        # 仓位变化
        result['position'] = result['signal'].ffill().fillna(0)
        result['position_change'] = result['position'].diff()
        
        # 标记交易动作
        result['action'] = 'HOLD'
        result.loc[result['position_change'] > 0, 'action'] = 'BUY'
        result.loc[result['position_change'] < 0, 'action'] = 'SELL'
        result.loc[(result['position'] == 0) & (result['position'].shift(1) != 0), 'action'] = 'CLOSE'
        
        return result
    
    def get_trade_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        获取所有交易信号点
        
        Args:
            df: 包含仓位变化的DataFrame
            
        Returns:
            仅包含交易信号的DataFrame
        """
        trades = df[df['action'].isin(['BUY', 'SELL', 'CLOSE'])].copy()
        return trades
    
    def backfill_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        向前填充信号（持续持仓）
        
        Args:
            df: 包含信号的DataFrame
            
        Returns:
            填充后的DataFrame
        """
        result = df.copy()
        result['signal'] = result['signal'].replace(0, np.nan).ffill().fillna(0)
        return result


if __name__ == "__main__":
    # 测试代码
    print("信号生成模块加载成功")
    print("支持策略: 突破系统(Breakout)、均线交叉(MA Cross)")
