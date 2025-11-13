"""
技术指标模块
包含ATR, ADX, Hurst指数, 效率比率等趋势相关指标
"""
import pandas as pd
import numpy as np
from typing import Tuple


class TechnicalIndicators:
    """技术指标计算类"""
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        计算平均真实波动幅度（ATR）
        
        Args:
            df: 包含high, low, close的DataFrame
            period: 计算周期
            
        Returns:
            ATR序列
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # 真实波动幅度
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR是TR的移动平均
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算平均趋向指数（ADX）及其组成部分
        
        Args:
            df: 包含high, low, close的DataFrame
            period: 计算周期
            
        Returns:
            (ADX, +DI, -DI)
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # 方向移动
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        # 只保留正向移动
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # 当+DM和-DM同时出现时，只保留较大的
        plus_dm[(plus_dm < minus_dm)] = 0
        minus_dm[(minus_dm < plus_dm)] = 0
        
        # 计算TR
        tr = TechnicalIndicators.calculate_atr(df, period=1) * period
        
        # 平滑处理
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # 计算DX和ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def calculate_hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
        """
        计算Hurst指数（衡量时间序列的趋势性）
        H > 0.5: 趋势性
        H = 0.5: 随机游走
        H < 0.5: 均值回归
        
        Args:
            series: 价格序列
            max_lag: 最大滞后期
            
        Returns:
            Hurst指数
        """
        lags = range(2, max_lag)
        tau = []
        
        for lag in lags:
            # 计算滞后差分的标准差
            std = np.std(np.subtract(series[lag:].values, series[:-lag].values))
            tau.append(std)
        
        # 线性回归 log(tau) ~ log(lag)
        try:
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = poly[0]
        except:
            hurst = 0.5  # 默认值
        
        return hurst
    
    @staticmethod
    def calculate_efficiency_ratio(df: pd.DataFrame, period: int = 10) -> pd.Series:
        """
        计算效率比率（Efficiency Ratio）
        衡量价格变动的效率
        
        Args:
            df: 包含close的DataFrame
            period: 计算周期
            
        Returns:
            效率比率序列
        """
        close = df['close']
        
        # 净价格变化
        change = abs(close - close.shift(period))
        
        # 波动性（价格变化的绝对值之和）
        volatility = close.diff().abs().rolling(window=period).sum()
        
        # 效率比率
        er = change / volatility
        er = er.fillna(0)
        
        return er
    
    @staticmethod
    def calculate_moving_averages(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> Tuple[pd.Series, pd.Series]:
        """
        计算快速和慢速移动平均线
        
        Args:
            df: 包含close的DataFrame
            fast: 快速MA周期
            slow: 慢速MA周期
            
        Returns:
            (快速MA, 慢速MA)
        """
        fast_ma = df['close'].rolling(window=fast).mean()
        slow_ma = df['close'].rolling(window=slow).mean()
        
        return fast_ma, slow_ma
    
    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算布林带
        
        Args:
            df: 包含close的DataFrame
            period: 计算周期
            std_dev: 标准差倍数
            
        Returns:
            (上轨, 中轨, 下轨)
        """
        close = df['close']
        
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        计算相对强弱指数（RSI）
        
        Args:
            df: 包含close的DataFrame
            period: 计算周期
            
        Returns:
            RSI序列
        """
        close = df['close']
        delta = close.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算MACD指标
        
        Args:
            df: 包含close的DataFrame
            fast: 快速EMA周期
            slow: 慢速EMA周期
            signal: 信号线周期
            
        Returns:
            (MACD线, 信号线, 柱状图)
        """
        close = df['close']
        
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return macd, signal_line, histogram


if __name__ == "__main__":
    # 测试代码
    print("技术指标模块加载成功")
    print("包含指标: ATR, ADX, Hurst, ER, MA, BB, RSI, MACD")
