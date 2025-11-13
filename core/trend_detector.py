"""
趋势识别模块
综合多个指标判断市场趋势强度
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from .indicators import TechnicalIndicators
import yaml


class TrendDetector:
    """趋势检测器"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """初始化趋势检测器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.trend_config = self.config['trend_detection']
        self.indicators = TechnicalIndicators()
    
    def calculate_trend_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算综合趋势得分
        
        Args:
            df: 包含OHLCV的DataFrame
            
        Returns:
            包含趋势得分及各组成部分的DataFrame
        """
        result = df.copy()
        
        # 1. ADX趋势强度
        adx, plus_di, minus_di = self.indicators.calculate_adx(
            df, period=self.trend_config['adx_period']
        )
        result['adx'] = adx
        result['plus_di'] = plus_di
        result['minus_di'] = minus_di
        
        # ADX得分：归一化到0-1
        adx_score = np.clip(adx / 50, 0, 1)
        result['adx_score'] = adx_score
        
        # 2. Hurst指数（滚动计算）
        hurst_values = self._rolling_hurst(
            result['close'], 
            window=self.trend_config['hurst_window']
        )
        result['hurst'] = hurst_values
        
        # Hurst得分：H>0.5表示趋势性
        hurst_score = np.clip((hurst_values - 0.5) * 2, 0, 1)
        result['hurst_score'] = hurst_score
        
        # 3. 效率比率
        er = self.indicators.calculate_efficiency_ratio(
            df, period=self.trend_config['er_period']
        )
        result['efficiency_ratio'] = er
        result['er_score'] = er  # ER本身就在0-1之间
        
        # 4. 趋势方向（基于DI）
        result['trend_direction'] = np.where(plus_di > minus_di, 1, -1)
        
        # 5. 价格动量
        momentum = result['close'].pct_change(20)
        momentum_score = np.clip(abs(momentum) * 10, 0, 1)
        result['momentum_score'] = momentum_score
        
        # 6. 综合趋势得分（加权平均）
        weights = {
            'adx': 0.35,
            'hurst': 0.25,
            'er': 0.20,
            'momentum': 0.20
        }
        
        result['trend_score'] = (
            weights['adx'] * result['adx_score'] +
            weights['hurst'] * result['hurst_score'] +
            weights['er'] * result['er_score'] +
            weights['momentum'] * result['momentum_score']
        )
        
        # 7. 趋势状态（强/弱/无）
        result['trend_state'] = self._classify_trend(
            result['trend_score'],
            result['trend_direction']
        )
        
        return result
    
    def _rolling_hurst(self, series: pd.Series, window: int = 100) -> pd.Series:
        """
        滚动计算Hurst指数
        
        Args:
            series: 价格序列
            window: 滚动窗口
            
        Returns:
            Hurst指数序列
        """
        hurst_values = []
        
        for i in range(len(series)):
            if i < window:
                hurst_values.append(np.nan)
            else:
                window_data = series.iloc[i-window:i]
                hurst = self.indicators.calculate_hurst_exponent(
                    window_data, max_lag=min(50, window//2)
                )
                hurst_values.append(hurst)
        
        return pd.Series(hurst_values, index=series.index)
    
    def _classify_trend(self, trend_score: pd.Series, direction: pd.Series) -> pd.Series:
        """
        分类趋势状态
        
        Args:
            trend_score: 趋势得分
            direction: 趋势方向
            
        Returns:
            趋势状态（strong_up, weak_up, neutral, weak_down, strong_down）
        """
        threshold_strong = self.trend_config['trend_score_threshold']
        threshold_weak = threshold_strong * 0.6
        
        conditions = [
            (trend_score >= threshold_strong) & (direction > 0),
            (trend_score >= threshold_weak) & (trend_score < threshold_strong) & (direction > 0),
            (trend_score < threshold_weak),
            (trend_score >= threshold_weak) & (trend_score < threshold_strong) & (direction < 0),
            (trend_score >= threshold_strong) & (direction < 0),
        ]
        
        choices = ['strong_up', 'weak_up', 'neutral', 'weak_down', 'strong_down']
        
        return pd.Series(
            np.select(conditions, choices, default='neutral'),
            index=trend_score.index
        )
    
    def get_trending_assets(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        识别所有资产的趋势强度并排序
        
        Args:
            data_dict: 资产数据字典
            
        Returns:
            包含各资产趋势评分的DataFrame
        """
        results = []
        
        for symbol, df in data_dict.items():
            try:
                trend_df = self.calculate_trend_score(df)
                
                # 获取最新趋势数据
                latest = trend_df.iloc[-1]
                
                results.append({
                    'symbol': symbol,
                    'trend_score': latest['trend_score'],
                    'trend_state': latest['trend_state'],
                    'adx': latest['adx'],
                    'hurst': latest['hurst'],
                    'efficiency_ratio': latest['efficiency_ratio'],
                    'direction': 'UP' if latest['trend_direction'] > 0 else 'DOWN'
                })
            except Exception as e:
                print(f"处理 {symbol} 时出错: {e}")
        
        results_df = pd.DataFrame(results)
        results_df.sort_values('trend_score', ascending=False, inplace=True)
        
        return results_df
    
    def is_trending(self, df: pd.DataFrame) -> bool:
        """
        判断当前是否处于趋势状态
        
        Args:
            df: 包含趋势指标的DataFrame
            
        Returns:
            是否趋势
        """
        if 'trend_score' not in df.columns:
            df = self.calculate_trend_score(df)
        
        latest_score = df['trend_score'].iloc[-1]
        threshold = self.trend_config['trend_score_threshold']
        
        return latest_score >= threshold


if __name__ == "__main__":
    # 测试代码
    print("趋势检测模块加载成功")
    print("功能: 综合ADX、Hurst、ER等指标计算趋势得分")
