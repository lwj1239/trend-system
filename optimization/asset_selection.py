"""
资产筛选模块
根据趋势强度和稳健性筛选资产池
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import yaml


class AssetSelector:
    """资产筛选器"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """初始化资产筛选器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def calculate_trend_quality(self, df: pd.DataFrame) -> float:
        """
        计算资产的趋势质量分数
        
        Args:
            df: 包含价格数据的DataFrame
            
        Returns:
            趋势质量分数（0-1）
        """
        if len(df) < 60:
            return 0.0
        
        returns = df['close'].pct_change().dropna()
        
        # 1. 收益率稳定性
        sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
        sharpe_score = np.clip(abs(sharpe) / 2, 0, 1)
        
        # 2. 趋势持续性（R²）
        x = np.arange(len(df))
        y = np.log(df['close'].values)
        
        try:
            # 线性回归
            coeffs = np.polyfit(x, y, 1)
            y_pred = coeffs[0] * x + coeffs[1]
            
            # R²
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            r_squared_score = max(0, r_squared)
        except:
            r_squared_score = 0
        
        # 3. 波动率适中性（太高或太低都不好）
        volatility = returns.std() * np.sqrt(252)
        vol_score = 1 - abs(volatility - 0.5) / 0.5  # 目标波动率50%
        vol_score = np.clip(vol_score, 0, 1)
        
        # 综合得分
        quality_score = (
            0.4 * sharpe_score +
            0.4 * r_squared_score +
            0.2 * vol_score
        )
        
        return quality_score
    
    def calculate_liquidity_score(self, df: pd.DataFrame) -> float:
        """
        计算流动性分数
        
        Args:
            df: 包含成交量数据的DataFrame
            
        Returns:
            流动性分数（0-1）
        """
        if 'volume' not in df.columns or len(df) < 30:
            return 0.5  # 默认中等流动性
        
        # 平均成交量
        avg_volume = df['volume'].tail(30).mean()
        
        # 成交量稳定性
        volume_std = df['volume'].tail(30).std()
        volume_cv = volume_std / avg_volume if avg_volume > 0 else 1
        
        # 流动性分数：成交量越大、变异系数越小越好
        liquidity_score = 1 / (1 + volume_cv)
        
        return np.clip(liquidity_score, 0, 1)
    
    def calculate_stability_score(self, returns: pd.Series, window: int = 60) -> float:
        """
        计算收益稳定性分数
        
        Args:
            returns: 收益率序列
            window: 计算窗口
            
        Returns:
            稳定性分数（0-1）
        """
        if len(returns) < window:
            return 0.0
        
        recent_returns = returns.tail(window)
        
        # 1. 收益率的偏度（越接近0越好）
        skewness = recent_returns.skew()
        skew_score = 1 / (1 + abs(skewness))
        
        # 2. 收益率的峰度（越接近3越好，正态分布）
        kurtosis = recent_returns.kurtosis()
        kurt_score = 1 / (1 + abs(kurtosis - 3))
        
        # 3. 正收益率比例
        positive_ratio = (recent_returns > 0).mean()
        
        # 综合稳定性分数
        stability = (skew_score + kurt_score + positive_ratio) / 3
        
        return stability
    
    def rank_assets(
        self,
        data_dict: Dict[str, pd.DataFrame],
        trend_scores: Dict[str, float] = None
    ) -> pd.DataFrame:
        """
        对资产进行综合评分和排名
        
        Args:
            data_dict: 资产数据字典
            trend_scores: 趋势得分字典（可选）
            
        Returns:
            包含各项评分的DataFrame
        """
        results = []
        
        for symbol, df in data_dict.items():
            try:
                returns = df['close'].pct_change().dropna()
                
                # 计算各项指标
                trend_quality = self.calculate_trend_quality(df)
                liquidity = self.calculate_liquidity_score(df)
                stability = self.calculate_stability_score(returns)
                
                # 如果提供了趋势得分，使用它
                if trend_scores and symbol in trend_scores:
                    trend_score = trend_scores[symbol]
                else:
                    trend_score = trend_quality
                
                # 综合评分
                composite_score = (
                    0.4 * trend_score +
                    0.3 * trend_quality +
                    0.2 * stability +
                    0.1 * liquidity
                )
                
                results.append({
                    'symbol': symbol,
                    'composite_score': composite_score,
                    'trend_score': trend_score,
                    'trend_quality': trend_quality,
                    'stability': stability,
                    'liquidity': liquidity
                })
            
            except Exception as e:
                print(f"评估 {symbol} 失败: {e}")
        
        results_df = pd.DataFrame(results)
        results_df.sort_values('composite_score', ascending=False, inplace=True)
        results_df.reset_index(drop=True, inplace=True)
        
        return results_df
    
    def select_assets(
        self,
        data_dict: Dict[str, pd.DataFrame],
        trend_scores: Dict[str, float] = None,
        top_n: int = None,
        min_score: float = 0.5
    ) -> List[str]:
        """
        筛选资产
        
        Args:
            data_dict: 资产数据字典
            trend_scores: 趋势得分字典
            top_n: 选择前N个资产
            min_score: 最低综合得分
            
        Returns:
            选中的资产列表
        """
        if top_n is None:
            top_n = self.config['portfolio']['max_assets']
        
        # 排名
        ranked = self.rank_assets(data_dict, trend_scores)
        
        # 过滤
        selected = ranked[ranked['composite_score'] >= min_score]
        
        # 选择前N个
        selected = selected.head(top_n)
        
        return selected['symbol'].tolist()
    
    def print_asset_report(self, ranked_df: pd.DataFrame):
        """打印资产评估报告"""
        print("\n" + "="*80)
        print("资产评估报告")
        print("="*80)
        print(f"{'排名':<6} {'代码':<8} {'综合得分':<10} {'趋势':<8} {'质量':<8} {'稳定性':<10} {'流动性':<10}")
        print("-"*80)
        
        for idx, row in ranked_df.iterrows():
            print(f"{idx+1:<6} {row['symbol']:<8} {row['composite_score']:>8.3f}   "
                  f"{row['trend_score']:>6.3f}   {row['trend_quality']:>6.3f}   "
                  f"{row['stability']:>8.3f}   {row['liquidity']:>8.3f}")
        
        print("="*80 + "\n")


if __name__ == "__main__":
    # 测试代码
    print("资产筛选模块加载成功")
    print("功能: 趋势质量、流动性、稳定性评估")
