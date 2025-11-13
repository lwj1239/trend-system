"""
投资组合分配模块
多币种组合优化（风险平价/趋势权重）
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import yaml


class PortfolioAllocator:
    """投资组合分配器"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """初始化投资组合分配器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.portfolio_config = self.config['portfolio']
    
    def calculate_correlation_matrix(
        self, 
        returns_dict: Dict[str, pd.Series],
        window: int = 60
    ) -> pd.DataFrame:
        """
        计算资产相关性矩阵
        
        Args:
            returns_dict: 各资产收益率字典
            window: 计算窗口
            
        Returns:
            相关性矩阵
        """
        # 合并所有收益率
        returns_df = pd.DataFrame(returns_dict)
        
        # 计算滚动相关性
        corr_matrix = returns_df.tail(window).corr()
        
        return corr_matrix
    
    def filter_by_correlation(
        self,
        assets: List[str],
        returns_dict: Dict[str, pd.Series]
    ) -> List[str]:
        """
        根据相关性过滤资产（去除高相关资产）
        
        Args:
            assets: 资产列表
            returns_dict: 收益率字典
            
        Returns:
            过滤后的资产列表
        """
        if len(assets) <= 1:
            return assets
        
        # 计算相关性矩阵
        returns_df = pd.DataFrame({asset: returns_dict[asset] for asset in assets})
        corr_matrix = returns_df.corr()
        
        threshold = self.portfolio_config['correlation_threshold']
        
        # 去除高相关资产
        selected = []
        for asset in assets:
            # 检查与已选资产的相关性
            if not selected:
                selected.append(asset)
            else:
                max_corr = max(abs(corr_matrix.loc[asset, sel]) for sel in selected)
                if max_corr < threshold:
                    selected.append(asset)
        
        return selected
    
    def risk_parity_allocation(
        self,
        returns_dict: Dict[str, pd.Series],
        window: int = 60
    ) -> Dict[str, float]:
        """
        风险平价配置（Equal Risk Contribution）
        
        Args:
            returns_dict: 各资产收益率字典
            window: 计算窗口
            
        Returns:
            各资产权重字典
        """
        assets = list(returns_dict.keys())
        
        # 计算波动率
        volatilities = {}
        for asset, returns in returns_dict.items():
            vol = returns.tail(window).std() * np.sqrt(252)  # 年化波动率
            volatilities[asset] = vol
        
        # 风险平价：权重与波动率成反比
        inv_vol = {asset: 1/vol if vol > 0 else 0 for asset, vol in volatilities.items()}
        total_inv_vol = sum(inv_vol.values())
        
        if total_inv_vol == 0:
            # 如果所有波动率都为0，则等权
            weights = {asset: 1/len(assets) for asset in assets}
        else:
            weights = {asset: iv/total_inv_vol for asset, iv in inv_vol.items()}
        
        return weights
    
    def trend_weighted_allocation(
        self,
        trend_scores: Dict[str, float],
        returns_dict: Dict[str, pd.Series] = None
    ) -> Dict[str, float]:
        """
        基于趋势得分的权重分配
        
        Args:
            trend_scores: 各资产趋势得分字典
            returns_dict: 各资产收益率字典（可选，用于风险调整）
            
        Returns:
            各资产权重字典
        """
        assets = list(trend_scores.keys())
        
        if returns_dict is not None:
            # 结合风险调整
            volatilities = {}
            for asset in assets:
                vol = returns_dict[asset].tail(60).std() * np.sqrt(252)
                volatilities[asset] = vol if vol > 0 else 0.01
            
            # 趋势得分 / 波动率
            risk_adjusted_scores = {
                asset: trend_scores[asset] / volatilities[asset]
                for asset in assets
            }
        else:
            risk_adjusted_scores = trend_scores
        
        # 归一化为权重
        total_score = sum(risk_adjusted_scores.values())
        
        if total_score == 0:
            weights = {asset: 1/len(assets) for asset in assets}
        else:
            weights = {
                asset: score/total_score 
                for asset, score in risk_adjusted_scores.items()
            }
        
        return weights
    
    def allocate_portfolio(
        self,
        trend_scores: Dict[str, float],
        returns_dict: Dict[str, pd.Series],
        method: str = None
    ) -> Dict[str, float]:
        """
        分配投资组合权重
        
        Args:
            trend_scores: 趋势得分字典
            returns_dict: 收益率字典
            method: 分配方法（risk_parity或trend_weighted）
            
        Returns:
            权重字典
        """
        if method is None:
            method = self.portfolio_config['allocation_method']
        
        # 只选择趋势得分最高的资产
        max_assets = self.portfolio_config['max_assets']
        top_assets = sorted(
            trend_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:max_assets]
        top_assets = [asset for asset, _ in top_assets]
        
        # 过滤高相关资产
        filtered_assets = self.filter_by_correlation(
            top_assets,
            returns_dict
        )
        
        # 只保留过滤后的资产
        filtered_scores = {a: trend_scores[a] for a in filtered_assets}
        filtered_returns = {a: returns_dict[a] for a in filtered_assets}
        
        # 根据方法分配权重
        if method == 'risk_parity':
            weights = self.risk_parity_allocation(filtered_returns)
        elif method == 'trend_weighted':
            weights = self.trend_weighted_allocation(filtered_scores, filtered_returns)
        else:
            # 等权
            weights = {asset: 1/len(filtered_assets) for asset in filtered_assets}
        
        # 确保权重和为1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {asset: w/total_weight for asset, w in weights.items()}
        
        return weights
    
    def rebalance_check(self, last_rebalance_date: pd.Timestamp, current_date: pd.Timestamp) -> bool:
        """
        检查是否需要再平衡
        
        Args:
            last_rebalance_date: 上次再平衡日期
            current_date: 当前日期
            
        Returns:
            是否需要再平衡
        """
        rebalance_period = self.portfolio_config['rebalance_period']
        days_since_rebalance = (current_date - last_rebalance_date).days
        
        return days_since_rebalance >= rebalance_period
    
    def calculate_portfolio_metrics(
        self,
        weights: Dict[str, float],
        returns_dict: Dict[str, pd.Series],
        window: int = 60
    ) -> Dict[str, float]:
        """
        计算投资组合指标
        
        Args:
            weights: 权重字典
            returns_dict: 收益率字典
            window: 计算窗口
            
        Returns:
            指标字典
        """
        # 构建投资组合收益率
        portfolio_returns = pd.Series(0, index=list(returns_dict.values())[0].index)
        
        for asset, weight in weights.items():
            if asset in returns_dict:
                portfolio_returns += weight * returns_dict[asset]
        
        # 计算指标
        recent_returns = portfolio_returns.tail(window)
        
        metrics = {
            'expected_return': recent_returns.mean() * 252,  # 年化
            'volatility': recent_returns.std() * np.sqrt(252),  # 年化
            'sharpe_ratio': (recent_returns.mean() / recent_returns.std() * np.sqrt(252)) if recent_returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown((1 + recent_returns).cumprod())
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """计算最大回撤"""
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return abs(drawdown.min())


if __name__ == "__main__":
    # 测试代码
    print("投资组合分配模块加载成功")
    print("支持方法: 风险平价(Risk Parity)、趋势权重(Trend Weighted)")
