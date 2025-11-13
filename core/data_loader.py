"""
数据加载与预处理模块
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import yaml


class DataLoader:
    """数据加载器"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """初始化数据加载器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path("data")
        self.interval = self.config['data']['interval']
        self.symbols = self.config['assets']['symbols']
        self.quote = self.config['assets']['quote_currency']
    
    def load_single_asset(self, symbol: str) -> pd.DataFrame:
        """
        加载单个资产的历史数据
        
        Args:
            symbol: 资产代码（如BTC）
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        file_path = self.data_dir / f"{symbol.lower()}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # 确保列名标准化
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必需列: {col}")
        
        return df
    
    def load_all_assets(self) -> Dict[str, pd.DataFrame]:
        """
        加载所有资产的历史数据
        
        Returns:
            资产代码到DataFrame的字典
        """
        data = {}
        for symbol in self.symbols:
            try:
                df = self.load_single_asset(symbol)
                data[symbol] = df
                print(f"✓ 加载 {symbol}: {len(df)} 条数据")
            except Exception as e:
                print(f"✗ 加载 {symbol} 失败: {e}")
        
        return data
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理
        
        Args:
            df: 原始数据
            
        Returns:
            处理后的数据
        """
        df = df.copy()
        
        # 去除缺失值
        df.dropna(inplace=True)
        
        # 去除异常值（价格为0或负数）
        df = df[(df['close'] > 0) & (df['volume'] >= 0)]
        
        # 去除重复索引
        df = df[~df.index.duplicated(keep='first')]
        
        # 按时间排序
        df.sort_index(inplace=True)
        
        return df
    
    def align_timestamps(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        对齐多个资产的时间戳
        
        Args:
            data_dict: 资产数据字典
            
        Returns:
            对齐后的数据字典
        """
        # 找到所有资产的共同时间范围
        common_index = None
        
        for symbol, df in data_dict.items():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        # 对齐所有数据到共同时间
        aligned_data = {}
        for symbol, df in data_dict.items():
            aligned_data[symbol] = df.loc[common_index]
        
        print(f"对齐后的时间范围: {common_index[0]} 至 {common_index[-1]}")
        print(f"共 {len(common_index)} 个数据点")
        
        return aligned_data
    
    def get_returns(self, df: pd.DataFrame, period: int = 1) -> pd.Series:
        """
        计算收益率
        
        Args:
            df: 价格数据
            period: 周期
            
        Returns:
            收益率序列
        """
        return df['close'].pct_change(period)
    
    def resample_data(self, df: pd.DataFrame, new_interval: str) -> pd.DataFrame:
        """
        重采样数据到不同周期
        
        Args:
            df: 原始数据
            new_interval: 新周期（如'4H', '1D'）
            
        Returns:
            重采样后的数据
        """
        resampled = df.resample(new_interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        return resampled.dropna()


if __name__ == "__main__":
    # 测试代码
    loader = DataLoader()
    
    # 加载单个资产
    try:
        btc_data = loader.load_single_asset('BTC')
        print(f"\nBTC数据预览:\n{btc_data.head()}")
        print(f"\n数据形状: {btc_data.shape}")
    except Exception as e:
        print(f"测试加载BTC失败: {e}")
