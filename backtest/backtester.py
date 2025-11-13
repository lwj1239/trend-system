"""
回测引擎
支持多币种、多参数组合的回测
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import yaml


class Backtester:
    """回测引擎"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """初始化回测引擎"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.backtest_config = self.config['backtest']
        self.initial_capital = self.backtest_config['initial_capital']
        self.commission = self.backtest_config['commission']
        self.slippage = self.backtest_config['slippage']
        
        # 回测状态
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.cash_history = []
    
    def reset(self):
        """重置回测状态"""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.cash_history = []
    
    def calculate_trade_cost(self, price: float, size: float) -> float:
        """
        计算交易成本
        
        Args:
            price: 价格
            size: 数量
            
        Returns:
            总成本（含滑点和手续费）
        """
        trade_value = abs(price * size)
        commission_cost = trade_value * self.commission
        slippage_cost = trade_value * self.slippage
        
        return commission_cost + slippage_cost
    
    def execute_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        size: float,
        timestamp: pd.Timestamp
    ) -> Dict:
        """
        执行交易
        
        Args:
            symbol: 资产代码
            action: 交易动作（BUY/SELL）
            price: 成交价格
            size: 数量
            timestamp: 时间戳
            
        Returns:
            交易记录字典
        """
        # 计算交易成本
        cost = self.calculate_trade_cost(price, size)
        
        # 计算交易金额
        trade_value = price * size
        
        if action == 'BUY':
            # 买入
            self.capital -= (trade_value + cost)
            self.positions[symbol] = self.positions.get(symbol, 0) + size
            
        elif action == 'SELL':
            # 卖出
            self.capital += (trade_value - cost)
            self.positions[symbol] = self.positions.get(symbol, 0) - size
        
        # 记录交易
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': action,
            'price': price,
            'size': size,
            'value': trade_value,
            'cost': cost,
            'capital': self.capital
        }
        
        self.trades.append(trade)
        
        return trade
    
    def calculate_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        计算投资组合总价值
        
        Args:
            prices: 当前价格字典
            
        Returns:
            总价值
        """
        position_value = sum(
            self.positions.get(symbol, 0) * prices.get(symbol, 0)
            for symbol in set(self.positions.keys()) | set(prices.keys())
        )
        
        total_value = self.capital + position_value
        
        return total_value
    
    def run_backtest(
        self,
        data_dict: Dict[str, pd.DataFrame],
        signals_dict: Dict[str, pd.DataFrame],
        position_sizes: Dict[str, pd.Series] = None
    ) -> pd.DataFrame:
        """
        运行回测
        
        Args:
            data_dict: 资产数据字典
            signals_dict: 信号数据字典
            position_sizes: 仓位大小字典（可选）
            
        Returns:
            回测结果DataFrame
        """
        self.reset()
        
        # 获取所有时间戳的并集
        all_timestamps = set()
        for df in data_dict.values():
            all_timestamps.update(df.index)
        
        all_timestamps = sorted(all_timestamps)
        
        print(f"开始回测...")
        print(f"初始资金: ${self.initial_capital:,.2f}")
        print(f"回测期间: {all_timestamps[0]} 至 {all_timestamps[-1]}")
        print(f"交易资产: {list(data_dict.keys())}")
        
        # 逐时间步模拟
        for timestamp in all_timestamps:
            current_prices = {}
            
            # 获取当前价格
            for symbol, df in data_dict.items():
                if timestamp in df.index:
                    current_prices[symbol] = df.loc[timestamp, 'close']
            
            # 处理信号
            for symbol in signals_dict.keys():
                if symbol not in data_dict:
                    continue
                
                if timestamp not in signals_dict[symbol].index:
                    continue
                
                signal_row = signals_dict[symbol].loc[timestamp]
                current_position = self.positions.get(symbol, 0)
                
                # 根据信号执行交易
                if 'position' in signal_row:
                    target_position = signal_row['position']
                    
                    # 计算需要调整的仓位
                    position_change = target_position - current_position
                    
                    if abs(position_change) > 0:
                        # 确定仓位大小
                        if position_sizes and symbol in position_sizes:
                            if timestamp in position_sizes[symbol].index:
                                size = position_sizes[symbol].loc[timestamp]
                            else:
                                size = abs(position_change)
                        else:
                            size = abs(position_change)
                        
                        # 执行交易
                        if position_change > 0:
                            self.execute_trade(
                                symbol, 'BUY',
                                current_prices[symbol],
                                size, timestamp
                            )
                        elif position_change < 0:
                            self.execute_trade(
                                symbol, 'SELL',
                                current_prices[symbol],
                                size, timestamp
                            )
            
            # 记录权益
            portfolio_value = self.calculate_portfolio_value(current_prices)
            self.equity_curve.append({
                'timestamp': timestamp,
                'portfolio_value': portfolio_value,
                'cash': self.capital
            })
        
        # 转换为DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        print(f"\n回测完成！")
        print(f"最终权益: ${equity_df['portfolio_value'].iloc[-1]:,.2f}")
        print(f"总收益: {(equity_df['portfolio_value'].iloc[-1] / self.initial_capital - 1):.2%}")
        print(f"交易次数: {len(self.trades)}")
        
        return equity_df
    
    def get_trades_df(self) -> pd.DataFrame:
        """获取交易记录DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        
        trades_df = pd.DataFrame(self.trades)
        trades_df.set_index('timestamp', inplace=True)
        
        # 计算每笔交易的收益
        trades_df['pnl'] = 0.0
        
        for symbol in trades_df['symbol'].unique():
            symbol_trades = trades_df[trades_df['symbol'] == symbol]
            
            # 简化：假设每次平仓计算收益
            buy_price = None
            for idx, trade in symbol_trades.iterrows():
                if trade['action'] == 'BUY':
                    buy_price = trade['price']
                elif trade['action'] == 'SELL' and buy_price:
                    pnl = (trade['price'] - buy_price) * trade['size'] - trade['cost']
                    trades_df.loc[idx, 'pnl'] = pnl
                    buy_price = None
        
        return trades_df


if __name__ == "__main__":
    # 测试代码
    print("回测引擎加载成功")
    print("支持: 多币种、多参数、完整交易记录")
