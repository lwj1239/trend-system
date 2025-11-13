"""
报告生成模块
"""
import pandas as pd
from typing import Dict
from datetime import datetime
import os


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, output_dir: str = "reports"):
        """初始化报告生成器"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_html_report(
        self,
        metrics: Dict,
        equity_curve: pd.Series,
        trades_df: pd.DataFrame = None,
        save_path: str = None
    ) -> str:
        """
        生成HTML格式报告
        
        Args:
            metrics: 绩效指标字典
            equity_curve: 权益曲线
            trades_df: 交易记录
            save_path: 保存路径
            
        Returns:
            HTML内容
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
        html = self._create_html_template(metrics, equity_curve, trades_df)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"HTML报告已生成: {save_path}")
        return html
    
    def _create_html_template(
        self,
        metrics: Dict,
        equity_curve: pd.Series,
        trades_df: pd.DataFrame = None
    ) -> str:
        """创建HTML模板"""
        
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>趋势跟踪系统回测报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-card.positive {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        .metric-card.negative {{
            background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        }}
        .metric-label {{
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 趋势跟踪系统回测报告</h1>
        <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>📊 核心绩效指标</h2>
        <div class="metrics-grid">
            <div class="metric-card {('positive' if metrics.get('total_return', 0) > 0 else 'negative')}">
                <div class="metric-label">总收益率</div>
                <div class="metric-value">{metrics.get('total_return', 0):.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">年化收益率</div>
                <div class="metric-value">{metrics.get('annualized_return', 0):.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">夏普比率</div>
                <div class="metric-value">{metrics.get('sharpe_ratio', 0):.2f}</div>
            </div>
            <div class="metric-card negative">
                <div class="metric-label">最大回撤</div>
                <div class="metric-value">{metrics.get('max_drawdown', 0):.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">索提诺比率</div>
                <div class="metric-value">{metrics.get('sortino_ratio', 0):.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">卡玛比率</div>
                <div class="metric-value">{metrics.get('calmar_ratio', 0):.2f}</div>
            </div>
        </div>
        
        {self._create_trade_stats_html(metrics, trades_df) if trades_df is not None else ''}
        
        <h2>📈 权益曲线统计</h2>
        <table>
            <tr>
                <th>指标</th>
                <th>数值</th>
            </tr>
            <tr>
                <td>起始权益</td>
                <td>${equity_curve.iloc[0]:,.2f}</td>
            </tr>
            <tr>
                <td>最终权益</td>
                <td>${equity_curve.iloc[-1]:,.2f}</td>
            </tr>
            <tr>
                <td>最高权益</td>
                <td>${equity_curve.max():,.2f}</td>
            </tr>
            <tr>
                <td>最低权益</td>
                <td>${equity_curve.min():,.2f}</td>
            </tr>
            <tr>
                <td>年化波动率</td>
                <td>{metrics.get('volatility', 0):.2%}</td>
            </tr>
        </table>
        
        <p style="margin-top: 40px; text-align: center; color: #7f8c8d;">
            趋势跟踪量化系统 © 2024
        </p>
    </div>
</body>
</html>
"""
        return html
    
    def _create_trade_stats_html(self, metrics: Dict, trades_df: pd.DataFrame) -> str:
        """创建交易统计HTML"""
        return f"""
        <h2>💼 交易统计</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">交易次数</div>
                <div class="metric-value">{metrics.get('num_trades', 0)}</div>
            </div>
            <div class="metric-card {('positive' if metrics.get('win_rate', 0) > 0.5 else '')}">
                <div class="metric-label">胜率</div>
                <div class="metric-value">{metrics.get('win_rate', 0):.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">盈利因子</div>
                <div class="metric-value">{metrics.get('profit_factor', 0):.2f}</div>
            </div>
        </div>
        """
    
    def save_trades_to_csv(self, trades_df: pd.DataFrame, filename: str = None):
        """保存交易记录到CSV"""
        if filename is None:
            filename = f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        trades_df.to_csv(filepath)
        print(f"交易记录已保存: {filepath}")


if __name__ == "__main__":
    print("报告生成模块加载成功")
    print("支持格式: HTML、CSV")
