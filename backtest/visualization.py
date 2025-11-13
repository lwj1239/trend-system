"""
可视化模块
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os
import platform


# 设置中文字体
def setup_chinese_font():
    """配置支持中文的字体"""
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']
    
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 初始化字体设置
setup_chinese_font()


class Visualizer:
    """可视化工具"""
    
    def __init__(self, output_dir: str = "reports/figures"):
        """初始化可视化工具"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置样式
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 确保中文字体生效(样式设置后重新配置)
        setup_chinese_font()
    
    def plot_equity_curve(
        self,
        equity_curve: pd.Series,
        title: str = "权益曲线",
        save_path: str = None
    ):
        """绘制权益曲线"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(equity_curve.index, equity_curve.values, linewidth=2)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('权益')
        ax.grid(True, alpha=0.3)
        
        # 格式化y轴
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_drawdown(
        self,
        equity_curve: pd.Series,
        title: str = "回撤曲线",
        save_path: str = None
    ):
        """绘制回撤曲线"""
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.fill_between(drawdown.index, 0, drawdown.values, alpha=0.3, color='red')
        ax.plot(drawdown.index, drawdown.values, linewidth=2, color='darkred')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('回撤')
        ax.grid(True, alpha=0.3)
        
        # 格式化y轴为百分比
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_returns_distribution(
        self,
        returns: pd.Series,
        title: str = "收益率分布",
        save_path: str = None
    ):
        """绘制收益率分布"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 直方图
        ax1.hist(returns, bins=50, alpha=0.7, edgecolor='black')
        ax1.set_title('收益率直方图')
        ax1.set_xlabel('收益率')
        ax1.set_ylabel('频数')
        ax1.axvline(returns.mean(), color='red', linestyle='--', label=f'均值: {returns.mean():.2%}')
        ax1.legend()
        
        # Q-Q图
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q图')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_parameter_heatmap(
        self,
        heatmap_data: pd.DataFrame,
        param1_name: str,
        param2_name: str,
        title: str = "参数热力图",
        save_path: str = None
    ):
        """绘制参数优化热力图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                    center=heatmap_data.median().median(), ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(param1_name)
        ax.set_ylabel(param2_name)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_trend_scores(
        self,
        trend_scores_df: pd.DataFrame,
        title: str = "资产趋势得分",
        save_path: str = None
    ):
        """绘制趋势得分柱状图"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(trend_scores_df['symbol'], trend_scores_df['trend_score'])
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('资产')
        ax.set_ylabel('趋势得分')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加阈值线
        threshold = 0.6
        ax.axhline(threshold, color='red', linestyle='--', label=f'阈值: {threshold}')
        ax.legend()
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_multiple_equity_curves(
        self,
        curves_dict: Dict[str, pd.Series],
        title: str = "多策略对比",
        save_path: str = None
    ):
        """绘制多条权益曲线对比"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for label, curve in curves_dict.items():
            # 归一化到初始值1
            normalized = curve / curve.iloc[0]
            ax.plot(normalized.index, normalized.values, label=label, linewidth=2)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('归一化权益')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_monthly_returns(
        self,
        returns: pd.Series,
        title: str = "月度收益热力图",
        save_path: str = None
    ):
        """绘制月度收益热力图"""
        # 计算月度收益
        monthly_returns = returns.resample('M').sum()
        
        # 重塑为年x月矩阵
        monthly_returns_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })
        
        pivot = monthly_returns_df.pivot(index='year', columns='month', values='return')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn',
                    center=0, ax=ax, cbar_kws={'label': '收益率'})
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('月份')
        ax.set_ylabel('年份')
        
        # 设置月份标签
        month_labels = ['1月', '2月', '3月', '4月', '5月', '6月',
                       '7月', '8月', '9月', '10月', '11月', '12月']
        ax.set_xticklabels(month_labels[:len(pivot.columns)])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


if __name__ == "__main__":
    print("可视化模块加载成功")
    print("支持图表: 权益曲线、回撤、热力图、分布图等")
