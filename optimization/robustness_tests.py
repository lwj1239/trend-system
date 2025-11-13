"""
稳健性测试模块
包括参数扰动测试、Walk-Forward分析、蒙特卡洛模拟
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Tuple
import yaml


class RobustnessTests:
    """稳健性测试器"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """初始化稳健性测试器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def parameter_sensitivity_test(
        self,
        objective_func: Callable,
        base_params: Dict,
        perturbation_range: float = 0.1,
        n_samples: int = 100
    ) -> Dict:
        """
        参数敏感性测试
        
        Args:
            objective_func: 目标函数
            base_params: 基准参数
            perturbation_range: 扰动范围（比例）
            n_samples: 采样次数
            
        Returns:
            测试结果字典
        """
        base_score = objective_func(base_params)
        
        results = {
            'base_score': base_score,
            'base_params': base_params.copy(),
            'sensitivity': {},
            'perturbed_scores': []
        }
        
        # 对每个参数进行敏感性分析
        for param_name, param_value in base_params.items():
            param_scores = []
            param_values = []
            
            # 在参数周围采样
            for _ in range(n_samples):
                perturbation = np.random.uniform(-perturbation_range, perturbation_range)
                new_value = param_value * (1 + perturbation)
                
                # 确保参数在合理范围内
                if isinstance(param_value, int):
                    new_value = max(1, int(new_value))
                else:
                    new_value = max(0.01, new_value)
                
                # 创建扰动后的参数
                perturbed_params = base_params.copy()
                perturbed_params[param_name] = new_value
                
                try:
                    score = objective_func(perturbed_params)
                    param_scores.append(score)
                    param_values.append(new_value)
                except:
                    continue
            
            # 计算敏感性（分数的标准差）
            if param_scores:
                sensitivity = np.std(param_scores) / (abs(base_score) + 1e-6)
                results['sensitivity'][param_name] = {
                    'sensitivity': sensitivity,
                    'mean_score': np.mean(param_scores),
                    'std_score': np.std(param_scores),
                    'min_score': np.min(param_scores),
                    'max_score': np.max(param_scores)
                }
        
        # 全局扰动测试
        print("执行全局扰动测试...")
        for _ in range(n_samples):
            perturbed_params = {}
            for param_name, param_value in base_params.items():
                perturbation = np.random.uniform(-perturbation_range, perturbation_range)
                new_value = param_value * (1 + perturbation)
                
                if isinstance(param_value, int):
                    new_value = max(1, int(new_value))
                else:
                    new_value = max(0.01, new_value)
                
                perturbed_params[param_name] = new_value
            
            try:
                score = objective_func(perturbed_params)
                results['perturbed_scores'].append(score)
            except:
                continue
        
        # 统计全局稳健性
        if results['perturbed_scores']:
            results['global_robustness'] = {
                'mean_score': np.mean(results['perturbed_scores']),
                'std_score': np.std(results['perturbed_scores']),
                'score_degradation': base_score - np.mean(results['perturbed_scores']),
                'pct_above_base': np.mean([s > base_score for s in results['perturbed_scores']])
            }
        
        return results
    
    def walk_forward_analysis(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        optimize_func: Callable,
        train_period: int = 252,
        test_period: int = 63,
        step_size: int = 63
    ) -> Dict:
        """
        Walk-Forward分析
        
        Args:
            data: 完整数据
            strategy_func: 策略函数(data, params) -> score
            optimize_func: 优化函数(data) -> best_params
            train_period: 训练期长度
            test_period: 测试期长度
            step_size: 步进大小
            
        Returns:
            WFA结果
        """
        results = {
            'in_sample_scores': [],
            'out_sample_scores': [],
            'optimal_params': [],
            'periods': []
        }
        
        data_length = len(data)
        start_idx = 0
        
        print(f"开始Walk-Forward分析...")
        print(f"训练期: {train_period}天, 测试期: {test_period}天, 步长: {step_size}天")
        
        iteration = 0
        while start_idx + train_period + test_period <= data_length:
            iteration += 1
            
            # 训练集
            train_start = start_idx
            train_end = start_idx + train_period
            train_data = data.iloc[train_start:train_end]
            
            # 测试集
            test_start = train_end
            test_end = test_start + test_period
            test_data = data.iloc[test_start:test_end]
            
            print(f"\n迭代 {iteration}:")
            print(f"  训练集: {train_data.index[0]} 至 {train_data.index[-1]}")
            print(f"  测试集: {test_data.index[0]} 至 {test_data.index[-1]}")
            
            try:
                # 在训练集上优化参数
                best_params = optimize_func(train_data)
                
                # 计算训练集表现
                is_score = strategy_func(train_data, best_params)
                
                # 计算测试集表现
                os_score = strategy_func(test_data, best_params)
                
                results['in_sample_scores'].append(is_score)
                results['out_sample_scores'].append(os_score)
                results['optimal_params'].append(best_params)
                results['periods'].append({
                    'train_start': train_data.index[0],
                    'train_end': train_data.index[-1],
                    'test_start': test_data.index[0],
                    'test_end': test_data.index[-1]
                })
                
                print(f"  训练集得分: {is_score:.4f}")
                print(f"  测试集得分: {os_score:.4f}")
                print(f"  最优参数: {best_params}")
                
            except Exception as e:
                print(f"  迭代失败: {e}")
            
            # 移动窗口
            start_idx += step_size
        
        # 计算汇总统计
        if results['out_sample_scores']:
            results['summary'] = {
                'mean_is_score': np.mean(results['in_sample_scores']),
                'mean_os_score': np.mean(results['out_sample_scores']),
                'std_os_score': np.std(results['out_sample_scores']),
                'overfitting_ratio': np.mean(results['in_sample_scores']) / np.mean(results['out_sample_scores']) if np.mean(results['out_sample_scores']) != 0 else np.inf,
                'consistency': np.mean([s > 0 for s in results['out_sample_scores']])
            }
            
            print(f"\n=== Walk-Forward分析汇总 ===")
            print(f"平均训练集得分: {results['summary']['mean_is_score']:.4f}")
            print(f"平均测试集得分: {results['summary']['mean_os_score']:.4f}")
            print(f"过拟合比率: {results['summary']['overfitting_ratio']:.2f}")
            print(f"一致性(正收益比例): {results['summary']['consistency']:.2%}")
        
        return results
    
    def monte_carlo_simulation(
        self,
        trades: pd.DataFrame,
        n_simulations: int = 1000,
        bootstrap: bool = True
    ) -> Dict:
        """
        蒙特卡洛模拟
        
        Args:
            trades: 交易记录DataFrame（需包含'return'列）
            n_simulations: 模拟次数
            bootstrap: 是否使用bootstrap采样
            
        Returns:
            模拟结果
        """
        if 'return' not in trades.columns:
            raise ValueError("trades DataFrame必须包含'return'列")
        
        returns = trades['return'].values
        n_trades = len(returns)
        
        simulated_returns = []
        simulated_sharpe = []
        simulated_max_dd = []
        
        print(f"执行{n_simulations}次蒙特卡洛模拟...")
        
        for i in range(n_simulations):
            if bootstrap:
                # Bootstrap采样
                sim_returns = np.random.choice(returns, size=n_trades, replace=True)
            else:
                # 随机打乱顺序
                sim_returns = np.random.permutation(returns)
            
            # 计算指标
            total_return = (1 + sim_returns).prod() - 1
            sharpe = (sim_returns.mean() / sim_returns.std() * np.sqrt(252)) if sim_returns.std() > 0 else 0
            
            # 计算最大回撤
            cum_returns = (1 + sim_returns).cumprod()
            running_max = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - running_max) / running_max
            max_dd = abs(drawdown.min())
            
            simulated_returns.append(total_return)
            simulated_sharpe.append(sharpe)
            simulated_max_dd.append(max_dd)
            
            if (i + 1) % 100 == 0:
                print(f"  完成 {i+1}/{n_simulations}")
        
        results = {
            'simulated_returns': simulated_returns,
            'simulated_sharpe': simulated_sharpe,
            'simulated_max_dd': simulated_max_dd,
            'statistics': {
                'return_mean': np.mean(simulated_returns),
                'return_std': np.std(simulated_returns),
                'return_5pct': np.percentile(simulated_returns, 5),
                'return_95pct': np.percentile(simulated_returns, 95),
                'sharpe_mean': np.mean(simulated_sharpe),
                'sharpe_5pct': np.percentile(simulated_sharpe, 5),
                'max_dd_mean': np.mean(simulated_max_dd),
                'max_dd_95pct': np.percentile(simulated_max_dd, 95)
            }
        }
        
        print(f"\n=== 蒙特卡洛模拟结果 ===")
        print(f"平均收益: {results['statistics']['return_mean']:.2%}")
        print(f"收益5%分位: {results['statistics']['return_5pct']:.2%}")
        print(f"收益95%分位: {results['statistics']['return_95pct']:.2%}")
        print(f"平均夏普比率: {results['statistics']['sharpe_mean']:.2f}")
        print(f"平均最大回撤: {results['statistics']['max_dd_mean']:.2%}")
        print(f"最大回撤95%分位: {results['statistics']['max_dd_95pct']:.2%}")
        
        return results


if __name__ == "__main__":
    # 测试代码
    print("稳健性测试模块加载成功")
    print("功能: 参数扰动、Walk-Forward分析、蒙特卡洛模拟")
