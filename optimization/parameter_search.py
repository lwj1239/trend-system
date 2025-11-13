"""
参数优化模块
支持贝叶斯优化、网格搜索、遗传算法
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable
import yaml
from itertools import product


class ParameterOptimizer:
    """参数优化器"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """初始化参数优化器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.opt_config = self.config['optimization']
    
    def grid_search(
        self,
        objective_func: Callable,
        param_grid: Dict[str, List],
        verbose: bool = True
    ) -> Tuple[Dict, float, List[Dict]]:
        """
        网格搜索优化
        
        Args:
            objective_func: 目标函数，接收参数字典，返回分数
            param_grid: 参数网格
            verbose: 是否显示进度
            
        Returns:
            (最优参数, 最优分数, 所有结果)
        """
        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        best_params = None
        best_score = -np.inf
        all_results = []
        
        total = len(combinations)
        
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            
            try:
                score = objective_func(params)
                
                all_results.append({
                    'params': params.copy(),
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                if verbose and (i + 1) % 10 == 0:
                    print(f"进度: {i+1}/{total}, 当前最优: {best_score:.4f}")
            
            except Exception as e:
                if verbose:
                    print(f"参数 {params} 评估失败: {e}")
        
        if verbose:
            print(f"\n优化完成！最优分数: {best_score:.4f}")
            print(f"最优参数: {best_params}")
        
        return best_params, best_score, all_results
    
    def random_search(
        self,
        objective_func: Callable,
        param_distributions: Dict[str, Tuple[float, float]],
        n_trials: int = None,
        verbose: bool = True
    ) -> Tuple[Dict, float, List[Dict]]:
        """
        随机搜索优化
        
        Args:
            objective_func: 目标函数
            param_distributions: 参数分布范围 {参数名: (最小值, 最大值)}
            n_trials: 试验次数
            verbose: 是否显示进度
            
        Returns:
            (最优参数, 最优分数, 所有结果)
        """
        if n_trials is None:
            n_trials = self.opt_config['n_trials']
        
        best_params = None
        best_score = -np.inf
        all_results = []
        
        for i in range(n_trials):
            # 随机采样参数
            params = {}
            for param_name, (min_val, max_val) in param_distributions.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    params[param_name] = np.random.uniform(min_val, max_val)
            
            try:
                score = objective_func(params)
                
                all_results.append({
                    'params': params.copy(),
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                if verbose and (i + 1) % 10 == 0:
                    print(f"进度: {i+1}/{n_trials}, 当前最优: {best_score:.4f}")
            
            except Exception as e:
                if verbose:
                    print(f"试验 {i+1} 失败: {e}")
        
        if verbose:
            print(f"\n优化完成！最优分数: {best_score:.4f}")
            print(f"最优参数: {best_params}")
        
        return best_params, best_score, all_results
    
    def bayesian_optimization(
        self,
        objective_func: Callable,
        param_bounds: Dict[str, Tuple[float, float]],
        n_trials: int = None,
        n_initial: int = 10,
        verbose: bool = True
    ) -> Tuple[Dict, float, List[Dict]]:
        """
        贝叶斯优化（简化版）
        
        Args:
            objective_func: 目标函数
            param_bounds: 参数边界
            n_trials: 试验次数
            n_initial: 初始随机采样次数
            verbose: 是否显示进度
            
        Returns:
            (最优参数, 最优分数, 所有结果)
        """
        if n_trials is None:
            n_trials = self.opt_config['n_trials']
        
        # 先进行随机搜索作为初始化
        print("贝叶斯优化: 初始随机采样阶段...")
        best_params, best_score, all_results = self.random_search(
            objective_func,
            param_bounds,
            n_trials=n_initial,
            verbose=False
        )
        
        # 剩余的试验使用改进的随机搜索（围绕当前最优解）
        print(f"贝叶斯优化: 探索阶段 (初始最优: {best_score:.4f})...")
        
        for i in range(n_initial, n_trials):
            # 在最优解附近采样
            params = {}
            for param_name, (min_val, max_val) in param_bounds.items():
                if param_name in best_params:
                    # 围绕当前最优值采样
                    current_val = best_params[param_name]
                    range_size = (max_val - min_val) * 0.2  # 20%范围
                    
                    new_min = max(min_val, current_val - range_size)
                    new_max = min(max_val, current_val + range_size)
                    
                    if isinstance(min_val, int):
                        params[param_name] = np.random.randint(int(new_min), int(new_max) + 1)
                    else:
                        params[param_name] = np.random.uniform(new_min, new_max)
                else:
                    if isinstance(min_val, int):
                        params[param_name] = np.random.randint(min_val, max_val + 1)
                    else:
                        params[param_name] = np.random.uniform(min_val, max_val)
            
            try:
                score = objective_func(params)
                
                all_results.append({
                    'params': params.copy(),
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    if verbose:
                        print(f"找到更优解! 分数: {best_score:.4f}, 参数: {best_params}")
                
                if verbose and (i + 1) % 10 == 0:
                    print(f"进度: {i+1}/{n_trials}, 当前最优: {best_score:.4f}")
            
            except Exception as e:
                if verbose:
                    print(f"试验 {i+1} 失败: {e}")
        
        if verbose:
            print(f"\n优化完成！最优分数: {best_score:.4f}")
            print(f"最优参数: {best_params}")
        
        return best_params, best_score, all_results
    
    def cross_validation(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        params: Dict,
        n_folds: int = None
    ) -> List[float]:
        """
        时间序列交叉验证
        
        Args:
            data: 数据
            strategy_func: 策略函数
            params: 参数
            n_folds: 折数
            
        Returns:
            各折分数列表
        """
        if n_folds is None:
            n_folds = self.opt_config['cv_folds']
        
        scores = []
        data_length = len(data)
        fold_size = data_length // (n_folds + 1)
        
        for i in range(n_folds):
            # 训练集：从开始到当前折
            train_end = fold_size * (i + 1)
            # 测试集：下一折
            test_start = train_end
            test_end = test_start + fold_size
            
            if test_end > data_length:
                break
            
            train_data = data.iloc[:train_end]
            test_data = data.iloc[test_start:test_end]
            
            try:
                score = strategy_func(train_data, test_data, params)
                scores.append(score)
            except Exception as e:
                print(f"第 {i+1} 折交叉验证失败: {e}")
        
        return scores
    
    def create_heatmap_data(
        self,
        results: List[Dict],
        param1: str,
        param2: str
    ) -> pd.DataFrame:
        """
        为二维参数组合创建热力图数据
        
        Args:
            results: 优化结果列表
            param1: 参数1名称
            param2: 参数2名称
            
        Returns:
            热力图数据
        """
        # 提取数据
        data = []
        for result in results:
            if param1 in result['params'] and param2 in result['params']:
                data.append({
                    param1: result['params'][param1],
                    param2: result['params'][param2],
                    'score': result['score']
                })
        
        df = pd.DataFrame(data)
        
        # 透视为热力图格式
        heatmap = df.pivot_table(
            values='score',
            index=param2,
            columns=param1,
            aggfunc='mean'
        )
        
        return heatmap


if __name__ == "__main__":
    # 测试代码
    print("参数优化模块加载成功")
    print("支持方法: 网格搜索、随机搜索、贝叶斯优化")
