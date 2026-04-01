import numpy as np
from typing import List, Dict, Any

class DynamicWeightAdjuster:
    """
    动态权重调整器
    基于论文第4.4节设计
    """
    
    def __init__(self, config: dict):
        self.config = config
        weight_config = config.get('weight_adjustment', {})
        
        # 业务类型标准权重
        self.standard_weights = {
            'edge_ai': np.array(weight_config.get('edge_ai_weights', [1.0, 0.0, 0.0])),
            'compute_scheduling': np.array(weight_config.get('compute_scheduling_weights', [0.4, 0.3, 0.3]))
        }
        
        # 过渡参数
        self.transition_rate = weight_config.get('transition_rate', 0.1)
        
        # 紧迫性敏感系数
        self.urgency_sensitivity = np.array(weight_config.get('urgency_sensitivity', [0.5, 0.3, 0.2]))
        
        # 历史目标值记录
        self.history_values = {
            'min': np.zeros(3),  # 三个目标的最小值
            'max': np.ones(3)    # 三个目标的最大值
        }
        
        # 当前权重
        self.current_weights = self.standard_weights['edge_ai'].copy()
        
    def adjust(self, request_type: str, network_state: Dict[str, Any]) -> np.ndarray:
        """
        调整权重
        
        Args:
            request_type: 业务类型 ('edge_ai' 或 'compute_scheduling')
            network_state: 网络状态
            
        Returns:
            调整后的权重向量
        """
        # 获取目标业务类型的标准权重
        target_weights = self.standard_weights[request_type]
        
        # 平滑过渡到目标权重
        self.current_weights = (
            self.transition_rate * target_weights + 
            (1 - self.transition_rate) * self.current_weights
        )
        
        # 根据网络状态实时调整
        urgency_factors = self.calculate_urgency_factors(network_state)
        adjusted_weights = self.current_weights * urgency_factors
        
        # 归一化
        adjusted_weights = adjusted_weights / np.sum(adjusted_weights)
        
        return adjusted_weights
    
    def calculate_urgency_factors(self, network_state: Dict[str, Any]) -> np.ndarray:
        """
        计算紧迫性因子
        
        Args:
            network_state: 网络状态
            
        Returns:
            紧迫性因子向量
        """
        # 计算当前目标值
        current_values = self.calculate_current_objective_values(network_state)
        
        # 更新历史记录
        self.update_history_values(current_values)
        
        # 计算紧迫性因子
        urgency_factors = np.ones(3)  # 初始化为1
        
        for i in range(3):
            if self.history_values['max'][i] > self.history_values['min'][i]:
                # 归一化当前值
                normalized_value = (
                    (current_values[i] - self.history_values['min'][i]) /
                    (self.history_values['max'][i] - self.history_values['min'][i])
                )
                # 计算紧迫性因子
                urgency_factors[i] = 1 + self.urgency_sensitivity[i] * normalized_value
        
        return urgency_factors
    
    def calculate_current_objective_values(self, network_state: Dict[str, Any]) -> np.ndarray:
        """
        计算当前目标值
        
        Args:
            network_state: 网络状态
            
        Returns:
            目标值向量 [时延相关, 负载均衡相关, 能耗相关]
        """
        nodes = network_state.get('nodes', [])
        links = network_state.get('links', [])
        
        # 1. 时延相关指标（平均链路时延）
        total_delay = 0
        valid_links = 0
        for link in links:
            if 'delay' in link:
                total_delay += link['delay']
                valid_links += 1
        
        delay_metric = total_delay / max(valid_links, 1)
        
        # 2. 负载均衡相关指标（节点负载方差）
        compute_loads = []
        for node in nodes:
            if node.get('is_compute_node', False):
                compute_loads.append(node.get('current_load', 0.0))
        
        if compute_loads:
            load_variance = np.var(compute_loads)
        else:
            load_variance = 0.0
        
        # 3. 能耗相关指标（平均能耗系数）
        total_energy = 0
        compute_nodes = 0
        for node in nodes:
            if node.get('is_compute_node', False):
                total_energy += node.get('energy_coefficient', 0.0)
                compute_nodes += 1
        
        energy_metric = total_energy / max(compute_nodes, 1)
        
        return np.array([delay_metric, load_variance, energy_metric])
    
    def update_history_values(self, current_values: np.ndarray):
        """
        更新历史记录
        
        Args:
            current_values: 当前目标值
        """
        # 更新最小值
        self.history_values['min'] = np.minimum(
            self.history_values['min'], current_values
        )
        
        # 更新最大值
        self.history_values['max'] = np.maximum(
            self.history_values['max'], current_values
        )
    
    def reset(self):
        """重置权重调整器"""
        self.current_weights = self.standard_weights['edge_ai'].copy()
        self.history_values = {
            'min': np.zeros(3),
            'max': np.ones(3)
        }