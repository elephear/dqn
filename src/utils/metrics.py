"""
指标计算函数
用于计算实验评价指标
"""

import numpy as np
from typing import List, Dict, Any

def calculate_delay(path_delays: List[float]) -> Dict[str, float]:
    """
    计算时延相关指标
    
    Args:
        path_delays: 路径时延列表
        
    Returns:
        时延指标字典
    """
    if not path_delays:
        return {
            'average': 0.0,
            'min': 0.0,
            'max': 0.0,
            'std': 0.0,
            'percentile_95': 0.0
        }
    
    delays = np.array(path_delays)
    
    return {
        'average': float(np.mean(delays)),
        'min': float(np.min(delays)),
        'max': float(np.max(delays)),
        'std': float(np.std(delays)),
        'percentile_95': float(np.percentile(delays, 95))
    }

def calculate_success_rate(successful_requests: int, total_requests: int) -> Dict[str, float]:
    """
    计算成功率相关指标
    
    Args:
        successful_requests: 成功请求数
        total_requests: 总请求数
        
    Returns:
        成功率指标字典
    """
    if total_requests == 0:
        return {
            'rate': 0.0,
            'successful': 0,
            'total': 0,
            'failed': 0
        }
    
    success_rate = successful_requests / total_requests
    
    return {
        'rate': float(success_rate),
        'successful': successful_requests,
        'total': total_requests,
        'failed': total_requests - successful_requests
    }

def calculate_load_balance(node_loads: Dict[int, float]) -> Dict[str, float]:
    """
    计算负载均衡指标
    
    Args:
        node_loads: 节点负载字典 {节点ID: 负载值}
        
    Returns:
        负载均衡指标字典
    """
    if not node_loads:
        return {
            'variance': 0.0,
            'std': 0.0,
            'jains_fairness': 1.0,
            'min_load': 0.0,
            'max_load': 0.0,
            'avg_load': 0.0,
            'imbalance_ratio': 0.0
        }
    
    loads = list(node_loads.values())
    avg_load = np.mean(loads)
    
    # 计算负载方差
    load_variance = np.var(loads)
    
    # 计算负载标准差
    load_std = np.std(loads)
    
    # 计算Jain's公平指数
    sum_loads = sum(loads)
    sum_squared_loads = sum(l**2 for l in loads)
    
    if sum_squared_loads > 0:
        jains_fairness = (sum_loads ** 2) / (len(loads) * sum_squared_loads)
    else:
        jains_fairness = 1.0
    
    # 计算不平衡比率
    if avg_load > 0:
        imbalance_ratio = (max(loads) - min(loads)) / avg_load
    else:
        imbalance_ratio = 0.0
    
    return {
        'variance': float(load_variance),
        'std': float(load_std),
        'jains_fairness': float(jains_fairness),
        'min_load': float(min(loads)),
        'max_load': float(max(loads)),
        'avg_load': float(avg_load),
        'imbalance_ratio': float(imbalance_ratio)
    }

def calculate_energy_consumption(node_powers: Dict[int, float], 
                                link_powers: Dict[tuple, float]) -> Dict[str, float]:
    """
    计算能耗指标
    
    Args:
        node_powers: 节点功耗字典
        link_powers: 链路功耗字典
        
    Returns:
        能耗指标字典
    """
    total_node_power = sum(node_powers.values()) if node_powers else 0.0
    total_link_power = sum(link_powers.values()) if link_powers else 0.0
    total_power = total_node_power + total_link_power
    
    return {
        'total': float(total_power),
        'node': float(total_node_power),
        'link': float(total_link_power),
        'average_node': float(total_node_power / len(node_powers)) if node_powers else 0.0,
        'average_link': float(total_link_power / len(link_powers)) if link_powers else 0.0
    }

def calculate_convergence_speed(learning_curve: List[float], 
                               target_value: float = 0.9,
                               window_size: int = 10) -> Dict[str, Any]:
    """
    计算收敛速度指标
    
    Args:
        learning_curve: 学习曲线值列表
        target_value: 目标值
        window_size: 稳定窗口大小
        
    Returns:
        收敛速度指标字典
    """
    if not learning_curve:
        return {
            'convergence_step': -1,
            'final_value': 0.0,
            'stability': False,
            'oscillation': 0.0
        }
    
    # 找到首次达到目标值的步数
    convergence_step = -1
    for i, value in enumerate(learning_curve):
        if value >= target_value:
            convergence_step = i
            break
    
    # 计算最终值
    final_value = learning_curve[-1] if learning_curve else 0.0
    
    # 检查稳定性（最后window_size个值的标准差）
    if len(learning_curve) >= window_size:
        last_values = learning_curve[-window_size:]
        stability_std = np.std(last_values)
        is_stable = stability_std < 0.05  # 标准差小于0.05认为稳定
    else:
        stability_std = np.std(learning_curve) if learning_curve else 0.0
        is_stable = False
    
    # 计算振荡程度（整个曲线的标准差）
    oscillation = np.std(learning_curve) if learning_curve else 0.0
    
    return {
        'convergence_step': convergence_step,
        'final_value': float(final_value),
        'stability': is_stable,
        'stability_std': float(stability_std),
        'oscillation': float(oscillation),
        'curve_length': len(learning_curve)
    }

def calculate_comprehensive_metrics(metrics_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    计算综合指标
    
    Args:
        metrics_data: 包含各种原始指标的数据
        
    Returns:
        综合指标字典
    """
    # 提取数据
    delays = metrics_data.get('delays', [])
    successful_requests = metrics_data.get('successful_requests', 0)
    total_requests = metrics_data.get('total_requests', 0)
    node_loads = metrics_data.get('node_loads', {})
    node_powers = metrics_data.get('node_powers', {})
    link_powers = metrics_data.get('link_powers', {})
    learning_curve = metrics_data.get('learning_curve', [])
    
    # 计算各项指标
    delay_metrics = calculate_delay(delays)
    success_metrics = calculate_success_rate(successful_requests, total_requests)
    load_balance_metrics = calculate_load_balance(node_loads)
    energy_metrics = calculate_energy_consumption(node_powers, link_powers)
    convergence_metrics = calculate_convergence_speed(learning_curve)
    
    # 计算综合评分（加权平均）
    weights = {
        'success_rate': 0.4,
        'delay': 0.3,
        'load_balance': 0.2,
        'energy': 0.1
    }
    
    # 归一化各项指标
    normalized_success = success_metrics['rate']  # 已经在[0,1]范围内
    normalized_delay = 1.0 - min(delay_metrics['average'] / 100.0, 1.0)  # 假设最大时延100ms
    normalized_load_balance = load_balance_metrics['jains_fairness']  # 已经在[0,1]范围内
    normalized_energy = 1.0 - min(energy_metrics['total'] / 1000.0, 1.0)  # 假设最大能耗1000
    
    # 计算综合评分
    comprehensive_score = (
        weights['success_rate'] * normalized_success +
        weights['delay'] * normalized_delay +
        weights['load_balance'] * normalized_load_balance +
        weights['energy'] * normalized_energy
    )
    
    return {
        'delay': delay_metrics,
        'success_rate': success_metrics,
        'load_balance': load_balance_metrics,
        'energy': energy_metrics,
        'convergence': convergence_metrics,
        'comprehensive_score': float(comprehensive_score),
        'normalized_metrics': {
            'success': float(normalized_success),
            'delay': float(normalized_delay),
            'load_balance': float(normalized_load_balance),
            'energy': float(normalized_energy)
        }
    }