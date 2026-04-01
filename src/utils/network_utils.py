"""
网络工具函数
用于构建图神经网络所需的矩阵
"""

import numpy as np
from typing import Dict, List, Tuple, Any

def build_adjacency_matrix(network_topology: Dict[str, Any]) -> np.ndarray:
    """
    构建邻接矩阵
    
    Args:
        network_topology: 网络拓扑结构
        
    Returns:
        邻接矩阵 (n x n)
    """
    # 从网络拓扑中提取节点和链路信息
    nodes = network_topology.get('nodes', [])
    links = network_topology.get('links', [])
    
    n = len(nodes)
    adj_matrix = np.zeros((n, n), dtype=np.float32)
    
    # 构建邻接矩阵
    for link in links:
        src = link.get('src', 0)
        dst = link.get('dst', 0)
        weight = link.get('weight', 1.0)  # 可以使用时延、带宽等作为权重
        
        if 0 <= src < n and 0 <= dst < n:
            adj_matrix[src, dst] = weight
            adj_matrix[dst, src] = weight  # 假设是无向图
    
    return adj_matrix

def build_feature_matrix(network_topology: Dict[str, Any]) -> np.ndarray:
    """
    构建节点特征矩阵
    
    Args:
        network_topology: 网络拓扑结构
        
    Returns:
        节点特征矩阵 (n x d)
    """
    nodes = network_topology.get('nodes', [])
    
    if not nodes:
        return np.zeros((0, 4), dtype=np.float32)
    
    # 提取节点特征：算力、内存、带宽、时延等
    features = []
    for node in nodes:
        # 基本特征
        compute_capacity = node.get('compute_capacity', 0.0)
        memory = node.get('memory', 0.0)
        bandwidth = node.get('bandwidth', 0.0)
        delay = node.get('delay', 0.0)
        
        # 归一化特征
        feature_vector = [
            compute_capacity / 100.0 if compute_capacity > 0 else 0.0,  # 归一化到[0,1]
            memory / 1000.0 if memory > 0 else 0.0,  # 归一化到[0,1]
            bandwidth / 100.0 if bandwidth > 0 else 0.0,  # 归一化到[0,1]
            delay / 50.0 if delay > 0 else 0.0  # 归一化到[0,1]
        ]
        
        features.append(feature_vector)
    
    return np.array(features, dtype=np.float32)

def normalize_adjacency_matrix(adj_matrix: np.ndarray) -> np.ndarray:
    """
    归一化邻接矩阵
    
    Args:
        adj_matrix: 原始邻接矩阵
        
    Returns:
        归一化的邻接矩阵
    """
    # 添加自环
    adj_matrix_with_self_loop = adj_matrix + np.eye(adj_matrix.shape[0])
    
    # 计算度矩阵
    degree_matrix = np.diag(np.sum(adj_matrix_with_self_loop, axis=1))
    
    # 计算度矩阵的-1/2次方
    degree_matrix_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(degree_matrix)))
    
    # 对称归一化
    normalized_adj = degree_matrix_inv_sqrt @ adj_matrix_with_self_loop @ degree_matrix_inv_sqrt
    
    return normalized_adj

def extract_topology_from_simulation(env) -> Dict[str, Any]:
    """
    从仿真环境中提取网络拓扑
    
    Args:
        env: 仿真环境对象
        
    Returns:
        网络拓扑字典
    """
    # 这是一个示例实现，实际实现需要根据仿真环境的具体结构调整
    topology = {
        'nodes': [],
        'links': []
    }
    
    try:
        # 尝试从环境中获取节点信息
        if hasattr(env, 'nodes'):
            for node_id, node_info in env.nodes.items():
                topology['nodes'].append({
                    'id': node_id,
                    'compute_capacity': node_info.get('compute_capacity', 50.0),
                    'memory': node_info.get('memory', 100.0),
                    'bandwidth': node_info.get('bandwidth', 50.0),
                    'delay': node_info.get('delay', 10.0)
                })
        
        # 尝试从环境中获取链路信息
        if hasattr(env, 'links'):
            for link_id, link_info in env.links.items():
                topology['links'].append({
                    'src': link_info.get('src', 0),
                    'dst': link_info.get('dst', 0),
                    'bandwidth': link_info.get('bandwidth', 100.0),
                    'delay': link_info.get('delay', 5.0),
                    'weight': link_info.get('delay', 5.0)  # 使用时延作为权重
                })
    except:
        # 如果无法提取，返回一个简单的默认拓扑
        topology = create_default_topology(10)
    
    return topology

def create_default_topology(num_nodes: int = 10) -> Dict[str, Any]:
    """
    创建默认的网络拓扑
    
    Args:
        num_nodes: 节点数量
        
    Returns:
        默认网络拓扑
    """
    import random
    
    topology = {
        'nodes': [],
        'links': []
    }
    
    # 创建节点
    for i in range(num_nodes):
        is_compute_node = random.random() < 0.3  # 30%的节点是算力节点
        
        topology['nodes'].append({
            'id': i,
            'compute_capacity': random.uniform(10, 100) if is_compute_node else 0.0,
            'memory': random.uniform(50, 200),
            'bandwidth': random.uniform(10, 100),
            'delay': random.uniform(1, 20),
            'is_compute_node': is_compute_node
        })
    
    # 创建链路（随机连接）
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < 0.3:  # 30%的概率创建链路
                delay = random.uniform(1, 20)
                topology['links'].append({
                    'src': i,
                    'dst': j,
                    'bandwidth': random.uniform(10, 100),
                    'delay': delay,
                    'weight': delay
                })
    
    return topology