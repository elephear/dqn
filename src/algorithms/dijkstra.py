import heapq
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

class DijkstraScheduler:
    """
    Dijkstra最短路径调度算法
    基于论文第5.2.1节的传统Dijkstra算法
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.name = "Dijkstra"
        
    def schedule(self, request: Dict[str, Any], 
                network_state: Dict[str, Any]) -> Tuple[Optional[int], Optional[list], Optional[float]]:
        """
        Dijkstra调度
        
        Args:
            request: 业务请求
            network_state: 网络状态
            
        Returns:
            (target_node, path, cost) - 调度决策
        """
        src = request['src']
        
        # 获取所有算力节点
        compute_nodes = []
        for node in network_state['nodes']:
            if node['is_compute_node']:
                # 检查算力约束
                if node['compute_capacity'] >= request['compute_requirement']:
                    compute_nodes.append(node['id'])
        
        if not compute_nodes:
            return None, None, None
        
        # 构建图
        graph = self._build_graph(network_state)
        
        # 对每个算力节点运行Dijkstra，选择总时延最小的
        best_target = None
        best_path = None
        best_cost = float('inf')
        
        for target in compute_nodes:
            # 计算从源节点到目标节点的最短路径
            path, cost = self._dijkstra_shortest_path(graph, src, target)
            
            if path and cost < best_cost:
                best_target = target
                best_path = path
                best_cost = cost
        
        return best_target, best_path, best_cost
    
    def _build_graph(self, network_state: Dict[str, Any]) -> Dict[int, List[Tuple[int, float]]]:
        """
        构建图结构
        
        Args:
            network_state: 网络状态
            
        Returns:
            邻接表表示的图
        """
        graph = {}
        
        for link in network_state['links']:
            src = link['src']
            dst = link['dst']
            delay = link['delay']
            
            if src not in graph:
                graph[src] = []
            if dst not in graph:
                graph[dst] = []
            
            # 使用时延作为边的权重
            graph[src].append((dst, delay))
            graph[dst].append((src, delay))  # 无向图
        
        return graph
    
    def _dijkstra_shortest_path(self, graph: Dict[int, List[Tuple[int, float]]], 
                               start: int, end: int) -> Tuple[Optional[List[int]], float]:
        """
        Dijkstra最短路径算法
        
        Args:
            graph: 图（邻接表）
            start: 起始节点
            end: 目标节点
            
        Returns:
            (路径, 总成本)
        """
        if start not in graph or end not in graph:
            return None, float('inf')
        
        # 初始化距离和前驱节点
        distances = {node: float('inf') for node in graph}
        predecessors = {node: None for node in graph}
        distances[start] = 0
        
        # 优先队列
        pq = [(0, start)]
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            # 如果已经找到目标节点，可以提前结束
            if current_node == end:
                break
            
            # 如果当前距离大于已知距离，跳过
            if current_dist > distances[current_node]:
                continue
            
            # 遍历邻居
            for neighbor, weight in graph.get(current_node, []):
                distance = current_dist + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))
        
        # 如果无法到达目标节点
        if distances[end] == float('inf'):
            return None, float('inf')
        
        # 重建路径
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()
        
        return path, distances[end]

class ImprovedDijkstraScheduler(DijkstraScheduler):
    """
    改进的Dijkstra算法
    基于论文第5.2.1节的改进Dijkstra算法
    考虑算力节点负载
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "Improved-Dijkstra"
        
    def schedule(self, request: Dict[str, Any], 
                network_state: Dict[str, Any]) -> Tuple[Optional[int], Optional[list], Optional[float]]:
        """
        改进的Dijkstra调度
        
        Args:
            request: 业务请求
            network_state: 网络状态
            
        Returns:
            (target_node, path, cost) - 调度决策
        """
        src = request['src']
        
        # 获取所有算力节点
        compute_nodes = []
        node_info = {}
        
        for node in network_state['nodes']:
            if node['is_compute_node']:
                # 检查算力约束
                if node['compute_capacity'] >= request['compute_requirement']:
                    node_id = node['id']
                    compute_nodes.append(node_id)
                    node_info[node_id] = {
                        'load': node['current_load'],
                        'capacity': node['compute_capacity']
                    }
        
        if not compute_nodes:
            return None, None, None
        
        # 构建图（考虑链路利用率）
        graph = self._build_improved_graph(network_state)
        
        # 对每个算力节点运行改进的Dijkstra
        best_target = None
        best_path = None
        best_score = float('inf')
        
        for target in compute_nodes:
            # 计算从源节点到目标节点的路径
            path, path_cost = self._dijkstra_shortest_path(graph, src, target)
            
            if path:
                # 计算综合得分：路径时延 + 节点负载惩罚
                node_load = node_info[target]['load']
                load_penalty = node_load * 10  # 负载惩罚系数
                total_score = path_cost + load_penalty
                
                if total_score < best_score:
                    best_target = target
                    best_path = path
                    best_score = total_score
        
        return best_target, best_path, best_score
    
    def _build_improved_graph(self, network_state: Dict[str, Any]) -> Dict[int, List[Tuple[int, float]]]:
        """
        构建改进的图结构（考虑链路利用率）
        
        Args:
            network_state: 网络状态
            
        Returns:
            邻接表表示的图
        """
        graph = {}
        
        for link in network_state['links']:
            src = link['src']
            dst = link['dst']
            delay = link['delay']
            utilization = link.get('utilization', 0.0)
            
            if src not in graph:
                graph[src] = []
            if dst not in graph:
                graph[dst] = []
            
            # 改进的权重：时延 × (1 + 利用率)
            # 利用率越高，权重越大，避免拥塞链路
            weight = delay * (1 + utilization)
            
            graph[src].append((dst, weight))
            graph[dst].append((src, weight))  # 无向图
        
        return graph