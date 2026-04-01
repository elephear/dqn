import random
import numpy as np
from typing import List, Dict, Any, Tuple
from enum import Enum

class RequestType(Enum):
    """业务请求类型"""
    EDGE_AI = "edge_ai"  # 边缘AI推理业务
    COMPUTE_SCHEDULING = "compute_scheduling"  # 算力跨域调度业务

class TrafficGenerator:
    """
    业务流量生成器
    基于论文第5.1.3节设计
    """
    
    def __init__(self, config: dict):
        self.config = config
        traffic_config = config.get('traffic', {})
        
        # 业务类型配置
        self.request_types = [RequestType(t) for t in traffic_config.get('request_types', ['edge_ai', 'compute_scheduling'])]
        
        # 边缘AI业务配置
        edge_ai_config = traffic_config.get('edge_ai', {})
        self.edge_ai_compute_range = edge_ai_config.get('compute_requirement_range', [5, 20])
        self.edge_ai_delay_range = edge_ai_config.get('delay_tolerance_range', [20, 50])
        self.edge_ai_data_range = edge_ai_config.get('data_size_range', [1, 10])
        
        # 算力跨域调度业务配置
        compute_config = traffic_config.get('compute_scheduling', {})
        self.compute_compute_range = compute_config.get('compute_requirement_range', [20, 50])
        self.compute_delay_range = compute_config.get('delay_tolerance_range', [50, 100])
        self.compute_data_range = compute_config.get('data_size_range', [10, 100])
        
        # 到达率配置
        self.arrival_rate_range = traffic_config.get('arrival_rate_range', [10, 50])
        
        # 随机种子
        self.random_seed = 42
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
    def generate_request(self, network_state: Dict[str, Any], 
                        request_type: RequestType = None) -> Dict[str, Any]:
        """
        生成单个业务请求
        
        Args:
            network_state: 网络状态（用于选择源目的节点）
            request_type: 业务类型，如果为None则随机选择
            
        Returns:
            业务请求字典
        """
        if request_type is None:
            request_type = random.choice(self.request_types)
        
        # 选择源节点和目的节点
        nodes = network_state['nodes']
        
        # 源节点：随机选择
        src_node = random.choice(nodes)['id']
        
        # 目的节点：随机选择（可以与源节点相同，实际中应该不同）
        dst_node = random.choice(nodes)['id']
        while dst_node == src_node and len(nodes) > 1:
            dst_node = random.choice(nodes)['id']
        
        # 根据业务类型生成请求参数
        if request_type == RequestType.EDGE_AI:
            compute_req = random.uniform(*self.edge_ai_compute_range)
            delay_tol = random.uniform(*self.edge_ai_delay_range)
            data_size = random.uniform(*self.edge_ai_data_range)
        else:  # COMPUTE_SCHEDULING
            compute_req = random.uniform(*self.compute_compute_range)
            delay_tol = random.uniform(*self.compute_delay_range)
            data_size = random.uniform(*self.compute_data_range)
        
        # 生成请求ID
        request_id = f"req_{random.randint(1000, 9999)}"
        
        request = {
            'id': request_id,
            'type': request_type.value,
            'src': src_node,
            'dst': dst_node,
            'compute_requirement': compute_req,  # GFLOPS
            'delay_tolerance': delay_tol,  # ms
            'data_size': data_size,  # MB
            'arrival_time': self._get_current_time(),
            'status': 'pending'  # pending, scheduled, completed, failed
        }
        
        return request
    
    def generate_request_batch(self, network_state: Dict[str, Any], 
                              num_requests: int = 100,
                              arrival_rate: float = None) -> List[Dict[str, Any]]:
        """
        生成一批业务请求
        
        Args:
            network_state: 网络状态
            num_requests: 请求数量
            arrival_rate: 到达率（flows/s），如果为None则随机
            
        Returns:
            业务请求列表
        """
        if arrival_rate is None:
            arrival_rate = random.uniform(*self.arrival_rate_range)
        
        requests = []
        current_time = 0.0
        
        # 使用泊松过程生成到达时间
        for i in range(num_requests):
            # 生成到达时间间隔（指数分布）
            inter_arrival_time = np.random.exponential(1.0 / arrival_rate)
            current_time += inter_arrival_time
            
            # 生成请求
            request = self.generate_request(network_state)
            request['arrival_time'] = current_time
            
            requests.append(request)
        
        # 按到达时间排序
        requests.sort(key=lambda x: x['arrival_time'])
        
        return requests
    
    def generate_traffic_scenarios(self, network_state: Dict[str, Any],
                                  arrival_rates: List[float] = None) -> Dict[float, List[Dict[str, Any]]]:
        """
        生成多种到达率的流量场景
        
        Args:
            network_state: 网络状态
            arrival_rates: 到达率列表，如果为None则使用配置
            
        Returns:
            字典：{到达率: 请求列表}
        """
        if arrival_rates is None:
            arrival_rates = self.config.get('experiment', {}).get('arrival_rates', [10, 30, 50])
        
        scenarios = {}
        num_requests = 1000  # 每个场景生成1000个请求
        
        for rate in arrival_rates:
            requests = self.generate_request_batch(
                network_state, num_requests, rate
            )
            scenarios[rate] = requests
        
        return scenarios
    
    def update_request_status(self, request: Dict[str, Any], 
                             status: str, 
                             completion_time: float = None,
                             actual_delay: float = None,
                             target_node: int = None,
                             path: List[int] = None) -> Dict[str, Any]:
        """
        更新请求状态
        
        Args:
            request: 原始请求
            status: 新状态 ('scheduled', 'completed', 'failed')
            completion_time: 完成时间
            actual_delay: 实际时延
            target_node: 目标算力节点
            path: 调度路径
            
        Returns:
            更新后的请求
        """
        updated_request = request.copy()
        updated_request['status'] = status
        
        if completion_time is not None:
            updated_request['completion_time'] = completion_time
        
        if actual_delay is not None:
            updated_request['actual_delay'] = actual_delay
            # 检查是否满足时延约束
            delay_tolerance = request.get('delay_tolerance', float('inf'))
            updated_request['delay_violation'] = actual_delay > delay_tolerance
        
        if target_node is not None:
            updated_request['target_node'] = target_node
        
        if path is not None:
            updated_request['path'] = path
        
        return updated_request
    
    def calculate_request_metrics(self, requests: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算请求相关指标
        
        Args:
            requests: 请求列表
            
        Returns:
            指标字典
        """
        if not requests:
            return {}
        
        # 分类统计
        edge_ai_requests = [r for r in requests if r.get('type') == 'edge_ai']
        compute_requests = [r for r in requests if r.get('type') == 'compute_scheduling']
        
        # 计算成功率
        completed_requests = [r for r in requests if r.get('status') == 'completed']
        success_rate = len(completed_requests) / len(requests) if requests else 0
        
        # 计算平均时延
        delays = []
        for req in completed_requests:
            if 'actual_delay' in req:
                delays.append(req['actual_delay'])
        
        avg_delay = np.mean(delays) if delays else 0
        
        # 计算时延违反率
        violations = []
        for req in requests:
            if 'delay_violation' in req:
                violations.append(1 if req['delay_violation'] else 0)
        
        violation_rate = np.mean(violations) if violations else 0
        
        # 按业务类型统计
        edge_ai_success = len([r for r in edge_ai_requests if r.get('status') == 'completed']) / len(edge_ai_requests) if edge_ai_requests else 0
        compute_success = len([r for r in compute_requests if r.get('status') == 'completed']) / len(compute_requests) if compute_requests else 0
        
        metrics = {
            'total_requests': len(requests),
            'edge_ai_requests': len(edge_ai_requests),
            'compute_requests': len(compute_requests),
            'success_rate': success_rate,
            'edge_ai_success_rate': edge_ai_success,
            'compute_success_rate': compute_success,
            'avg_delay': avg_delay,
            'delay_violation_rate': violation_rate,
            'avg_compute_requirement': np.mean([r.get('compute_requirement', 0) for r in requests]) if requests else 0,
            'avg_data_size': np.mean([r.get('data_size', 0) for r in requests]) if requests else 0
        }
        
        return metrics
    
    def _get_current_time(self) -> float:
        """
        获取当前时间（模拟）
        
        Returns:
            当前时间
        """
        # 这里简化实现，实际应该使用仿真时间
        import time
        return time.time()
    
    def reset(self):
        """重置生成器"""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)