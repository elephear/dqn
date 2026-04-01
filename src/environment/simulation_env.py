import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from .network_generator import NetworkGenerator
from .traffic_generator import TrafficGenerator, RequestType

class SimulationEnvironment:
    """
    仿真环境主类
    基于论文第5章实验环境设计
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # 初始化生成器
        self.network_generator = NetworkGenerator(config)
        self.traffic_generator = TrafficGenerator(config)
        
        # 当前状态
        self.current_network = None
        self.current_requests = []
        self.completed_requests = []
        self.failed_requests = []
        
        # 仿真时间
        self.simulation_time = 0.0
        self.time_step = 0.1  # 时间步长（秒）
        
        # 性能指标
        self.metrics_history = []
        
        # 调度器（将在外部设置）
        self.scheduler = None
        
    def setup(self, network_size: int = 30, 
              topology_type: str = 'random',
              initial_requests: int = 100):
        """
        设置仿真环境
        
        Args:
            network_size: 网络规模
            topology_type: 拓扑类型
            initial_requests: 初始请求数量
        """
        # 生成网络
        self.current_network = self.network_generator.generate_network(
            network_size, topology_type
        )
        
        # 生成初始请求
        self.current_requests = self.traffic_generator.generate_request_batch(
            self.current_network, initial_requests
        )
        
        # 重置仿真时间
        self.simulation_time = 0.0
        self.completed_requests = []
        self.failed_requests = []
        self.metrics_history = []
        
        print(f"仿真环境设置完成:")
        print(f"  网络规模: {network_size} 节点")
        print(f"  拓扑类型: {topology_type}")
        print(f"  初始请求: {initial_requests}")
        
        # 打印网络指标
        network_metrics = self.network_generator.get_network_metrics(self.current_network)
        print(f"  网络指标: {network_metrics}")
    
    def step(self, scheduler = None) -> Dict[str, Any]:
        """
        执行一个仿真步
        
        Args:
            scheduler: 调度器实例
            
        Returns:
            步进结果
        """
        if scheduler:
            self.scheduler = scheduler
        
        # 更新仿真时间
        self.simulation_time += self.time_step
        
        # 更新网络状态（模拟动态变化）
        self.current_network = self.network_generator.update_network_state(
            self.current_network, self.time_step
        )
        
        # 处理到达的请求
        new_requests = self._get_arriving_requests()
        self.current_requests.extend(new_requests)
        
        # 处理当前请求
        processed_requests = []
        remaining_requests = []
        
        for request in self.current_requests:
            # 检查请求是否已经到达
            if request['arrival_time'] <= self.simulation_time:
                # 如果请求还未调度，尝试调度
                if request.get('status') == 'pending':
                    if self.scheduler:
                        # 使用调度器进行调度
                        target_node, path, q_value = self.scheduler.schedule(
                            request, self.current_network
                        )
                        
                        if target_node is not None and path is not None:
                            # 调度成功
                            request = self.traffic_generator.update_request_status(
                                request, 'scheduled',
                                target_node=target_node,
                                path=path
                            )
                            
                            # 计算实际时延（简化）
                            actual_delay = self._calculate_request_delay(request, path)
                            
                            # 检查是否满足约束
                            if self._check_request_constraints(request, actual_delay):
                                # 请求完成
                                request = self.traffic_generator.update_request_status(
                                    request, 'completed',
                                    completion_time=self.simulation_time,
                                    actual_delay=actual_delay
                                )
                                self.completed_requests.append(request)
                            else:
                                # 请求失败（违反约束）
                                request = self.traffic_generator.update_request_status(
                                    request, 'failed',
                                    actual_delay=actual_delay
                                )
                                self.failed_requests.append(request)
                        else:
                            # 调度失败
                            request = self.traffic_generator.update_request_status(
                                request, 'failed'
                            )
                            self.failed_requests.append(request)
                        
                        processed_requests.append(request)
                    else:
                        # 无调度器，请求保持pending
                        remaining_requests.append(request)
                else:
                    # 请求已处理，移动到相应列表
                    if request['status'] == 'completed':
                        self.completed_requests.append(request)
                    elif request['status'] == 'failed':
                        self.failed_requests.append(request)
                    else:
                        remaining_requests.append(request)
            else:
                # 请求尚未到达
                remaining_requests.append(request)
        
        # 更新当前请求列表
        self.current_requests = remaining_requests
        
        # 收集指标
        step_metrics = self._collect_step_metrics()
        self.metrics_history.append(step_metrics)
        
        # 检查仿真是否结束
        done = self._check_simulation_done()
        
        step_result = {
            'simulation_time': self.simulation_time,
            'network_state': self.current_network,
            'current_requests': len(self.current_requests),
            'completed_requests': len(self.completed_requests),
            'failed_requests': len(self.failed_requests),
            'metrics': step_metrics,
            'done': done
        }
        
        return step_result
    
    def run(self, scheduler = None, max_steps: int = 1000) -> List[Dict[str, Any]]:
        """
        运行完整仿真
        
        Args:
            scheduler: 调度器实例
            max_steps: 最大步数
            
        Returns:
            仿真历史记录
        """
        if scheduler:
            self.scheduler = scheduler
        
        history = []
        
        print(f"开始仿真，最大步数: {max_steps}")
        
        for step in range(max_steps):
            step_result = self.step()
            history.append(step_result)
            
            # 打印进度
            if step % 100 == 0:
                print(f"  步数 {step}: 完成请求 {len(self.completed_requests)}, "
                      f"失败请求 {len(self.failed_requests)}, "
                      f"等待请求 {len(self.current_requests)}")
            
            if step_result['done']:
                print(f"仿真提前结束于步数 {step}")
                break
        
        # 打印最终结果
        final_metrics = self.get_overall_metrics()
        print(f"仿真完成:")
        print(f"  总请求: {final_metrics['total_requests']}")
        print(f"  成功率: {final_metrics['success_rate']:.2%}")
        print(f"  平均时延: {final_metrics['avg_delay']:.2f} ms")
        print(f"  时延违反率: {final_metrics['delay_violation_rate']:.2%}")
        
        return history
    
    def _get_arriving_requests(self) -> List[Dict[str, Any]]:
        """
        获取在当前时间步到达的请求
        
        Returns:
            到达的请求列表
        """
        arriving_requests = []
        
        # 这里简化实现：随机生成新请求
        # 在实际系统中，应该根据到达率生成
        if np.random.random() < 0.3:  # 30%的概率生成新请求
            new_request = self.traffic_generator.generate_request(
                self.current_network
            )
            new_request['arrival_time'] = self.simulation_time
            arriving_requests.append(new_request)
        
        return arriving_requests
    
    def _calculate_request_delay(self, request: Dict[str, Any], 
                                path: List[int]) -> float:
        """
        计算请求的实际时延
        
        Args:
            request: 请求
            path: 调度路径
            
        Returns:
            实际时延
        """
        # 简化实现：计算路径时延加上处理时延
        path_delay = 0.0
        
        # 计算路径时延
        link_map = {}
        for link in self.current_network['links']:
            src = link['src']
            dst = link['dst']
            delay = link['delay']
            link_map[(src, dst)] = delay
            link_map[(dst, src)] = delay
        
        for i in range(len(path) - 1):
            src = path[i]
            dst = path[i + 1]
            path_delay += link_map.get((src, dst), 0.0)
        
        # 计算处理时延（与计算需求相关）
        compute_requirement = request.get('compute_requirement', 0)
        # 假设处理速度为 10 GFLOPS/ms
        processing_delay = compute_requirement / 10.0
        
        # 总时延
        total_delay = path_delay + processing_delay
        
        return total_delay
    
    def _check_request_constraints(self, request: Dict[str, Any], 
                                  actual_delay: float) -> bool:
        """
        检查请求约束是否满足
        
        Args:
            request: 请求
            actual_delay: 实际时延
            
        Returns:
            是否满足约束
        """
        # 检查时延约束
        delay_tolerance = request.get('delay_tolerance', float('inf'))
        if actual_delay > delay_tolerance:
            return False
        
        # 检查带宽约束（简化）
        data_size = request.get('data_size', 0)
        # 这里简化处理，假设带宽足够
        
        return True
    
    def _collect_step_metrics(self) -> Dict[str, Any]:
        """
        收集当前步的指标
        
        Returns:
            指标字典
        """
        # 合并所有请求
        all_requests = (
            self.current_requests + 
            self.completed_requests + 
            self.failed_requests
        )
        
        # 计算请求指标
        request_metrics = self.traffic_generator.calculate_request_metrics(all_requests)
        
        # 计算网络指标
        network_metrics = self.network_generator.get_network_metrics(self.current_network)
        
        # 计算负载均衡指标
        load_balance_metric = self._calculate_load_balance()
        
        # 计算能耗指标
        energy_metric = self._calculate_energy_consumption()
        
        # 合并指标
        step_metrics = {
            'simulation_time': self.simulation_time,
            **request_metrics,
            **network_metrics,
            'load_balance': load_balance_metric,
            'energy_consumption': energy_metric,
            'cache_hit_rate': self._get_cache_hit_rate() if self.scheduler else 0
        }
        
        return step_metrics
    
    def _calculate_load_balance(self) -> float:
        """
        计算负载均衡指标
        
        Returns:
            负载均衡指标（负载方差）
        """
        compute_loads = []
        for node in self.current_network['nodes']:
            if node['is_compute_node']:
                compute_loads.append(node['current_load'])
        
        if len(compute_loads) > 1:
            return np.var(compute_loads)
        else:
            return 0.0
    
    def _calculate_energy_consumption(self) -> float:
        """
        计算能耗指标
        
        Returns:
            总能耗
        """
        total_energy = 0.0
        
        for node in self.current_network['nodes']:
            if node['is_compute_node']:
                # 能耗 = 能耗系数 × 计算能力 × 负载
                energy_coeff = node.get('energy_coefficient', 0.0)
                compute_capacity = node.get('compute_capacity', 0.0)
                load = node.get('current_load', 0.0)
                
                total_energy += energy_coeff * compute_capacity * load
        
        return total_energy
    
    def _get_cache_hit_rate(self) -> float:
        """
        获取缓存命中率
        
        Returns:
            缓存命中率
        """
        if self.scheduler and hasattr(self.scheduler, 'path_cache'):
            cache_stats = self.scheduler.path_cache.get_stats()
            return cache_stats.get('hit_rate', 0.0)
        return 0.0
    
    def _check_simulation_done(self) -> bool:
        """
        检查仿真是否结束
        
        Returns:
            是否结束
        """
        # 结束条件：所有请求都已处理或仿真时间超过阈值
        if len(self.current_requests) == 0:
            return True
        
        # 如果仿真时间超过100秒，也结束
        if self.simulation_time > 100.0:
            return True
        
        return False
    
    def get_overall_metrics(self) -> Dict[str, Any]:
        """
        获取整体仿真指标
        
        Returns:
            整体指标字典
        """
        if not self.metrics_history:
            return {}
        
        # 使用最后一步的指标作为整体指标
        final_metrics = self.metrics_history[-1]
        
        # 添加一些累积指标
        all_requests = self.completed_requests + self.failed_requests
        final_metrics['total_processed'] = len(all_requests)
        final_metrics['completion_rate'] = len(self.completed_requests) / len(all_requests) if all_requests else 0
        
        return final_metrics
    
    def reset(self):
        """重置仿真环境"""
        self.current_network = None
        self.current_requests = []
        self.completed_requests = []
        self.failed_requests = []
        self.simulation_time = 0.0
        self.metrics_history = []
        
    def visualize(self, save_path: str = None):
        """
        可视化当前网络状态
        
        Args:
            save_path: 保存路径
        """
        if self.current_network:
            self.network_generator.visualize_network(
                self.current_network, save_path
            )