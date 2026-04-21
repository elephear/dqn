import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional
from .gcn import GCNFeatureExtractor
from .dqn import DQNAgent
from ..algorithms.weight_adjuster import DynamicWeightAdjuster
from ..algorithms.path_cache import FastPathCache

class GCN_DQN_Scheduler:
    """
    GCN-DQN算力感知调度算法主类
    基于论文第4章设计
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # 设备配置
        self.device = self._get_device(config)
        print(f"使用设备: {self.device}")
        
        # 初始化各模块
        self.gcn_extractor = GCNFeatureExtractor(config, device=self.device).to(self.device)
        self.dqn_agent = DQNAgent(config)
        self.weight_adjuster = DynamicWeightAdjuster(config)
        self.path_cache = FastPathCache(config)
        
        # 状态跟踪
        self.current_state = None
        self.history_states = []
        
        # 训练模式
        self.training_mode = True
    
    def _get_device(self, config: dict) -> torch.device:
        """
        获取可用设备
        
        Args:
            config: 配置字典
            
        Returns:
            torch.device: 设备对象
        """
        device_config = config.get('device', {})
        device_type = device_config.get('type', 'auto')
        
        if device_type == 'cuda' and torch.cuda.is_available():
            device_id = device_config.get('device_id', 0)
            return torch.device(f'cuda:{device_id}')
        elif device_type == 'cpu':
            return torch.device('cpu')
        else:
            # 自动选择
            if torch.cuda.is_available():
                return torch.device('cuda:0')
            else:
                return torch.device('cpu')
        
    def schedule(self, request: Dict[str, Any], 
                 network_state: Dict[str, Any]) -> Tuple[Optional[int], Optional[list], Optional[float]]:
        """
        调度主函数
        
        Args:
            request: 业务请求 {
                'type': 'edge_ai' 或 'compute_scheduling',
                'src': 源节点ID,
                'dst': 目的节点ID,
                'compute_requirement': 计算需求(GFLOPS),
                'delay_tolerance': 时延容忍度(ms),
                'data_size': 数据大小(MB)
            }
            network_state: 网络状态
            
        Returns:
            (target_node, path, q_value) - 调度决策
        """
        # 1. 更新当前状态
        self.current_state = network_state
        self.history_states.append(network_state)
        
        # 2. 提取图特征
        try:
            h_G, _ = self.gcn_extractor.extract_features(network_state)
            # 确保特征在正确的设备上
            h_G = h_G.to(self.device)
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None, None, None
        
        # 3. 动态调整权重
        weights = self.weight_adjuster.adjust(request['type'], network_state)
        weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 4. 检查缓存
        cached_decision = self.path_cache.lookup(
            request['src'], request['dst'], network_state
        )
        if cached_decision:
            target_node, path, q_value = cached_decision
            # 验证缓存决策的可行性
            if self._validate_decision(target_node, path, request, network_state):
                return target_node, path, q_value
        
        # 5. 获取可能的动作
        possible_actions = self._get_possible_actions(request, network_state)
        if not possible_actions:
            print("无可用动作")
            return None, None, None
        
        # 6. DQN决策
        action = self.dqn_agent.select_action(
            h_G.unsqueeze(0),  # 添加批次维度
            possible_actions,
            weights_tensor,
            training=self.training_mode
        )
        
        if action is None:
            print("DQN决策失败")
            return None, None, None
        
        # 7. 解码动作
        target_node, next_hop = self._decode_action(action, request, network_state)
        
        # 8. 构建完整路径
        path = self._construct_path(request['src'], target_node, next_hop, network_state)
        
        # 9. 计算Q值（用于缓存）
        with torch.no_grad():
            q_value = self.dqn_agent.q_network(
                h_G.unsqueeze(0),
                action.unsqueeze(0),
                weights_tensor
            ).item()
        
        # 10. 更新缓存
        decision = (target_node, path, q_value)
        self.path_cache.update(
            request['src'], request['dst'], network_state, decision
        )
        
        # 11. 如果是训练模式，收集经验
        if self.training_mode:
            self._collect_experience(request, action, network_state)
        
        return target_node, path, q_value
    
    def _get_possible_actions(self, request: Dict[str, Any], 
                             network_state: Dict[str, Any]) -> list:
        """
        获取可能的动作
        
        Args:
            request: 业务请求
            network_state: 网络状态
            
        Returns:
            可能的动作列表
        """
        possible_actions = []
        
        # 获取算力节点
        compute_nodes = []
        for node in network_state.get('nodes', []):
            if node.get('is_compute_node', False):
                # 检查算力约束
                if node.get('compute_capacity', 0) >= request.get('compute_requirement', 0):
                    compute_nodes.append(node['id'])
        
        if not compute_nodes:
            return []
        
        # 对于每个可能的算力节点，获取从源节点出发的下一跳
        src_node = request['src']
        for target_node in compute_nodes:
            # 获取从源节点到目标节点的邻居
            neighbors = self._get_neighbors(src_node, network_state)
            
            for neighbor in neighbors:
                # 编码动作: (目标节点索引, 下一跳索引)
                # 这里简化处理，使用节点ID
                action = torch.tensor([target_node, neighbor], dtype=torch.long, device=self.device)
                possible_actions.append(action)
        
        return possible_actions
    
    def _decode_action(self, action: torch.Tensor, request: Dict[str, Any],
                      network_state: Dict[str, Any]) -> Tuple[int, int]:
        """
        解码动作
        
        Args:
            action: 动作张量
            request: 业务请求
            network_state: 网络状态
            
        Returns:
            (target_node, next_hop)
        """
        # 动作编码: (目标节点ID, 下一跳ID)
        target_node = action[0].item()
        next_hop = action[1].item()
        
        return target_node, next_hop
    
    def _construct_path(self, src: int, target: int, next_hop: int,
                       network_state: Dict[str, Any]) -> list:
        """
        构建完整路径
        
        Args:
            src: 源节点
            target: 目标节点
            next_hop: 下一跳
            network_state: 网络状态
            
        Returns:
            路径节点列表
        """
        # 这里简化实现：使用最短路径算法
        # 在实际系统中，应该使用更复杂的路径规划
        
        path = [src]
        
        # 如果下一跳不是目标节点，添加到路径
        if next_hop != target:
            path.append(next_hop)
        
        # 添加目标节点
        path.append(target)
        
        return path
    
    def _validate_decision(self, target_node: int, path: list,
                          request: Dict[str, Any], 
                          network_state: Dict[str, Any]) -> bool:
        """
        验证决策的可行性
        
        Args:
            target_node: 目标节点
            path: 路径
            request: 业务请求
            network_state: 网络状态
            
        Returns:
            是否可行
        """
        # 1. 检查目标节点是否存在且为算力节点
        target_node_info = None
        for node in network_state.get('nodes', []):
            if node['id'] == target_node and node.get('is_compute_node', False):
                target_node_info = node
                break
        
        if target_node_info is None:
            return False
        
        # 2. 检查算力约束
        compute_requirement = request.get('compute_requirement', 0)
        if target_node_info.get('compute_capacity', 0) < compute_requirement:
            return False
        
        # 3. 检查路径连通性
        if not self._check_path_connectivity(path, network_state):
            return False
        
        # 4. 检查时延约束
        total_delay = self._calculate_path_delay(path, network_state)
        delay_tolerance = request.get('delay_tolerance', float('inf'))
        if total_delay > delay_tolerance:
            return False
        
        return True
    
    def _check_path_connectivity(self, path: list, 
                                network_state: Dict[str, Any]) -> bool:
        """
        检查路径连通性
        
        Args:
            path: 路径
            network_state: 网络状态
            
        Returns:
            是否连通
        """
        if len(path) < 2:
            return True
        
        # 获取邻接关系
        adjacency = {}
        for link in network_state.get('links', []):
            src = link.get('src')
            dst = link.get('dst')
            if src not in adjacency:
                adjacency[src] = []
            if dst not in adjacency:
                adjacency[dst] = []
            adjacency[src].append(dst)
            adjacency[dst].append(src)  # 无向图
        
        # 检查路径中相邻节点是否连通
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            
            if current not in adjacency or next_node not in adjacency[current]:
                return False
        
        return True
    
    def _calculate_path_delay(self, path: list, 
                             network_state: Dict[str, Any]) -> float:
        """
        计算路径总时延
        
        Args:
            path: 路径
            network_state: 网络状态
            
        Returns:
            总时延
        """
        total_delay = 0.0
        
        if len(path) < 2:
            return total_delay
        
        # 构建链路映射
        link_map = {}
        for link in network_state.get('links', []):
            src = link.get('src')
            dst = link.get('dst')
            delay = link.get('delay', 0.0)
            link_map[(src, dst)] = delay
            link_map[(dst, src)] = delay  # 无向图
        
        # 累加路径时延
        for i in range(len(path) - 1):
            src = path[i]
            dst = path[i + 1]
            delay = link_map.get((src, dst), 0.0)
            total_delay += delay
        
        return total_delay
    
    def _get_neighbors(self, node_id: int, 
                      network_state: Dict[str, Any]) -> list:
        """
        获取节点的邻居
        
        Args:
            node_id: 节点ID
            network_state: 网络状态
            
        Returns:
            邻居节点ID列表
        """
        neighbors = []
        
        for link in network_state.get('links', []):
            src = link.get('src')
            dst = link.get('dst')
            
            if src == node_id:
                neighbors.append(dst)
            elif dst == node_id:
                neighbors.append(src)
        
        return neighbors
    
    def _collect_experience(self, request: Dict[str, Any], 
                           action: torch.Tensor,
                           network_state: Dict[str, Any]):
        """
        收集经验（简化实现）
        
        Args:
            request: 业务请求
            action: 采取的动作
            network_state: 网络状态
        """
        # 这里简化实现，实际应该根据调度结果计算奖励
        # 并存储到经验回放缓冲区
        
        # 计算奖励
        reward = self._calculate_reward(request, action, network_state)
        
        # 获取下一个状态（这里简化处理）
        next_state = network_state  # 实际应该更新网络状态
        
        # 存储经验
        # 注意：这里需要状态特征，而不是原始网络状态
        try:
            h_G, _ = self.gcn_extractor.extract_features(network_state)
            self.dqn_agent.replay_buffer.push(
                h_G, action, reward, h_G, False  # done=False
            )
        except Exception as e:
            print(f"经验收集失败: {e}")
    
    def _calculate_reward(self, request: Dict[str, Any],
                         action: torch.Tensor,
                         network_state: Dict[str, Any]) -> float:
        """
        计算奖励
        
        Args:
            request: 业务请求
            action: 采取的动作
            network_state: 网络状态
            
        Returns:
            奖励值
        """
        # 基于论文第4.3.2节的奖励函数设计
        
        request_type = request.get('type', 'edge_ai')
        
        if request_type == 'edge_ai':
            # 边缘AI推理业务：主要关注时延
            # 这里简化实现
            return 1.0  # 固定奖励
        else:
            # 算力跨域调度业务：多目标优化
            # 这里简化实现
            return 0.5  # 固定奖励
    
    def train(self):
        """切换到训练模式"""
        self.training_mode = True
        self.dqn_agent.q_network.train()
    
    def eval(self):
        """切换到评估模式"""
        self.training_mode = False
        self.dqn_agent.q_network.eval()
    
    def save(self, path: str):
        """保存模型"""
        self.dqn_agent.save(path)
    
    def load(self, path: str):
        """加载模型"""
        self.dqn_agent.load(path)