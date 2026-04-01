import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
from .replay_buffer import ReplayBuffer

class QNetwork(nn.Module):
    """
    Q网络
    基于论文第4.3.3节的双流架构设计
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        dqn_config = config.get('dqn', {})
        
        # 状态流: 处理图级特征
        state_dim = dqn_config.get('state_dim', 32)
        hidden_dim = dqn_config.get('hidden_dim', 128)
        
        self.state_stream = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # 动作流: 处理动作特征
        # 动作编码: (目标节点, 下一跳) -> one-hot向量
        # 假设最大节点数为100，最大邻居数为10
        max_nodes = config.get('network', {}).get('max_nodes', 100)
        max_neighbors = 10  # 假设最大邻居数
        
        action_dim = max_nodes + max_neighbors  # one-hot编码维度
        
        self.action_embedding = nn.Embedding(action_dim, hidden_dim // 2)
        
        # 融合层
        fusion_dim = (hidden_dim // 2) * 2
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Q值输出
        )
        
    def forward(self, state_features: torch.Tensor, action: torch.Tensor, 
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state_features: 图级特征向量 (batch_size, state_dim)
            action: 动作索引 (batch_size,) 或 (batch_size, 2)
            weights: 多目标权重 (batch_size, num_objectives)
            
        Returns:
            Q值 (batch_size, 1)
        """
        # 状态流处理
        state_features = self.state_stream(state_features)
        
        # 动作流处理
        if action.dim() == 2:  # (batch_size, 2) -> (目标节点, 下一跳)
            # 将二维动作编码为一维索引
            max_nodes = self.config.get('network', {}).get('max_nodes', 100)
            action_flat = action[:, 0] * max_nodes + action[:, 1]
        else:  # 已经是一维索引
            action_flat = action
            
        action_features = self.action_embedding(action_flat)
        
        # 融合状态和动作特征
        combined = torch.cat([state_features, action_features], dim=-1)
        
        # 如果提供了权重，将其作为额外特征
        if weights is not None:
            combined = torch.cat([combined, weights], dim=-1)
        
        # 输出Q值
        q_value = self.fusion_layers(combined)
        
        return q_value
    
    def get_best_action(self, state_features: torch.Tensor, 
                        possible_actions: List[torch.Tensor],
                        weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取最佳动作
        
        Args:
            state_features: 图级特征向量
            possible_actions: 可能的动作列表
            weights: 多目标权重
            
        Returns:
            best_action: 最佳动作
            best_q_value: 对应的Q值
        """
        if not possible_actions:
            return None, torch.tensor(-float('inf'))
        
        # 计算所有可能动作的Q值
        q_values = []
        for action in possible_actions:
            # 扩展维度以匹配批次
            action_tensor = action.unsqueeze(0) if action.dim() == 1 else action
            q_value = self.forward(state_features, action_tensor, weights)
            q_values.append(q_value)
        
        # 选择最大Q值对应的动作
        q_values_tensor = torch.cat(q_values, dim=0)
        best_idx = torch.argmax(q_values_tensor)
        best_action = possible_actions[best_idx]
        best_q_value = q_values_tensor[best_idx]
        
        return best_action, best_q_value

class DQNAgent:
    """
    DQN智能体
    基于论文第4.3节设计
    """
    
    def __init__(self, config: dict):
        self.config = config
        dqn_config = config.get('dqn', {})
        
        # 主网络和目标网络
        self.q_network = QNetwork(config)
        self.target_network = QNetwork(config)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=dqn_config.get('learning_rate', 0.001)
        )
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(
            capacity=dqn_config.get('replay_buffer_size', 10000)
        )
        
        # 探索策略参数
        self.epsilon = dqn_config.get('epsilon_max', 1.0)
        self.epsilon_min = dqn_config.get('epsilon_min', 0.01)
        self.epsilon_decay = dqn_config.get('epsilon_decay', 0.995)
        self.gamma = dqn_config.get('gamma', 0.99)
        
        # 训练计数器
        self.train_step = 0
        self.target_update_freq = dqn_config.get('target_update_freq', 100)
        
    def select_action(self, state_features: torch.Tensor, 
                      possible_actions: List[torch.Tensor],
                      weights: Optional[torch.Tensor] = None,
                      training: bool = True) -> torch.Tensor:
        """
        选择动作（ε-greedy策略）
        
        Args:
            state_features: 图级特征向量
            possible_actions: 可能的动作列表
            weights: 多目标权重
            training: 是否在训练模式
            
        Returns:
            选择的动作
        """
        if not possible_actions:
            return None
        
        if training and np.random.random() < self.epsilon:
            # 探索：随机选择动作
            action_idx = np.random.randint(len(possible_actions))
            action = possible_actions[action_idx]
        else:
            # 利用：选择最大Q值动作
            action, _ = self.q_network.get_best_action(
                state_features, possible_actions, weights
            )
        
        # 衰减ε
        if training:
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay
            )
        
        return action
    
    def train_step(self, batch_size: int = 32) -> Optional[float]:
        """
        训练一步
        
        Args:
            batch_size: 批次大小
            
        Returns:
            损失值（如果进行了训练）
        """
        # 从回放缓冲区采样
        batch = self.replay_buffer.sample(batch_size)
        if batch is None:
            return None
        
        states, actions, rewards, next_states, dones = batch
        
        # 计算当前Q值
        current_q = self.q_network(states, actions)
        
        # 计算目标Q值（Double DQN）
        with torch.no_grad():
            # 选择动作（主网络）
            next_actions = []
            for i in range(len(next_states)):
                # 这里需要根据状态获取可能的动作
                # 简化处理：使用当前动作
                next_actions.append(actions[i])
            next_actions = torch.stack(next_actions)
            
            # 评估Q值（目标网络）
            next_q = self.target_network(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 计算Huber损失
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 更新训练步数
        self.train_step += 1
        
        # 定期更新目标网络
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def save(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step
        }, path)
    
    def load(self, path: str):
        """
        加载模型
        
        Args:
            path: 加载路径
        """
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']