import random
import numpy as np
from collections import deque
from typing import Tuple, Optional
import torch

class ReplayBuffer:
    """
    经验回放缓冲区
    存储(state, action, reward, next_state, done)元组
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """
        存储经验
        
        Args:
            state: 当前状态
            action: 采取的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
        """
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple:
        """
        随机采样一批经验
        
        Args:
            batch_size: 批次大小
            
        Returns:
            批次的(state, action, reward, next_state, done)
        """
        if len(self.buffer) < batch_size:
            return None
            
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量
        states = torch.stack(states) if isinstance(states[0], torch.Tensor) else torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states) if isinstance(next_states[0], torch.Tensor) else torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)