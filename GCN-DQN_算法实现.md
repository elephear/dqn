# 基于GCN-DQN的算力感知调度算法实现

本文档实现了《面向差异化业务的算力感知多目标调度策略研究》第4章中的算法过程，包括GCN特征提取、DQN决策优化、动态权重调整和快速路径缓存等核心模块。

## 1. 算法总体框架

```python
class GCN_DQN_Scheduler:
    """
    GCN-DQN算力感知调度算法主类
    """
    def __init__(self, network_graph, config):
        self.graph = network_graph  # 网络图结构
        self.config = config        # 算法配置参数
        
        # 初始化各模块
        self.gcn_extractor = GCNFeatureExtractor(config)
        self.dqn_agent = DQNAgent(config)
        self.weight_adjuster = DynamicWeightAdjuster(config)
        self.path_cache = FastPathCache(config)
        
        # 状态跟踪
        self.current_state = None
        self.history_states = []
        
    def schedule(self, request):
        """
        调度主函数
        输入: request - 业务请求 (type, src, dst, requirements)
        输出: (target_node, path, q_value) - 调度决策
        """
        # 1. 获取当前网络状态
        network_state = self.get_network_state()
        
        # 2. 提取图特征
        graph_features = self.gcn_extractor.extract(network_state)
        
        # 3. 动态调整权重
        weights = self.weight_adjuster.adjust(request.type, network_state)
        
        # 4. 检查缓存
        cached_decision = self.path_cache.lookup(request.src, request.dst, network_state)
        if cached_decision and self.validate_decision(cached_decision, network_state):
            return cached_decision
        
        # 5. DQN决策
        action = self.dqn_agent.select_action(graph_features, request, weights)
        
        # 6. 更新缓存
        self.path_cache.update(request.src, request.dst, network_state, action)
        
        return action
```

## 2. 图卷积网络特征提取模块

### 2.1 图结构建模

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNFeatureExtractor(nn.Module):
    """
    两层GCN特征提取器
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 第一层GCN
        self.gcn1 = GCNLayer(
            in_features=config.node_feature_dim,
            out_features=config.gcn_hidden_dim
        )
        
        # 第二层GCN
        self.gcn2 = GCNLayer(
            in_features=config.gcn_hidden_dim,
            out_features=config.gcn_output_dim
        )
        
        # 图级特征聚合（均值池化）
        self.pooling = GlobalMeanPooling()
        
        # 特征归一化
        self.norm = nn.LayerNorm(config.gcn_output_dim)
        
    def forward(self, A, X):
        """
        前向传播
        输入:
            A: 邻接矩阵 (|V| x |V|)
            X: 节点特征矩阵 (|V| x d)
        输出:
            h_G: 图级特征向量 (d')
        """
        # 添加自连接
        A_hat = A + torch.eye(A.size(0))
        
        # 计算归一化邻接矩阵
        D_hat = torch.diag(torch.sum(A_hat, dim=1))
        D_hat_inv_sqrt = torch.inverse(torch.sqrt(D_hat))
        A_norm = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt
        
        # 第一层GCN
        H1 = F.relu(self.gcn1(A_norm, X))
        
        # 第二层GCN
        Z = F.relu(self.gcn2(A_norm, H1))
        
        # 图级特征聚合
        h_G = self.pooling(Z)
        
        # 特征归一化
        h_G = self.norm(h_G)
        
        return h_G
    
    def extract(self, network_state):
        """
        从网络状态提取特征
        """
        # 构建邻接矩阵和特征矩阵
        A = self.build_adjacency_matrix(network_state)
        X = self.build_feature_matrix(network_state)
        
        # 提取特征
        with torch.no_grad():
            features = self.forward(A, X)
        
        return features.cpu().numpy()

class GCNLayer(nn.Module):
    """
    单层GCN实现
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, A, X):
        """
        GCN层前向传播: H = σ(AXW)
        """
        # 消息传递: AX
        support = torch.matmul(A, X)
        
        # 线性变换: (AX)W
        output = self.linear(support)
        
        return output

class GlobalMeanPooling(nn.Module):
    """
    全局均值池化
    """
    def forward(self, Z):
        """
        输入: Z (|V| x d')
        输出: h_G (d')
        """
        return torch.mean(Z, dim=0)
```

### 2.2 网络状态到图表示的转换

```python
def build_adjacency_matrix(self, network_state):
    """
    从网络状态构建邻接矩阵
    """
    n_nodes = len(network_state.nodes)
    A = torch.zeros((n_nodes, n_nodes))
    
    for i, node_i in enumerate(network_state.nodes):
        for j, node_j in enumerate(network_state.nodes):
            if node_j in network_state.adjacency_list[node_i]:
                # 链路存在，权重为1（无向图）
                A[i, j] = 1
                A[j, i] = 1
    
    return A

def build_feature_matrix(self, network_state):
    """
    构建节点特征矩阵
    特征包括: [计算能力, 当前负载, 能耗系数, 节点类型]
    """
    features = []
    for node in network_state.nodes:
        if node.is_compute_node:
            node_features = [
                node.compute_capacity,      # 计算能力 (GFLOPS)
                node.current_load,          # 当前负载 [0,1]
                node.energy_coefficient,    # 能耗系数
                1.0                         # 算力节点标识
            ]
        else:
            node_features = [
                0.0,                        # 无计算能力
                0.0,                        # 无负载
                0.0,                        # 无能耗系数
                0.0                         # 转发节点标识
            ]
        features.append(node_features)
    
    return torch.tensor(features, dtype=torch.float32)
```

## 3. 深度Q网络决策优化模块

### 3.1 DQN智能体实现

```python
class DQNAgent:
    """
    深度Q网络智能体
    """
    def __init__(self, config):
        self.config = config
        
        # 主网络和目标网络
        self.q_network = QNetwork(config)
        self.target_network = QNetwork(config)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate
        )
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
        
        # 探索策略
        self.epsilon = config.epsilon_max
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        
        # 训练计数器
        self.train_step = 0
        
    def select_action(self, state_features, request, weights):
        """
        选择动作（ε-greedy策略）
        """
        # 计算所有可能动作的Q值
        possible_actions = self.get_possible_actions(request)
        
        if np.random.random() < self.epsilon:
            # 探索：随机选择动作
            action_idx = np.random.randint(len(possible_actions))
        else:
            # 利用：选择最大Q值动作
            q_values = []
            for action in possible_actions:
                q_value = self.q_network(state_features, action, weights)
                q_values.append(q_value)
            
            action_idx = np.argmax(q_values)
        
        # 衰减ε
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )
        
        return possible_actions[action_idx]
    
    def train_step(self, batch):
        """
        训练一步
        """
        states, actions, rewards, next_states, dones = batch
        
        # 计算当前Q值
        current_q = self.q_network(states, actions)
        
        # 计算目标Q值（Double DQN）
        with torch.no_grad():
            # 选择动作（主网络）
            next_actions = self.q_network.get_best_actions(next_states)
            # 评估Q值（目标网络）
            next_q = self.target_network(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.config.gamma * next_q
        
        # 计算损失（Huber损失）
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 定期更新目标网络
        if self.train_step % self.config.target_update_freq == 0:
            self.update_target_network()
        
        self.train_step += 1
        
        return loss.item()
    
    def update_target_network(self):
        """
        软更新目标网络
        """
        tau = self.config.tau
        for target_param, param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )
```

### 3.2 Q网络架构

```python
class QNetwork(nn.Module):
    """
    双流Q网络架构
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 状态流（处理图特征）
        self.state_stream = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU()
        )
        
        # 动作流（处理动作特征）
        self.action_embedding = nn.Embedding(
            config.action_space_size,
            config.action_embed_dim
        )
        
        self.action_stream = nn.Sequential(
            nn.Linear(config.action_embed_dim, config.hidden_dim),
            nn.ReLU()
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)  # Q值输出
        )
        
        # 权重调整层（多目标优化）
        self.weight_layer = nn.Linear(config.num_objectives, config.hidden_dim)
        
    def forward(self, state, action, weights=None):
        """
        前向传播
        输入:
            state: 状态特征 (batch_size, state_dim)
            action: 动作索引 (batch_size,)
            weights: 多目标权重 (batch_size, num_objectives)
        输出:
            q_value: Q值 (batch_size, 1)
        """
        # 状态流
        state_features = self.state_stream(state)
        
        # 动作流
        action_embedded = self.action_embedding(action)
        action_features = self.action_stream(action_embedded)
        
        # 权重调整（如果提供）
        if weights is not None:
            weight_features = self.weight_layer(weights)
            state_features = state_features + weight_features
        
        # 特征融合
        combined = torch.cat([state_features, action_features], dim=1)
        q_value = self.fusion_layer(combined)
        
        return q_value
```

### 3.3 奖励函数实现

```python
class RewardFunction:
    """
    差异化业务奖励函数
    """
    def __init__(self, config):
        self.config = config
        
        # 奖励参数
        self.R_success = config.R_success          # 成功完成奖励
        self.R_violation = config.R_violation      # 违反约束惩罚
        self.alpha = config.alpha                  # 时延惩罚系数
        self.beta = config.beta                    # 中间步骤惩罚系数
        
    def compute_reward(self, state, action, next_state, done, business_type):
        """
        计算奖励
        """
        if business_type == "edge_ai":
            return self._edge_ai_reward(state, action, next_state, done)
        elif business_type == "cross_domain":
            return self._cross_domain_reward(state, action, next_state, done)
        else:
            raise ValueError(f"未知业务类型: {business_type}")
    
    def _edge_ai_reward(self, state, action, next_state, done):
        """
        边缘AI推理业务奖励
        """
        if done:
            if self._check_constraints_violated(next_state):
                # 违反约束
                return self.R_violation
            else:
                # 成功完成，奖励与时延负相关
                total_delay = self._compute_total_delay(state, action, next_state)
                return self.R_success - self.alpha * total_delay
        else:
            # 中间步骤，惩罚步长时间
            step_delay = self._compute_step_delay(state, action)
            return -self.beta * step_delay
    
    def _cross_domain_reward(self, state, action, next_state, done):
        """
        算力跨域调度业务奖励
        """
        if done:
            if self._check_constraints_violated(next_state):
                # 违反约束
                return self.R_violation
            else:
                # 成功完成，多目标优化
                f1 = self._compute_computing_accessibility(state, action)
                f2 = self._compute_load_balance(state, action)
                f3 = self._compute_energy_consumption(state, action)
                
                return (self.R_success 
                        - self.config.gamma1 * f1
                        - self.config.gamma2 * f2
                        - self.config.gamma3 * f3)
        else:
            # 中间步骤，惩罚最大链路利用率
            max_utilization = self._compute_max_link_utilization(state, action)
            return -self.config.delta * max_utilization
```

## 4. 动态权重调整机制

```python
class DynamicWeightAdjuster:
    """
    动态权重调整器
    """
    def __init__(self, config):
        self.config = config
        
        # 初始权重（根据业务类型）
        self.weights = {
            "edge_ai": {
                "delay": 1.0,      # 时延最小化
                "load_balance": 0.0,
                "energy": 0.0
            },
            "cross_domain": {
                "delay": 0.4,      # 算力可达性
                "load_balance": 0.3,  # 负载均衡
                "energy": 0.3      # 能耗优化
            }
        }
        
        # 历史目标值记录
        self.history_values = {
            "delay": {"min": float('inf'), "max": 0.0},
            "load_balance": {"min": float('inf'), "max": 0.0},
            "energy": {"min": float('inf'), "max": 0.0}
        }
        
        # 敏感系数
        self.alpha = {
            "delay": 0.5,
            "load_balance": 0.3,
            "energy": 0.2
        }
        
    def adjust(self, business_type, network_state):
        """
        调整权重
        """
        # 获取基础权重
        base_weights = self.weights[business_type].copy()
        
        # 计算当前目标值
        current_values = self._compute_current_values(network_state)
        
        # 更新历史记录
        self._update_history(current_values)
        
        # 计算紧迫性因子
        urgency_factors = {}
        for objective in base_weights.keys():
            if base_weights[objective] > 0:
                current_val = current_values[objective]
                min_val = self.history_values[objective]["min"]
                max_val = self.history_values[objective]["max"]
                
                if max_val > min_val:
                    # 归一化紧迫性
                    normalized = (current_val - min_val) / (max_val - min_val)
                    urgency = 1 + self.alpha[objective] * normalized
                else:
                    urgency = 1.0
                
                urgency_factors[objective] = urgency
        
        # 调整权重
        adjusted_weights = {}
        total_weight = 0
        
        for objective, base_weight in base_weights.items():
            if base_weight > 0:
                adjusted = base_weight * urgency_factors.get(objective, 1.0)
                adjusted_weights[objective] = adjusted
                total_weight += adjusted
            else:
                adjusted_weights[objective] = 0.0
        
        # 归一化
        if total_weight > 0:
            for objective in adjusted_weights:
                adjusted_weights[objective] /= total_weight
        
        return adjusted_weights
    
        from_weights = self.weights[from_type]
        to_weights = self.weights[to_type]
        
        # 平滑过渡
        for objective in from_weights.keys():
            current_weight = from_weights[objective]
            target_weight = to_weights[objective]
            
            # 线性插值
            new_weight = (1 - transition_rate) * current_weight + transition_rate * target_weight
            from_weights[objective] = new_weight
        
        return from_weights
    
    def _compute_current_values(self, network_state):
        """
        计算当前各目标值
        """
        values = {}
        
        # 计算平均时延
        total_delay = 0
        count = 0
        for link in network_state.links:
            total_delay += link.delay * link.utilization
            count += 1
        values["delay"] = total_delay / count if count > 0 else 0
        
        # 计算负载均衡度（链路利用率方差）
        utilizations = [link.utilization for link in network_state.links]
        values["load_balance"] = np.var(utilizations) if utilizations else 0
        
        # 计算能耗
        total_energy = 0
        for node in network_state.nodes:
            if node.is_compute_node:
                total_energy += node.energy_coefficient * node.current_load
        values["energy"] = total_energy
        
        return values
    
    def _update_history(self, current_values):
        """
        更新历史记录
        """
        for objective, value in current_values.items():
            if value < self.history_values[objective]["min"]:
                self.history_values[objective]["min"] = value
            if value > self.history_values[objective]["max"]:
                self.history_values[objective]["max"] = value


## 5. 快速路径缓存技术

```python
import hashlib
from collections import OrderedDict

class FastPathCache:
    """
    快速路径缓存技术
    维护两级缓存：节点对缓存和状态缓存
    """
    def __init__(self, config):
        self.config = config
        
        # 节点对缓存：存储常用源-目的节点对的最优路径
        self.node_pair_cache = OrderedDict()
        
        # 状态缓存：存储特定网络状态下的决策结果
        self.state_cache = OrderedDict()
        
        # 缓存大小限制
        self.max_node_pair_cache = config.max_node_pair_cache
        self.max_state_cache = config.max_state_cache
        
        # 状态变化阈值
        self.state_change_threshold = config.state_change_threshold
        
        # 缓存命中统计
        self.hits = 0
        self.misses = 0
        
    def lookup(self, src, dst, network_state):
        """
        查找缓存
        返回: (target_node, path, q_value) 或 None
        """
        # 1. 检查节点对缓存
        node_pair_key = (src, dst)
        if node_pair_key in self.node_pair_cache:
            cached = self.node_pair_cache[node_pair_key]
            
            # 验证缓存有效性
            if self._validate_cache_entry(cached, network_state):
                # 更新LRU顺序
                self.node_pair_cache.move_to_end(node_pair_key)
                self.hits += 1
                return cached
        
        # 2. 检查状态缓存
        state_hash = self._hash_network_state(network_state)
        state_key = (src, dst, state_hash)
        
        if state_key in self.state_cache:
            cached = self.state_cache[state_key]
            
            # 验证缓存有效性
            if self._validate_cache_entry(cached, network_state):
                # 更新LRU顺序
                self.state_cache.move_to_end(state_key)
                self.hits += 1
                return cached
        
        self.misses += 1
        return None
    
    def update(self, src, dst, network_state, decision):
        """
        更新缓存
        """
        # 1. 更新节点对缓存
        node_pair_key = (src, dst)
        self.node_pair_cache[node_pair_key] = decision
        self.node_pair_cache.move_to_end(node_pair_key)
        
        # 维护缓存大小（LRU淘汰）
        if len(self.node_pair_cache) > self.max_node_pair_cache:
            self.node_pair_cache.popitem(last=False)
        
        # 2. 更新状态缓存
        state_hash = self._hash_network_state(network_state)
        state_key = (src, dst, state_hash)
        self.state_cache[state_key] = decision
        self.state_cache.move_to_end(state_key)
        
        # 维护缓存大小
        if len(self.state_cache) > self.max_state_cache:
            self.state_cache.popitem(last=False)
    
    def _validate_cache_entry(self, cached_decision, current_state):
        """
        验证缓存条目的有效性
        """
        target_node, path, q_value = cached_decision
        
        # 检查路径可行性
        if not self._check_path_feasibility(path, current_state):
            return False
        
        # 检查状态变化是否超过阈值
        cached_state_hash = cached_decision.get('state_hash', None)
        if cached_state_hash:
            current_state_hash = self._hash_network_state(current_state)
            # 计算状态差异
            state_diff = self._compute_state_difference(cached_state_hash, current_state_hash)
            if state_diff > self.state_change_threshold:
                return False
        
        return True
    
    def _check_path_feasibility(self, path, network_state):
        """
        检查路径是否满足约束条件
        """
        # 检查带宽约束
        for i in range(len(path) - 1):
            src_node = path[i]
            dst_node = path[i + 1]
            
            # 查找链路
            link = network_state.get_link(src_node, dst_node)
            if not link:
                return False
            
            # 检查带宽是否足够
            if link.utilization > 0.9:  # 利用率超过90%认为不可用
                return False
        
        return True
    
    def _hash_network_state(self, network_state):
        """
        计算网络状态的哈希值
        """
        # 提取关键状态特征
        state_features = []
        
        # 节点负载
        for node in network_state.nodes:
            if node.is_compute_node:
                state_features.append(f"{node.id}:{node.current_load:.3f}")
        
        # 链路利用率
        for link in network_state.links:
            state_features.append(f"{link.src}-{link.dst}:{link.utilization:.3f}")
        
        # 排序确保一致性
        state_features.sort()
        
        # 计算哈希
        state_str = "|".join(state_features)
        return hashlib.md5(state_str.encode()).hexdigest()
    
    def _compute_state_difference(self, hash1, hash2):
        """
        计算两个状态哈希的差异（简化版）
        """
        # 在实际实现中，可以计算状态向量的欧氏距离
        # 这里使用哈希比较作为简化
        return 0.0 if hash1 == hash2 else 1.0
    
    def warm_up(self, historical_data):
        """
        缓存预热：使用历史数据预填充缓存
        """
        for data in historical_data:
            src = data['src']
            dst = data['dst']
            state = data['network_state']
            decision = data['decision']
            
            self.update(src, dst, state, decision)
    
    def get_cache_stats(self):
        """
        获取缓存统计信息
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'node_pair_cache_size': len(self.node_pair_cache),
            'state_cache_size': len(self.state_cache)
        }
```

## 6. 算法训练与部署流程

### 6.1 离线训练阶段

```python
def train_gcn_dqn(config, training_data):
    """
    GCN-DQN离线训练
    """
    # 初始化各组件
    scheduler = GCN_DQN_Scheduler(config.network_graph, config)
    
    # 预训练GCN（无监督学习）
    print("预训练GCN特征提取器...")
    pretrain_gcn(scheduler.gcn_extractor, training_data)
    
    # 联合训练GCN-DQN
    print("开始联合训练GCN-DQN...")
    
    for epoch in range(config.num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch in create_training_batches(training_data, config.batch_size):
            # 前向传播
            states, actions, rewards, next_states = process_batch(batch)
            
            # 提取特征
            state_features = scheduler.gcn_extractor.extract_batch(states)
            next_state_features = scheduler.gcn_extractor.extract_batch(next_states)
            
            # DQN训练
            batch_loss = scheduler.dqn_agent.train_batch(
                state_features, actions, rewards, next_state_features
            )
            
            epoch_loss += batch_loss
            num_batches += 1
        
        # 打印训练进度
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {avg_loss:.4f}")
        
        # 定期保存模型
        if (epoch + 1) % config.save_interval == 0:
            save_model(scheduler, f"model_epoch_{epoch+1}.pth")
    
    return scheduler
```

### 6.2 在线部署阶段

```python
class OnlineScheduler:
    """
    在线调度器
    """
    def __init__(self, model_path, config):
        # 加载预训练模型
        self.scheduler = load_model(model_path)
        self.config = config
        
        # 在线微调参数
        self.online_buffer = ReplayBuffer(config.online_buffer_size)
        self.fine_tune_interval = config.fine_tune_interval
        self.request_count = 0
        
    def process_request(self, request):
        """
        处理业务请求
        """
        # 1. 调度决策
        decision = self.scheduler.schedule(request)
        
        # 2. 执行调度
        result = self._execute_scheduling(decision)
        
        # 3. 收集反馈
        reward = self._compute_reward(result, request)
        
        # 4. 存储经验
        self._store_experience(request, decision, reward, result)
        
        # 5. 定期在线微调
        self.request_count += 1
        if self.request_count % self.fine_tune_interval == 0:
            self._online_fine_tune()
        
        return decision, result
    
    def _online_fine_tune(self):
        """
        在线微调
        """
        if len(self.online_buffer) < self.config.min_batch_size:
            return
        
        # 采样批次数据
        batch = self.online_buffer.sample(self.config.online_batch_size)
        
        # 微调模型
        self.scheduler.dqn_agent.fine_tune(batch)
        
        # 更新缓存（根据新模型调整缓存内容）
        self._update_cache_with_new_policy()
```

### 6.3 训练超参数配置

```python
class Config:
    """
    算法配置参数
    """
    # 网络参数
    node_feature_dim = 4          # 节点特征维度
    gcn_hidden_dim = 64           # GCN隐藏层维度
    gcn_output_dim = 32           # GCN输出维度
    
    # DQN参数
    state_dim = 32                # 状态特征维度
    action_space_size = 100       # 动作空间大小
    action_embed_dim = 16         # 动作嵌入维度
    hidden_dim = 128              # Q网络隐藏层维度
    
    # 训练参数
    learning_rate = 0.001         # 学习率
    gamma = 0.99                  # 折扣因子
    epsilon_max = 1.0             # 初始探索率
    epsilon_min = 0.01            # 最小探索率
    epsilon_decay = 0.995         # 探索率衰减
    
    # 经验回放
    replay_buffer_size = 10000    # 回放缓冲区大小
    batch_size = 32               # 批次大小
    target_update_freq = 100      # 目标网络更新频率
    tau = 0.01                    # 软更新系数
    
    # 动态权重调整
    num_objectives = 3            # 优化目标数量
    
    # 缓存参数
    max_node_pair_cache = 1000    # 节点对缓存大小
    max_state_cache = 5000        # 状态缓存大小
    state_change_threshold = 0.1  # 状态变化阈值
    
    # 训练流程
    num_epochs = 100              # 训练轮数
    save_interval = 10            # 模型保存间隔
    
    # 在线部署
    online_buffer_size = 5000     # 在线经验缓冲区大小
    fine_tune_interval = 100      # 在线微调间隔
    min_batch_size = 32           # 最小微调批次大小
    online_batch_size = 16        # 在线微调批次大小
```

## 7. 算法复杂度分析

### 7.1 时间复杂度分析

1. **GCN特征提取**：
   - 邻接矩阵归一化：$O(|V|^2)$
   - 两层GCN前向传播：$O(L \cdot (|E| \cdot d + |V| \cdot d^2))$
   - 其中 $L=2$ 为层数，$d$ 为特征维度

2. **DQN决策**：
   - Q网络前向传播：$O(d_{hidden}^2)$
   - 动作选择：$O(|A|)$，其中 $|A|$ 为动作空间大小

3. **缓存查询**：
   - 哈希表查找：$O(1)$
   - 路径验证：$O(|P|)$，其中 $|P|$ 为路径长度

**总时间复杂度**：$O(|V|^2 + |E| \cdot d + d_{hidden}^2 + |A|)$

### 7.2 空间复杂度分析

1. **模型参数**：
   - GCN参数：$O(|V| \cdot d + d^2)$
   - DQN参数：$O(d_{state}^2 + |A| \cdot d_{action})$

2. **状态存储**：
   - 网络状态：$O(|V| + |E|)$
   - 特征表示：$O(|V| \cdot d)$

3. **缓存存储**：
   - 节点对缓存：$O(N_{node\_pair})$
   - 状态缓存：$O(N_{state})$

**总空间复杂度**：$O(|V|^2 + |E| + d^2 + N_{cache})$

## 8. 总结

本文档完整实现了基于GCN-DQN的算力感知调度算法，包含以下核心模块：

1. **GCN特征提取模块**：利用图卷积网络从网络-算力联合状态中提取拓扑特征
2. **DQN决策优化模块**：采用双流Q网络架构进行路径-节点组合决策
3. **动态权重调整机制**：根据业务类型和实时网络状态自适应调整多目标权重
4. **快速路径缓存技术**：通过两级缓存提升算法实时性与决策效率

算法支持差异化业务（边缘AI推理和算力跨域调度）的统一调度，实现了网络资源与算力资源的协同优化。通过离线训练与在线部署相结合的方式，既保证了决策质量，又满足了实时性要求。

该实现可作为实际系统开发的基础框架，也可用于学术研究的仿真验证。
