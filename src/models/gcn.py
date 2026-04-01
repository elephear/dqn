import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class GCNLayer(nn.Module):
    """
    图卷积网络层
    实现: H = σ(AXW)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            A: 归一化邻接矩阵 (|V| x |V|)
            X: 节点特征矩阵 (|V| x in_features)
            
        Returns:
            输出特征矩阵 (|V| x out_features)
        """
        # 消息传递: AX
        support = torch.matmul(A, X)
        
        # 线性变换: (AX)W
        output = self.linear(support)
        
        return output

class GCNFeatureExtractor(nn.Module):
    """
    两层GCN特征提取器
    基于论文第4.2节设计
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        gcn_config = config.get('gcn', {})
        
        # 第一层GCN
        self.gcn1 = GCNLayer(
            in_features=gcn_config.get('node_feature_dim', 4),
            out_features=gcn_config.get('hidden_dim', 64)
        )
        
        # 第二层GCN
        self.gcn2 = GCNLayer(
            in_features=gcn_config.get('hidden_dim', 64),
            out_features=gcn_config.get('output_dim', 32)
        )
        
        # Dropout层
        self.dropout = nn.Dropout(gcn_config.get('dropout', 0.1))
        
        # 激活函数
        activation = gcn_config.get('activation', 'relu')
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            self.activation = F.relu
            
        # 层归一化
        self.norm1 = nn.LayerNorm(gcn_config.get('hidden_dim', 64))
        self.norm2 = nn.LayerNorm(gcn_config.get('output_dim', 32))
        
    def forward(self, A: torch.Tensor, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            A: 邻接矩阵 (|V| x |V|)
            X: 节点特征矩阵 (|V| x d)
            
        Returns:
            h_G: 图级特征向量 (d')
            Z: 节点级特征矩阵 (|V| x d')
        """
        # 添加自连接: Ã = A + I
        A_hat = A + torch.eye(A.size(0), device=A.device)
        
        # 计算归一化邻接矩阵: D^{-1/2}ÃD^{-1/2}
        D_hat = torch.diag(torch.sum(A_hat, dim=1))
        D_hat_inv_sqrt = torch.inverse(torch.sqrt(D_hat))
        A_norm = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt
        
        # 第一层GCN: H^{(1)} = σ(ÃXW^{(0)})
        H1 = self.gcn1(A_norm, X)
        H1 = self.norm1(H1)
        H1 = self.activation(H1)
        H1 = self.dropout(H1)
        
        # 第二层GCN: Z = σ(ÃH^{(1)}W^{(1)})
        Z = self.gcn2(A_norm, H1)
        Z = self.norm2(Z)
        Z = self.activation(Z)
        
        # 图级特征聚合: 均值池化
        h_G = torch.mean(Z, dim=0)
        
        return h_G, Z
    
    def extract_features(self, network_state: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从网络状态提取特征
        
        Args:
            network_state: 网络状态字典
            
        Returns:
            h_G: 图级特征向量
            Z: 节点级特征矩阵
        """
        # 构建邻接矩阵和特征矩阵
        A = self.build_adjacency_matrix(network_state)
        X = self.build_feature_matrix(network_state)
        
        # 提取特征
        with torch.no_grad():
            h_G, Z = self.forward(A, X)
        
        return h_G, Z
    
    def build_adjacency_matrix(self, network_state: dict) -> torch.Tensor:
        """
        从网络状态构建邻接矩阵
        
        Args:
            network_state: 网络状态字典
            
        Returns:
            邻接矩阵 (|V| x |V|)
        """
        nodes = network_state.get('nodes', [])
        adjacency_list = network_state.get('adjacency_list', {})
        
        n_nodes = len(nodes)
        A = torch.zeros((n_nodes, n_nodes))
        
        # 构建节点ID到索引的映射
        node_id_to_idx = {node['id']: i for i, node in enumerate(nodes)}
        
        # 填充邻接矩阵
        for i, node_i in enumerate(nodes):
            node_id = node_i['id']
            if node_id in adjacency_list:
                for neighbor_id in adjacency_list[node_id]:
                    if neighbor_id in node_id_to_idx:
                        j = node_id_to_idx[neighbor_id]
                        A[i, j] = 1
                        A[j, i] = 1  # 无向图
        
        return A
    
    def build_feature_matrix(self, network_state: dict) -> torch.Tensor:
        """
        构建节点特征矩阵
        特征包括: [计算能力, 当前负载, 能耗系数, 节点类型]
        
        Args:
            network_state: 网络状态字典
            
        Returns:
            节点特征矩阵 (|V| x d)
        """
        nodes = network_state.get('nodes', [])
        features = []
        
        for node in nodes:
            if node.get('is_compute_node', False):
                node_features = [
                    node.get('compute_capacity', 0.0),      # 计算能力 (GFLOPS)
                    node.get('current_load', 0.0),          # 当前负载 [0,1]
                    node.get('energy_coefficient', 0.0),    # 能耗系数
                    1.0                                     # 算力节点标识
                ]
            else:
                node_features = [
                    0.0,  # 无计算能力
                    0.0,  # 无负载
                    0.0,  # 无能耗系数
                    0.0   # 转发节点标识
                ]
            features.append(node_features)
        
        return torch.tensor(features, dtype=torch.float32)