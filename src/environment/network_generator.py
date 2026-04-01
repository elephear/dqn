import random
import numpy as np
from typing import List, Dict, Any, Tuple
import networkx as nx

class NetworkGenerator:
    """
    网络拓扑生成器
    基于论文第5.1.2节设计
    """
    
    def __init__(self, config: dict):
        self.config = config
        network_config = config.get('network', {})
        
        self.min_nodes = network_config.get('min_nodes', 10)
        self.max_nodes = network_config.get('max_nodes', 100)
        self.compute_node_ratio = network_config.get('compute_node_ratio', 0.3)
        self.bandwidth_range = network_config.get('bandwidth_range', [10, 100])
        self.delay_range = network_config.get('delay_range', [1, 20])
        self.compute_capacity_range = network_config.get('compute_capacity_range', [10, 100])
        
    def generate_network(self, num_nodes: int = None, 
                        topology_type: str = 'random') -> Dict[str, Any]:
        """
        生成网络拓扑
        
        Args:
            num_nodes: 节点数量，如果为None则在[min_nodes, max_nodes]范围内随机
            topology_type: 拓扑类型 ('random', 'scale_free', 'small_world')
            
        Returns:
            网络状态字典
        """
        if num_nodes is None:
            num_nodes = random.randint(self.min_nodes, self.max_nodes)
        
        # 生成图结构
        if topology_type == 'scale_free':
            graph = nx.barabasi_albert_graph(num_nodes, m=2)
        elif topology_type == 'small_world':
            graph = nx.watts_strogatz_graph(num_nodes, k=4, p=0.3)
        else:  # random
            # 随机连接，确保连通性
            graph = nx.connected_watts_strogatz_graph(num_nodes, k=3, p=0.3)
        
        # 创建节点
        nodes = []
        node_ids = list(graph.nodes())
        
        # 确定算力节点
        num_compute_nodes = max(1, int(num_nodes * self.compute_node_ratio))
        compute_node_ids = random.sample(node_ids, num_compute_nodes)
        
        for node_id in node_ids:
            is_compute_node = node_id in compute_node_ids
            
            node = {
                'id': node_id,
                'is_compute_node': is_compute_node,
                'current_load': random.uniform(0, 0.8) if is_compute_node else 0.0
            }
            
            if is_compute_node:
                node['compute_capacity'] = random.uniform(*self.compute_capacity_range)
                node['energy_coefficient'] = random.uniform(0.5, 2.0)  # J/GFLOPS
            else:
                node['compute_capacity'] = 0.0
                node['energy_coefficient'] = 0.0
            
            nodes.append(node)
        
        # 创建链路
        links = []
        adjacency_list = {}
        
        for edge in graph.edges():
            src, dst = edge
            
            # 随机生成链路属性
            bandwidth = random.uniform(*self.bandwidth_range)
            delay = random.uniform(*self.delay_range)
            utilization = random.uniform(0, 0.7)  # 初始利用率
            
            link = {
                'src': src,
                'dst': dst,
                'bandwidth': bandwidth,
                'delay': delay,
                'utilization': utilization,
                'available_bandwidth': bandwidth * (1 - utilization)
            }
            
            links.append(link)
            
            # 更新邻接表
            if src not in adjacency_list:
                adjacency_list[src] = []
            if dst not in adjacency_list:
                adjacency_list[dst] = []
            
            adjacency_list[src].append(dst)
            adjacency_list[dst].append(src)
        
        # 构建网络状态
        network_state = {
            'nodes': nodes,
            'links': links,
            'adjacency_list': adjacency_list,
            'num_nodes': num_nodes,
            'num_links': len(links),
            'topology_type': topology_type
        }
        
        return network_state
    
    def generate_multiscale_networks(self, sizes: List[int] = None) -> Dict[int, Dict[str, Any]]:
        """
        生成多规模网络
        
        Args:
            sizes: 网络规模列表，如果为None则使用[10, 30, 100]
            
        Returns:
            字典：{规模: 网络状态}
        """
        if sizes is None:
            sizes = self.config.get('experiment', {}).get('network_sizes', [10, 30, 100])
        
        networks = {}
        for size in sizes:
            networks[size] = self.generate_network(size)
        
        return networks
    
    def update_network_state(self, network_state: Dict[str, Any], 
                            time_step: int = 1) -> Dict[str, Any]:
        """
        更新网络状态（模拟动态变化）
        
        Args:
            network_state: 当前网络状态
            time_step: 时间步长
            
        Returns:
            更新后的网络状态
        """
        updated_state = network_state.copy()
        
        # 更新节点负载
        for node in updated_state['nodes']:
            if node['is_compute_node']:
                # 模拟负载变化
                load_change = random.uniform(-0.1, 0.1)
                new_load = node['current_load'] + load_change
                node['current_load'] = max(0.0, min(1.0, new_load))
        
        # 更新链路利用率
        for link in updated_state['links']:
            # 模拟利用率变化
            utilization_change = random.uniform(-0.05, 0.05)
            new_utilization = link['utilization'] + utilization_change
            link['utilization'] = max(0.0, min(0.95, new_utilization))
            link['available_bandwidth'] = link['bandwidth'] * (1 - link['utilization'])
        
        return updated_state
    
    def get_network_metrics(self, network_state: Dict[str, Any]) -> Dict[str, float]:
        """
        计算网络指标
        
        Args:
            network_state: 网络状态
            
        Returns:
            网络指标字典
        """
        nodes = network_state['nodes']
        links = network_state['links']
        
        # 计算平均节点度
        degrees = []
        for node in nodes:
            node_id = node['id']
            degree = len(network_state['adjacency_list'].get(node_id, []))
            degrees.append(degree)
        
        avg_degree = np.mean(degrees) if degrees else 0
        
        # 计算平均链路时延
        delays = [link['delay'] for link in links]
        avg_delay = np.mean(delays) if delays else 0
        
        # 计算平均带宽
        bandwidths = [link['bandwidth'] for link in links]
        avg_bandwidth = np.mean(bandwidths) if bandwidths else 0
        
        # 计算算力节点平均计算能力
        compute_capacities = []
        for node in nodes:
            if node['is_compute_node']:
                compute_capacities.append(node['compute_capacity'])
        
        avg_compute_capacity = np.mean(compute_capacities) if compute_capacities else 0
        
        metrics = {
            'num_nodes': len(nodes),
            'num_links': len(links),
            'num_compute_nodes': sum(1 for node in nodes if node['is_compute_node']),
            'avg_degree': avg_degree,
            'avg_delay': avg_delay,
            'avg_bandwidth': avg_bandwidth,
            'avg_compute_capacity': avg_compute_capacity,
            'graph_density': len(links) / (len(nodes) * (len(nodes) - 1) / 2) if len(nodes) > 1 else 0
        }
        
        return metrics
    
    def visualize_network(self, network_state: Dict[str, Any], save_path: str = None):
        """
        可视化网络拓扑
        
        Args:
            network_state: 网络状态
            save_path: 保存路径，如果为None则显示
        """
        try:
            import matplotlib.pyplot as plt
            
            # 创建图
            G = nx.Graph()
            
            # 添加节点
            for node in network_state['nodes']:
                G.add_node(node['id'], 
                          is_compute=node['is_compute_node'],
                          load=node['current_load'])
            
            # 添加边
            for link in network_state['links']:
                G.add_edge(link['src'], link['dst'],
                          delay=link['delay'],
                          bandwidth=link['bandwidth'])
            
            # 设置节点颜色
            node_colors = []
            for node in G.nodes():
                node_data = G.nodes[node]
                if node_data['is_compute']:
                    # 算力节点：根据负载着色
                    load = node_data['load']
                    node_colors.append(plt.cm.Reds(load))
                else:
                    # 转发节点：蓝色
                    node_colors.append('lightblue')
            
            # 设置节点大小
            node_sizes = []
            for node in G.nodes():
                node_data = G.nodes[node]
                if node_data['is_compute']:
                    node_sizes.append(300)
                else:
                    node_sizes.append(200)
            
            # 绘制
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G, seed=42)
            
            # 绘制边
            nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.5)
            
            # 绘制节点
            nx.draw_networkx_nodes(G, pos, 
                                  node_color=node_colors,
                                  node_size=node_sizes,
                                  edgecolors='black',
                                  linewidths=1)
            
            # 绘制标签
            nx.draw_networkx_labels(G, pos, font_size=10)
            
            plt.title(f"Network Topology ({len(G.nodes())} nodes, {len(G.edges())} edges)")
            plt.axis('off')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                
        except ImportError:
            print("可视化需要matplotlib库，请先安装: pip install matplotlib")