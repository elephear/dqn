import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import heapq

class GeneticAlgorithmScheduler:
    """
    遗传算法调度器
    基于论文第5.2.1节的遗传算法
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.name = "GA"
        
        # GA参数
        ga_config = config.get('dqn', {})  # 使用DQN配置中的参数
        self.population_size = ga_config.get('replay_buffer_size', 10000) // 200  # 简化计算
        if self.population_size < 10:
            self.population_size = 50  # 默认值
        
        self.generations = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elitism_count = 2
        
    def schedule(self, request: Dict[str, Any], 
                network_state: Dict[str, Any]) -> Tuple[Optional[int], Optional[list], Optional[float]]:
        """
        遗传算法调度
        
        Args:
            request: 业务请求
            network_state: 网络状态
            
        Returns:
            (target_node, path, fitness) - 调度决策
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
                        'capacity': node['compute_capacity'],
                        'energy': node.get('energy_coefficient', 0.0)
                    }
        
        if not compute_nodes:
            return None, None, None
        
        # 构建图
        graph = self._build_graph(network_state)
        
        # 如果图太小，使用简单方法
        if len(graph) < 5:
            return self._simple_schedule(src, compute_nodes, graph, node_info, request)
        
        # 运行遗传算法
        best_solution = self._genetic_algorithm(src, compute_nodes, graph, node_info, request)
        
        if best_solution:
            target_node, path = best_solution
            # 计算适应度（取负值，因为遗传算法最大化适应度）
            fitness = -self._calculate_fitness(target_node, path, node_info, request)
            return target_node, path, fitness
        
        return None, None, None
    
    def _simple_schedule(self, src: int, compute_nodes: List[int],
                        graph: Dict[int, List[Tuple[int, float]]],
                        node_info: Dict[int, Dict[str, float]],
                        request: Dict[str, Any]) -> Tuple[Optional[int], Optional[list], Optional[float]]:
        """
        简单调度（当图太小时使用）
        """
        best_target = None
        best_path = None
        best_fitness = float('inf')
        
        for target in compute_nodes:
            # 使用Dijkstra找到路径
            path, _ = self._dijkstra_shortest_path(graph, src, target)
            
            if path:
                fitness = self._calculate_fitness(target, path, node_info, request)
                
                if fitness < best_fitness:
                    best_target = target
                    best_path = path
                    best_fitness = fitness
        
        return best_target, best_path, best_fitness
    
    def _genetic_algorithm(self, src: int, compute_nodes: List[int],
                          graph: Dict[int, List[Tuple[int, float]]],
                          node_info: Dict[int, Dict[str, float]],
                          request: Dict[str, Any]) -> Optional[Tuple[int, List[int]]]:
        """
        遗传算法主函数
        
        Returns:
            最佳解 (target_node, path)
        """
        # 初始化种群
        population = self._initialize_population(src, compute_nodes, graph)
        
        for generation in range(self.generations):
            # 评估适应度
            fitness_scores = []
            for individual in population:
                if individual:
                    target, path = individual
                    fitness = self._calculate_fitness(target, path, node_info, request)
                    fitness_scores.append((fitness, individual))
                else:
                    fitness_scores.append((float('inf'), None))
            
            # 排序（最小化适应度）
            fitness_scores.sort(key=lambda x: x[0])
            
            # 选择精英
            elites = [ind for _, ind in fitness_scores[:self.elitism_count] if ind is not None]
            
            # 如果已经找到可行解，可以提前结束
            if fitness_scores[0][0] < float('inf'):
                best_fitness, best_individual = fitness_scores[0]
                if generation > self.generations // 2 and best_fitness < 100:  # 阈值
                    return best_individual
            
            # 选择父代（锦标赛选择）
            parents = self._tournament_selection(population, fitness_scores)
            
            # 生成子代
            offspring = []
            
            # 保留精英
            offspring.extend(elites)
            
            # 交叉和变异
            while len(offspring) < self.population_size:
                if len(parents) >= 2:
                    parent1, parent2 = random.sample(parents, 2)
                    
                    if random.random() < self.crossover_rate:
                        child = self._crossover(parent1, parent2, graph)
                    else:
                        child = parent1 if random.random() < 0.5 else parent2
                    
                    if random.random() < self.mutation_rate:
                        child = self._mutate(child, compute_nodes, graph)
                    
                    if child:
                        offspring.append(child)
                else:
                    # 如果没有足够的父代，随机生成
                    individual = self._generate_random_individual(src, compute_nodes, graph)
                    if individual:
                        offspring.append(individual)
            
            # 更新种群
            population = offspring[:self.population_size]
        
        # 返回最佳解
        fitness_scores = []
        for individual in population:
            if individual:
                target, path = individual
                fitness = self._calculate_fitness(target, path, node_info, request)
                fitness_scores.append((fitness, individual))
        
        if fitness_scores:
            fitness_scores.sort(key=lambda x: x[0])
            return fitness_scores[0][1]
        
        return None
    
    def _initialize_population(self, src: int, compute_nodes: List[int],
                              graph: Dict[int, List[Tuple[int, float]]]) -> List[Optional[Tuple[int, List[int]]]]:
        """
        初始化种群
        
        Returns:
            种群列表，每个个体为 (target_node, path)
        """
        population = []
        
        for _ in range(self.population_size):
            individual = self._generate_random_individual(src, compute_nodes, graph)
            population.append(individual)
        
        return population
    
    def _generate_random_individual(self, src: int, compute_nodes: List[int],
                                   graph: Dict[int, List[Tuple[int, float]]]) -> Optional[Tuple[int, List[int]]]:
        """
        生成随机个体
        """
        if not compute_nodes:
            return None
        
        # 随机选择目标节点
        target = random.choice(compute_nodes)
        
        # 随机生成路径（使用随机游走）
        path = self._random_walk_path(graph, src, target, max_steps=20)
        
        if path:
            return (target, path)
        
        return None
    
    def _random_walk_path(self, graph: Dict[int, List[Tuple[int, float]]],
                         start: int, end: int, max_steps: int = 20) -> Optional[List[int]]:
        """
        随机游走生成路径
        """
        path = [start]
        current = start
        visited = {start}
        
        for _ in range(max_steps):
            if current == end:
                return path
            
            # 获取邻居
            neighbors = [n for n, _ in graph.get(current, [])]
            if not neighbors:
                break
            
            # 优先选择未访问的邻居
            unvisited = [n for n in neighbors if n not in visited]
            if unvisited:
                next_node = random.choice(unvisited)
            else:
                next_node = random.choice(neighbors)
            
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        
        # 如果到达目标，返回路径
        if current == end:
            return path
        
        # 尝试使用最短路径补全
        try:
            shortest_path, _ = self._dijkstra_shortest_path(graph, current, end)
            if shortest_path:
                # 去掉重复的当前节点
                if shortest_path[0] == current:
                    shortest_path = shortest_path[1:]
                return path + shortest_path
        except:
            pass
        
        return None
    
    def _calculate_fitness(self, target: int, path: List[int],
                          node_info: Dict[int, Dict[str, float]],
                          request: Dict[str, Any]) -> float:
        """
        计算适应度（越小越好）
        
        Args:
            target: 目标节点
            path: 路径
            node_info: 节点信息
            request: 请求
            
        Returns:
            适应度值
        """
        if not path:
            return float('inf')
        
        # 1. 路径长度惩罚
        path_length_penalty = len(path) - 1  # 跳数
        
        # 2. 节点负载惩罚
        node_load = node_info.get(target, {}).get('load', 1.0)
        load_penalty = node_load * 10
        
        # 3. 能耗惩罚
        energy_coeff = node_info.get(target, {}).get('energy', 1.0)
        energy_penalty = energy_coeff * 5
        
        # 4. 计算需求匹配度
        compute_req = request.get('compute_requirement', 0)
        node_capacity = node_info.get(target, {}).get('capacity', 0)
        capacity_penalty = 0 if node_capacity >= compute_req else 1000
        
        # 总适应度
        fitness = path_length_penalty + load_penalty + energy_penalty + capacity_penalty
        
        return fitness
    
    def _tournament_selection(self, population: List, 
                             fitness_scores: List[Tuple[float, Any]], 
                             tournament_size: int = 3) -> List:
        """
        锦标赛选择
        """
        selected = []
        
        for _ in range(len(population)):
            # 随机选择 tournament_size 个个体
            tournament = random.sample(list(zip(population, [f for f, _ in fitness_scores])), 
                                      min(tournament_size, len(population)))
            
            # 选择适应度最好的（最小）
            tournament.sort(key=lambda x: x[1])
            selected.append(tournament[0][0])
        
        return selected
    
    def _crossover(self, parent1: Tuple[int, List[int]], 
                  parent2: Tuple[int, List[int]],
                  graph: Dict[int, List[Tuple[int, float]]]) -> Optional[Tuple[int, List[int]]]:
        """
        交叉操作
        """
        target1, path1 = parent1
        target2, path2 = parent2
        
        # 随机选择交叉点
        if len(path1) < 2 or len(path2) < 2:
            return parent1 if random.random() < 0.5 else parent2
        
        # 单点交叉：交换部分路径
        crossover_point1 = random.randint(1, len(path1) - 1)
        crossover_point2 = random.randint(1, len(path2) - 1)
        
        # 创建新路径
        new_path = path1[:crossover_point1] + path2[crossover_point2:]
        
        # 确保路径连通
        new_path = self._repair_path(new_path, graph)
        
        if new_path:
            # 随机选择目标节点
            new_target = target1 if random.random() < 0.5 else target2
            return (new_target, new_path)
        
        return parent1 if random.random() < 0.5 else parent2
    
    def _mutate(self, individual: Tuple[int, List[int]],
               compute_nodes: List[int],
               graph: Dict[int, List[Tuple[int, float]]]) -> Tuple[int, List[int]]:
        """
        变异操作
        """
        target, path = individual
        
        mutation_type = random.random()
        
        if mutation_type < 0.3 and len(path) > 1:
            # 路径变异：随机改变一个节点
            idx = random.randint(1, len(path) - 1)
            current = path[idx - 1]
            
            # 获取当前节点的邻居
            neighbors = [n for n, _ in graph.get(current, [])]
            if neighbors:
                path[idx] = random.choice(neighbors)
                # 修复路径
                path = self._repair_path(path, graph)
        
        elif mutation_type < 0.6 and compute_nodes:
            # 目标节点变异
            new_target = random.choice(compute_nodes)
            target = new_target
        
        else:
            # 随机插入/删除节点
            if len(path) > 2 and random.random() < 0.5:
                # 删除一个中间节点
                idx = random.randint(1, len(path) - 2)
                del path[idx]
            elif len(path) < 10:
                # 插入一个节点
                idx = random.randint(1, len(path) - 1)
                current = path[idx - 1]
                neighbors = [n for n, _ in graph.get(current, [])]
                if neighbors:
                    new_node = random.choice(neighbors)
                    path.insert(idx, new_node)
        
        return (target, path)
    
    def _repair_path(self, path: List[int], 
                    graph: Dict[int, List[Tuple[int, float]]]) -> Optional[List[int]]:
        """
        修复路径，确保连通性
        """
        if not path:
            return None
        
        repaired = [path[0]]
        
        for i in range(1, len(path)):
            current = repaired[-1]
            next_node = path[i]
            
            # 检查是否连通
            neighbors = [n for n, _ in graph.get(current, [])]
            
            if next_node in neighbors:
                repaired.append(next_node)
            else:
                # 尝试找到最短路径连接
                try:
                    shortest_path, _ = self._dijkstra_shortest_path(graph, current, next_node)
                    if shortest_path:
                        # 去掉重复的当前节点
                        if shortest_path[0] == current:
                            shortest_path = shortest_path[1:]
                        repaired.extend(shortest_path)
                    else:
                        # 无法连接，返回None
                        return None
                except:
                    return None
        
        return repaired
    
    def _build_graph(self, network_state: Dict[str, Any]) -> Dict[int, List[Tuple[int, float]]]:
        """
        构建图结构
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
            
            graph[src].append((dst, delay))
            graph[dst].append((src, delay))  # 无向图
        
        return graph
    
    def _dijkstra_shortest_path(self, graph: Dict[int, List[Tuple[int, float]]], 
                               start: int, end: int) -> Tuple[Optional[List[int]], float]:
        """
        Dijkstra最短路径算法
        """
        if start not in graph or end not in graph:
            return None, float('inf')
        
        distances = {node: float('inf') for node in graph}
        predecessors = {node: None for node in graph}
        distances[start] = 0
        
        pq = [(0, start)]
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_node == end:
                break
            
            if current_dist > distances[current_node]:
                continue
            
            for neighbor, weight in graph.get(current_node, []):
                distance = current_dist + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))
        
        if distances[end] == float('inf'):
            return None, float('inf')
        
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()
        
        return path, distances[end]