import hashlib
from collections import OrderedDict
from typing import Tuple, Optional, Dict, Any
import numpy as np

class FastPathCache:
    """
    快速路径缓存
    基于论文第4.5节设计
    """
    
    def __init__(self, config: dict):
        self.config = config
        cache_config = config.get('cache', {})
        
        # 缓存大小
        self.max_size = cache_config.get('max_size', 1000)
        
        # LRU缓存
        self.cache = OrderedDict()
        
        # 验证阈值
        self.validation_threshold = cache_config.get('validation_threshold', 0.1)
        
        # 统计信息
        self.stats = {
            'hits': 0,
            'misses': 0,
            'updates': 0,
            'invalidations': 0
        }
    
    def lookup(self, src_node: int, dst_node: int, 
               network_state: Dict[str, Any]) -> Optional[Tuple]:
        """
        查找缓存
        
        Args:
            src_node: 源节点ID
            dst_node: 目的节点ID
            network_state: 网络状态
            
        Returns:
            缓存的决策 (target_node, path, q_value)，如果未命中则返回None
        """
        # 生成缓存键
        cache_key = self._generate_key(src_node, dst_node, network_state)
        
        if cache_key in self.cache:
            # 缓存命中
            cached_entry = self.cache[cache_key]
            
            # 验证缓存有效性
            if self._validate_entry(cached_entry, network_state):
                # 移动到最近使用位置
                self.cache.move_to_end(cache_key)
                self.stats['hits'] += 1
                return cached_entry['decision']
            else:
                # 缓存失效
                del self.cache[cache_key]
                self.stats['invalidations'] += 1
        
        # 缓存未命中
        self.stats['misses'] += 1
        return None
    
    def update(self, src_node: int, dst_node: int, 
               network_state: Dict[str, Any], decision: Tuple):
        """
        更新缓存
        
        Args:
            src_node: 源节点ID
            dst_node: 目的节点ID
            network_state: 网络状态
            decision: 调度决策 (target_node, path, q_value)
        """
        # 生成缓存键
        cache_key = self._generate_key(src_node, dst_node, network_state)
        
        # 创建缓存条目
        cache_entry = {
            'decision': decision,
            'network_state_hash': self._hash_network_state(network_state),
            'timestamp': self._get_timestamp()
        }
        
        # 更新缓存
        self.cache[cache_key] = cache_entry
        self.cache.move_to_end(cache_key)
        
        # 如果超过最大大小，移除最旧的条目
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
        
        self.stats['updates'] += 1
    
    def _generate_key(self, src_node: int, dst_node: int, 
                     network_state: Dict[str, Any]) -> str:
        """
        生成缓存键
        
        Args:
            src_node: 源节点ID
            dst_node: 目的节点ID
            network_state: 网络状态
            
        Returns:
            缓存键字符串
        """
        # 使用源-目的节点对和网络状态哈希作为键
        state_hash = self._hash_network_state(network_state)
        return f"{src_node}-{dst_node}-{state_hash}"
    
    def _hash_network_state(self, network_state: Dict[str, Any]) -> str:
        """
        计算网络状态的哈希值
        
        Args:
            network_state: 网络状态
            
        Returns:
            哈希字符串
        """
        # 提取关键状态信息进行哈希
        hash_data = []
        
        # 节点负载信息
        nodes = network_state.get('nodes', [])
        for node in nodes:
            if node.get('is_compute_node', False):
                hash_data.append(f"n{node['id']}:{node.get('current_load', 0.0):.3f}")
        
        # 链路利用率信息
        links = network_state.get('links', [])
        for link in links:
            if 'utilization' in link:
                hash_data.append(f"l{link['src']}-{link['dst']}:{link['utilization']:.3f}")
        
        # 排序以确保一致性
        hash_data.sort()
        
        # 计算MD5哈希
        hash_str = "|".join(hash_data)
        return hashlib.md5(hash_str.encode()).hexdigest()[:8]
    
    def _validate_entry(self, cache_entry: Dict[str, Any], 
                       current_state: Dict[str, Any]) -> bool:
        """
        验证缓存条目是否仍然有效
        
        Args:
            cache_entry: 缓存条目
            current_state: 当前网络状态
            
        Returns:
            是否有效
        """
        # 计算当前状态哈希
        current_hash = self._hash_network_state(current_state)
        cached_hash = cache_entry['network_state_hash']
        
        # 如果哈希相同，直接有效
        if current_hash == cached_hash:
            return True
        
        # 否则，需要更详细的验证
        # 这里简化处理：如果网络状态变化小于阈值，则认为有效
        state_change = self._calculate_state_change(cache_entry, current_state)
        return state_change < self.validation_threshold
    
    def _calculate_state_change(self, cache_entry: Dict[str, Any], 
                               current_state: Dict[str, Any]) -> float:
        """
        计算网络状态变化
        
        Args:
            cache_entry: 缓存条目
            current_state: 当前网络状态
            
        Returns:
            状态变化量
        """
        # 这里简化实现：计算节点负载的平均变化
        # 在实际实现中，需要更精确的状态变化计算
        
        current_nodes = {node['id']: node for node in current_state.get('nodes', [])}
        
        # 从缓存条目中无法直接获取历史状态，这里返回一个保守值
        # 在实际系统中，缓存条目应包含状态快照
        return 0.5  # 保守估计，假设有较大变化
    
    def _get_timestamp(self) -> float:
        """
        获取当前时间戳
        
        Returns:
            时间戳
        """
        import time
        return time.time()
    
    def get_stats(self) -> Dict[str, int]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        stats = self.stats.copy()
        stats['size'] = len(self.cache)
        stats['hit_rate'] = (
            stats['hits'] / max(stats['hits'] + stats['misses'], 1)
        )
        return stats
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'updates': 0,
            'invalidations': 0
        }
    
    def warmup(self, historical_data: list):
        """
        缓存预热
        
        Args:
            historical_data: 历史数据列表，每个元素为(src, dst, state, decision)
        """
        for src, dst, state, decision in historical_data:
            self.update(src, dst, state, decision)