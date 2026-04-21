#!/usr/bin/env python3
"""
简化实验 - 测试基本功能而不需要PyTorch
"""

import sys
import os
import json
import time
import random
from typing import Dict, Any, List

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("=" * 60)
print("简化实验 - 测试GCN-DQN算法框架的基本功能")
print("=" * 60)

# 创建模拟的网络状态
def create_mock_network_state(num_nodes=10):
    """创建模拟网络状态"""
    network_state = {
        "topology": {
            "nodes": list(range(num_nodes)),
            "edges": [(i, (i+1) % num_nodes) for i in range(num_nodes)] + 
                     [(i, (i+2) % num_nodes) for i in range(num_nodes)],
        },
        "link_states": {
            f"{i}-{j}": {
                "bandwidth": random.randint(10, 100),
                "delay": random.randint(1, 20),
                "utilization": random.random() * 0.8
            }
            for i in range(num_nodes)
            for j in range(num_nodes)
            if i != j and random.random() < 0.3
        },
        "node_resources": {
            i: {
                "compute_capacity": random.randint(10, 100),
                "current_load": random.random() * 0.7,
                "energy_coefficient": random.random() * 0.5 + 0.5
            }
            for i in range(num_nodes)
        }
    }
    return network_state

# 创建模拟的业务请求
def create_mock_request(network_state):
    """创建模拟业务请求"""
    nodes = list(network_state["topology"]["nodes"])
    request_types = ["edge_ai", "compute_scheduling"]
    
    return {
        "request_id": f"req_{int(time.time() * 1000)}",
        "src_node": random.choice(nodes),
        "dst_node": random.choice([n for n in nodes if n != random.choice(nodes)]),
        "request_type": random.choice(request_types),
        "compute_requirement": random.randint(5, 50),
        "delay_tolerance": random.randint(20, 100),
        "data_size": random.randint(1, 100),
        "priority": random.choice(["high", "medium", "low"])
    }

# 模拟调度算法
class MockScheduler:
    """模拟调度器 - 用于测试"""
    
    def __init__(self, config):
        self.config = config
        self.cache = {}
        self.stats = {
            "total_requests": 0,
            "successful_decisions": 0,
            "cache_hits": 0,
            "average_decision_time": 0
        }
    
    def schedule(self, request, network_state):
        """模拟调度决策"""
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        # 简单的决策逻辑
        nodes = list(network_state["topology"]["nodes"])
        target_node = random.choice(nodes)
        
        # 构建简单路径
        src = request["src_node"]
        dst = request["dst_node"]
        
        # 简单路径构建（随机选择）
        path = [src]
        current = src
        max_hops = 5
        
        while current != dst and len(path) < max_hops:
            # 随机选择下一个节点
            possible_next = [n for n in nodes if n != current]
            if not possible_next:
                break
            next_node = random.choice(possible_next)
            path.append(next_node)
            current = next_node
        
        decision = {
            "target_node": target_node,
            "path": path,
            "q_value": random.random(),
            "decision_time": time.time() - start_time,
            "algorithm": "mock_scheduler"
        }
        
        self.stats["successful_decisions"] += 1
        return decision
    
    def get_stats(self):
        """获取统计信息"""
        if self.stats["total_requests"] > 0:
            success_rate = self.stats["successful_decisions"] / self.stats["total_requests"]
        else:
            success_rate = 0
            
        return {
            **self.stats,
            "success_rate": success_rate
        }

# 运行简化实验
def run_simple_experiment():
    """运行简化实验"""
    print("\n1. 创建模拟配置...")
    config = {
        "network": {
            "min_nodes": 10,
            "max_nodes": 50
        },
        "experiment": {
            "num_runs": 3,
            "max_steps": 20
        }
    }
    
    print("2. 创建模拟调度器...")
    scheduler = MockScheduler(config)
    
    print("3. 生成模拟网络状态...")
    network_state = create_mock_network_state(num_nodes=15)
    print(f"   节点数: {len(network_state['topology']['nodes'])}")
    print(f"   边数: {len(network_state['topology']['edges'])}")
    print(f"   链路状态数: {len(network_state['link_states'])}")
    
    print("4. 运行模拟调度...")
    results = []
    
    for step in range(config["experiment"]["max_steps"]):
        # 生成业务请求
        request = create_mock_request(network_state)
        
        # 调度决策
        decision = scheduler.schedule(request, network_state)
        
        # 记录结果
        result = {
            "step": step,
            "request_id": request["request_id"],
            "request_type": request["request_type"],
            "decision_time": decision["decision_time"],
            "path_length": len(decision["path"]),
            "success": decision["path"][-1] == request["dst_node"] if decision["path"] else False
        }
        results.append(result)
        
        if step % 5 == 0:
            print(f"   步骤 {step}: 请求 {request['request_id']}, 决策时间 {decision['decision_time']:.4f}s")
    
    print("5. 分析结果...")
    
    # 计算统计信息
    total_steps = len(results)
    successful_decisions = sum(1 for r in results if r["success"])
    avg_decision_time = sum(r["decision_time"] for r in results) / total_steps if total_steps > 0 else 0
    avg_path_length = sum(r["path_length"] for r in results) / total_steps if total_steps > 0 else 0
    
    print(f"\n实验结果统计:")
    print(f"   总步数: {total_steps}")
    print(f"   成功决策数: {successful_decisions}")
    print(f"   成功率: {successful_decisions/total_steps:.2%}")
    print(f"   平均决策时间: {avg_decision_time:.4f}秒")
    print(f"   平均路径长度: {avg_path_length:.2f}跳")
    
    # 按请求类型分析
    request_types = {}
    for r in results:
        req_type = r["request_type"]
        if req_type not in request_types:
            request_types[req_type] = {"count": 0, "success": 0}
        request_types[req_type]["count"] += 1
        if r["success"]:
            request_types[req_type]["success"] += 1
    
    print(f"\n按请求类型分析:")
    for req_type, stats in request_types.items():
        success_rate = stats["success"] / stats["count"] if stats["count"] > 0 else 0
        print(f"   {req_type}: {stats['count']}次请求, 成功率: {success_rate:.2%}")
    
    # 调度器统计
    scheduler_stats = scheduler.get_stats()
    print(f"\n调度器统计:")
    for key, value in scheduler_stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    # 保存结果
    output_dir = "results/simple_experiment"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "simple_experiment_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": config,
            "results": results,
            "statistics": {
                "total_steps": total_steps,
                "successful_decisions": successful_decisions,
                "success_rate": successful_decisions/total_steps,
                "avg_decision_time": avg_decision_time,
                "avg_path_length": avg_path_length,
                "request_types": request_types
            },
            "scheduler_stats": scheduler_stats,
            "timestamp": time.time()
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n6. 结果已保存到: {output_file}")
    print("\n简化实验完成！")
    
    return {
        "success_rate": successful_decisions/total_steps,
        "avg_decision_time": avg_decision_time,
        "output_file": output_file
    }

if __name__ == "__main__":
    try:
        results = run_simple_experiment()
        print(f"\n实验总结:")
        print(f"  成功率: {results['success_rate']:.2%}")
        print(f"  平均决策时间: {results['avg_decision_time']:.4f}秒")
        print(f"  结果文件: {results['output_file']}")
    except Exception as e:
        print(f"实验运行出错: {e}")
        import traceback
        traceback.print_exc()