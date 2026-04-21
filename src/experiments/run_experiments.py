#!/usr/bin/env python3
"""
GCN-DQN算力感知调度算法实验运行脚本
统一入口，支持多种实验类型

使用方法：
python run_experiments.py --experiment comparative --network_size 30 --arrival_rate 30 --num_runs 5
"""

import argparse
import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# 调试：打印路径
if __name__ == "__main__":
    print(f"当前目录: {current_dir}")
    print(f"项目根目录: {project_root}")
    print(f"Python路径: {sys.path[:3]}")

try:
    from src.experiments.experiment_runner import ExperimentRunner
except ImportError as e:
    print(f"导入错误: {e}")
    print("尝试直接导入experiment_runner...")
    # 尝试直接导入
    import importlib.util
    spec = importlib.util.spec_from_file_location("experiment_runner",
                                                  os.path.join(current_dir, "experiment_runner.py"))
    experiment_runner = importlib.util.module_from_spec(spec)
    sys.modules["experiment_runner"] = experiment_runner
    spec.loader.exec_module(experiment_runner)
    ExperimentRunner = experiment_runner.ExperimentRunner


def run_comparative_experiment(args):
    """运行对比实验"""
    print("=" * 60)
    print("运行对比实验")
    print("=" * 60)
    
    runner = ExperimentRunner("config/config.yaml")
    
    results = runner.run_comparative_experiment(
        network_size=args.network_size,
        arrival_rate=args.arrival_rate,
        num_runs=args.num_runs
    )
    
    print(f"\n对比实验完成！")
    print(f"结果已保存到 results/ 目录")
    return results


def run_scalability_experiment(args):
    """运行可扩展性实验"""
    print("=" * 60)
    print("运行可扩展性实验")
    print("=" * 60)
    
    runner = ExperimentRunner("config/config.yaml")
    
    # 使用配置中的网络规模或参数
    network_sizes = args.network_sizes
    if not network_sizes:
        # 从配置中读取默认值
        import yaml
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        network_sizes = config.get('experiment', {}).get('network_sizes', [10, 30, 100])
    
    results = runner.run_scalability_experiment(
        network_sizes=network_sizes,
        arrival_rate=args.arrival_rate,
        num_runs=args.num_runs
    )
    
    print(f"\n可扩展性实验完成！")
    print(f"测试的网络规模: {network_sizes}")
    print(f"结果已保存到 results/ 目录")
    return results


def run_ablation_experiment(args):
    """运行消融实验"""
    print("=" * 60)
    print("运行消融实验")
    print("=" * 60)
    
    runner = ExperimentRunner("config/config.yaml")
    
    ablation_types = args.ablation_types
    if not ablation_types:
        ablation_types = ["no_cache", "no_weight_adjust", "no_gcn", "baseline"]
    
    results = runner.run_ablation_experiment(
        ablation_types=ablation_types,
        network_size=args.network_size,
        arrival_rate=args.arrival_rate,
        num_runs=args.num_runs
    )
    
    print(f"\n消融实验完成！")
    print(f"测试的消融类型: {ablation_types}")
    print(f"结果已保存到 results/ 目录")
    return results


def run_training_experiment(args):
    """运行训练实验（GCN-DQN训练）"""
    print("=" * 60)
    print("运行GCN-DQN训练实验")
    print("=" * 60)
    
    from src.models.scheduler import GCN_DQN_Scheduler
    from src.environment.simulation_env import SimulationEnvironment
    from src.utils.config_loader import load_config
    
    # 加载配置
    config = load_config("config/config.yaml")
    
    # 创建调度器和环境
    scheduler = GCN_DQN_Scheduler(config)
    env = SimulationEnvironment(config)
    
    # 设置环境
    env.setup(network_size=args.network_size, initial_requests=200)
    
    # 训练模式
    scheduler.train()
    
    print(f"开始训练GCN-DQN调度器...")
    print(f"网络规模: {args.network_size} 节点")
    print(f"训练轮数: {args.training_episodes}")
    print(f"每轮最大步数: {args.max_steps}")
    
    # 训练循环
    for episode in range(args.training_episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(args.max_steps):
            # 获取业务请求
            request = env.generate_request()
            
            # 调度决策
            decision = scheduler.schedule(request, env.get_network_state())
            
            # 执行决策并获取奖励
            reward, done = env.step(decision)
            total_reward += reward
            
            # 收集经验
            scheduler._collect_experience(request, decision, reward, done)
            
            if done:
                break
        
        # 每100轮进行一次训练
        if episode % 100 == 0:
            loss = scheduler.train_step(batch_size=32)
            if loss is not None:
                print(f"Episode {episode}: Total Reward = {total_reward:.2f}, Loss = {loss:.4f}")
            else:
                print(f"Episode {episode}: Total Reward = {total_reward:.2f}")
    
    # 保存模型
    model_path = f"results/models/gcn_dqn_model_{args.network_size}n.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    scheduler.save(model_path)
    
    print(f"\n训练完成！模型已保存到: {model_path}")
    return {"model_path": model_path, "episodes": args.training_episodes}


def main():
    parser = argparse.ArgumentParser(
        description="GCN-DQN算力感知调度算法实验运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行对比实验
  python run_experiments.py --experiment comparative --network_size 30 --arrival_rate 30 --num_runs 5
  
  # 运行可扩展性实验
  python run_experiments.py --experiment scalability --arrival_rate 30 --num_runs 3
  
  # 运行消融实验
  python run_experiments.py --experiment ablation --network_size 30 --arrival_rate 30 --num_runs 3
  
  # 运行训练实验
  python run_experiments.py --experiment training --network_size 30 --training_episodes 1000
        """
    )
    
    parser.add_argument("--experiment", type=str, default="comparative",
                       choices=["comparative", "scalability", "ablation", "training"],
                       help="实验类型: comparative(对比), scalability(可扩展性), ablation(消融), training(训练)")
    
    # 通用参数
    parser.add_argument("--network_size", type=int, default=30,
                       help="网络规模（节点数），默认30")
    parser.add_argument("--arrival_rate", type=float, default=30.0,
                       help="业务到达率（flows/s），默认30.0")
    parser.add_argument("--num_runs", type=int, default=5,
                       help="重复运行次数，默认5")
    
    # 可扩展性实验专用参数
    parser.add_argument("--network_sizes", type=int, nargs="+",
                       help="可扩展性实验的网络规模列表，如: 10 30 100")
    
    # 消融实验专用参数
    parser.add_argument("--ablation_types", type=str, nargs="+",
                       help="消融实验类型列表，如: no_cache no_weight_adjust no_gcn")
    
    # 训练实验专用参数
    parser.add_argument("--training_episodes", type=int, default=1000,
                       help="训练轮数，默认1000")
    parser.add_argument("--max_steps", type=int, default=100,
                       help="每轮最大步数，默认100")
    
    args = parser.parse_args()
    
    # 根据实验类型调用相应的函数
    if args.experiment == "comparative":
        results = run_comparative_experiment(args)
    elif args.experiment == "scalability":
        results = run_scalability_experiment(args)
    elif args.experiment == "ablation":
        results = run_ablation_experiment(args)
    elif args.experiment == "training":
        results = run_training_experiment(args)
    else:
        print(f"未知的实验类型: {args.experiment}")
        return 1
    
    print(f"\n实验 '{args.experiment}' 执行完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())