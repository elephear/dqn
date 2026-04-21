#!/usr/bin/env python3
"""
测试GPU训练功能
"""

import torch
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_gpu_availability():
    """测试GPU可用性"""
    print("=" * 50)
    print("GPU可用性测试")
    print("=" * 50)
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  内存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("警告: CUDA不可用，将使用CPU")

def test_model_gpu():
    """测试模型在GPU上的运行"""
    print("\n" + "=" * 50)
    print("模型GPU测试")
    print("=" * 50)
    
    try:
        from src.utils.config_loader import load_config
        from src.models.scheduler import GCN_DQN_Scheduler
        
        # 加载配置
        config = load_config()
        
        # 创建调度器
        print("创建GCN-DQN调度器...")
        scheduler = GCN_DQN_Scheduler(config)
        
        print(f"调度器设备: {scheduler.device}")
        print(f"GCN提取器设备: {next(scheduler.gcn_extractor.parameters()).device}")
        print(f"DQN网络设备: {next(scheduler.dqn_agent.q_network.parameters()).device}")
        
        # 测试简单的张量运算
        print("\n测试张量运算...")
        if torch.cuda.is_available():
            # 创建测试张量
            x = torch.randn(10, 10).cuda()
            y = torch.randn(10, 10).cuda()
            z = torch.matmul(x, y)
            print(f"GPU张量运算成功: {z.shape}")
            print(f"张量设备: {z.device}")
        else:
            print("GPU不可用，跳过张量运算测试")
            
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_gpu():
    """测试训练功能"""
    print("\n" + "=" * 50)
    print("训练功能测试")
    print("=" * 50)
    
    try:
        # 创建一个简单的神经网络在GPU上训练
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            
            # 创建简单的神经网络
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 50),
                torch.nn.ReLU(),
                torch.nn.Linear(50, 1)
            ).to(device)
            
            # 创建优化器
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # 创建测试数据
            x = torch.randn(100, 10).to(device)
            y = torch.randn(100, 1).to(device)
            
            # 训练几步
            model.train()
            for i in range(5):
                optimizer.zero_grad()
                output = model(x)
                loss = torch.nn.functional.mse_loss(output, y)
                loss.backward()
                optimizer.step()
                
                print(f"训练步 {i+1}: 损失 = {loss.item():.6f}")
            
            print("GPU训练测试成功!")
            return True
        else:
            print("GPU不可用，跳过训练测试")
            return True
            
    except Exception as e:
        print(f"训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("开始GPU训练环境测试")
    print("=" * 50)
    
    # 测试1: GPU可用性
    test_gpu_availability()
    
    # 测试2: 模型GPU支持
    model_ok = test_model_gpu()
    
    # 测试3: 训练功能
    training_ok = test_training_gpu()
    
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    
    if model_ok and training_ok:
        print("✅ 所有测试通过!")
        if torch.cuda.is_available():
            print("✅ GPU训练环境配置成功!")
        else:
            print("⚠️  GPU不可用，但CPU环境正常")
    else:
        print("❌ 测试失败，请检查配置")
    
    return model_ok and training_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)