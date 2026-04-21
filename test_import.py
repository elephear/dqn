#!/usr/bin/env python3
"""
测试导入和基本功能
"""

import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("测试导入...")

# 测试基本导入
try:
    import numpy as np
    print(f"✓ NumPy版本: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy导入失败: {e}")

try:
    import pandas as pd
    print(f"✓ Pandas版本: {pd.__version__}")
except ImportError as e:
    print(f"✗ Pandas导入失败: {e}")

try:
    import torch
    print(f"✓ PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"✗ PyTorch导入失败: {e}")

try:
    import yaml
    print(f"✓ PyYAML可用")
except ImportError as e:
    print(f"✗ PyYAML导入失败: {e}")

# 测试配置文件
config_path = "config/config.yaml"
if os.path.exists(config_path):
    print(f"✓ 配置文件存在: {config_path}")
    try:
        with open(config_path, 'r') as f:
            content = f.read(100)
            print(f"  配置文件预览: {content[:50]}...")
    except Exception as e:
        print(f"✗ 读取配置文件失败: {e}")
else:
    print(f"✗ 配置文件不存在: {config_path}")

# 测试src目录结构
src_dir = "src"
if os.path.exists(src_dir):
    print(f"✓ src目录存在")
    # 列出关键文件
    key_files = [
        "models/scheduler.py",
        "models/gcn.py", 
        "models/dqn.py",
        "experiments/experiment_runner.py",
        "algorithms/weight_adjuster.py",
        "algorithms/path_cache.py"
    ]
    
    for file in key_files:
        full_path = os.path.join(src_dir, file)
        if os.path.exists(full_path):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (缺失)")
else:
    print(f"✗ src目录不存在")

print("\n导入测试完成！")