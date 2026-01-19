#!/usr/bin/env python3
"""
无攻击场景实验：测试去作弊版FDCR在0%恶意客户端下的FPR和过滤比例
目的：验证FINCH+Fisher在异构/Non-IID下是否会产生高误报
"""

import subprocess
import time
from datetime import datetime

def run_experiment(alpha, server_type, seed=0):
    """运行单个实验"""
    csv_name = f"no_attack_alpha{alpha}_{server_type}_seed{seed}"
    
    cmd = [
        "python", "main.py",
        "--device_id", "0",
        "--task", "label_skew",
        "--dataset", "fl_cifar10",
        "--attack_type", "backdoor",  # 使用backdoor类型但设置恶意率为0
        "--optim", "FedFish",
        "--server", server_type,
        "--seed", str(seed),
        "--csv_log",
        "--csv_name", csv_name,
        "DATASET.beta", str(alpha),
        "attack.bad_client_rate", "0.0",  # 0%恶意客户端
        "attack.backdoor.evils", "base_backdoor"  # 使用base_backdoor但不会有恶意客户端
    ]
    
    print(f"\n{'='*60}")
    print(f"运行实验: {csv_name}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start_time
    
    print(f"\n实验完成，耗时: {elapsed/60:.1f} 分钟")
    print(f"返回码: {result.returncode}\n")
    
    return result.returncode == 0

def main():
    print("="*70)
    print("FDCR 无攻击场景实验 (0% 恶意客户端)")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 实验配置
    experiments = [
        # (alpha, server_type, description)
        (0.9, "OurRandomControlNoCheat", "α=0.9 (接近IID) - 去作弊版"),
        (0.1, "OurRandomControlNoCheat", "α=0.1 (高度异构) - 去作弊版"),
    ]
    
    total = len(experiments)
    success_count = 0
    
    for idx, (alpha, server_type, desc) in enumerate(experiments, 1):
        print(f"\n[{idx}/{total}] 开始实验...")
        print(f"配置: {desc}")
        
        success = run_experiment(alpha, server_type, seed=0)
        if success:
            success_count += 1
            print(f"✓ 实验成功")
        else:
            print(f"✗ 实验失败")
    
    print("\n" + "="*70)
    print("所有实验完成！")
    print(f"成功: {success_count}/{total}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    print("\n下一步：运行分析脚本")
    print("python analyze_no_attack.py")

if __name__ == "__main__":
    main()
