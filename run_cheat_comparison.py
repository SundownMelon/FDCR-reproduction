"""
FDCR 作弊 vs 去作弊 对比实验脚本

实验设计：
- 2×2×2 = 8个实验配置
- 攻击类型: base_backdoor, dba_backdoor
- 数据分布: α=0.9 (接近IID), α=0.1 (高度异构)
- 服务器: OurRandomControl (作弊版), OurRandomControlNoCheat (去作弊版)

评估指标:
- 检测指标: TPR, FPR, Precision, F1
- 任务指标: ACC, ASR
"""

import subprocess
import os
import sys
import time
import csv
from datetime import datetime


def run_experiment(attack_type, alpha, server_type, seed=0):
    """运行单个实验 - 实时显示输出"""
    
    # 构建实验名称
    exp_name = f"{attack_type}_alpha{alpha}_{server_type}_seed{seed}"
    
    # 构建命令
    cmd = [
        "python", "-u", "main.py",  # -u 禁用缓冲，实时输出
        "--device_id", "0",
        "--task", "label_skew",
        "--dataset", "fl_cifar10",
        "--attack_type", "backdoor",
        "--optim", "FedFish",
        "--server", server_type,
        "--seed", str(seed),
        "--csv_log",
        "--csv_name", exp_name,
        "DATASET.beta", str(alpha),
        f"attack.backdoor.evils", attack_type
    ]
    
    print(f"\n{'='*60}")
    print(f"运行实验: {exp_name}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    sys.stdout.flush()
    
    start_time = time.time()
    
    try:
        # 使用Popen实时显示输出
        process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True
        )
        
        process.wait()
        elapsed = time.time() - start_time
        
        if process.returncode == 0:
            print(f"\n✅ 实验完成: {exp_name} (耗时: {elapsed/60:.1f}分钟)")
            return True, elapsed
        else:
            print(f"\n❌ 实验失败: {exp_name} (返回码: {process.returncode})")
            return False, elapsed
            
    except Exception as e:
        print(f"\n❌ 实验异常: {exp_name}, 错误: {e}")
        return False, 0


def main():
    """运行所有对比实验"""
    
    print("="*70)
    print("FDCR 作弊 vs 去作弊 对比实验")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 实验配置
    attack_types = ["base_backdoor", "dba_backdoor"]
    alphas = [0.9, 0.1]
    server_types = ["OurRandomControl", "OurRandomControlNoCheat"]
    
    # 记录实验结果
    results = []
    
    total_experiments = len(attack_types) * len(alphas) * len(server_types)
    current = 0
    
    for attack_type in attack_types:
        for alpha in alphas:
            for server_type in server_types:
                current += 1
                print(f"\n[{current}/{total_experiments}] 开始实验...")
                
                success, elapsed = run_experiment(
                    attack_type=attack_type,
                    alpha=alpha,
                    server_type=server_type,
                    seed=0
                )
                
                results.append({
                    'attack_type': attack_type,
                    'alpha': alpha,
                    'server_type': server_type,
                    'success': success,
                    'elapsed_minutes': elapsed / 60
                })
    
    # 打印实验摘要
    print("\n" + "="*70)
    print("实验摘要")
    print("="*70)
    
    success_count = sum(1 for r in results if r['success'])
    print(f"成功: {success_count}/{total_experiments}")
    
    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"  {status} {r['attack_type']} + α={r['alpha']} + {r['server_type']}: {r['elapsed_minutes']:.1f}分钟")
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n运行分析脚本获取详细指标:")
    print("  python analyze_cheat_comparison.py")


if __name__ == "__main__":
    main()
