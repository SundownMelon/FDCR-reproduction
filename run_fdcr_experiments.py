#!/usr/bin/env python
"""
FDCR复现实验启动脚本 (单GPU版本)

实验配置：
- 攻击模式: base_backdoor (集中触发), dba_backdoor (分布式触发)
- Non-IID强度: α=0.9 (接近IID), α=0.1 (高度异构)
- 数据集: CIFAR-10
- 随机种子: 0 (单次运行)

使用方法:
  # 运行所有4种配置
  python run_fdcr_experiments.py

  # 快速测试 (只运行一个配置)
  python run_fdcr_experiments.py --quick-test

  # 预览命令但不执行
  python run_fdcr_experiments.py --dry-run

  # 从指定配置继续运行 (如果之前中断)
  python run_fdcr_experiments.py --start-from 2
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime


# 实验配置 (4种组合)
EXPERIMENT_CONFIGS = [
    # (编号, 攻击模式, alpha值, 描述)
    (1, 'base_backdoor', 0.9, '集中触发 + 接近IID'),
    (2, 'base_backdoor', 0.1, '集中触发 + 高度异构'),
    (3, 'dba_backdoor', 0.9, 'DBA分布式触发 + 接近IID'),
    (4, 'dba_backdoor', 0.1, 'DBA分布式触发 + 高度异构'),
]

DEFAULT_SEED = 0
DEFAULT_DATASET = 'fl_cifar10'


def build_single_command(attack: str, alpha: float, seed: int, dataset: str) -> list:
    """构建单个实验的命令行参数"""
    csv_name = f"{attack}_alpha{alpha}_seed{seed}"
    
    cmd = [
        sys.executable, 'main.py',
        '--device_id', '0',  # 固定使用GPU 0
        '--task', 'label_skew',
        '--dataset', dataset,
        '--attack_type', 'backdoor',
        '--optim', 'FedFish',
        '--server', 'OurRandomControl',
        '--seed', str(seed),
        '--csv_log',
        '--csv_name', csv_name,
        # yacs 配置参数需要 key value 分开
        'DATASET.beta', str(alpha),
        'attack.backdoor.evils', attack,
    ]
    
    return cmd


def run_experiment(exp_id: int, attack: str, alpha: float, desc: str,
                   seed: int, dataset: str, dry_run: bool = False) -> bool:
    """运行单个实验"""
    cmd = build_single_command(attack, alpha, seed, dataset)
    
    print(f"\n{'='*60}")
    print(f"实验 {exp_id}/4: {desc}")
    print(f"{'='*60}")
    print(f"  攻击模式: {attack}")
    print(f"  Alpha: {alpha}")
    print(f"  种子: {seed}")
    print(f"  数据集: {dataset}")
    print(f"{'='*60}")
    print(f"命令: {' '.join(cmd)}")
    
    if dry_run:
        print("[预览模式] 跳过执行")
        return True
    
    start_time = datetime.now()
    print(f"开始时间: {start_time.strftime('%H:%M:%S')}")
    
    try:
        result = subprocess.run(cmd)
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"结束时间: {end_time.strftime('%H:%M:%S')} (耗时: {duration})")
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n用户中断实验")
        return False
    except Exception as e:
        print(f"实验失败: {e}")
        return False


def run_all_experiments(seed: int, dataset: str, start_from: int, dry_run: bool = False):
    """顺序运行所有实验配置"""
    
    print(f"\n{'#'*60}")
    print(f"FDCR 复现实验 (单GPU顺序执行)")
    print(f"{'#'*60}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总实验数: 4")
    print(f"随机种子: {seed}")
    print(f"数据集: {dataset}")
    if start_from > 1:
        print(f"从实验 {start_from} 开始")
    print(f"{'#'*60}\n")
    
    results = []
    
    for exp_id, attack, alpha, desc in EXPERIMENT_CONFIGS:
        if exp_id < start_from:
            print(f"[跳过] 实验 {exp_id}: {desc}")
            continue
        
        success = run_experiment(exp_id, attack, alpha, desc, seed, dataset, dry_run)
        results.append({
            'id': exp_id,
            'attack': attack,
            'alpha': alpha,
            'desc': desc,
            'success': success
        })
        
        if not success and not dry_run:
            print(f"\n实验 {exp_id} 失败，是否继续? (y/n): ", end='')
            try:
                choice = input().strip().lower()
                if choice != 'y':
                    print("用户选择停止")
                    break
            except:
                break
    
    # 打印汇总
    print(f"\n{'='*60}")
    print(f"实验完成汇总")
    print(f"{'='*60}")
    
    for r in results:
        status = "✓ 成功" if r['success'] else "✗ 失败"
        print(f"  实验 {r['id']}: {r['desc']} - {status}")
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


def run_quick_test(seed: int, dry_run: bool = False):
    """快速测试 - 只运行第一个配置"""
    print("\n[快速测试] 只运行: 集中触发 + 接近IID")
    return run_experiment(1, 'base_backdoor', 0.9, '集中触发 + 接近IID', 
                         seed, DEFAULT_DATASET, dry_run)


def main():
    parser = argparse.ArgumentParser(
        description='FDCR复现实验 (单GPU版本)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='随机种子 (默认: 0)')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET,
                        help='数据集 (默认: fl_cifar10)')
    parser.add_argument('--dry-run', action='store_true', 
                        help='预览命令但不执行')
    parser.add_argument('--quick-test', action='store_true', 
                        help='快速测试 (只运行一个配置)')
    parser.add_argument('--start-from', type=int, default=1, choices=[1,2,3,4],
                        help='从第几个实验开始 (1-4, 用于断点续跑)')
    
    args = parser.parse_args()
    
    if args.quick_test:
        run_quick_test(args.seed, args.dry_run)
    else:
        run_all_experiments(args.seed, args.dataset, args.start_from, args.dry_run)
    
    # 提示生成汇总表
    if not args.dry_run:
        print(f"\n{'='*60}")
        print("生成结果汇总表:")
        print("  python run_experiments.py --summary-only")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
