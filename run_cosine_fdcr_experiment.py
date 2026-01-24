"""
Cosine-FDCR 端到端对比实验

实验组：
1. Baseline FDCR (V_k MSE)
2. Cosine-FDCR (1 - cos)
3. Normalized-MSE FDCR (V_k / ||gI||²)

使用方法:
    python run_cosine_fdcr_experiment.py --variant cos --slice head
"""

import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime


def run_experiment(variant: str, slice_mode: str, rounds: int = 100, beta: float = 0.9):
    """运行单个实验变体"""
    
    # 映射 variant 到 server name
    server_map = {
        'baseline': 'OurRandomControlNoCheat',
        'cos': 'CosineFDCR_Head' if slice_mode == 'head' else 'CosineFDCR_LastBlock',
        'nmse': 'NormalizedMSE_FDCR',
        'unit': 'UnitNormMSE_FDCR',
    }
    
    server_name = server_map.get(variant, variant)
    
    print(f"\n{'='*60}")
    print(f"运行实验: {variant} ({server_name})")
    print(f"Slice: {slice_mode}, Rounds: {rounds}, Beta: {beta}")
    print(f"{'='*60}")
    
    # 构建命令行参数
    cmd = [
        'python', 'main.py',
        '--server', server_name,
        '--attack_type', 'backdoor',
        '--dataset', 'fl_cifar10',
        '--optim', 'FedFish',
        'DATASET.communication_epoch', str(rounds),
        'DATASET.beta', str(beta),
        'attack.bad_client_rate', '0.3',
    ]
    
    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"实验失败: {result.stderr}")
        return None
    
    print(result.stdout)
    return {'variant': variant, 'status': 'completed'}


def collect_metrics(results_dir: str) -> pd.DataFrame:
    """收集实验指标"""
    # TODO: 从日志文件中收集指标
    pass


def generate_summary_table(variants: list, output_dir: str):
    """生成对照表"""
    rows = []
    
    for variant in variants:
        # TODO: 收集每个变体的指标
        rows.append({
            'variant': variant,
            'mean_TPR_last50': 0.0,
            'mean_FPR_last50': 0.0,
            'final_ACC': 0.0,
            'final_ASR': 0.0,
            'mean_filtered_ratio': 0.0,
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'summary_table.csv'), index=False)
    print(f"对照表已保存: {os.path.join(output_dir, 'summary_table.csv')}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Cosine-FDCR 端到端对比实验')
    parser.add_argument('--variant', type=str, default='cos',
                        choices=['baseline', 'cos', 'nmse', 'unit'],
                        help='实验变体')
    parser.add_argument('--slice', type=str, default='head',
                        choices=['head', 'last_block'],
                        help='slice 模式')
    parser.add_argument('--rounds', type=int, default=100, help='通信轮数')
    parser.add_argument('--beta', type=float, default=0.9, help='Non-IID 程度')
    parser.add_argument('--all', action='store_true', help='运行所有变体')
    parser.add_argument('--save_dir', type=str, default='outputs/cosine_fdcr',
                        help='输出目录')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 60)
    print("Cosine-FDCR 端到端对比实验")
    print("=" * 60)
    
    if args.all:
        variants = ['baseline', 'cos', 'nmse']
        for variant in variants:
            run_experiment(variant, args.slice, args.rounds, args.beta)
        generate_summary_table(variants, args.save_dir)
    else:
        run_experiment(args.variant, args.slice, args.rounds, args.beta)


if __name__ == '__main__':
    main()
