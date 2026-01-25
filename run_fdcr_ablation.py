"""
FDCR 消融实验：对比 cos / nmse / unit 变体

验证"尺度不变是否足够"：
- cos: 1 - cos(gI, center)
- nmse: V_k / ||gI||²  
- unit: MSE(gI/||gI||, center/||center||)

使用方法:
    python run_fdcr_ablation.py --variant nmse --slice head
    python run_fdcr_ablation.py --variant unit --slice head
    python run_fdcr_ablation.py --all
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime


def run_single_experiment(variant: str, slice_mode: str, save_dir: str):
    """运行单个消融实验"""
    
    # 映射 variant 到 server name
    server_map = {
        'baseline': 'OurRandomControlNoCheat',
        'cos': 'CosineFDCR_Head' if slice_mode == 'head' else 'CosineFDCR_LastBlock',
        'nmse': 'NormalizedMSE_FDCR',
        'unit': 'UnitNormMSE_FDCR',
    }
    
    server_name = server_map.get(variant)
    if server_name is None:
        print(f"未知变体: {variant}")
        return None
    
    print(f"\n{'='*60}")
    print(f"消融实验: {variant} ({server_name})")
    print(f"Slice: {slice_mode}")
    print(f"{'='*60}")
    
    # 构建命令
    cmd = [
        'python', 'main.py',
        '--server', server_name,
        '--attack_type', 'backdoor',
        '--dataset', 'fl_cifar10',
        '--optim', 'FedFish',
        'DATASET.communication_epoch', '100',
        'DATASET.beta', '0.9',
        'attack.bad_client_rate', '0.3',
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"实验失败")
        return None
    
    return {'variant': variant, 'server': server_name, 'status': 'completed'}


def analyze_all_results(save_dir: str):
    """分析并汇总所有变体结果"""
    import pandas as pd
    
    base_path = r"data\label_skew\base_backdoor\0.3\fl_cifar10\0.9"
    
    variants = {
        'cos': 'CosineFDCR_Head',
        'nmse': 'NormalizedMSE_FDCR',
        'unit': 'UnitNormMSE_FDCR',
        'baseline': 'OurRandomControlNoCheat',
    }
    
    results = []
    
    for variant_name, server_name in variants.items():
        exp_dir = os.path.join(base_path, server_name, 'FedFish', 'para1')
        summary_file = os.path.join(exp_dir, 'experiment_summary.csv')
        
        if os.path.exists(summary_file):
            df = pd.read_csv(summary_file)
            metrics = dict(zip(df['metric'], df['value']))
            
            # 计算 TPR/FPR（从 aggregation_weight）
            # ...（可以复用 analyze_cosine_fdcr_results.py 的逻辑）
            
            results.append({
                'variant': variant_name,
                'server': server_name,
                'acc': metrics.get('steady_state_acc', 0),
                'asr': metrics.get('steady_state_asr', 0),
                'detection_acc': metrics.get('detection_accuracy', 0),
            })
    
    if results:
        summary_df = pd.DataFrame(results)
        summary_file = os.path.join(save_dir, 'ablation_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"\n汇总结果已保存: {summary_file}")
        print(summary_df.to_string(index=False))
    
    return results


def main():
    parser = argparse.ArgumentParser(description='FDCR 尺度不变性消融实验')
    parser.add_argument('--variant', type=str, default='nmse',
                        choices=['baseline', 'cos', 'nmse', 'unit'],
                        help='实验变体')
    parser.add_argument('--slice', type=str, default='head',
                        choices=['head', 'last_block'],
                        help='slice 模式')
    parser.add_argument('--all', action='store_true',
                        help='运行所有消融变体 (nmse, unit)')
    parser.add_argument('--analyze', action='store_true',
                        help='仅分析已有结果')
    parser.add_argument('--save_dir', type=str, default='outputs/ablation',
                        help='输出目录')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 60)
    print("FDCR 尺度不变性消融实验")
    print("=" * 60)
    
    if args.analyze:
        analyze_all_results(args.save_dir)
    elif args.all:
        for variant in ['nmse', 'unit']:
            run_single_experiment(variant, args.slice, args.save_dir)
        analyze_all_results(args.save_dir)
    else:
        run_single_experiment(args.variant, args.slice, args.save_dir)


if __name__ == '__main__':
    main()
