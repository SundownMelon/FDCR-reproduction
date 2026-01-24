"""
A2 实验脚本：分层 V_k FDCR 对比实验

运行三个变体并对比 TPR/FPR/ACC/ASR:
- Baseline: 原 FDCR (global V_k)
- Variant-1: V_{k,last_block}
- Variant-2: V_{k,head+last}

使用方法:
    python run_A2_experiment.py [--rounds N] [--beta BETA]
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from typing import Dict, List


def run_single_experiment(server_name: str, rounds: int = 100, beta: float = 0.9,
                          attack: str = 'base_backdoor') -> Dict:
    """运行单个实验变体"""
    os.environ['FDCR_STEP1_LOGGING'] = '0'  # 不记录详细日志
    
    from main import main, parse_args
    
    sys.argv = [
        'run_A2_experiment.py',
        '--server', server_name,
        '--attack_type', 'backdoor',
        '--dataset', 'fl_cifar10',
        '--optim', 'FedFish',
        f'DATASET.communication_epoch', str(rounds),
        f'DATASET.beta', str(beta),
        f'attack.backdoor.evils', attack,
        f'attack.bad_client_rate', '0.3',
    ]
    
    print(f"\n{'='*60}")
    print(f"运行实验: {server_name}")
    print(f"{'='*60}")
    
    try:
        main()
        
        # 获取结果 (从 fed_server.detection_results)
        # 这里需要修改 main.py 返回结果，暂时返回空
        return {'server': server_name, 'status': 'completed'}
        
    except Exception as e:
        print(f"实验失败: {e}")
        return {'server': server_name, 'status': 'failed', 'error': str(e)}


def analyze_detection_results(results_path: str) -> Dict:
    """分析检测结果"""
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # 计算指标
    total_rounds = len(results)
    
    tpr_list = []  # True Positive Rate (correctly identified malicious)
    fpr_list = []  # False Positive Rate (benign incorrectly flagged)
    
    for r in results:
        actual_malicious = set(r['actual_malicious_idx'])
        predicted_malicious = set(r['evil_idx'])
        n_clients = len(r['benign_idx']) + len(r['evil_idx'])
        actual_benign = set(range(n_clients)) - actual_malicious
        
        # True Positives
        tp = len(predicted_malicious & actual_malicious)
        # False Positives
        fp = len(predicted_malicious & actual_benign)
        # True Negatives
        tn = len(actual_benign - predicted_malicious)
        # False Negatives
        fn = len(actual_malicious - predicted_malicious)
        
        tpr = tp / len(actual_malicious) if len(actual_malicious) > 0 else 0
        fpr = fp / len(actual_benign) if len(actual_benign) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    return {
        'mean_tpr': np.mean(tpr_list),
        'std_tpr': np.std(tpr_list),
        'mean_fpr': np.mean(fpr_list),
        'std_fpr': np.std(fpr_list),
        'total_rounds': total_rounds
    }


def main_cli():
    import argparse
    
    parser = argparse.ArgumentParser(description='A2: 分层 V_k FDCR 对比实验')
    parser.add_argument('--rounds', type=int, default=100, help='通信轮数')
    parser.add_argument('--beta', type=float, default=0.9, help='Non-IID 程度')
    parser.add_argument('--attack', type=str, default='base_backdoor', help='攻击类型')
    parser.add_argument('--variants', nargs='+', 
                        default=['OurRandomControlNoCheat', 'LayerwiseFDCR_Last', 'LayerwiseFDCR_HeadLast'],
                        help='要测试的服务器变体')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("A2: 分层 V_k FDCR 对比实验")
    print("=" * 60)
    print(f"通信轮数: {args.rounds}")
    print(f"Non-IID (β): {args.beta}")
    print(f"攻击类型: {args.attack}")
    print(f"测试变体: {args.variants}")
    print()
    
    results = {}
    
    for variant in args.variants:
        result = run_single_experiment(
            server_name=variant,
            rounds=args.rounds,
            beta=args.beta,
            attack=args.attack
        )
        results[variant] = result
    
    # 保存汇总结果
    output_dir = 'logs/A2_results'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'A2_comparison_{timestamp}.json')
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存: {output_path}")


if __name__ == '__main__':
    main_cli()
