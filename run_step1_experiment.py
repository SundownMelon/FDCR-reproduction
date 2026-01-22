"""
Step 1 实验运行脚本：FDCR 管线显微镜

运行去作弊版 FDCR，记录每轮每个客户端的关键中间量，
用于后续 Step 2 的信号体检分析。

使用方法:
    python run_step1_experiment.py [--rounds N] [--beta BETA] [--attack TYPE]

输出目录: logs/step1/{run_id}/
"""

import os
import sys
import argparse
from datetime import datetime


def run_experiment(rounds: int = 100, beta: float = 0.9, attack: str = 'base_backdoor'):
    """
    运行 Step 1 实验。
    
    Args:
        rounds: 通信轮数
        beta: Non-IID 程度 (Dirichlet α)
        attack: 攻击类型 ('base_backdoor' 或 'dba_backdoor')
    """
    # 启用 Step 1 日志记录
    os.environ['FDCR_STEP1_LOGGING'] = '1'
    
    # 导入 main 函数
    from main import main, parse_args
    
    # 构建参数
    sys.argv = [
        'run_step1_experiment.py',
        '--server', 'OurRandomControlNoCheat',
        '--attack_type', 'backdoor',
        '--dataset', 'fl_cifar10',
        '--optim', 'FedFish',
        f'DATASET.communication_epoch', str(rounds),
        f'DATASET.beta', str(beta),
        f'attack.backdoor.evils', attack,
        f'attack.bad_client_rate', '0.3',
    ]
    
    print("=" * 60)
    print(f"Step 1 FDCR 管线显微镜实验")
    print("=" * 60)
    print(f"  通信轮数: {rounds}")
    print(f"  Non-IID (β): {beta}")
    print(f"  攻击类型: {attack}")
    print(f"  服务器: OurRandomControlNoCheat (去作弊版)")
    print(f"  日志输出: logs/step1/")
    print("=" * 60)
    
    # 运行实验
    main()


def main_cli():
    parser = argparse.ArgumentParser(
        description='Step 1: FDCR 管线显微镜实验'
    )
    parser.add_argument(
        '--rounds', type=int, default=100,
        help='通信轮数 (默认: 100)'
    )
    parser.add_argument(
        '--beta', type=float, default=0.9,
        help='Non-IID 程度 (Dirichlet α，默认: 0.9)'
    )
    parser.add_argument(
        '--attack', type=str, default='base_backdoor',
        choices=['base_backdoor', 'dba_backdoor'],
        help='攻击类型 (默认: base_backdoor)'
    )
    
    args = parser.parse_args()
    run_experiment(rounds=args.rounds, beta=args.beta, attack=args.attack)


if __name__ == '__main__':
    main_cli()
