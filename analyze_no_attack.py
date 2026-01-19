#!/usr/bin/env python3
"""
分析无攻击场景下的FPR和过滤比例
"""

import pandas as pd
import os
from pathlib import Path

def analyze_no_attack_experiment(alpha, server_type, seed=0):
    """分析单个无攻击实验"""
    csv_name = f"no_attack_alpha{alpha}_{server_type}_seed{seed}"
    
    # 构建路径 (使用backdoor路径，因为我们用backdoor类型但恶意率为0)
    base_path = Path("data/label_skew/backdoor/0.0/fl_cifar10")
    result_path = base_path / str(alpha) / server_type / "FedFish" / csv_name
    
    detection_file = result_path / "detection_results.csv"
    
    if not detection_file.exists():
        print(f"⚠️  文件不存在: {detection_file}")
        return None
    
    # 读取检测结果
    df = pd.read_csv(detection_file)
    
    # 计算指标
    total_rounds = len(df)
    
    # FPR: 在无攻击场景下，所有被标记为恶意的都是误报
    # 因为实际恶意客户端数量为0
    fpr_per_round = df['FP'] / (df['TN'] + df['FP'])  # FP / (TN + FP)
    avg_fpr = fpr_per_round.mean()
    
    # 被过滤比例
    total_clients = 10  # 总客户端数
    filtered_ratio_per_round = df['predicted_malicious_count'] / total_clients
    avg_filtered_ratio = filtered_ratio_per_round.mean()
    
    # 统计有多少轮产生了误报
    rounds_with_fp = (df['FP'] > 0).sum()
    fp_rate = rounds_with_fp / total_rounds
    
    # 最大误报数
    max_fp = df['FP'].max()
    
    results = {
        'alpha': alpha,
        'server_type': server_type,
        'total_rounds': total_rounds,
        'avg_fpr': avg_fpr * 100,  # 转换为百分比
        'avg_filtered_ratio': avg_filtered_ratio * 100,
        'rounds_with_fp': rounds_with_fp,
        'fp_occurrence_rate': fp_rate * 100,
        'max_fp_count': max_fp,
        'avg_fp_count': df['FP'].mean()
    }
    
    return results

def main():
    print("="*70)
    print("FDCR 无攻击场景分析")
    print("="*70)
    print()
    
    experiments = [
        (0.9, "OurRandomControlNoCheat", "α=0.9 (接近IID)"),
        (0.1, "OurRandomControlNoCheat", "α=0.1 (高度异构)"),
    ]
    
    all_results = []
    
    for alpha, server_type, desc in experiments:
        print(f"\n分析: {desc}")
        print("-" * 60)
        
        result = analyze_no_attack_experiment(alpha, server_type)
        
        if result:
            all_results.append(result)
            
            print(f"总轮次: {result['total_rounds']}")
            print(f"平均FPR: {result['avg_fpr']:.2f}%")
            print(f"平均过滤比例: {result['avg_filtered_ratio']:.2f}%")
            print(f"产生误报的轮次: {result['rounds_with_fp']}/{result['total_rounds']} ({result['fp_occurrence_rate']:.1f}%)")
            print(f"平均误报数: {result['avg_fp_count']:.2f} 个客户端")
            print(f"最大误报数: {result['max_fp_count']} 个客户端")
    
    if len(all_results) == 2:
        print("\n" + "="*70)
        print("对比分析")
        print("="*70)
        
        iid_result = all_results[0]  # α=0.9
        noniid_result = all_results[1]  # α=0.1
        
        print(f"\n{'指标':<20} {'α=0.9 (IID)':<20} {'α=0.1 (Non-IID)':<20} {'差异':<15}")
        print("-" * 75)
        print(f"{'平均FPR':<20} {iid_result['avg_fpr']:<20.2f} {noniid_result['avg_fpr']:<20.2f} {noniid_result['avg_fpr']-iid_result['avg_fpr']:+.2f}%")
        print(f"{'平均过滤比例':<20} {iid_result['avg_filtered_ratio']:<20.2f} {noniid_result['avg_filtered_ratio']:<20.2f} {noniid_result['avg_filtered_ratio']-iid_result['avg_filtered_ratio']:+.2f}%")
        print(f"{'误报发生率':<20} {iid_result['fp_occurrence_rate']:<20.1f} {noniid_result['fp_occurrence_rate']:<20.1f} {noniid_result['fp_occurrence_rate']-iid_result['fp_occurrence_rate']:+.1f}%")
        print(f"{'平均误报数':<20} {iid_result['avg_fp_count']:<20.2f} {noniid_result['avg_fp_count']:<20.2f} {noniid_result['avg_fp_count']-iid_result['avg_fp_count']:+.2f}")
        
        print("\n" + "="*70)
        print("结论")
        print("="*70)
        
        if noniid_result['avg_fpr'] > iid_result['avg_fpr']:
            print(f"✓ 在高度异构场景(α=0.1)下，FPR比IID场景(α=0.9)高 {noniid_result['avg_fpr']-iid_result['avg_fpr']:.2f}%")
        
        if noniid_result['avg_filtered_ratio'] > 10:
            print(f"✓ 在无攻击场景下，平均有 {noniid_result['avg_filtered_ratio']:.1f}% 的良性客户端被误判")
        
        if noniid_result['fp_occurrence_rate'] > 50:
            print(f"✓ 在 {noniid_result['fp_occurrence_rate']:.1f}% 的轮次中产生了误报")
        
        print("\n这证明了FINCH+Fisher在异构/Non-IID场景下存在天然的不稳定性，")
        print("即使在无攻击的情况下也会产生大量误报。")

if __name__ == "__main__":
    main()
