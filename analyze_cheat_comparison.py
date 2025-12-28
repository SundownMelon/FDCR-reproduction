"""
FDCR 作弊 vs 去作弊 对比分析脚本

分析指标:
- 检测指标: TPR, FPR, Precision, F1, 完美检测率
- 任务指标: ACC, ASR
"""

import os
import csv
import numpy as np
from collections import defaultdict

# 真实的恶意客户端索引 (ground truth)
ACTUAL_MALICIOUS = {7, 8, 9}
ACTUAL_BENIGN = {0, 1, 2, 3, 4, 5, 6}


def parse_indices(indices_str):
    """解析分号分隔的索引字符串"""
    if not indices_str or indices_str.strip() == '':
        return set()
    return set(int(x) for x in indices_str.split(';') if x.strip())


def analyze_detection_results(csv_path):
    """分析单个实验的检测结果"""
    results = {
        'total_rounds': 0,
        'perfect_detection': 0,
        'true_positives': 0,
        'false_positives': 0,
        'true_negatives': 0,
        'false_negatives': 0,
    }
    
    if not os.path.exists(csv_path):
        return None
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            detected_benign = parse_indices(row['benign_indices'])
            detected_malicious = parse_indices(row['malicious_indices'])
            
            results['total_rounds'] += 1
            
            tp = len(detected_malicious & ACTUAL_MALICIOUS)
            fp = len(detected_malicious & ACTUAL_BENIGN)
            tn = len(detected_benign & ACTUAL_BENIGN)
            fn = len(detected_benign & ACTUAL_MALICIOUS)
            
            results['true_positives'] += tp
            results['false_positives'] += fp
            results['true_negatives'] += tn
            results['false_negatives'] += fn
            
            if tp == 3 and fp == 0:
                results['perfect_detection'] += 1
    
    return results


def calculate_metrics(results):
    """计算TPR/FPR/Precision/F1"""
    if results is None:
        return None
        
    tp = results['true_positives']
    fp = results['false_positives']
    tn = results['true_negatives']
    fn = results['false_negatives']
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'TPR': tpr,
        'FPR': fpr,
        'Precision': precision,
        'F1': f1,
        'Perfect_Rate': results['perfect_detection'] / results['total_rounds'] if results['total_rounds'] > 0 else 0
    }


def analyze_experiment_summary(csv_path):
    """分析实验摘要获取ACC和ASR"""
    if not os.path.exists(csv_path):
        return None
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if rows:
            last_row = rows[-1]
            return {
                'ACC': float(last_row.get('test_acc', 0)) * 100,
                'ASR': float(last_row.get('backdoor_acc', 0)) * 100
            }
    return None


def find_experiment_paths(attack_type, alpha, server_type):
    """查找实验结果路径"""
    base_path = f"data/label_skew/{attack_type}/0.3/fl_cifar10/{alpha}/{server_type}/FedFish"
    
    # 查找实验目录
    if not os.path.exists(base_path):
        return None, None
    
    # 查找包含正确server_type的目录
    exp_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and server_type in d]
    if not exp_dirs:
        # 如果没找到，尝试所有目录
        exp_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not exp_dirs:
        return None, None
    
    exp_dir = os.path.join(base_path, exp_dirs[0])
    
    detection_path = os.path.join(exp_dir, "detection_results.csv")
    summary_path = os.path.join(exp_dir, "experiment_summary.csv")
    
    return detection_path, summary_path


def main():
    """主分析函数"""
    
    print("="*90)
    print("FDCR 作弊 vs 去作弊 对比分析")
    print("="*90)
    
    # 实验配置
    attack_types = ["base_backdoor", "dba_backdoor"]
    alphas = [0.9, 0.1]
    server_types = ["OurRandomControl", "OurRandomControlNoCheat"]
    
    # 收集所有结果
    all_results = []
    
    for attack_type in attack_types:
        for alpha in alphas:
            for server_type in server_types:
                detection_path, summary_path = find_experiment_paths(attack_type, alpha, server_type)
                
                detection_results = analyze_detection_results(detection_path) if detection_path else None
                detection_metrics = calculate_metrics(detection_results)
                task_metrics = analyze_experiment_summary(summary_path) if summary_path else None
                
                version = "作弊版" if server_type == "OurRandomControl" else "去作弊版"
                
                all_results.append({
                    'attack': attack_type.replace('_backdoor', ''),
                    'alpha': alpha,
                    'version': version,
                    'server': server_type,
                    'detection': detection_metrics,
                    'task': task_metrics
                })
    
    # 打印检测指标对比表
    print("\n" + "="*90)
    print("检测指标对比 (TPR/FPR/Precision/F1)")
    print("="*90)
    print(f"{'攻击类型':<12} {'α':<6} {'版本':<10} {'TPR':>8} {'FPR':>8} {'Precision':>10} {'F1':>8} {'完美率':>8}")
    print("-"*90)
    
    for r in all_results:
        if r['detection']:
            m = r['detection']
            print(f"{r['attack']:<12} {r['alpha']:<6} {r['version']:<10} "
                  f"{m['TPR']*100:>7.1f}% {m['FPR']*100:>7.1f}% {m['Precision']*100:>9.1f}% "
                  f"{m['F1']*100:>7.1f}% {m['Perfect_Rate']*100:>7.1f}%")
        else:
            print(f"{r['attack']:<12} {r['alpha']:<6} {r['version']:<10} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'N/A':>8} {'N/A':>8}")
    
    # 打印任务指标对比表
    print("\n" + "="*90)
    print("任务指标对比 (ACC/ASR)")
    print("="*90)
    print(f"{'攻击类型':<12} {'α':<6} {'版本':<10} {'ACC':>10} {'ASR':>10}")
    print("-"*90)
    
    for r in all_results:
        if r['task']:
            t = r['task']
            print(f"{r['attack']:<12} {r['alpha']:<6} {r['version']:<10} "
                  f"{t['ACC']:>9.2f}% {t['ASR']:>9.2f}%")
        else:
            print(f"{r['attack']:<12} {r['alpha']:<6} {r['version']:<10} {'N/A':>10} {'N/A':>10}")
    
    # 打印差异分析
    print("\n" + "="*90)
    print("作弊 vs 去作弊 差异分析")
    print("="*90)
    
    for attack_type in attack_types:
        for alpha in alphas:
            cheat = next((r for r in all_results if r['attack'] == attack_type.replace('_backdoor', '') 
                         and r['alpha'] == alpha and r['version'] == '作弊版'), None)
            no_cheat = next((r for r in all_results if r['attack'] == attack_type.replace('_backdoor', '') 
                            and r['alpha'] == alpha and r['version'] == '去作弊版'), None)
            
            if cheat and no_cheat and cheat['detection'] and no_cheat['detection']:
                print(f"\n{attack_type} + α={alpha}:")
                
                tpr_diff = cheat['detection']['TPR'] - no_cheat['detection']['TPR']
                fpr_diff = cheat['detection']['FPR'] - no_cheat['detection']['FPR']
                f1_diff = cheat['detection']['F1'] - no_cheat['detection']['F1']
                
                print(f"  TPR差异: {tpr_diff*100:+.1f}% (作弊: {cheat['detection']['TPR']*100:.1f}%, 去作弊: {no_cheat['detection']['TPR']*100:.1f}%)")
                print(f"  FPR差异: {fpr_diff*100:+.1f}% (作弊: {cheat['detection']['FPR']*100:.1f}%, 去作弊: {no_cheat['detection']['FPR']*100:.1f}%)")
                print(f"  F1差异:  {f1_diff*100:+.1f}% (作弊: {cheat['detection']['F1']*100:.1f}%, 去作弊: {no_cheat['detection']['F1']*100:.1f}%)")
                
                if cheat['task'] and no_cheat['task']:
                    acc_diff = cheat['task']['ACC'] - no_cheat['task']['ACC']
                    asr_diff = cheat['task']['ASR'] - no_cheat['task']['ASR']
                    print(f"  ACC差异: {acc_diff:+.2f}% (作弊: {cheat['task']['ACC']:.2f}%, 去作弊: {no_cheat['task']['ACC']:.2f}%)")
                    print(f"  ASR差异: {asr_diff:+.2f}% (作弊: {cheat['task']['ASR']:.2f}%, 去作弊: {no_cheat['task']['ASR']:.2f}%)")
    
    # 结论
    print("\n" + "="*90)
    print("结论")
    print("="*90)
    
    # 计算平均差异
    tpr_diffs = []
    f1_diffs = []
    
    for attack_type in attack_types:
        for alpha in alphas:
            cheat = next((r for r in all_results if r['attack'] == attack_type.replace('_backdoor', '') 
                         and r['alpha'] == alpha and r['version'] == '作弊版'), None)
            no_cheat = next((r for r in all_results if r['attack'] == attack_type.replace('_backdoor', '') 
                            and r['alpha'] == alpha and r['version'] == '去作弊版'), None)
            
            if cheat and no_cheat and cheat['detection'] and no_cheat['detection']:
                tpr_diffs.append(cheat['detection']['TPR'] - no_cheat['detection']['TPR'])
                f1_diffs.append(cheat['detection']['F1'] - no_cheat['detection']['F1'])
    
    if tpr_diffs:
        avg_tpr_diff = np.mean(tpr_diffs) * 100
        avg_f1_diff = np.mean(f1_diffs) * 100
        
        print(f"平均TPR差异: {avg_tpr_diff:+.1f}%")
        print(f"平均F1差异:  {avg_f1_diff:+.1f}%")
        
        if avg_tpr_diff > 10:
            print("\n⚠️  结论: FDCR的高检测性能主要来自'作弊'路径（使用先验知识替换恶意客户端的Fisher信息）")
            print("   去除作弊逻辑后，检测性能显著下降，说明方法本身的检测能力有限。")
        elif avg_tpr_diff < 5:
            print("\n✅ 结论: FDCR的检测性能主要来自方法本身，而非'作弊'路径。")
            print("   去除作弊逻辑后，检测性能基本保持，说明Fisher信息聚类确实有效。")
        else:
            print("\n⚡ 结论: FDCR的检测性能部分依赖'作弊'路径。")
            print("   去除作弊逻辑后有一定性能下降，但方法本身仍有一定检测能力。")
    else:
        print("⚠️  没有足够的实验数据进行分析，请先运行实验。")


if __name__ == "__main__":
    main()
