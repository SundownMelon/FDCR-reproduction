"""
FDCR Detection Metrics Analysis Script
分析所有实验配置的TPR/FPR/Precision指标
"""

import os
import csv
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
        'perfect_detection': 0,  # TPR=100% and FPR=0%
        'true_positives': 0,     # 正确识别的恶意客户端数
        'false_positives': 0,   # 错误标记为恶意的良性客户端数
        'true_negatives': 0,    # 正确识别的良性客户端数
        'false_negatives': 0,   # 漏检的恶意客户端数
        'errors': []            # 记录错误的轮次
    }
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row['epoch'])
            detected_benign = parse_indices(row['benign_indices'])
            detected_malicious = parse_indices(row['malicious_indices'])
            
            results['total_rounds'] += 1
            
            # 计算TP/FP/TN/FN
            tp = len(detected_malicious & ACTUAL_MALICIOUS)  # 正确检测到的恶意
            fp = len(detected_malicious & ACTUAL_BENIGN)     # 误报（良性被标为恶意）
            tn = len(detected_benign & ACTUAL_BENIGN)        # 正确识别的良性
            fn = len(detected_benign & ACTUAL_MALICIOUS)     # 漏检（恶意被标为良性）
            
            results['true_positives'] += tp
            results['false_positives'] += fp
            results['true_negatives'] += tn
            results['false_negatives'] += fn
            
            # 检查是否完美检测
            if tp == 3 and fp == 0:
                results['perfect_detection'] += 1
            else:
                results['errors'].append({
                    'epoch': epoch,
                    'detected_malicious': detected_malicious,
                    'detected_benign': detected_benign,
                    'tp': tp, 'fp': fp, 'fn': fn
                })
    
    return results

def calculate_metrics(results):
    """计算TPR/FPR/Precision"""
    tp = results['true_positives']
    fp = results['false_positives']
    tn = results['true_negatives']
    fn = results['false_negatives']
    
    # TPR (True Positive Rate / Recall / Sensitivity)
    # = TP / (TP + FN) = 正确检测的恶意 / 所有实际恶意
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # FPR (False Positive Rate)
    # = FP / (FP + TN) = 误报的良性 / 所有实际良性
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Precision
    # = TP / (TP + FP) = 正确检测的恶意 / 所有被标记为恶意的
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # F1 Score
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'TPR': tpr,
        'FPR': fpr,
        'Precision': precision,
        'F1': f1,
        'Perfect_Detection_Rate': results['perfect_detection'] / results['total_rounds']
    }

def main():
    experiments = [
        {
            'name': 'base_backdoor + α=0.9 (接近IID)',
            'path': 'data/label_skew/base_backdoor/0.3/fl_cifar10/0.9/OurRandomControl/FedFish/base_backdoor_alpha0.9_seed0/detection_results.csv'
        },
        {
            'name': 'base_backdoor + α=0.1 (高度异构)',
            'path': 'data/label_skew/base_backdoor/0.3/fl_cifar10/0.1/OurRandomControl/FedFish/base_backdoor_alpha0.1_seed0/detection_results.csv'
        },
        {
            'name': 'dba_backdoor + α=0.9 (接近IID)',
            'path': 'data/label_skew/dba_backdoor/0.3/fl_cifar10/0.9/OurRandomControl/FedFish/dba_backdoor_alpha0.9_seed0/detection_results.csv'
        },
        {
            'name': 'dba_backdoor + α=0.1 (高度异构)',
            'path': 'data/label_skew/dba_backdoor/0.3/fl_cifar10/0.1/OurRandomControl/FedFish/dba_backdoor_alpha0.1_seed0/detection_results.csv'
        }
    ]
    
    print("=" * 80)
    print("FDCR Detection Metrics Analysis")
    print("真实恶意客户端: {7, 8, 9}  |  真实良性客户端: {0, 1, 2, 3, 4, 5, 6}")
    print("=" * 80)
    
    all_results = []
    
    for exp in experiments:
        if not os.path.exists(exp['path']):
            print(f"\n⚠️  {exp['name']}: 文件不存在")
            continue
            
        print(f"\n📊 {exp['name']}")
        print("-" * 60)
        
        results = analyze_detection_results(exp['path'])
        metrics = calculate_metrics(results)
        
        print(f"  总轮次: {results['total_rounds']}")
        print(f"  完美检测轮次: {results['perfect_detection']}/{results['total_rounds']} ({metrics['Perfect_Detection_Rate']*100:.1f}%)")
        print(f"  TPR (召回率): {metrics['TPR']*100:.2f}%")
        print(f"  FPR (误报率): {metrics['FPR']*100:.2f}%")
        print(f"  Precision (精确率): {metrics['Precision']*100:.2f}%")
        print(f"  F1 Score: {metrics['F1']*100:.2f}%")
        
        if results['errors']:
            print(f"\n  ⚠️  检测错误的轮次 ({len(results['errors'])}个):")
            for err in results['errors'][:5]:  # 只显示前5个
                print(f"    Epoch {err['epoch']}: 检测到恶意={err['detected_malicious']}, "
                      f"TP={err['tp']}, FP={err['fp']}, FN={err['fn']}")
            if len(results['errors']) > 5:
                print(f"    ... 还有 {len(results['errors'])-5} 个错误轮次")
        
        all_results.append({
            'name': exp['name'],
            'results': results,
            'metrics': metrics
        })
    
    # 汇总表格
    print("\n" + "=" * 80)
    print("📋 汇总表格")
    print("=" * 80)
    print(f"{'实验配置':<35} {'TPR':>8} {'FPR':>8} {'Precision':>10} {'F1':>8} {'完美率':>8}")
    print("-" * 80)
    
    for r in all_results:
        m = r['metrics']
        print(f"{r['name']:<35} {m['TPR']*100:>7.1f}% {m['FPR']*100:>7.1f}% {m['Precision']*100:>9.1f}% {m['F1']*100:>7.1f}% {m['Perfect_Detection_Rate']*100:>7.1f}%")
    
    print("\n" + "=" * 80)
    print("📝 结论分析")
    print("=" * 80)
    
    avg_tpr = sum(r['metrics']['TPR'] for r in all_results) / len(all_results)
    avg_fpr = sum(r['metrics']['FPR'] for r in all_results) / len(all_results)
    avg_precision = sum(r['metrics']['Precision'] for r in all_results) / len(all_results)
    
    print(f"  平均 TPR: {avg_tpr*100:.2f}%")
    print(f"  平均 FPR: {avg_fpr*100:.2f}%")
    print(f"  平均 Precision: {avg_precision*100:.2f}%")
    
    if avg_fpr < 0.05 and avg_tpr > 0.95:
        print("\n  ✅ FDCR 检测效果优秀:")
        print("     - 高TPR表明能有效识别恶意客户端")
        print("     - 低FPR表明不会过度过滤良性客户端")
        print("     - 高Precision表明被标记为恶意的客户端确实是恶意的")
    elif avg_fpr > 0.1:
        print("\n  ⚠️  警告: FPR较高，可能存在过度过滤问题")
    
    print()

if __name__ == '__main__':
    main()
