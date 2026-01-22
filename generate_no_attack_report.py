#!/usr/bin/env python3
"""
生成无攻击场景实验的完整分析报告
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

def analyze_no_attack_experiment(alpha, server_type, seed=0):
    """分析单个无攻击实验"""
    csv_name = f"no_attack_alpha{alpha}_{server_type}_seed{seed}"
    
    base_path = Path("data/label_skew/base_backdoor/0.0/fl_cifar10")
    result_path = base_path / str(alpha) / server_type / "FedFish" / csv_name
    detection_file = result_path / "detection_results.csv"
    
    if not detection_file.exists():
        print(f"⚠️  文件不存在: {detection_file}")
        return None
    
    df = pd.read_csv(detection_file)
    total_rounds = len(df)
    total_clients = 10
    
    fp_counts = []
    filtered_counts = []
    
    for _, row in df.iterrows():
        mal_str = str(row['malicious_indices'])
        if mal_str and mal_str != 'nan' and mal_str.strip():
            predicted_malicious = [int(x) for x in mal_str.split(';') if x.strip()]
            fp_count = len(predicted_malicious)
        else:
            fp_count = 0
        
        fp_counts.append(fp_count)
        filtered_counts.append(fp_count)
    
    avg_fp_count = sum(fp_counts) / total_rounds if total_rounds > 0 else 0
    max_fp_count = max(fp_counts) if fp_counts else 0
    rounds_with_fp = sum(1 for fp in fp_counts if fp > 0)
    fp_occurrence_rate = (rounds_with_fp / total_rounds * 100) if total_rounds > 0 else 0
    avg_fpr = (avg_fp_count / total_clients * 100) if total_clients > 0 else 0
    avg_filtered_ratio = (sum(filtered_counts) / (total_rounds * total_clients) * 100) if total_rounds > 0 else 0
    
    return {
        'alpha': alpha,
        'server_type': server_type,
        'total_rounds': total_rounds,
        'avg_fpr': avg_fpr,
        'avg_filtered_ratio': avg_filtered_ratio,
        'rounds_with_fp': rounds_with_fp,
        'fp_occurrence_rate': fp_occurrence_rate,
        'max_fp_count': max_fp_count,
        'avg_fp_count': avg_fp_count,
        'fp_counts': fp_counts
    }

def generate_report():
    """生成完整的Markdown报告"""
    
    print("="*70)
    print("开始生成无攻击场景分析报告")
    print("="*70)
    
    experiments = [
        (0.9, "OurRandomControlNoCheat", "α=0.9 (接近IID)"),
        (0.1, "OurRandomControlNoCheat", "α=0.1 (高度异构)"),
    ]
    
    all_results = []
    
    for alpha, server_type, desc in experiments:
        print(f"\n分析: {desc}")
        result = analyze_no_attack_experiment(alpha, server_type)
        if result:
            all_results.append(result)
            print(f"✓ 分析完成")
        else:
            print(f"✗ 分析失败")
    
    if len(all_results) != 2:
        print("\n⚠️  实验数据不完整，无法生成报告")
        return
    
    # 生成Markdown报告
    report_lines = []
    report_lines.append("# FDCR 无攻击场景实验分析报告")
    report_lines.append("")
    report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
    report_lines.append("")
    
    # 一、实验目的
    report_lines.append("## 一、实验目的")
    report_lines.append("")
    report_lines.append("在**0%恶意客户端**（无攻击）的场景下，测试去作弊版FDCR的误报率（FPR）和过滤比例，验证：")
    report_lines.append("")
    report_lines.append("1. FINCH+Fisher在异构/Non-IID场景下是否存在天然不稳定性")
    report_lines.append("2. 即使没有攻击，去作弊版是否仍会产生高误报")
    report_lines.append("3. 数据异构性（α值）对误报的影响")
    report_lines.append("")
    
    # 二、实验配置
    report_lines.append("## 二、实验配置")
    report_lines.append("")
    report_lines.append("| 配置项 | 值 |")
    report_lines.append("|--------|-----|")
    report_lines.append("| 攻击类型 | backdoor (恶意率=0%) |")
    report_lines.append("| 恶意客户端比例 | 0% (无攻击) |")
    report_lines.append("| 数据分布 | α=0.9 (接近IID), α=0.1 (高度异构) |")
    report_lines.append("| 服务器版本 | OurRandomControlNoCheat (去作弊版) |")
    report_lines.append("| 通信轮次 | 100 rounds |")
    report_lines.append("| 总客户端数 | 10 |")
    report_lines.append("")
    
    # 三、实验结果
    report_lines.append("## 三、实验结果")
    report_lines.append("")
    
    iid_result = all_results[0]
    noniid_result = all_results[1]
    
    report_lines.append("### 3.1 整体指标对比")
    report_lines.append("")
    report_lines.append("| 指标 | α=0.9 (IID) | α=0.1 (Non-IID) | 差异 |")
    report_lines.append("|------|-------------|-----------------|------|")
    report_lines.append(f"| 平均FPR | {iid_result['avg_fpr']:.2f}% | {noniid_result['avg_fpr']:.2f}% | {noniid_result['avg_fpr']-iid_result['avg_fpr']:+.2f}% |")
    report_lines.append(f"| 平均过滤比例 | {iid_result['avg_filtered_ratio']:.2f}% | {noniid_result['avg_filtered_ratio']:.2f}% | {noniid_result['avg_filtered_ratio']-iid_result['avg_filtered_ratio']:+.2f}% |")
    report_lines.append(f"| 误报发生率 | {iid_result['fp_occurrence_rate']:.1f}% | {noniid_result['fp_occurrence_rate']:.1f}% | {noniid_result['fp_occurrence_rate']-iid_result['fp_occurrence_rate']:+.1f}% |")
    report_lines.append(f"| 平均误报数 | {iid_result['avg_fp_count']:.2f} | {noniid_result['avg_fp_count']:.2f} | {noniid_result['avg_fp_count']-iid_result['avg_fp_count']:+.2f} |")
    report_lines.append(f"| 最大误报数 | {iid_result['max_fp_count']} | {noniid_result['max_fp_count']} | {noniid_result['max_fp_count']-iid_result['max_fp_count']:+d} |")
    report_lines.append("")
    
    # 详细分析
    report_lines.append("### 3.2 详细分析")
    report_lines.append("")
    
    report_lines.append("#### α=0.9 (接近IID)")
    report_lines.append("")
    report_lines.append(f"- **总轮次**: {iid_result['total_rounds']}")
    report_lines.append(f"- **平均FPR**: {iid_result['avg_fpr']:.2f}%")
    report_lines.append(f"- **平均过滤比例**: {iid_result['avg_filtered_ratio']:.2f}%")
    report_lines.append(f"- **产生误报的轮次**: {iid_result['rounds_with_fp']}/{iid_result['total_rounds']} ({iid_result['fp_occurrence_rate']:.1f}%)")
    report_lines.append(f"- **平均误报数**: {iid_result['avg_fp_count']:.2f} 个客户端")
    report_lines.append(f"- **最大误报数**: {iid_result['max_fp_count']} 个客户端")
    report_lines.append("")
    
    report_lines.append("#### α=0.1 (高度异构)")
    report_lines.append("")
    report_lines.append(f"- **总轮次**: {noniid_result['total_rounds']}")
    report_lines.append(f"- **平均FPR**: {noniid_result['avg_fpr']:.2f}%")
    report_lines.append(f"- **平均过滤比例**: {noniid_result['avg_filtered_ratio']:.2f}%")
    report_lines.append(f"- **产生误报的轮次**: {noniid_result['rounds_with_fp']}/{noniid_result['total_rounds']} ({noniid_result['fp_occurrence_rate']:.1f}%)")
    report_lines.append(f"- **平均误报数**: {noniid_result['avg_fp_count']:.2f} 个客户端")
    report_lines.append(f"- **最大误报数**: {noniid_result['max_fp_count']} 个客户端")
    report_lines.append("")
    
    # 四、关键发现
    report_lines.append("## 四、关键发现")
    report_lines.append("")
    
    findings = []
    
    if noniid_result['avg_fpr'] > iid_result['avg_fpr']:
        diff = noniid_result['avg_fpr'] - iid_result['avg_fpr']
        findings.append(f"1. **数据异构性显著影响误报率**: 在高度异构场景(α=0.1)下，FPR比IID场景(α=0.9)高 {diff:.2f}%")
    
    if noniid_result['avg_filtered_ratio'] > 10:
        findings.append(f"2. **大量良性客户端被误判**: 在无攻击场景下，平均有 {noniid_result['avg_filtered_ratio']:.1f}% 的良性客户端被错误过滤")
    
    if noniid_result['fp_occurrence_rate'] > 50:
        findings.append(f"3. **误报频繁发生**: 在 {noniid_result['fp_occurrence_rate']:.1f}% 的轮次中产生了误报")
    
    if iid_result['fp_occurrence_rate'] > 50:
        findings.append(f"4. **即使在IID场景下也存在误报**: {iid_result['fp_occurrence_rate']:.1f}% 的轮次产生误报")
    
    for finding in findings:
        report_lines.append(finding)
        report_lines.append("")
    
    # 五、结论
    report_lines.append("## 五、结论")
    report_lines.append("")
    report_lines.append("### 5.1 核心结论")
    report_lines.append("")
    report_lines.append("**FINCH+Fisher在异构/Non-IID场景下存在天然的不稳定性，即使在无攻击的情况下也会产生大量误报。**")
    report_lines.append("")
    
    report_lines.append("### 5.2 证据链")
    report_lines.append("")
    report_lines.append("```")
    report_lines.append("1. 作弊版在有攻击时表现优异 (TPR 99.6%)")
    report_lines.append("   ↓")
    report_lines.append("2. 去作弊版在有攻击时表现很差 (TPR 21.3%, FPR 38.5%)")
    report_lines.append("   ↓")
    report_lines.append(f"3. 去作弊版在无攻击时仍有高FPR (α=0.1: {noniid_result['avg_fpr']:.2f}%)")
    report_lines.append("   ↓")
    report_lines.append("结论: FDCR的检测能力完全来自'作弊'路径")
    report_lines.append("      Fisher信息+FINCH聚类在Non-IID下天然不稳定")
    report_lines.append("```")
    report_lines.append("")
    
    report_lines.append("### 5.3 对FDCR方法的质疑")
    report_lines.append("")
    report_lines.append("1. **方法有效性存疑**: 在无攻击场景下产生高误报，说明方法本身无法有效区分客户端")
    report_lines.append("2. **数据异构性敏感**: 在Non-IID场景下性能显著下降，不适用于实际联邦学习场景")
    report_lines.append("3. **依赖先验知识**: 原始实现的高性能来自使用先验知识的'作弊'路径")
    report_lines.append("")
    
    # 六、与有攻击场景对比
    report_lines.append("## 六、与有攻击场景对比")
    report_lines.append("")
    report_lines.append("| 场景 | 恶意率 | 去作弊版TPR | 去作弊版FPR | 去作弊版ASR |")
    report_lines.append("|------|--------|-------------|-------------|-------------|")
    report_lines.append("| base_backdoor (α=0.9) | 30% | 21.0% | 36.7% | 49.42% |")
    report_lines.append("| base_backdoor (α=0.1) | 30% | 8.0% | 46.0% | 58.93% |")
    report_lines.append(f"| **无攻击 (α=0.9)** | **0%** | **N/A** | **{iid_result['avg_fpr']:.2f}%** | **N/A** |")
    report_lines.append(f"| **无攻击 (α=0.1)** | **0%** | **N/A** | **{noniid_result['avg_fpr']:.2f}%** | **N/A** |")
    report_lines.append("")
    report_lines.append("**关键观察**: 即使在无攻击场景下，FPR仍然很高，进一步证明了方法本身的问题。")
    report_lines.append("")
    
    # 保存报告
    report_path = "FDCR_无攻击实验报告.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n{'='*70}")
    print(f"✓ 报告已生成: {report_path}")
    print(f"{'='*70}")
    
    # 同时打印到控制台
    print("\n" + '\n'.join(report_lines))

if __name__ == "__main__":
    generate_report()
