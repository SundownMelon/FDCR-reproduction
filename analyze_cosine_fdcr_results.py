"""
分析 CosineFDCR 实验结果：计算 FPR、TPR 及过滤代价

从 aggregation_weight.csv 推算检测指标
"""

import pandas as pd
import numpy as np
import re
import os

# 配置
DATA_DIR = r"c:\Users\l'x\Downloads\FDCR-main\FDCR-main\data\label_skew\base_backdoor\0.3\fl_cifar10\0.9\CosineFDCR_Head\FedFish\para1"
MALICIOUS_IDX = [7, 8, 9]  # 恶意客户端索引
BENIGN_IDX = [0, 1, 2, 3, 4, 5, 6]  # 良性客户端索引
NUM_CLIENTS = 10


def parse_aggregation_weights(filepath: str) -> pd.DataFrame:
    """解析 aggregation_weight.csv"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # 解析格式：每两行一组（epoch: 和 weights）
    lines = content.strip().split('\n')
    rows = []
    
    for i in range(0, len(lines), 2):
        epoch_line = lines[i].strip()
        if not epoch_line:
            continue
        
        epoch = int(epoch_line.replace(':', ''))
        
        if i + 1 < len(lines):
            weight_line = lines[i + 1].strip()
            weights = [float(w) for w in weight_line.split(',') if w.strip()]
            rows.append({'epoch': epoch, 'weights': weights})
    
    return pd.DataFrame(rows)


def compute_detection_metrics(weights_df: pd.DataFrame) -> pd.DataFrame:
    """计算每轮的检测指标"""
    results = []
    
    for _, row in weights_df.iterrows():
        epoch = row['epoch']
        weights = row['weights']
        
        if len(weights) != NUM_CLIENTS:
            continue
        
        # 被过滤的客户端：权重 = 0
        filtered_idx = [i for i, w in enumerate(weights) if w == 0]
        
        # 计算 TP, FP, TN, FN
        TP = len([i for i in filtered_idx if i in MALICIOUS_IDX])
        FP = len([i for i in filtered_idx if i in BENIGN_IDX])
        TN = len([i for i in BENIGN_IDX if i not in filtered_idx])
        FN = len([i for i in MALICIOUS_IDX if i not in filtered_idx])
        
        # 计算指标
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        
        results.append({
            'epoch': epoch,
            'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
            'TPR': TPR, 'FPR': FPR, 'Precision': Precision,
            'num_filtered': len(filtered_idx),
            'num_filtered_benign': FP
        })
    
    return pd.DataFrame(results)


def analyze_results():
    """主分析函数"""
    print("=" * 60)
    print("Cosine-FDCR 检测指标详细分析")
    print("=" * 60)
    
    # 解析数据
    weights_file = os.path.join(DATA_DIR, 'aggregation_weight.csv')
    weights_df = parse_aggregation_weights(weights_file)
    
    print(f"总共 {len(weights_df)} 轮数据")
    
    # 计算检测指标
    metrics_df = compute_detection_metrics(weights_df)
    
    # 全局统计
    print("\n【全局统计（100轮平均）】")
    print(f"  TPR (检测率): {metrics_df['TPR'].mean():.2%}")
    print(f"  FPR (误杀率): {metrics_df['FPR'].mean():.2%}")
    print(f"  Precision:    {metrics_df['Precision'].mean():.2%}")
    print(f"  每轮平均过滤数: {metrics_df['num_filtered'].mean():.2f}")
    print(f"  每轮误杀良性数: {metrics_df['num_filtered_benign'].mean():.2f}")
    
    # 后50轮统计（稳定阶段）
    last50 = metrics_df[metrics_df['epoch'] >= 50]
    print("\n【后50轮统计（稳定阶段）】")
    print(f"  TPR (检测率): {last50['TPR'].mean():.2%}")
    print(f"  FPR (误杀率): {last50['FPR'].mean():.2%}")
    print(f"  Precision:    {last50['Precision'].mean():.2%}")
    print(f"  每轮平均过滤数: {last50['num_filtered'].mean():.2f}")
    print(f"  每轮误杀良性数: {last50['num_filtered_benign'].mean():.2f}")
    
    # 保存详细结果
    output_file = os.path.join(DATA_DIR, 'detection_metrics_detailed.csv')
    metrics_df.to_csv(output_file, index=False)
    print(f"\n详细指标已保存: {output_file}")
    
    return metrics_df, last50


if __name__ == '__main__':
    metrics_df, last50 = analyze_results()
