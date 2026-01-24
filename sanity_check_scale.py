"""
Sanity Check: 验证 gI/I/ratio 的口径一致性

检查项：
1. gI 是否等于 g * I（逐元素乘法）
2. I 是否在 [0,1] 范围内
3. shrink_ratio = norm(gI)/norm(g) 是否在 [0,1]

使用方法:
    python sanity_check_scale.py --run_id <run_id>
"""

import os
import json
import torch
import numpy as np
from typing import Dict, Tuple


def run_sanity_check(run_dir: str, rounds_to_check: list = None):
    """运行 sanity check"""
    
    # 加载元数据
    with open(os.path.join(run_dir, 'run_manifest.json'), 'r') as f:
        manifest = json.load(f)
    
    n_clients = manifest['config']['parti_num']
    n_rounds = manifest['config']['communication_epoch']
    
    if rounds_to_check is None:
        rounds_to_check = [0, 49, 99]  # 抽查开始、中间、结束
    
    print("=" * 70)
    print("Sanity Check: gI/I/ratio 口径验证")
    print("=" * 70)
    
    all_issues = []
    
    for round_idx in rounds_to_check:
        round_dir = os.path.join(run_dir, f'round_{round_idx:03d}')
        tensors = torch.load(os.path.join(round_dir, 'tensors.pt'))
        
        print(f"\n【Round {round_idx}】")
        print("-" * 50)
        
        for layer in ['head', 'last_block']:
            print(f"\n  [{layer}]")
            
            # 收集所有客户端的数据
            g_list = []
            gI_list = []
            I_list = []
            delta_w_list = []
            
            for i in range(n_clients):
                client_key = f'client_{i}'
                if client_key not in tensors:
                    continue
                
                t = tensors[client_key]
                
                gI_key = f'g_weighted_{layer}_full'
                I_key = f'I_minmax_{layer}_full'
                dw_key = f'delta_w_true_{layer}_full'
                
                if gI_key in t and I_key in t and dw_key in t:
                    gI_list.append(t[gI_key].float())
                    I_list.append(t[I_key].float())
                    delta_w_list.append(t[dw_key].float())
            
            if len(gI_list) == 0:
                print(f"    ⚠️ 无数据")
                continue
            
            # Stack for batch analysis
            gI = torch.stack(gI_list)  # [n_clients, d]
            I = torch.stack(I_list)
            delta_w = torch.stack(delta_w_list)
            
            # ============ Check 1: I 的范围 ============
            I_min = float(I.min())
            I_max = float(I.max())
            I_mean = float(I.mean())
            
            print(f"    I 范围: min={I_min:.6f}, max={I_max:.6f}, mean={I_mean:.6f}")
            
            if I_max > 1 + 1e-6:
                msg = f"Round {round_idx}, {layer}: I_max={I_max:.6f} > 1"
                print(f"    ❌ {msg}")
                all_issues.append(msg)
            elif I_min < -1e-6:
                msg = f"Round {round_idx}, {layer}: I_min={I_min:.6f} < 0"
                print(f"    ❌ {msg}")
                all_issues.append(msg)
            else:
                print(f"    ✅ I ∈ [0, 1]")
            
            # ============ Check 2: gI ≈ delta_w * I ============
            # 计算 g * I（逐元素乘法）
            computed_gI = delta_w * I
            diff = gI - computed_gI
            max_abs_diff = float(diff.abs().max())
            mean_abs_diff = float(diff.abs().mean())
            
            print(f"    gI vs delta_w*I: max_diff={max_abs_diff:.6e}, mean_diff={mean_abs_diff:.6e}")
            
            if max_abs_diff > 1e-3:
                msg = f"Round {round_idx}, {layer}: gI ≠ delta_w*I, max_diff={max_abs_diff:.6e}"
                print(f"    ❌ {msg}")
                all_issues.append(msg)
            else:
                print(f"    ✅ gI ≈ delta_w * I")
            
            # ============ Check 3: ratio = norm(gI) / norm(delta_w) ============
            norm_gI = torch.norm(gI, p=2, dim=1)  # [n_clients]
            norm_dw = torch.norm(delta_w, p=2, dim=1)
            
            ratio = norm_gI / (norm_dw + 1e-12)
            
            ratio_min = float(ratio.min())
            ratio_max = float(ratio.max())
            ratio_mean = float(ratio.mean())
            ratio_p50 = float(torch.quantile(ratio, 0.5))
            ratio_p95 = float(torch.quantile(ratio, 0.95))
            
            print(f"    ratio 统计: min={ratio_min:.4f}, max={ratio_max:.4f}, mean={ratio_mean:.4f}")
            print(f"    ratio 分位: p50={ratio_p50:.4f}, p95={ratio_p95:.4f}")
            
            if ratio_max > 1 + 1e-3:
                msg = f"Round {round_idx}, {layer}: ratio_max={ratio_max:.4f} > 1"
                print(f"    ❌ {msg}")
                all_issues.append(msg)
            else:
                print(f"    ✅ ratio ∈ [0, 1]")
            
            # ============ 额外检查：I 的分布 ============
            I_below_01 = float((I < 0.1).float().mean())
            I_below_001 = float((I < 0.01).float().mean())
            print(f"    I 稀疏性: {I_below_01:.1%} < 0.1, {I_below_001:.1%} < 0.01")
    
    # ============ 汇总 ============
    print("\n" + "=" * 70)
    print("【汇总】")
    print("=" * 70)
    
    if len(all_issues) == 0:
        print("✅ 所有检查通过")
    else:
        print(f"❌ 发现 {len(all_issues)} 个问题:")
        for issue in all_issues:
            print(f"  - {issue}")
    
    return all_issues


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Sanity Check: gI/I/ratio')
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='logs/step1')
    
    args = parser.parse_args()
    
    if args.run_id:
        run_dir = os.path.join(args.log_dir, args.run_id)
    else:
        runs = sorted([d for d in os.listdir(args.log_dir) 
                       if os.path.isdir(os.path.join(args.log_dir, d))])
        run_dir = os.path.join(args.log_dir, runs[-1])
        print(f"使用最新运行: {runs[-1]}")
    
    issues = run_sanity_check(run_dir)
    
    if issues:
        print("\n⚠️ 存在口径问题，需要检查数据记录逻辑")


if __name__ == '__main__':
    main()
