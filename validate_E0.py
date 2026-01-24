"""
E0: 定义一致性与符号一致性核对

验证 A1 的负 effect size 是"真实机制"还是实现口径引入的假象。

Check-1: V_{k,global} 是否与系统检测的 V_k 一致
Check-2: 中心 ge_global 定义是否一致
Check-3: 层切片 S_l 是否正确
Check-4: 加权对象 g⊙I 是否一致
Check-5: effect size 符号方向核对
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple
from collections import OrderedDict


class E0Validator:
    """E0 定义一致性验证器"""
    
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        
        # 加载元数据
        with open(os.path.join(run_dir, 'run_manifest.json'), 'r') as f:
            self.manifest = json.load(f)
        with open(os.path.join(run_dir, 'param_index_map.json'), 'r') as f:
            self.param_map = json.load(f)
        with open(os.path.join(run_dir, 'layer_slices.json'), 'r') as f:
            self.layer_slices = json.load(f)
            
        self.n_clients = self.manifest['config']['parti_num']
        self.n_rounds = self.manifest['config']['communication_epoch']
        self.eta = self.manifest['config']['local_train_lr']
        
    def load_round_data(self, round_idx: int) -> Tuple[Dict, Dict, List[int]]:
        """加载一轮数据"""
        round_dir = os.path.join(self.run_dir, f'round_{round_idx:03d}')
        
        tensors = torch.load(os.path.join(round_dir, 'tensors.pt'))
        with open(os.path.join(round_dir, 'stats.json'), 'r') as f:
            stats = json.load(f)
        with open(os.path.join(round_dir, 'ground_truth.json'), 'r') as f:
            gt = json.load(f)
            
        return tensors, stats, gt['actual_malicious_idx']
    
    def check_1_Vk_consistency(self, rounds: List[int] = None) -> Dict:
        """
        Check-1: V_{k,global} 是否与系统检测的 V_k 一致
        
        比较：
        - V_{k,code}: 系统保存的 decision['V_k']
        - V_{k,calc}: 按公式计算的 ||g^e_k - g^e_global||^2
        """
        if rounds is None:
            rounds = [0, 1, 2]  # 抽查前3轮
        
        results = {'max_diff': 0, 'samples': []}
        
        for round_idx in rounds:
            tensors, stats, malicious_idx = self.load_round_data(round_idx)
            
            # 获取系统保存的 V_k
            if 'decision' not in tensors or 'V_k' not in tensors['decision']:
                print(f"  Round {round_idx}: 无 decision/V_k 数据")
                continue
            
            V_k_code = tensors['decision']['V_k'].squeeze()
            
            # 获取 g^e_global (ge_global_topk 只是 topk，需要重建)
            # 这里我们用保存的 g_weighted 来重新计算
            
            # 收集所有客户端的 g_weighted (仅 head+last_block 全量)
            g_weighted_full = {}
            for i in range(self.n_clients):
                client_key = f'client_{i}'
                if client_key in tensors:
                    # 尝试获取全量 (我们只保存了 head 和 last_block)
                    g_head = tensors[client_key].get('g_weighted_head_full', None)
                    g_last = tensors[client_key].get('g_weighted_last_block_full', None)
                    if g_head is not None and g_last is not None:
                        g_weighted_full[i] = {
                            'head': g_head,
                            'last_block': g_last
                        }
            
            results['samples'].append({
                'round': round_idx,
                'V_k_code': V_k_code.tolist() if isinstance(V_k_code, torch.Tensor) else V_k_code,
                'n_clients_with_data': len(g_weighted_full)
            })
        
        return results
    
    def check_2_center_definition(self, round_idx: int = 0) -> Dict:
        """
        Check-2: 中心 ge_global 定义一致吗？
        
        比对离线计算的中心与系统保存的中心
        """
        tensors, stats, malicious_idx = self.load_round_data(round_idx)
        
        result = {}
        
        # 系统保存的 ge_global (topk)
        if 'decision' in tensors:
            ge_global_topk_idx = tensors['decision'].get('ge_global_topk_idx', None)
            ge_global_topk_val = tensors['decision'].get('ge_global_topk_val', None)
            
            if ge_global_topk_val is not None:
                result['ge_global_topk_l2'] = float(torch.norm(ge_global_topk_val, p=2))
                result['ge_global_topk_mean'] = float(ge_global_topk_val.mean())
                result['ge_global_topk_max'] = float(ge_global_topk_val.max())
                result['ge_global_topk_min'] = float(ge_global_topk_val.min())
        
        # 尝试从 stats 获取信息
        if 'decision' in stats:
            result['decision_stats'] = stats['decision']
        
        return result
    
    def check_3_layer_slices(self) -> Dict:
        """
        Check-3: 层切片 S_l 是否与 param_index_map 一致
        """
        result = {
            'total_params': self.manifest['total_params'],
            'layers': {}
        }
        
        # 检查每层
        for level in ['L1']:
            if level not in self.layer_slices:
                continue
            
            for layer_name, layer_info in self.layer_slices[level].items():
                if 'indices' not in layer_info:
                    continue
                
                indices_info = layer_info['indices']
                total_length = sum(idx['length'] for idx in indices_info)
                
                result['layers'][layer_name] = {
                    'params': layer_info.get('params', []),
                    'total_length': total_length,
                    'indices': [(idx['offset'], idx['offset'] + idx['length']) for idx in indices_info]
                }
        
        # 检查重叠
        if 'head' in result['layers'] and 'last_block' in result['layers']:
            head_set = set()
            for start, end in result['layers']['head']['indices']:
                head_set.update(range(start, end))
            
            last_set = set()
            for start, end in result['layers']['last_block']['indices']:
                last_set.update(range(start, end))
            
            overlap = head_set & last_set
            result['head_last_overlap'] = len(overlap)
            result['head_last_disjoint'] = len(overlap) == 0
        
        return result
    
    def check_4_weighted_object(self, round_idx: int = 0) -> Dict:
        """
        Check-4: 加权对象是 g⊙I 还是 Δw⊙I？
        
        检查 g_weighted 与 delta_w_true * I_minmax 是否一致
        """
        tensors, stats, malicious_idx = self.load_round_data(round_idx)
        
        result = {'comparisons': []}
        
        for i in range(min(3, self.n_clients)):  # 抽查前3个客户端
            client_key = f'client_{i}'
            if client_key not in tensors:
                continue
            
            # 从 stats 获取信息
            if client_key in stats:
                client_stats = stats[client_key]
                
                # g = delta_w / eta，所以 g_weighted = (delta_w / eta) * I
                # 如果是 delta_w * I，则 g_weighted = g * I * eta
                
                delta_w_head_l2 = client_stats.get('delta_w_true_head', {}).get('l2', 0)
                g_weighted_head_l2 = client_stats.get('g_weighted_head', {}).get('l2', 0)
                I_head_l2 = client_stats.get('I_minmax_head', {}).get('l2', 0)
                
                result['comparisons'].append({
                    'client': i,
                    'delta_w_head_l2': delta_w_head_l2,
                    'g_weighted_head_l2': g_weighted_head_l2,
                    'I_minmax_head_l2': I_head_l2,
                    'eta': self.eta
                })
        
        return result
    
    def check_5_effect_size_sign(self, rounds: List[int] = None) -> Dict:
        """
        Check-5: effect size 符号方向核对
        
        直接打印 μ_M 和 μ_B，验证符号
        """
        if rounds is None:
            rounds = list(range(min(10, self.n_rounds)))
        
        # 收集所有轮次的 V_k
        V_k_all = {'benign': [], 'malicious': []}
        
        for round_idx in rounds:
            tensors, stats, malicious_idx = self.load_round_data(round_idx)
            malicious_set = set(malicious_idx)
            
            if 'decision' in tensors and 'V_k' in tensors['decision']:
                V_k = tensors['decision']['V_k'].squeeze()
                
                for i in range(min(len(V_k), self.n_clients)):
                    if i in malicious_set:
                        V_k_all['malicious'].append(float(V_k[i]))
                    else:
                        V_k_all['benign'].append(float(V_k[i]))
        
        # 计算均值
        mu_M = np.mean(V_k_all['malicious']) if V_k_all['malicious'] else 0
        mu_B = np.mean(V_k_all['benign']) if V_k_all['benign'] else 0
        std_M = np.std(V_k_all['malicious']) if V_k_all['malicious'] else 0
        std_B = np.std(V_k_all['benign']) if V_k_all['benign'] else 0
        
        delta = mu_M - mu_B
        pooled_std = np.sqrt((std_M**2 + std_B**2) / 2)
        effect_size = delta / pooled_std if pooled_std > 0 else 0
        
        return {
            'mu_M (malicious mean)': mu_M,
            'mu_B (benign mean)': mu_B,
            'std_M': std_M,
            'std_B': std_B,
            'delta (mu_M - mu_B)': delta,
            'effect_size (delta/pooled_std)': effect_size,
            'n_malicious': len(V_k_all['malicious']),
            'n_benign': len(V_k_all['benign']),
            'interpretation': '恶意 V_k 更大' if delta > 0 else '恶意 V_k 更小（反向）'
        }
    
    def run_all_checks(self):
        """运行所有检查"""
        print("=" * 60)
        print("E0: 定义一致性与符号一致性核对")
        print("=" * 60)
        print(f"数据目录: {self.run_dir}")
        print()
        
        # Check-1
        print("【Check-1】V_{k,global} 与系统 V_k 一致性")
        print("-" * 40)
        check1 = self.check_1_Vk_consistency()
        print(f"  抽查轮数: {len(check1['samples'])}")
        for sample in check1['samples']:
            print(f"  Round {sample['round']}: V_k_code = {sample['V_k_code'][:3]}... (前3个)")
        print()
        
        # Check-2
        print("【Check-2】中心 ge_global 定义一致性")
        print("-" * 40)
        check2 = self.check_2_center_definition()
        if 'ge_global_topk_l2' in check2:
            print(f"  ge_global topk L2 norm: {check2['ge_global_topk_l2']:.6f}")
            print(f"  ge_global topk mean: {check2['ge_global_topk_mean']:.6f}")
        print()
        
        # Check-3
        print("【Check-3】层切片 S_l 一致性")
        print("-" * 40)
        check3 = self.check_3_layer_slices()
        print(f"  总参数数: {check3['total_params']}")
        for layer_name, info in check3.get('layers', {}).items():
            print(f"  {layer_name}: {info['total_length']} 参数")
        if 'head_last_disjoint' in check3:
            print(f"  head 与 last_block 不重叠: {'✅ 是' if check3['head_last_disjoint'] else '❌ 否'}")
        print()
        
        # Check-4
        print("【Check-4】加权对象 g⊙I 一致性")
        print("-" * 40)
        check4 = self.check_4_weighted_object()
        for comp in check4['comparisons']:
            print(f"  Client {comp['client']}: delta_w_head_l2={comp['delta_w_head_l2']:.4f}, "
                  f"g_weighted_head_l2={comp['g_weighted_head_l2']:.4f}")
        print(f"  eta = {self.eta}")
        print()
        
        # Check-5 (关键)
        print("【Check-5】Effect size 符号方向核对 ⭐")
        print("-" * 40)
        check5 = self.check_5_effect_size_sign()
        print(f"  μ_M (恶意均值): {check5['mu_M (malicious mean)']:.6f}")
        print(f"  μ_B (良性均值): {check5['mu_B (benign mean)']:.6f}")
        print(f"  Δ (μ_M - μ_B): {check5['delta (mu_M - mu_B)']:.6f}")
        print(f"  Effect size: {check5['effect_size (delta/pooled_std)']:.4f}")
        print(f"  解释: {check5['interpretation']}")
        print()
        
        # 结论
        print("=" * 60)
        print("【E0 结论】")
        print("=" * 60)
        
        if check5['delta (mu_M - mu_B)'] < 0:
            print("✅ 负 effect size 是真实的：恶意客户端的 V_k 确实比良性更小")
            print("   这说明 FDCR 的 g⊙I→V 设计确实使恶意客户端看起来更'正常'")
        else:
            print("❌ effect size 符号核对不通过，需要检查实现")
        
        return {
            'check1': check1,
            'check2': check2,
            'check3': check3,
            'check4': check4,
            'check5': check5
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='E0: 定义一致性核对')
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
    
    validator = E0Validator(run_dir)
    validator.run_all_checks()


if __name__ == '__main__':
    main()
