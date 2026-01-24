"""
A1: 分层 V_{k,l} 可分性诊断

验证信号是否能沿着 I → g^e → V 传递到最终判别量。
对每轮每客户端，计算分层版 V_k。

使用方法:
    python analyze_layerwise_Vk.py --run_id <run_id>
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class LayerwiseVkAnalyzer:
    """分层 V_k 分析器"""
    
    # 层级切片定义（与 fdcr_logger 一致）
    LAYER_PARAMS = {
        'head': ['cls.weight', 'cls.bias'],
        'last_block': ['l2.weight', 'l2.bias'],
        'head_last': ['cls.weight', 'cls.bias', 'l2.weight', 'l2.bias'],
        'l1': ['l1.weight', 'l1.bias'],
    }
    
    def __init__(self, run_dir: str):
        """初始化分析器"""
        self.run_dir = run_dir
        
        # 加载元数据
        with open(os.path.join(run_dir, 'run_manifest.json'), 'r') as f:
            self.manifest = json.load(f)
        with open(os.path.join(run_dir, 'param_index_map.json'), 'r') as f:
            self.param_map = json.load(f)
            
        self.n_clients = self.manifest['config']['parti_num']
        self.n_rounds = self.manifest['config']['communication_epoch']
        self.eta = self.manifest['config']['local_train_lr']
        
        # 构建层级索引
        self._build_layer_indices()
        
        # 创建输出目录
        self.output_dir = os.path.join(run_dir, 'analysis_A1')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _build_layer_indices(self):
        """构建每层的参数索引范围"""
        self.layer_indices = {}
        
        for layer_name, param_names in self.LAYER_PARAMS.items():
            indices = []
            for pname in param_names:
                if pname in self.param_map:
                    info = self.param_map[pname]
                    start = info['offset']
                    end = start + info['length']
                    indices.extend(range(start, end))
            self.layer_indices[layer_name] = indices
        
        # global = 所有参数
        self.layer_indices['global'] = list(range(self.manifest['total_params']))
        
    def load_round_data(self, round_idx: int) -> Tuple[Dict, Dict, List[int]]:
        """加载一轮数据"""
        round_dir = os.path.join(self.run_dir, f'round_{round_idx:03d}')
        
        tensors = torch.load(os.path.join(round_dir, 'tensors.pt'))
        with open(os.path.join(round_dir, 'stats.json'), 'r') as f:
            stats = json.load(f)
        with open(os.path.join(round_dir, 'ground_truth.json'), 'r') as f:
            gt = json.load(f)
            
        return tensors, stats, gt['actual_malicious_idx']
    
    def compute_layerwise_Vk(
        self,
        rounds: Optional[List[int]] = None,
        layers: List[str] = ['global', 'head', 'last_block', 'head_last']
    ) -> Dict:
        """
        计算分层版 V_{k,l}
        
        V_{k,l} = MSE(ge_global[l], ge_k[l])
        """
        if rounds is None:
            rounds = list(range(self.n_rounds))
        
        results = {layer: {'V_k': [], 'is_malicious': [], 'round': [], 'client': []} 
                   for layer in layers}
        
        for round_idx in rounds:
            tensors, stats, malicious_idx = self.load_round_data(round_idx)
            malicious_set = set(malicious_idx)
            
            # 获取每个客户端的 g_weighted 和 I_minmax
            client_g_weighted = {}
            client_I_minmax = {}
            
            for i in range(self.n_clients):
                client_key = f'client_{i}'
                if client_key in tensors:
                    # 使用 topk 重建全量向量（近似）
                    # 或者使用保存的 head/last_block 全量
                    g_head = tensors[client_key].get('g_weighted_head_full', None)
                    g_last = tensors[client_key].get('g_weighted_last_block_full', None)
                    I_head = tensors[client_key].get('I_minmax_head_full', None)
                    I_last = tensors[client_key].get('I_minmax_last_block_full', None)
                    
                    if g_head is not None and g_last is not None:
                        client_g_weighted[i] = {
                            'head': g_head.float(),
                            'last_block': g_last.float(),
                            'head_last': torch.cat([g_head.float(), g_last.float()])
                        }
                    if I_head is not None and I_last is not None:
                        client_I_minmax[i] = {
                            'head': I_head.float(),
                            'last_block': I_last.float(),
                            'head_last': torch.cat([I_head.float(), I_last.float()])
                        }
            
            # 如果有 decision 数据，使用它
            if 'decision' in tensors:
                ge_global_topk_val = tensors['decision'].get('ge_global_topk_val', None)
            else:
                ge_global_topk_val = None
            
            # 计算每层的加权聚合 (ge_global)
            for layer in layers:
                if layer == 'global':
                    # 对于 global，使用 decision 中的 V_k
                    if 'decision' in tensors and 'V_k' in tensors['decision']:
                        V_k_global = tensors['decision']['V_k'].squeeze()
                        for i in range(self.n_clients):
                            if i < len(V_k_global):
                                results[layer]['V_k'].append(float(V_k_global[i]))
                                results[layer]['is_malicious'].append(i in malicious_set)
                                results[layer]['round'].append(round_idx)
                                results[layer]['client'].append(i)
                    continue
                
                # 对于其他层，手动计算
                if len(client_g_weighted) < 2:
                    continue
                
                # 计算 ge_global (该层的加权聚合)
                valid_clients = [i for i in range(self.n_clients) if i in client_g_weighted and layer in client_g_weighted[i]]
                if len(valid_clients) == 0:
                    continue
                    
                freq = np.ones(len(valid_clients)) / len(valid_clients)  # 均匀权重
                
                ge_layer_global = torch.zeros_like(client_g_weighted[valid_clients[0]][layer])
                for idx, i in enumerate(valid_clients):
                    ge_layer_global += client_g_weighted[i][layer] * freq[idx]
                
                # 计算每个客户端的 V_{k,l}
                for i in valid_clients:
                    ge_k = client_g_weighted[i][layer]
                    # V_k = MSE(ge_global, ge_k) 或 L2 距离
                    V_k = float(torch.nn.functional.mse_loss(ge_layer_global, ge_k))
                    
                    results[layer]['V_k'].append(V_k)
                    results[layer]['is_malicious'].append(i in malicious_set)
                    results[layer]['round'].append(round_idx)
                    results[layer]['client'].append(i)
        
        return results
    
    def analyze_separability(self, Vk_results: Dict) -> Dict:
        """分析 V_k 的 bb vs bm 可分性"""
        summary = {}
        
        for layer, data in Vk_results.items():
            V_k = np.array(data['V_k'])
            is_malicious = np.array(data['is_malicious'])
            
            if len(V_k) == 0:
                continue
            
            # 分离 benign 和 malicious
            V_benign = V_k[~is_malicious]
            V_malicious = V_k[is_malicious]
            
            if len(V_benign) == 0 or len(V_malicious) == 0:
                continue
            
            # 计算统计量
            # 注意：对于 V_k，我们期望 benign 更相似（V_k 更小），malicious 更不同（V_k 更大）
            mean_bb = np.mean(V_benign)
            std_bb = np.std(V_benign)
            mean_bm = np.mean(V_malicious)
            std_bm = np.std(V_malicious)
            
            # delta = mean_bm - mean_bb (正值表示恶意更远，符合预期)
            delta = mean_bm - mean_bb
            pooled_std = np.sqrt((std_bb**2 + std_bm**2) / 2)
            effect_size = delta / pooled_std if pooled_std > 0 else 0
            
            summary[layer] = {
                'mean_benign': mean_bb,
                'std_benign': std_bb,
                'mean_malicious': mean_bm,
                'std_malicious': std_bm,
                'delta': delta,
                'effect_size': effect_size,
                'n_benign': len(V_benign),
                'n_malicious': len(V_malicious),
                'bb_variance': float(np.var(V_benign)),  # 良性内部方差
                'bm_separation': float(delta / std_bb if std_bb > 0 else 0)  # 相对分离度
            }
        
        return summary
    
    def plot_Vk_distributions(self, Vk_results: Dict, summary: Dict) -> str:
        """绘制 V_k 的 bb vs bm 分布图"""
        layers = [l for l in ['global', 'head', 'last_block', 'head_last'] if l in Vk_results]
        n_layers = len(layers)
        
        if n_layers == 0:
            return None
        
        fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
        if n_layers == 1:
            axes = [axes]
        
        for ax, layer in zip(axes, layers):
            V_k = np.array(Vk_results[layer]['V_k'])
            is_malicious = np.array(Vk_results[layer]['is_malicious'])
            
            V_benign = V_k[~is_malicious]
            V_malicious = V_k[is_malicious]
            
            # 直方图
            if len(V_benign) > 0:
                ax.hist(V_benign, bins=30, alpha=0.6, label=f'Benign (n={len(V_benign)})', color='blue')
            if len(V_malicious) > 0:
                ax.hist(V_malicious, bins=30, alpha=0.6, label=f'Malicious (n={len(V_malicious)})', color='red')
            
            ax.set_xlabel('V_k')
            ax.set_ylabel('Count')
            ax.set_title(f'{layer}\neffect={summary.get(layer, {}).get("effect_size", 0):.3f}')
            ax.legend()
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'Vk_distributions.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_effect_size_comparison(self, summary: Dict) -> str:
        """绘制各层 effect size 对比柱状图"""
        layers = ['global', 'head', 'last_block', 'head_last']
        layers = [l for l in layers if l in summary]
        
        if len(layers) == 0:
            return None
        
        effect_sizes = [summary[l]['effect_size'] for l in layers]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        colors = ['green' if es > 0.5 else 'orange' if es > 0.2 else 'red' for es in effect_sizes]
        bars = ax.bar(layers, effect_sizes, color=colors, alpha=0.7)
        
        # 添加参考线
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Large effect (0.8)')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium effect (0.5)')
        ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Small effect (0.2)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 添加数值标签
        for bar, es in zip(bars, effect_sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{es:.3f}',
                    ha='center', va='bottom' if height >= 0 else 'top')
        
        ax.set_ylabel('Effect Size (malicious - benign)')
        ax.set_xlabel('Layer')
        ax.set_title('V_k 层级可分性对比\n(正值 = 恶意客户端 V_k 更大 = 更容易检测)')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'Vk_effect_size.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_report(self) -> str:
        """生成完整分析报告"""
        print("=" * 60)
        print("A1: 分层 V_{k,l} 可分性诊断")
        print("=" * 60)
        print(f"数据目录: {self.run_dir}")
        print(f"客户端数: {self.n_clients}")
        print(f"通信轮数: {self.n_rounds}")
        print()
        
        # 计算分层 V_k
        print("正在计算分层 V_k...")
        Vk_results = self.compute_layerwise_Vk()
        
        # 分析可分性
        print("正在分析可分性...")
        summary = self.analyze_separability(Vk_results)
        
        # 打印结果
        print("\n【分层 V_k 可分性】")
        print("-" * 80)
        print(f"{'层级':15s} | {'Benign V_k':20s} | {'Malicious V_k':20s} | {'Δ':>10s} | {'Effect':>8s}")
        print("-" * 80)
        
        for layer in ['global', 'head', 'last_block', 'head_last']:
            if layer in summary:
                s = summary[layer]
                print(f"{layer:15s} | {s['mean_benign']:.6f}±{s['std_benign']:.6f} | "
                      f"{s['mean_malicious']:.6f}±{s['std_malicious']:.6f} | "
                      f"{s['delta']:>10.6f} | {s['effect_size']:>8.3f}")
        
        print()
        print("【可分性判定】")
        for layer in ['global', 'head', 'last_block', 'head_last']:
            if layer in summary:
                es = summary[layer]['effect_size']
                if es > 0.8:
                    verdict = "✅ 强可分"
                elif es > 0.5:
                    verdict = "✅ 中等可分"
                elif es > 0.2:
                    verdict = "⚠️ 弱可分"
                elif es > 0:
                    verdict = "⚠️ 极弱可分"
                else:
                    verdict = "❌ 不可分/反向"
                print(f"  {layer}: {verdict} (effect={es:.3f})")
        
        # 绘图
        print("\n正在生成图表...")
        dist_path = self.plot_Vk_distributions(Vk_results, summary)
        es_path = self.plot_effect_size_comparison(summary)
        
        if dist_path:
            print(f"分布图已保存: {dist_path}")
        if es_path:
            print(f"Effect size 图已保存: {es_path}")
        
        # 保存结果
        results = {
            'summary': summary,
            'Vk_results': {k: {kk: v if not isinstance(v, list) else v 
                               for kk, v in vv.items()} 
                          for k, vv in Vk_results.items()}
        }
        
        # 转换 numpy 类型
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64, np.float_)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64, np.int_)):
                return int(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        report_path = os.path.join(self.output_dir, 'A1_results.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(convert(results), f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存: {report_path}")
        print("=" * 60)
        
        # 结论
        print("\n【A1 结论】")
        if 'last_block' in summary and 'global' in summary:
            es_last = summary['last_block']['effect_size']
            es_global = summary['global']['effect_size']
            if es_last > es_global and es_last > 0.2:
                print("✅ 层级稀释假设成立：V_{k,last} 可分性显著高于 V_{k,global}")
                print(f"   V_{{k,last}} effect = {es_last:.3f}")
                print(f"   V_{{k,global}} effect = {es_global:.3f}")
                print("   → 建议在 A2 中测试用 V_{k,last} 替换 V_k 做 FINCH")
            else:
                print("❌ 层级稀释假设不成立或效果不明显")
                print(f"   V_{{k,last}} effect = {es_last:.3f}")
                print(f"   V_{{k,global}} effect = {es_global:.3f}")
        
        return report_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='A1: 分层 V_k 可分性诊断')
    parser.add_argument('--run_id', type=str, default=None, help='运行 ID')
    parser.add_argument('--log_dir', type=str, default='logs/step1', help='日志目录')
    
    args = parser.parse_args()
    
    # 查找运行目录
    if args.run_id:
        run_dir = os.path.join(args.log_dir, args.run_id)
    else:
        runs = sorted([d for d in os.listdir(args.log_dir) 
                       if os.path.isdir(os.path.join(args.log_dir, d))])
        if not runs:
            print("未找到任何运行记录")
            return
        run_dir = os.path.join(args.log_dir, runs[-1])
        print(f"使用最新运行: {runs[-1]}")
    
    analyzer = LayerwiseVkAnalyzer(run_dir)
    analyzer.generate_report()


if __name__ == '__main__':
    main()
