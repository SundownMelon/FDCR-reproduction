"""
Step 2: FDCR 信号体检分析

三板斧诊断：
1. 层级分解可分性 - 信号在哪个层最明显
2. 距离谱分析 - bb vs bm 的分离度
3. 替代归一化 - min-max 是否是可分性崩溃的罪魁祸首

使用方法:
    python analyze_step1_signals.py --run_id <run_id>
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Step2Analyzer:
    """Step 2 信号体检分析器"""
    
    def __init__(self, run_dir: str):
        """
        初始化分析器。
        
        Args:
            run_dir: Step 1 日志输出目录
        """
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
        
        # 创建输出目录
        self.output_dir = os.path.join(run_dir, 'analysis')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_round_data(self, round_idx: int) -> Tuple[Dict, Dict, List[int]]:
        """
        加载一轮的数据。
        
        Returns:
            (tensors, stats, malicious_idx)
        """
        round_dir = os.path.join(self.run_dir, f'round_{round_idx:03d}')
        
        tensors = torch.load(os.path.join(round_dir, 'tensors.pt'))
        
        with open(os.path.join(round_dir, 'stats.json'), 'r') as f:
            stats = json.load(f)
        with open(os.path.join(round_dir, 'ground_truth.json'), 'r') as f:
            gt = json.load(f)
            
        return tensors, stats, gt['actual_malicious_idx']
    
    def compute_cosine_similarity(self, v1: torch.Tensor, v2: torch.Tensor) -> float:
        """计算余弦相似度"""
        v1 = v1.float().flatten()
        v2 = v2.float().flatten()
        return float(torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)))
    
    def compute_spearman_correlation(self, v1: torch.Tensor, v2: torch.Tensor) -> float:
        """计算 Spearman 秩相关"""
        v1 = v1.float().flatten().numpy()
        v2 = v2.float().flatten().numpy()
        corr, _ = scipy_stats.spearmanr(v1, v2)
        return float(corr) if not np.isnan(corr) else 0.0
    
    def compute_topk_jaccard(self, idx1: torch.Tensor, idx2: torch.Tensor) -> float:
        """计算 top-k Jaccard 相似度"""
        set1 = set(idx1.numpy().tolist())
        set2 = set(idx2.numpy().tolist())
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def analyze_layer_separability(
        self,
        rounds: Optional[List[int]] = None,
        object_names: List[str] = ['I_minmax', 'F_raw'],
        layers: List[str] = ['global', 'head', 'last_block']
    ) -> Dict:
        """
        体检一：层级分解可分性分析
        
        对每个层级，计算 bb (benign-benign) vs bm (benign-malicious) 的相似度分布。
        """
        if rounds is None:
            rounds = list(range(min(10, self.n_rounds)))  # 默认前10轮
        
        results = defaultdict(lambda: defaultdict(lambda: {'bb': [], 'bm': []}))
        
        for round_idx in rounds:
            tensors, stats, malicious_idx = self.load_round_data(round_idx)
            malicious_set = set(malicious_idx)
            
            for obj_name in object_names:
                for layer in layers:
                    # 获取每个客户端的向量
                    client_vectors = {}
                    for i in range(self.n_clients):
                        client_key = f'client_{i}'
                        if layer == 'global':
                            key = f'{obj_name}_topk_val'
                        else:
                            key = f'{obj_name}_{layer}_full'
                        
                        if client_key in tensors and key in tensors[client_key]:
                            client_vectors[i] = tensors[client_key][key]
                    
                    # 计算两两相似度
                    for i in range(self.n_clients):
                        for j in range(i + 1, self.n_clients):
                            if i not in client_vectors or j not in client_vectors:
                                continue
                                
                            v1, v2 = client_vectors[i], client_vectors[j]
                            cos_sim = self.compute_cosine_similarity(v1, v2)
                            
                            # 判断是 bb 还是 bm
                            i_benign = i not in malicious_set
                            j_benign = j not in malicious_set
                            
                            if i_benign and j_benign:
                                results[obj_name][layer]['bb'].append(cos_sim)
                            elif i_benign or j_benign:  # 一个良性一个恶意
                                results[obj_name][layer]['bm'].append(cos_sim)
                            # mm (malicious-malicious) 暂不统计
        
        # 计算统计量
        summary = {}
        for obj_name in object_names:
            summary[obj_name] = {}
            for layer in layers:
                bb = results[obj_name][layer]['bb']
                bm = results[obj_name][layer]['bm']
                
                if len(bb) == 0 or len(bm) == 0:
                    continue
                    
                mean_bb, std_bb = np.mean(bb), np.std(bb)
                mean_bm, std_bm = np.mean(bm), np.std(bm)
                delta = mean_bb - mean_bm
                pooled_std = np.sqrt((std_bb**2 + std_bm**2) / 2)
                effect_size = delta / pooled_std if pooled_std > 0 else 0
                
                summary[obj_name][layer] = {
                    'mean_bb': mean_bb,
                    'std_bb': std_bb,
                    'mean_bm': mean_bm,
                    'std_bm': std_bm,
                    'delta': delta,
                    'effect_size': effect_size,
                    'n_bb': len(bb),
                    'n_bm': len(bm)
                }
        
        return {'raw': dict(results), 'summary': summary}
    
    def analyze_minmax_impact(
        self,
        rounds: Optional[List[int]] = None
    ) -> Dict:
        """
        体检三：分析 min-max 归一化的影响
        
        对比 F_raw 和 I_minmax 的可分性差异
        """
        if rounds is None:
            rounds = list(range(min(20, self.n_rounds)))
        
        # 收集每轮的 min_F / max_F
        minmax_info = defaultdict(list)
        
        for round_idx in rounds:
            tensors, stats, malicious_idx = self.load_round_data(round_idx)
            malicious_set = set(malicious_idx)
            
            for i in range(self.n_clients):
                client_key = f'client_{i}'
                if client_key in tensors:
                    min_F = tensors[client_key].get('min_F', 0)
                    max_F = tensors[client_key].get('max_F', 1)
                    is_malicious = i in malicious_set
                    
                    minmax_info['round'].append(round_idx)
                    minmax_info['client'].append(i)
                    minmax_info['min_F'].append(min_F)
                    minmax_info['max_F'].append(max_F)
                    minmax_info['range_F'].append(max_F - min_F)
                    minmax_info['is_malicious'].append(is_malicious)
        
        # 计算良性 vs 恶意的 range 差异
        benign_ranges = [r for r, m in zip(minmax_info['range_F'], minmax_info['is_malicious']) if not m]
        malicious_ranges = [r for r, m in zip(minmax_info['range_F'], minmax_info['is_malicious']) if m]
        
        return {
            'minmax_info': dict(minmax_info),
            'summary': {
                'benign_range_mean': np.mean(benign_ranges) if benign_ranges else 0,
                'benign_range_std': np.std(benign_ranges) if benign_ranges else 0,
                'malicious_range_mean': np.mean(malicious_ranges) if malicious_ranges else 0,
                'malicious_range_std': np.std(malicious_ranges) if malicious_ranges else 0,
            }
        }
    
    def plot_layer_separability(self, analysis_result: Dict, save_path: Optional[str] = None):
        """绘制层级可分性对比图"""
        summary = analysis_result['summary']
        
        fig, axes = plt.subplots(1, len(summary), figsize=(5 * len(summary), 4))
        if len(summary) == 1:
            axes = [axes]
        
        for ax, (obj_name, layer_data) in zip(axes, summary.items()):
            layers = list(layer_data.keys())
            effect_sizes = [layer_data[l]['effect_size'] for l in layers]
            
            colors = ['green' if es > 0.5 else 'orange' if es > 0.2 else 'red' for es in effect_sizes]
            
            ax.bar(layers, effect_sizes, color=colors, alpha=0.7)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect')
            ax.axhline(y=0.2, color='gray', linestyle=':', alpha=0.5, label='Small effect')
            ax.set_ylabel('Effect Size (bb-bm)')
            ax.set_title(f'{obj_name} 层级可分性')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'layer_separability.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_report(self) -> str:
        """生成完整的分析报告"""
        print("=" * 60)
        print("Step 2: FDCR 信号体检分析")
        print("=" * 60)
        print(f"Run ID: {self.manifest['run_id']}")
        print(f"数据集: {self.manifest['config']['dataset']}")
        print(f"攻击类型: {self.manifest['attack']['type']}")
        print(f"Non-IID (β): {self.manifest['config']['beta']}")
        print(f"总轮数: {self.n_rounds}")
        print()
        
        # 1. 层级可分性分析
        print("【体检一：层级分解可分性】")
        layer_result = self.analyze_layer_separability()
        
        for obj_name, layer_data in layer_result['summary'].items():
            print(f"\n{obj_name}:")
            for layer, data in layer_data.items():
                print(f"  {layer:15s} | bb={data['mean_bb']:.4f}±{data['std_bb']:.4f} | "
                      f"bm={data['mean_bm']:.4f}±{data['std_bm']:.4f} | "
                      f"Δ={data['delta']:.4f} | effect={data['effect_size']:.3f}")
        
        # 保存图表
        fig_path = self.plot_layer_separability(layer_result)
        print(f"\n层级可分性图表已保存: {fig_path}")
        
        # 2. min-max 影响分析
        print("\n【体检三：min-max 归一化影响】")
        minmax_result = self.analyze_minmax_impact()
        
        print(f"良性客户端 Fisher range: {minmax_result['summary']['benign_range_mean']:.6f} ± "
              f"{minmax_result['summary']['benign_range_std']:.6f}")
        print(f"恶意客户端 Fisher range: {minmax_result['summary']['malicious_range_mean']:.6f} ± "
              f"{minmax_result['summary']['malicious_range_std']:.6f}")
        
        # 保存结果
        results = {
            'layer_separability': layer_result,
            'minmax_impact': minmax_result
        }
        
        report_path = os.path.join(self.output_dir, 'step2_results.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            # 将 numpy 类型转换为 Python 原生类型
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert(v) for v in obj]
                return obj
            
            json.dump(convert(results), f, indent=2, ensure_ascii=False)
        
        print(f"\n分析结果已保存: {report_path}")
        print("=" * 60)
        
        return report_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Step 2: FDCR 信号体检分析')
    parser.add_argument('--run_id', type=str, default=None, help='运行 ID（默认使用最新）')
    parser.add_argument('--log_dir', type=str, default='logs/step1', help='日志目录')
    
    args = parser.parse_args()
    
    # 查找运行目录
    if args.run_id:
        run_dir = os.path.join(args.log_dir, args.run_id)
    else:
        # 使用最新的运行
        runs = sorted([d for d in os.listdir(args.log_dir) if os.path.isdir(os.path.join(args.log_dir, d))])
        if not runs:
            print("未找到任何运行记录")
            return
        run_dir = os.path.join(args.log_dir, runs[-1])
        print(f"使用最新运行: {runs[-1]}")
    
    analyzer = Step2Analyzer(run_dir)
    analyzer.generate_report()


if __name__ == '__main__':
    main()
