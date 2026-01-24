"""
FDCR 诊断实验：统一分析框架

实验 1: LOO 版 V_k（定位"中心/自包含"）
实验 2: Norm/Cosine 分解（定位"尺度 vs 方向"）
实验 3: 三种 x 全套对照
实验 4: Mask 相似度（定位"掩码几何"）

使用方法:
    python fdcr_diagnostics.py --run_id <run_id>
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class FDCRDiagnostics:
    """FDCR 诊断分析器"""
    
    def __init__(self, run_dir: str, cfg: Dict = None):
        self.run_dir = run_dir
        
        # 加载元数据
        with open(os.path.join(run_dir, 'run_manifest.json'), 'r') as f:
            self.manifest = json.load(f)
        with open(os.path.join(run_dir, 'param_index_map.json'), 'r') as f:
            self.param_map = json.load(f)
        
        self.n_clients = self.manifest['config']['parti_num']
        self.n_rounds = self.manifest['config']['communication_epoch']
        
        # 配置
        self.cfg = cfg or {
            'weight_mode': 'equal',
            'center_mode': 'allmean',
            'use_participants': True,
            'eps': 1e-12,
            'top_p': 0.1,  # 实验4 的 top-p%
        }
        
        # 构建层级索引
        self._build_slices()
        
        # 输出目录
        self.output_dir = os.path.join(run_dir, 'diagnostics')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 结果长表
        self.results = []
    
    def _build_slices(self):
        """构建层级切片索引"""
        self.slices = {}
        
        layer_params = {
            'head': ['cls.weight', 'cls.bias'],
            'last_block': ['l2.weight', 'l2.bias'],
            'head_last': ['cls.weight', 'cls.bias', 'l2.weight', 'l2.bias'],
        }
        
        for layer_name, param_names in layer_params.items():
            indices = []
            for pname in param_names:
                if pname in self.param_map:
                    info = self.param_map[pname]
                    start = info['offset']
                    end = start + info['length']
                    indices.extend(range(start, end))
            self.slices[layer_name] = torch.tensor(indices, dtype=torch.long)
        
        self.slices['global'] = torch.arange(self.manifest['total_params'])
    
    def load_round_data(self, round_idx: int) -> Tuple[Dict, List[int]]:
        """加载一轮数据，返回统一格式"""
        round_dir = os.path.join(self.run_dir, f'round_{round_idx:03d}')
        
        tensors = torch.load(os.path.join(round_dir, 'tensors.pt'))
        with open(os.path.join(round_dir, 'ground_truth.json'), 'r') as f:
            gt = json.load(f)
        
        # 转换为统一格式
        logs = {}
        for i in range(self.n_clients):
            client_key = f'client_{i}'
            if client_key not in tensors:
                continue
            
            t = tensors[client_key]
            
            # 构建各层的向量
            logs[i] = {
                'is_mal': i in gt['actual_malicious_idx'],
            }
            
            # 对于 head 和 last_block，使用完整向量
            for layer in ['head', 'last_block']:
                g_key = f'g_weighted_{layer}_full'
                I_key = f'I_minmax_{layer}_full'
                dw_key = f'delta_w_true_{layer}_full'
                
                if g_key in t:
                    logs[i][f'gI_{layer}'] = t[g_key].float()
                if I_key in t:
                    logs[i][f'I_{layer}'] = t[I_key].float()
                if dw_key in t:
                    logs[i][f'g_{layer}'] = t[dw_key].float()  # delta_w 作为 g
        
        return logs, gt['actual_malicious_idx']
    
    # ========== 实验 1: LOO V_k ==========
    
    def compute_V_all_and_loo(
        self,
        X: Dict[int, torch.Tensor],
        alpha: Dict[int, float],
        eps: float = 1e-12
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        计算 all-mean 距离和 LOO 距离
        
        X: {k: Tensor} 每个客户端的向量
        alpha: {k: float} 权重
        """
        clients = list(X.keys())
        
        # 计算加权和
        sum_x = torch.zeros_like(X[clients[0]])
        sum_alpha = 0.0
        for k in clients:
            sum_x = sum_x + alpha[k] * X[k]
            sum_alpha += alpha[k]
        
        # all-mean center
        center = sum_x / (sum_alpha + eps)
        
        V_all = {}
        V_loo = {}
        
        for k in clients:
            # V_all: MSE to all-mean center
            diff = X[k] - center
            V_all[k] = float((diff ** 2).mean())
            
            # V_loo: MSE to LOO center
            loo_sum = sum_x - alpha[k] * X[k]
            loo_alpha = sum_alpha - alpha[k]
            center_loo = loo_sum / (loo_alpha + eps)
            diff_loo = X[k] - center_loo
            V_loo[k] = float((diff_loo ** 2).mean())
        
        return V_all, V_loo
    
    # ========== 实验 2: Norm/Cosine ==========
    
    def compute_norm_and_cos(
        self,
        X: Dict[int, torch.Tensor],
        alpha: Dict[int, float],
        eps: float = 1e-12
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
        """
        计算幅值和方向相似度
        """
        clients = list(X.keys())
        
        # 计算中心
        sum_x = torch.zeros_like(X[clients[0]])
        sum_alpha = 0.0
        for k in clients:
            sum_x = sum_x + alpha[k] * X[k]
            sum_alpha += alpha[k]
        center = sum_x / (sum_alpha + eps)
        center_norm = torch.norm(center, p=2) + eps
        
        norm = {}
        cos_all = {}
        cos_loo = {}
        
        for k in clients:
            xk = X[k]
            xk_norm = torch.norm(xk, p=2) + eps
            
            # norm
            norm[k] = float(xk_norm)
            
            # cos_all
            cos_all[k] = float(torch.dot(xk, center) / (xk_norm * center_norm))
            
            # cos_loo
            loo_sum = sum_x - alpha[k] * X[k]
            loo_alpha = sum_alpha - alpha[k]
            center_loo = loo_sum / (loo_alpha + eps)
            center_loo_norm = torch.norm(center_loo, p=2) + eps
            cos_loo[k] = float(torch.dot(xk, center_loo) / (xk_norm * center_loo_norm))
        
        return norm, cos_all, cos_loo
    
    # ========== 实验 4: Mask 相似度 ==========
    
    def compute_top_p_sets(
        self,
        I_dict: Dict[int, torch.Tensor],
        top_p: float = 0.1
    ) -> Dict[int, Set[int]]:
        """计算每个客户端的 top-p% 重要参数集合"""
        top_sets = {}
        for k, I in I_dict.items():
            k_val = max(1, int(len(I) * top_p))
            _, indices = torch.topk(I, k_val)
            top_sets[k] = set(indices.tolist())
        return top_sets
    
    def avg_jaccard_by_group(
        self,
        top_sets: Dict[int, Set[int]],
        labels: Dict[int, bool],
        num_pairs: int = 200,
        seed: int = 0
    ) -> Dict[str, float]:
        """计算分组 Jaccard 相似度"""
        import random
        random.seed(seed)
        
        benign = [k for k, is_mal in labels.items() if not is_mal]
        malicious = [k for k, is_mal in labels.items() if is_mal]
        
        def jaccard(s1, s2):
            if len(s1) == 0 and len(s2) == 0:
                return 1.0
            inter = len(s1 & s2)
            union = len(s1 | s2)
            return inter / union if union > 0 else 0.0
        
        def sample_pairs(group, n):
            pairs = list(combinations(group, 2))
            if len(pairs) <= n:
                return pairs
            return random.sample(pairs, n)
        
        results = {}
        
        # BB
        bb_pairs = sample_pairs(benign, num_pairs)
        if bb_pairs:
            results['BB'] = np.mean([jaccard(top_sets[i], top_sets[j]) for i, j in bb_pairs])
        else:
            results['BB'] = 0.0
        
        # MM
        mm_pairs = sample_pairs(malicious, num_pairs)
        if mm_pairs:
            results['MM'] = np.mean([jaccard(top_sets[i], top_sets[j]) for i, j in mm_pairs])
        else:
            results['MM'] = 0.0
        
        # BM
        bm_pairs = [(b, m) for b in benign for m in malicious]
        if len(bm_pairs) > num_pairs:
            bm_pairs = random.sample(bm_pairs, num_pairs)
        if bm_pairs:
            results['BM'] = np.mean([jaccard(top_sets[i], top_sets[j]) for i, j in bm_pairs])
        else:
            results['BM'] = 0.0
        
        return results
    
    # ========== 主运行函数 ==========
    
    def run_all_experiments(self, rounds: Optional[List[int]] = None):
        """运行所有实验"""
        if rounds is None:
            rounds = list(range(self.n_rounds))
        
        x_types = ['gI', 'I', 'g']  # 三种特征对象
        slices_to_use = ['head', 'last_block', 'head_last']
        
        print("=" * 60)
        print("FDCR 诊断实验")
        print("=" * 60)
        
        for round_idx in rounds:
            logs, malicious_idx = self.load_round_data(round_idx)
            
            if len(logs) == 0:
                continue
            
            # 权重
            alpha = {k: 1.0 for k in logs.keys()}
            labels = {k: logs[k]['is_mal'] for k in logs.keys()}
            
            for x_type in x_types:
                for slice_name in slices_to_use:
                    key = f'{x_type}_{slice_name}'
                    
                    # 获取该层的向量
                    X = {}
                    for k, data in logs.items():
                        if key in data:
                            X[k] = data[key]
                    
                    if len(X) < 2:
                        continue
                    
                    # 实验 1: V_all, V_loo
                    V_all, V_loo = self.compute_V_all_and_loo(X, alpha, self.cfg['eps'])
                    
                    # 实验 2: norm, cos_all, cos_loo
                    norm, cos_all, cos_loo = self.compute_norm_and_cos(X, alpha, self.cfg['eps'])
                    
                    # 记录结果
                    for k in X.keys():
                        group = 'malicious' if labels[k] else 'benign'
                        
                        for metric, value in [
                            ('V_all', V_all[k]),
                            ('V_loo', V_loo[k]),
                            ('norm', norm[k]),
                            ('cos_all', cos_all[k]),
                            ('cos_loo', cos_loo[k]),
                        ]:
                            self.results.append({
                                'round': round_idx,
                                'x_type': x_type,
                                'slice': slice_name,
                                'metric': metric,
                                'group': group,
                                'client': k,
                                'value': value
                            })
            
            # 实验 4: Mask 相似度（仅对 I）
            for slice_name in slices_to_use:
                key = f'I_{slice_name}'
                I_dict = {k: logs[k][key] for k in logs.keys() if key in logs[k]}
                
                if len(I_dict) >= 2:
                    top_sets = self.compute_top_p_sets(I_dict, self.cfg['top_p'])
                    jaccard = self.avg_jaccard_by_group(top_sets, labels)
                    
                    for pair_type, val in jaccard.items():
                        self.results.append({
                            'round': round_idx,
                            'x_type': 'I',
                            'slice': slice_name,
                            'metric': f'Jaccard_{pair_type}',
                            'group': pair_type,
                            'client': -1,
                            'value': val
                        })
            
            if (round_idx + 1) % 20 == 0:
                print(f"  已处理 {round_idx + 1}/{len(rounds)} 轮")
        
        # 转换为 DataFrame
        self.df = pd.DataFrame(self.results)
        return self.df
    
    def compute_delta_by_round(self) -> pd.DataFrame:
        """计算每轮的 delta (malicious - benign)"""
        # 过滤出非 Jaccard 指标
        df_main = self.df[~self.df['metric'].str.startswith('Jaccard')]
        
        # 按 round, x_type, slice, metric, group 聚合
        agg = df_main.groupby(['round', 'x_type', 'slice', 'metric', 'group'])['value'].mean().reset_index()
        
        # pivot 得到 benign 和 malicious 列
        pivot = agg.pivot_table(
            index=['round', 'x_type', 'slice', 'metric'],
            columns='group',
            values='value'
        ).reset_index()
        
        # 计算 delta
        pivot['delta'] = pivot['malicious'] - pivot['benign']
        
        return pivot
    
    def summarize_by_phase(self, delta_df: pd.DataFrame) -> pd.DataFrame:
        """分阶段统计"""
        phases = [
            ('前10轮', 0, 10),
            ('前50轮', 0, 50),
            ('后50轮', 50, 100),
            ('全部', 0, 100),
        ]
        
        summary_rows = []
        for phase_name, start, end in phases:
            phase_df = delta_df[(delta_df['round'] >= start) & (delta_df['round'] < end)]
            
            for (x_type, slice_name, metric), group in phase_df.groupby(['x_type', 'slice', 'metric']):
                mean_delta = group['delta'].mean()
                pos_ratio = (group['delta'] > 0).mean()
                
                summary_rows.append({
                    'phase': phase_name,
                    'x_type': x_type,
                    'slice': slice_name,
                    'metric': metric,
                    'mean_delta': mean_delta,
                    'pos_ratio': pos_ratio,
                })
        
        return pd.DataFrame(summary_rows)
    
    def plot_delta_trends(self, delta_df: pd.DataFrame, save_dir: str = None):
        """绘制 delta 随轮次变化趋势"""
        if save_dir is None:
            save_dir = self.output_dir
        
        # 对 gI 的各层绘制
        x_type = 'gI'
        metrics = ['V_all', 'V_loo', 'norm', 'cos_all']
        slices = ['head', 'last_block']
        
        fig, axes = plt.subplots(len(metrics), len(slices), figsize=(12, 12))
        
        for i, metric in enumerate(metrics):
            for j, slice_name in enumerate(slices):
                ax = axes[i, j]
                
                subset = delta_df[
                    (delta_df['x_type'] == x_type) &
                    (delta_df['slice'] == slice_name) &
                    (delta_df['metric'] == metric)
                ]
                
                if len(subset) == 0:
                    ax.set_title(f'{metric} @ {slice_name}\n(无数据)')
                    continue
                
                rounds = subset['round'].values
                deltas = subset['delta'].values
                
                colors = ['green' if d > 0 else 'red' for d in deltas]
                ax.bar(rounds, deltas, color=colors, alpha=0.7, width=1)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax.set_xlabel('Round')
                ax.set_ylabel('Delta (M-B)')
                ax.set_title(f'{metric} @ {slice_name}')
        
        plt.suptitle(f'gI 各指标 Delta 趋势 (绿=恶意更大, 红=恶意更小)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'delta_trends_gI.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"图表已保存: {os.path.join(save_dir, 'delta_trends_gI.png')}")
    
    def plot_jaccard_trends(self, save_dir: str = None):
        """绘制 Jaccard 相似度趋势"""
        if save_dir is None:
            save_dir = self.output_dir
        
        jaccard_df = self.df[self.df['metric'].str.startswith('Jaccard')]
        
        if len(jaccard_df) == 0:
            print("无 Jaccard 数据")
            return
        
        slices = jaccard_df['slice'].unique()
        
        fig, axes = plt.subplots(1, len(slices), figsize=(5 * len(slices), 4))
        if len(slices) == 1:
            axes = [axes]
        
        for ax, slice_name in zip(axes, slices):
            subset = jaccard_df[jaccard_df['slice'] == slice_name]
            
            for pair_type in ['BB', 'MM', 'BM']:
                metric = f'Jaccard_{pair_type}'
                data = subset[subset['metric'] == metric]
                if len(data) > 0:
                    ax.plot(data['round'], data['value'], label=pair_type, alpha=0.7)
            
            ax.set_xlabel('Round')
            ax.set_ylabel('Jaccard Similarity')
            ax.set_title(f'Jaccard @ {slice_name}')
            ax.legend()
        
        plt.suptitle('Top-p% 参数集合 Jaccard 相似度')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'jaccard_trends.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"图表已保存: {os.path.join(save_dir, 'jaccard_trends.png')}")
    
    def generate_diagnosis(self, summary_df: pd.DataFrame) -> str:
        """生成诊断结论"""
        lines = []
        lines.append("=" * 60)
        lines.append("【诊断结论】")
        lines.append("=" * 60)
        
        # 检查 V_all vs V_loo
        for slice_name in ['head', 'last_block']:
            v_all_late = summary_df[
                (summary_df['phase'] == '后50轮') &
                (summary_df['x_type'] == 'gI') &
                (summary_df['slice'] == slice_name) &
                (summary_df['metric'] == 'V_all')
            ]['mean_delta'].values
            
            v_loo_late = summary_df[
                (summary_df['phase'] == '后50轮') &
                (summary_df['x_type'] == 'gI') &
                (summary_df['slice'] == slice_name) &
                (summary_df['metric'] == 'V_loo')
            ]['mean_delta'].values
            
            if len(v_all_late) > 0 and len(v_loo_late) > 0:
                if v_all_late[0] < 0 and v_loo_late[0] > 0:
                    lines.append(f"✅ [{slice_name}] 中心/自包含主导：V_all后期为负({v_all_late[0]:.4f})，但V_loo回正({v_loo_late[0]:.4f})")
                elif v_all_late[0] < 0 and v_loo_late[0] < 0:
                    lines.append(f"⚠️ [{slice_name}] V_loo仍为负，问题不在中心定义")
        
        # 检查 norm vs cos
        for slice_name in ['head', 'last_block']:
            norm_late = summary_df[
                (summary_df['phase'] == '后50轮') &
                (summary_df['x_type'] == 'gI') &
                (summary_df['slice'] == slice_name) &
                (summary_df['metric'] == 'norm')
            ]['mean_delta'].values
            
            cos_late = summary_df[
                (summary_df['phase'] == '后50轮') &
                (summary_df['x_type'] == 'gI') &
                (summary_df['slice'] == slice_name) &
                (summary_df['metric'] == 'cos_all')
            ]['mean_delta'].values
            
            if len(norm_late) > 0 and len(cos_late) > 0:
                if norm_late[0] < -0.1 and abs(cos_late[0]) < 0.1:
                    lines.append(f"✅ [{slice_name}] 尺度缩放主导：norm后期显著负({norm_late[0]:.4f})，cos接近0({cos_late[0]:.4f})")
                elif cos_late[0] > 0.1:
                    lines.append(f"✅ [{slice_name}] 方向贴均值主导：cos后期为正({cos_late[0]:.4f})")
        
        return "\n".join(lines)
    
    def run_full_analysis(self):
        """运行完整分析"""
        print("正在运行所有诊断实验...")
        self.run_all_experiments()
        
        print("\n正在计算 Delta...")
        delta_df = self.compute_delta_by_round()
        
        print("正在分阶段统计...")
        summary_df = self.summarize_by_phase(delta_df)
        
        print("正在生成图表...")
        self.plot_delta_trends(delta_df)
        self.plot_jaccard_trends()
        
        # 保存结果
        self.df.to_csv(os.path.join(self.output_dir, 'all_results.csv'), index=False)
        delta_df.to_csv(os.path.join(self.output_dir, 'delta_by_round.csv'), index=False)
        summary_df.to_csv(os.path.join(self.output_dir, 'summary_by_phase.csv'), index=False)
        
        print(f"\n结果已保存到: {self.output_dir}")
        
        # 生成诊断
        diagnosis = self.generate_diagnosis(summary_df)
        print(diagnosis)
        
        with open(os.path.join(self.output_dir, 'diagnosis.txt'), 'w', encoding='utf-8') as f:
            f.write(diagnosis)
        
        return summary_df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='FDCR 诊断实验')
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='logs/step1')
    parser.add_argument('--top_p', type=float, default=0.1, help='Top-p%% for Jaccard')
    
    args = parser.parse_args()
    
    if args.run_id:
        run_dir = os.path.join(args.log_dir, args.run_id)
    else:
        runs = sorted([d for d in os.listdir(args.log_dir) 
                       if os.path.isdir(os.path.join(args.log_dir, d))])
        run_dir = os.path.join(args.log_dir, runs[-1])
        print(f"使用最新运行: {runs[-1]}")
    
    cfg = {'top_p': args.top_p, 'eps': 1e-12}
    
    diagnostics = FDCRDiagnostics(run_dir, cfg)
    diagnostics.run_full_analysis()


if __name__ == '__main__':
    main()
