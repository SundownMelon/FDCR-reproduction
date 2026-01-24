"""
FDCR 幅值归因与尺度不变修复实验

实验 A: 幅值归因拆解（norm_g, norm_gI, I_mean, I_sparsity, shrink_ratio）
实验 B: 尺度不变异常度（score_cos, score_unit_mse, score_nmse）

使用方法:
    python fdcr_scale_attribution.py --run_id <run_id>
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ScaleAttributionAnalyzer:
    """幅值归因与尺度不变修复分析器"""
    
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
            'eps': 1e-12,
            'tau': 0.1,  # I_sparsity 阈值
            'top_p': 0.05,
        }
        
        # 构建层级切片
        self._build_slices()
        
        # 输出目录
        self.output_dir = os.path.join(run_dir, 'scale_attribution')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 结果
        self.amp_results = []
        self.scaleinv_results = []
    
    def _build_slices(self):
        """构建层级切片"""
        self.slices = {}
        layer_params = {
            'head': ['cls.weight', 'cls.bias'],
            'last_block': ['l2.weight', 'l2.bias'],
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
    
    def load_round_data(self, round_idx: int) -> Tuple[Dict, List[int]]:
        """加载一轮数据"""
        round_dir = os.path.join(self.run_dir, f'round_{round_idx:03d}')
        
        tensors = torch.load(os.path.join(round_dir, 'tensors.pt'))
        with open(os.path.join(round_dir, 'ground_truth.json'), 'r') as f:
            gt = json.load(f)
        
        logs = {}
        lr = self.manifest['config']['local_train_lr']  # 获取学习率
        
        for i in range(self.n_clients):
            client_key = f'client_{i}'
            if client_key not in tensors:
                continue
            
            t = tensors[client_key]
            logs[i] = {'is_mal': i in gt['actual_malicious_idx']}
            
            for layer in ['head', 'last_block']:
                # gI = g_weighted (已有)
                gI_key = f'g_weighted_{layer}_full'
                # I = I_minmax (已有)
                I_key = f'I_minmax_{layer}_full'
                # delta_w_true = w_new - w_prev
                dw_key = f'delta_w_true_{layer}_full'
                
                if gI_key in t:
                    logs[i][f'gI_{layer}'] = t[gI_key].float()
                if I_key in t:
                    logs[i][f'I_{layer}'] = t[I_key].float()
                if dw_key in t:
                    # 修正口径: grad = (w_prev - w_new) / lr = -delta_w / lr
                    delta_w = t[dw_key].float()
                    grad = -delta_w / lr
                    logs[i][f'g_{layer}'] = grad
        
        return logs, gt['actual_malicious_idx']
    
    # ========== 实验 A: 幅值归因 ==========
    
    def compute_amp_metrics(
        self,
        logs_t: Dict,
        K: List,
        layer: str,
        eps: float,
        tau: float = 0.1
    ) -> Dict:
        """计算幅值归因指标"""
        metrics = {}
        
        for k in K:
            if k not in logs_t:
                continue
            
            data = logs_t[k]
            g_key = f'g_{layer}'
            gI_key = f'gI_{layer}'
            I_key = f'I_{layer}'
            
            if g_key not in data or gI_key not in data or I_key not in data:
                continue
            
            g = data[g_key]
            gI = data[gI_key]
            I = data[I_key]
            
            # 1. norm_g
            norm_g = float(torch.norm(g, p=2))
            
            # 2. norm_gI
            norm_gI = float(torch.norm(gI, p=2))
            
            # 3. I_mean
            I_mean = float(I.mean())
            
            # 4. I_sparsity = frac(I < tau)
            I_sparsity = float((I < tau).float().mean())
            
            # 5. shrink_ratio = norm_gI / (norm_g + eps)
            shrink_ratio = norm_gI / (norm_g + eps)
            
            metrics[k] = {
                'norm_g': norm_g,
                'norm_gI': norm_gI,
                'I_mean': I_mean,
                'I_sparsity': I_sparsity,
                'shrink_ratio': shrink_ratio,
                'is_mal': data['is_mal']
            }
        
        return metrics
    
    # ========== 实验 B: 尺度不变异常度 ==========
    
    def compute_center(
        self,
        X: Dict[int, torch.Tensor],
        alpha: Dict[int, float],
        eps: float
    ) -> torch.Tensor:
        """计算加权中心"""
        clients = list(X.keys())
        if len(clients) == 0:
            return None
        
        sum_x = torch.zeros_like(X[clients[0]])
        sum_alpha = 0.0
        
        for k in clients:
            sum_x = sum_x + alpha[k] * X[k]
            sum_alpha += alpha[k]
        
        return sum_x / (sum_alpha + eps)
    
    def compute_scale_invariant_scores(
        self,
        logs_t: Dict,
        K: List,
        layer: str,
        weight_mode: str,
        eps: float
    ) -> Dict:
        """计算尺度不变异常度"""
        # 收集 gI
        X = {}
        labels = {}
        for k in K:
            if k not in logs_t:
                continue
            gI_key = f'gI_{layer}'
            if gI_key in logs_t[k]:
                X[k] = logs_t[k][gI_key]
                labels[k] = logs_t[k]['is_mal']
        
        if len(X) < 2:
            return {}
        
        # 权重
        if weight_mode == 'equal':
            alpha = {k: 1.0 for k in X.keys()}
        else:
            alpha = {k: 1.0 / len(X) for k in X.keys()}
        
        # 计算中心
        center = self.compute_center(X, alpha, eps)
        center_norm = torch.norm(center, p=2) + eps
        
        scores = {}
        
        for k in X.keys():
            xk = X[k]
            xk_norm = torch.norm(xk, p=2) + eps
            
            # 1. score_cos_all = 1 - cos(xk, center)
            cos_val = float(torch.dot(xk, center) / (xk_norm * center_norm))
            score_cos = 1.0 - cos_val
            
            # 2. score_unit_mse: ||xk/||xk|| - center/||center||||²
            xk_unit = xk / xk_norm
            center_unit = center / center_norm
            score_unit_mse = float(((xk_unit - center_unit) ** 2).mean())
            
            # 3. V_all: mean((xk - center)²)
            V_all = float(((xk - center) ** 2).mean())
            
            # 4. score_nmse = V_all / (||xk||² + eps)
            score_nmse = V_all / (float(xk_norm ** 2) + eps)
            
            scores[k] = {
                'score_cos_all': score_cos,
                'score_unit_mse': score_unit_mse,
                'score_nmse': score_nmse,
                'V_all': V_all,
                'is_mal': labels[k]
            }
        
        return scores
    
    # ========== 运行实验 ==========
    
    def run_experiments(self, rounds: Optional[List[int]] = None):
        """运行所有实验"""
        if rounds is None:
            rounds = list(range(self.n_rounds))
        
        print("=" * 60)
        print("FDCR 幅值归因与尺度不变修复实验")
        print("=" * 60)
        
        for round_idx in rounds:
            logs_t, malicious_idx = self.load_round_data(round_idx)
            K = list(logs_t.keys())
            
            for layer in ['head', 'last_block']:
                # 实验 A: 幅值归因
                amp_metrics = self.compute_amp_metrics(
                    logs_t, K, layer,
                    eps=self.cfg['eps'],
                    tau=self.cfg['tau']
                )
                
                for k, m in amp_metrics.items():
                    for metric_name in ['norm_g', 'norm_gI', 'I_mean', 'I_sparsity', 'shrink_ratio']:
                        self.amp_results.append({
                            'round': round_idx,
                            'slice': layer,
                            'client_id': k,
                            'is_mal': m['is_mal'],
                            'metric': metric_name,
                            'value': m[metric_name]
                        })
                
                # 实验 B: 尺度不变
                scaleinv_scores = self.compute_scale_invariant_scores(
                    logs_t, K, layer,
                    weight_mode=self.cfg['weight_mode'],
                    eps=self.cfg['eps']
                )
                
                for k, s in scaleinv_scores.items():
                    for metric_name in ['score_cos_all', 'score_unit_mse', 'score_nmse', 'V_all']:
                        self.scaleinv_results.append({
                            'round': round_idx,
                            'slice': layer,
                            'client_id': k,
                            'is_mal': s['is_mal'],
                            'metric': metric_name,
                            'value': s[metric_name]
                        })
            
            if (round_idx + 1) % 20 == 0:
                print(f"  已处理 {round_idx + 1}/{len(rounds)} 轮")
        
        # 转为 DataFrame
        self.df_amp = pd.DataFrame(self.amp_results)
        self.df_scaleinv = pd.DataFrame(self.scaleinv_results)
        
        return self.df_amp, self.df_scaleinv
    
    def compute_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 delta = mean(mal) - mean(benign)"""
        # 按 round, slice, metric, is_mal 聚合
        agg = df.groupby(['round', 'slice', 'metric', 'is_mal'])['value'].mean().reset_index()
        
        # pivot
        pivot = agg.pivot_table(
            index=['round', 'slice', 'metric'],
            columns='is_mal',
            values='value'
        ).reset_index()
        
        # 重命名
        pivot.columns = ['round', 'slice', 'metric', 'benign', 'malicious']
        pivot['delta'] = pivot['malicious'] - pivot['benign']
        
        return pivot
    
    def summarize_by_phase(self, delta_df: pd.DataFrame) -> pd.DataFrame:
        """分阶段统计"""
        phases = [
            ('前10轮', 0, 10),
            ('后50轮', 50, 100),
            ('全部', 0, 100),
        ]
        
        rows = []
        for phase_name, start, end in phases:
            phase_df = delta_df[(delta_df['round'] >= start) & (delta_df['round'] < end)]
            
            for (slice_name, metric), group in phase_df.groupby(['slice', 'metric']):
                mean_delta = group['delta'].mean()
                pos_ratio = (group['delta'] > 0).mean()
                
                rows.append({
                    'phase': phase_name,
                    'slice': slice_name,
                    'metric': metric,
                    'mean_delta': mean_delta,
                    'pos_ratio': pos_ratio
                })
        
        return pd.DataFrame(rows)
    
    def plot_amp_delta(self, delta_df: pd.DataFrame):
        """绘制幅值归因 delta 曲线"""
        for slice_name in ['head', 'last_block']:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            metrics = ['norm_g', 'norm_gI', 'shrink_ratio', 'I_mean', 'I_sparsity']
            
            for metric in metrics:
                subset = delta_df[(delta_df['slice'] == slice_name) & (delta_df['metric'] == metric)]
                if len(subset) > 0:
                    ax.plot(subset['round'], subset['delta'], label=metric, alpha=0.7)
            
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Round')
            ax.set_ylabel('Delta (Malicious - Benign)')
            ax.set_title(f'幅值归因 Delta @ {slice_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            path = os.path.join(self.output_dir, f'delta_amp_{slice_name}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"图表已保存: {path}")
    
    def plot_scaleinv_delta(self, delta_df: pd.DataFrame):
        """绘制尺度不变 delta 曲线"""
        for slice_name in ['head', 'last_block']:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            metrics = ['V_all', 'score_cos_all', 'score_unit_mse', 'score_nmse']
            colors = {'V_all': 'blue', 'score_cos_all': 'green', 'score_unit_mse': 'orange', 'score_nmse': 'red'}
            
            for metric in metrics:
                subset = delta_df[(delta_df['slice'] == slice_name) & (delta_df['metric'] == metric)]
                if len(subset) > 0:
                    ax.plot(subset['round'], subset['delta'], label=metric, 
                           color=colors.get(metric, None), alpha=0.7, linewidth=2)
            
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Round')
            ax.set_ylabel('Delta (Malicious - Benign)')
            ax.set_title(f'尺度不变异常度 Delta @ {slice_name}\n(绿=cos, 橙=unit_mse, 红=nmse, 蓝=V_all原始)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            path = os.path.join(self.output_dir, f'delta_scaleinv_{slice_name}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"图表已保存: {path}")
    
    def generate_diagnosis(self, amp_summary: pd.DataFrame, scaleinv_summary: pd.DataFrame) -> str:
        """生成诊断结论"""
        lines = []
        lines.append("=" * 60)
        lines.append("【幅值归因诊断】")
        lines.append("=" * 60)
        
        for slice_name in ['head', 'last_block']:
            # 获取后50轮数据
            late = amp_summary[(amp_summary['phase'] == '后50轮') & (amp_summary['slice'] == slice_name)]
            
            norm_g_delta = late[late['metric'] == 'norm_g']['mean_delta'].values
            shrink_delta = late[late['metric'] == 'shrink_ratio']['mean_delta'].values
            I_sparsity_delta = late[late['metric'] == 'I_sparsity']['mean_delta'].values
            
            if len(norm_g_delta) > 0 and len(shrink_delta) > 0:
                ng = norm_g_delta[0]
                sr = shrink_delta[0]
                
                lines.append(f"\n[{slice_name}]")
                lines.append(f"  norm_g delta: {ng:.6f}")
                lines.append(f"  shrink_ratio delta: {sr:.6f}")
                
                if ng < -0.01 and sr > -0.01:
                    lines.append(f"  → 主因: g 自身缩放 (update scaling)")
                elif ng > -0.01 and sr < -0.01:
                    lines.append(f"  → 主因: I 在压缩 (importance-mask)")
                elif ng < -0.01 and sr < -0.01:
                    lines.append(f"  → 叠加效应 (g 缩放 + I 压缩)")
                else:
                    lines.append(f"  → 无明显归因")
        
        lines.append("\n" + "=" * 60)
        lines.append("【尺度不变修复验证】")
        lines.append("=" * 60)
        
        for slice_name in ['head', 'last_block']:
            late = scaleinv_summary[(scaleinv_summary['phase'] == '后50轮') & (scaleinv_summary['slice'] == slice_name)]
            
            cos_delta = late[late['metric'] == 'score_cos_all']['mean_delta'].values
            cos_pos = late[late['metric'] == 'score_cos_all']['pos_ratio'].values
            v_all_delta = late[late['metric'] == 'V_all']['mean_delta'].values
            
            if len(cos_delta) > 0 and len(v_all_delta) > 0:
                lines.append(f"\n[{slice_name}]")
                lines.append(f"  V_all delta: {v_all_delta[0]:.6f}")
                lines.append(f"  score_cos delta: {cos_delta[0]:.6f} (pos_ratio: {cos_pos[0]:.1%})")
                
                if cos_delta[0] > 0 and cos_pos[0] > 0.5:
                    lines.append(f"  ✅ 修复成功! cosine 距离在后期恢复可分性")
                else:
                    lines.append(f"  ⚠️ 修复效果有限")
        
        return "\n".join(lines)
    
    def run_full_analysis(self):
        """运行完整分析"""
        print("正在运行实验...")
        self.run_experiments()
        
        print("\n正在计算 Delta...")
        delta_amp = self.compute_delta(self.df_amp)
        delta_scaleinv = self.compute_delta(self.df_scaleinv)
        
        print("正在分阶段统计...")
        summary_amp = self.summarize_by_phase(delta_amp)
        summary_scaleinv = self.summarize_by_phase(delta_scaleinv)
        
        print("正在生成图表...")
        self.plot_amp_delta(delta_amp)
        self.plot_scaleinv_delta(delta_scaleinv)
        
        # 保存结果
        self.df_amp.to_csv(os.path.join(self.output_dir, 'amp_metrics_long.csv'), index=False)
        self.df_scaleinv.to_csv(os.path.join(self.output_dir, 'scaleinv_scores_long.csv'), index=False)
        delta_amp.to_csv(os.path.join(self.output_dir, 'delta_amp_metrics.csv'), index=False)
        delta_scaleinv.to_csv(os.path.join(self.output_dir, 'delta_scaleinv.csv'), index=False)
        summary_amp.to_csv(os.path.join(self.output_dir, 'summary_amp.csv'), index=False)
        summary_scaleinv.to_csv(os.path.join(self.output_dir, 'summary_scaleinv.csv'), index=False)
        
        print(f"\n结果已保存到: {self.output_dir}")
        
        # 生成诊断
        diagnosis = self.generate_diagnosis(summary_amp, summary_scaleinv)
        print(diagnosis)
        
        with open(os.path.join(self.output_dir, 'diagnosis.txt'), 'w', encoding='utf-8') as f:
            f.write(diagnosis)
        
        return summary_amp, summary_scaleinv


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='FDCR 幅值归因与尺度不变修复')
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='logs/step1')
    parser.add_argument('--tau', type=float, default=0.1, help='I_sparsity 阈值')
    
    args = parser.parse_args()
    
    if args.run_id:
        run_dir = os.path.join(args.log_dir, args.run_id)
    else:
        runs = sorted([d for d in os.listdir(args.log_dir) 
                       if os.path.isdir(os.path.join(args.log_dir, d))])
        run_dir = os.path.join(args.log_dir, runs[-1])
        print(f"使用最新运行: {runs[-1]}")
    
    cfg = {'tau': args.tau, 'eps': 1e-12, 'weight_mode': 'equal'}
    
    analyzer = ScaleAttributionAnalyzer(run_dir, cfg)
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
