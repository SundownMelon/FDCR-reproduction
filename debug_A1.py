"""
可视化 V_k 随轮次的变化趋势

验证：恶意客户端的 V_k 是否随训练逐渐变小？
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

run_dir = 'logs/step1/20260122_165532'
n_clients = 10
n_rounds = 100

# 收集每轮的数据
round_data = {'round': [], 'mean_benign': [], 'mean_malicious': [], 'delta': []}

for round_idx in range(n_rounds):
    round_path = os.path.join(run_dir, f'round_{round_idx:03d}')
    
    tensors = torch.load(os.path.join(round_path, 'tensors.pt'))
    with open(os.path.join(round_path, 'ground_truth.json'), 'r') as f:
        gt = json.load(f)
    
    malicious_set = set(gt['actual_malicious_idx'])
    V_k = tensors['decision']['V_k'].squeeze()
    
    V_benign = [float(V_k[i]) for i in range(n_clients) if i not in malicious_set]
    V_malicious = [float(V_k[i]) for i in range(n_clients) if i in malicious_set]
    
    round_data['round'].append(round_idx)
    round_data['mean_benign'].append(np.mean(V_benign))
    round_data['mean_malicious'].append(np.mean(V_malicious))
    round_data['delta'].append(np.mean(V_malicious) - np.mean(V_benign))

# 绘图
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 图1：V_k 均值随轮次变化
ax1 = axes[0]
ax1.plot(round_data['round'], round_data['mean_benign'], 'b-', label='Benign', alpha=0.7)
ax1.plot(round_data['round'], round_data['mean_malicious'], 'r-', label='Malicious', alpha=0.7)
ax1.set_xlabel('Round')
ax1.set_ylabel('Mean V_k')
ax1.set_title('V_k 均值随轮次变化')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2：Delta (M-B) 随轮次变化
ax2 = axes[1]
colors = ['green' if d > 0 else 'red' for d in round_data['delta']]
ax2.bar(round_data['round'], round_data['delta'], color=colors, alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax2.set_xlabel('Round')
ax2.set_ylabel('Delta (Malicious - Benign)')
ax2.set_title('V_k 可分性随轮次变化\n(绿色=恶意更大=可检测, 红色=恶意更小=难检测)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(run_dir, 'analysis_A1', 'Vk_trend_by_round.png'), dpi=150, bbox_inches='tight')
print(f"图表已保存")

# 统计不同阶段的平均值
print("\n【分阶段统计】")
for start, end, name in [(0, 10, '前10轮'), (0, 50, '前50轮'), (50, 100, '后50轮'), (0, 100, '所有100轮')]:
    deltas = round_data['delta'][start:end]
    print(f"{name}: 平均Delta = {np.mean(deltas):.4f}, 正Delta比例 = {sum(1 for d in deltas if d > 0) / len(deltas):.1%}")
