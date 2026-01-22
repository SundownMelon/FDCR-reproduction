"""
FDCR 管线显微镜 - 日志记录模块

Step 1: 为 FDCR 的关键中间量提供完整记录，用于诊断 Fisher 信息的真实价值。

核心四件套：
- F_raw: 未归一化的 Fisher 信息向量  
- I_minmax: min-max 归一化后的重要性向量
- delta_w_true: 客户端更新 (w_k - w_global)，直接从模型计算
- g_weighted: 加权更新向量 g_k ⊙ I_k

层级切片协议 (SimpleCNN):
- L0: classifier (cls.*) vs backbone (其余)
- L1: head (cls.*) + last_block (l2.*) + l1 (l1.*)
- L2: 每个参数张量独立
"""

import os
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict


class FDCRLogger:
    """
    FDCR 管线显微镜日志记录器。
    
    在每轮通信中记录每个客户端的关键中间量，
    支持层级切片、top-k 稀疏表示和统计量计算。
    """
    
    # SimpleCNN 层级切片定义
    LAYER_SLICES = {
        'L0': {
            'classifier': ['cls.weight', 'cls.bias'],
            'backbone': ['feats.conv1.weight', 'feats.conv1.bias', 
                        'feats.conv2.weight', 'feats.conv2.bias',
                        'feats.fc1.weight', 'feats.fc1.bias',
                        'feats.fc2.weight', 'feats.fc2.bias',
                        'l1.weight', 'l1.bias', 'l2.weight', 'l2.bias']
        },
        'L1': {
            'head': ['cls.weight', 'cls.bias'],
            'last_block': ['l2.weight', 'l2.bias'],
            'l1': ['l1.weight', 'l1.bias']
        },
        # L2 由 param_index_map 动态生成
    }
    
    # 需要保存全量的层
    FULL_SAVE_LAYERS = ['head', 'last_block']
    
    def __init__(
        self,
        output_dir: str,
        cfg: Any,
        reference_net: torch.nn.Module,
        run_id: Optional[str] = None,
        top_k: int = 512
    ):
        """
        初始化日志记录器。
        
        Args:
            output_dir: 日志输出目录根路径
            cfg: 实验配置对象
            reference_net: 参考模型（用于生成参数索引映射）
            run_id: 运行标识符（默认使用时间戳）
            top_k: 保存的 top-k 数量
        """
        self.top_k = top_k
        self.cfg = cfg
        
        # 生成 run_id
        if run_id is None:
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_id = run_id
        
        # 创建输出目录
        self.output_dir = os.path.join(output_dir, 'step1', run_id)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 生成并保存参数索引映射
        self.param_index_map = self._build_param_index_map(reference_net)
        self._save_param_index_map()
        
        # 保存层级切片定义
        self._save_layer_slices()
        
        # 保存 run manifest
        self._save_manifest(cfg)
        
        # 每轮数据缓存
        self._round_data = {}
        
    def _build_param_index_map(self, net: torch.nn.Module) -> OrderedDict:
        """
        构建参数索引映射。
        
        为每个参数张量记录：offset, length, shape，
        确保扁平化顺序固定且可追溯。
        
        Args:
            net: 神经网络模型
            
        Returns:
            OrderedDict: 参数名 -> {offset, length, shape} 的有序字典
        """
        param_map = OrderedDict()
        offset = 0
        
        for name, param in net.named_parameters():
            length = param.numel()
            param_map[name] = {
                'offset': offset,
                'length': length,
                'shape': list(param.shape)
            }
            offset += length
            
        self.total_params = offset
        return param_map
    
    def _save_param_index_map(self):
        """保存参数索引映射到 JSON 文件。"""
        path = os.path.join(self.output_dir, 'param_index_map.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dict(self.param_index_map), f, indent=2)
            
    def _save_layer_slices(self):
        """保存层级切片定义到 JSON 文件。"""
        # 生成带索引范围的切片定义
        slices_with_indices = {}
        for level, layers in self.LAYER_SLICES.items():
            slices_with_indices[level] = {}
            for layer_name, param_names in layers.items():
                indices = []
                for pname in param_names:
                    if pname in self.param_index_map:
                        info = self.param_index_map[pname]
                        indices.append({
                            'param': pname,
                            'offset': info['offset'],
                            'length': info['length']
                        })
                slices_with_indices[level][layer_name] = {
                    'params': param_names,
                    'indices': indices
                }
        
        # 添加 L2（每个参数独立）
        slices_with_indices['L2'] = {}
        for pname, info in self.param_index_map.items():
            slices_with_indices['L2'][pname] = {
                'params': [pname],
                'indices': [{'param': pname, 'offset': info['offset'], 'length': info['length']}]
            }
        
        path = os.path.join(self.output_dir, 'layer_slices.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(slices_with_indices, f, indent=2)
            
    def _save_manifest(self, cfg: Any):
        """保存实验元数据到 manifest 文件。"""
        manifest = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'total_params': self.total_params,
            'top_k': self.top_k,
            'full_save_layers': self.FULL_SAVE_LAYERS,
            'config': {
                'dataset': cfg.DATASET.dataset if hasattr(cfg.DATASET, 'dataset') else 'unknown',
                'n_classes': cfg.DATASET.n_classes,
                'parti_num': cfg.DATASET.parti_num,
                'communication_epoch': cfg.DATASET.communication_epoch,
                'beta': cfg.DATASET.beta,
                'backbone': cfg.DATASET.backbone if hasattr(cfg.DATASET, 'backbone') else 'unknown',
                'local_epoch': cfg.OPTIMIZER.local_epoch,
                'local_train_lr': cfg.OPTIMIZER.local_train_lr,
                'local_train_batch': cfg.OPTIMIZER.local_train_batch,
            },
            'attack': {
                'type': cfg.attack.backdoor.evils if hasattr(cfg.attack, 'backdoor') else 'unknown',
                'bad_client_rate': cfg.attack.bad_client_rate if hasattr(cfg.attack, 'bad_client_rate') else 0,
            }
        }
        
        path = os.path.join(self.output_dir, 'run_manifest.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
            
    def _get_layer_slice(self, vector: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        获取向量的指定层级切片。
        
        Args:
            vector: 扁平化的参数向量
            layer_name: 层名称 (如 'head', 'last_block')
            
        Returns:
            对应层的向量切片
        """
        # 查找层对应的参数名列表
        param_names = None
        for level, layers in self.LAYER_SLICES.items():
            if layer_name in layers:
                param_names = layers[layer_name]
                break
        
        if param_names is None:
            raise ValueError(f"Unknown layer name: {layer_name}")
        
        # 收集索引
        indices = []
        for pname in param_names:
            if pname in self.param_index_map:
                info = self.param_index_map[pname]
                start = info['offset']
                end = start + info['length']
                indices.extend(range(start, end))
        
        return vector[indices]
    
    def _compute_stats(self, vector: torch.Tensor) -> Dict[str, float]:
        """
        计算向量的统计量。
        
        Args:
            vector: 输入向量
            
        Returns:
            包含 l1, l2, mean, std, min, max, p50, p90, p99 的字典
        """
        v = vector.detach().float().cpu()
        
        # 处理空向量或全零向量
        if v.numel() == 0:
            return {k: 0.0 for k in ['l1', 'l2', 'mean', 'std', 'min', 'max', 'p50', 'p90', 'p99']}
        
        return {
            'l1': float(torch.norm(v, p=1)),
            'l2': float(torch.norm(v, p=2)),
            'mean': float(v.mean()),
            'std': float(v.std()) if v.numel() > 1 else 0.0,
            'min': float(v.min()),
            'max': float(v.max()),
            'p50': float(torch.quantile(v, 0.50)),
            'p90': float(torch.quantile(v, 0.90)),
            'p99': float(torch.quantile(v, 0.99))
        }
    
    def _compute_topk(self, vector: torch.Tensor, k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 top-k 稀疏表示。
        
        Args:
            vector: 输入向量
            k: top-k 的 k 值（默认使用 self.top_k）
            
        Returns:
            (topk_idx, topk_val) 元组
        """
        if k is None:
            k = self.top_k
            
        v = vector.detach().float().cpu()
        k = min(k, v.numel())
        
        # 按绝对值取 top-k
        abs_v = torch.abs(v)
        topk_val, topk_idx = torch.topk(abs_v, k)
        
        # 返回原始值（保留符号）
        topk_val = v[topk_idx]
        
        return topk_idx, topk_val
    
    def log_round_client(
        self,
        round_idx: int,
        client_idx: int,
        F_raw: torch.Tensor,
        I_minmax: torch.Tensor,
        min_F: torch.Tensor,
        max_F: torch.Tensor,
        delta_w_true: torch.Tensor,
        g_weighted: torch.Tensor
    ):
        """
        记录一轮中某个客户端的核心四件套。
        
        Args:
            round_idx: 轮次索引
            client_idx: 客户端索引
            F_raw: 未归一化的 Fisher 信息
            I_minmax: min-max 归一化后的重要性向量
            min_F: Fisher 信息最小值
            max_F: Fisher 信息最大值
            delta_w_true: 客户端更新 (w_k - w_global)
            g_weighted: 加权更新向量
        """
        # 初始化该轮数据
        if round_idx not in self._round_data:
            self._round_data[round_idx] = {
                'tensors': {},
                'stats': {}
            }
        
        client_key = f'client_{client_idx}'
        
        # 计算并存储各对象的数据
        objects = {
            'F_raw': F_raw.detach().cpu(),
            'I_minmax': I_minmax.detach().cpu(),
            'delta_w_true': delta_w_true.detach().cpu(),
            'g_weighted': g_weighted.detach().cpu()
        }
        
        client_tensors = {
            'min_F': float(min_F),
            'max_F': float(max_F)
        }
        client_stats = {}
        
        for obj_name, vector in objects.items():
            # 计算全局统计量
            client_stats[f'{obj_name}_global'] = self._compute_stats(vector)
            
            # 计算 top-k
            topk_idx, topk_val = self._compute_topk(vector)
            client_tensors[f'{obj_name}_topk_idx'] = topk_idx
            client_tensors[f'{obj_name}_topk_val'] = topk_val
            
            # 对每个全量保存的层，计算层级统计量和全量向量
            for layer_name in self.FULL_SAVE_LAYERS:
                layer_slice = self._get_layer_slice(vector, layer_name)
                
                # 层级统计量
                client_stats[f'{obj_name}_{layer_name}'] = self._compute_stats(layer_slice)
                
                # 层级全量向量
                client_tensors[f'{obj_name}_{layer_name}_full'] = layer_slice
        
        self._round_data[round_idx]['tensors'][client_key] = client_tensors
        self._round_data[round_idx]['stats'][client_key] = client_stats
        
    def log_round_decision(
        self,
        round_idx: int,
        ge_global: torch.Tensor,
        V_k: torch.Tensor,
        cluster_labels: List[int],
        benign_idx: List[int],
        evil_idx: List[int],
        alpha_b: np.ndarray
    ):
        """
        记录一轮的 FDCR 决策量。
        
        Args:
            round_idx: 轮次索引
            ge_global: 加权更新聚合结果
            V_k: 每个客户端的差异度标量
            cluster_labels: 聚类标签
            benign_idx: 判定为良性的客户端索引
            evil_idx: 判定为恶意的客户端索引
            alpha_b: 最终聚合权重
        """
        if round_idx not in self._round_data:
            self._round_data[round_idx] = {'tensors': {}, 'stats': {}}
        
        # 计算 ge_global 的统计量
        ge_stats = self._compute_stats(ge_global.detach().cpu())
        
        # 存储决策量
        self._round_data[round_idx]['tensors']['decision'] = {
            'ge_global_topk_idx': self._compute_topk(ge_global.detach().cpu())[0],
            'ge_global_topk_val': self._compute_topk(ge_global.detach().cpu())[1],
            'V_k': V_k.detach().cpu() if isinstance(V_k, torch.Tensor) else torch.tensor(V_k),
            'alpha_b': torch.tensor(alpha_b) if isinstance(alpha_b, np.ndarray) else alpha_b
        }
        
        self._round_data[round_idx]['stats']['decision'] = {
            'ge_global': ge_stats,
            'cluster_labels': cluster_labels,
            'benign_idx': benign_idx,
            'evil_idx': list(evil_idx) if not isinstance(evil_idx, list) else evil_idx
        }
        
    def log_ground_truth(
        self,
        round_idx: int,
        actual_malicious_idx: List[int]
    ):
        """
        记录 ground truth 标签（仅用于离线分析）。
        
        Args:
            round_idx: 轮次索引
            actual_malicious_idx: 实际恶意客户端索引
        """
        if round_idx not in self._round_data:
            self._round_data[round_idx] = {'tensors': {}, 'stats': {}}
            
        self._round_data[round_idx]['ground_truth'] = {
            'actual_malicious_idx': actual_malicious_idx
        }
        
    def save_round(self, round_idx: int):
        """
        保存一轮的所有数据到磁盘。
        
        Args:
            round_idx: 轮次索引
        """
        if round_idx not in self._round_data:
            return
            
        round_dir = os.path.join(self.output_dir, f'round_{round_idx:03d}')
        os.makedirs(round_dir, exist_ok=True)
        
        data = self._round_data[round_idx]
        
        # 保存 tensors.pt
        tensors_path = os.path.join(round_dir, 'tensors.pt')
        torch.save(data['tensors'], tensors_path)
        
        # 保存 stats.json
        stats_path = os.path.join(round_dir, 'stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(data['stats'], f, indent=2)
            
        # 保存 ground_truth.json
        if 'ground_truth' in data:
            gt_path = os.path.join(round_dir, 'ground_truth.json')
            with open(gt_path, 'w', encoding='utf-8') as f:
                json.dump(data['ground_truth'], f, indent=2)
        
        # 清理内存
        del self._round_data[round_idx]
        
    def get_output_dir(self) -> str:
        """获取输出目录路径。"""
        return self.output_dir


def get_layer_slices(model_name: str = 'SimpleCNN') -> Dict:
    """
    获取指定模型的层级切片定义。
    
    Args:
        model_name: 模型名称
        
    Returns:
        层级切片定义字典
    """
    if model_name == 'SimpleCNN':
        return FDCRLogger.LAYER_SLICES
    else:
        raise ValueError(f"Unknown model: {model_name}")
