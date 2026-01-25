"""
Cosine-FDCR 服务器变体

实现三种 score 变体：
1. baseline (V_k): MSE discrepancy
2. cos: 1 - cos(gI, center)
3. nmse: V_k / ||gI||²

口径：grad = -delta_w / lr, gI = grad * I
"""

import torch
import torch.nn.functional as F
import numpy as np
import copy
import os
from Server.OurRandomControlNoCheat import OurRandomControlNoCheat
from utils.finch import FINCH


class CosineFDCR(OurRandomControlNoCheat):
    """Cosine-FDCR: 使用 cosine 距离替代 MSE"""
    
    NAME = 'CosineFDCR'
    
    def __init__(self, args, cfg):
        super().__init__(args, cfg)
        self.score_mode = 'cos'  # 'cos', 'nmse', 'unit', 'baseline'
        self.slice_mode = 'head'  # 'head', 'last_block', 'global'
        self.eps = 1e-12
        
        # 构建 slice 索引（延迟到第一次调用）
        self._slice_indices = None
    
    def _build_slice_indices(self, reference_net):
        """构建 slice 索引"""
        if self._slice_indices is not None:
            return
        
        self._slice_indices = {}
        offset = 0
        param_map = {}
        
        for name, param in reference_net.named_parameters():
            length = param.numel()
            param_map[name] = {'offset': offset, 'length': length}
            offset += length
        
        # 定义层级
        layer_params = {
            'head': ['cls.weight', 'cls.bias'],
            'last_block': ['l2.weight', 'l2.bias'],
        }
        
        for layer_name, pnames in layer_params.items():
            indices = []
            for pname in pnames:
                if pname in param_map:
                    info = param_map[pname]
                    start = info['offset']
                    end = start + info['length']
                    indices.extend(range(start, end))
            self._slice_indices[layer_name] = torch.tensor(indices, dtype=torch.long)
        
        self._slice_indices['global'] = torch.arange(offset)
    
    def compute_score(self, gI_list, freq, mode='cos'):
        """
        计算异常度 score
        
        Args:
            gI_list: list of gI tensors (已 slice)
            freq: 权重
            mode: 'cos', 'nmse', 'unit', 'baseline'
        
        Returns:
            scores: tensor [K]
        """
        K = len(gI_list)
        device = gI_list[0].device
        
        # 计算 center
        center = torch.zeros_like(gI_list[0])
        for gI, f in zip(gI_list, freq):
            center += gI * f
        
        center_norm = torch.norm(center, p=2) + self.eps
        
        scores = []
        for gI in gI_list:
            gI_norm = torch.norm(gI, p=2) + self.eps
            
            if mode == 'baseline':
                # V_k = MSE(gI, center)
                score = F.mse_loss(gI, center, reduction='mean')
            
            elif mode == 'cos':
                # score_cos = 1 - cos(gI, center)
                cos_val = torch.dot(gI, center) / (gI_norm * center_norm)
                score = 1.0 - cos_val
            
            elif mode == 'nmse':
                # score_nmse = V_k / ||gI||²
                V_k = F.mse_loss(gI, center, reduction='mean')
                score = V_k / (gI_norm ** 2 + self.eps)
            
            elif mode == 'unit':
                # score_unit = MSE(gI/||gI||, center/||center||)
                gI_unit = gI / gI_norm
                center_unit = center / center_norm
                score = F.mse_loss(gI_unit, center_unit, reduction='mean')
            
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            scores.append(float(score))
        
        return torch.tensor(scores)
    
    def server_update(self, **kwargs):
        """重写 server_update，使用 cosine score"""
        
        online_clients_list = kwargs['online_clients_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']
        
        # 初始化 slice 索引
        self._build_slice_indices(nets_list[0])
        
        default_net = copy.deepcopy(global_net)
        priloader_list = kwargs['priloader_list']
        
        freq = self.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)
        self.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                       global_net=default_net, freq=freq, except_part=[], global_only=True)
        
        # 计算 actual_malicious_idx（仅用于评估）
        bad_scale = int(self.cfg.DATASET.parti_num * self.cfg['attack'].bad_client_rate)
        good_scale = self.cfg.DATASET.parti_num - bad_scale
        client_type = np.repeat(True, good_scale).tolist() + (np.repeat(False, bad_scale)).tolist()
        actual_malicious_idx = [i for i, is_benign in enumerate(client_type) if not is_benign]
        
        local_fish_dict = kwargs['local_fish_dict']
        prev_net = copy.deepcopy(global_net)
        
        # 向量化网络参数
        vectorize_nets_list = []
        for query_net in nets_list:
            vectorize_net = torch.cat([p.view(-1) for p in query_net.parameters()]).detach()
            vectorize_nets_list.append(vectorize_net)
        prev_vectorize_net = torch.cat([p.view(-1) for p in prev_net.parameters()]).detach()
        
        # 获取 slice 索引
        slice_idx = self._slice_indices[self.slice_mode]
        
        # 计算 grad 和 gI
        gI_list = []
        for query_index, _ in enumerate(nets_list):
            # grad = (prev - new) / lr = -delta_w / lr
            delta_w = vectorize_nets_list[query_index] - prev_vectorize_net
            grad = -delta_w / self.learning_rate
            
            # I (min-max normalized Fisher)
            query_fish_dict = local_fish_dict[query_index]
            query_fish = torch.cat([p.view(-1) for p in query_fish_dict.values()]).detach()
            I = (query_fish - torch.min(query_fish)) / (torch.max(query_fish) - torch.min(query_fish) + self.eps)
            
            # gI = grad * I
            gI = grad * I
            
            # slice
            gI_sliced = gI[slice_idx]
            gI_list.append(gI_sliced)
        
        # 计算 score
        scores = self.compute_score(gI_list, freq, mode=self.score_mode)
        div_score = scores.view(-1, 1)
        self.div_score = div_score  # 供 training.py 记录
        
        # FINCH 聚类
        fin = FINCH()
        fin.fit(div_score)
        
        benign_idx = list(range(len(online_clients_list)))
        evils_idx = []
        cluster_labels = [0] * len(online_clients_list)
        
        if len(fin.partitions) == 0:
            reconstructed_freq = freq
        else:
            select_partitions = (fin.partitions)['parition_0']
            evils_center = max(select_partitions['cluster_centers'])
            evils_center_idx = np.where(select_partitions['cluster_centers'] == evils_center)[0]
            evils_idx = select_partitions['cluster_core_indices'][int(evils_center_idx)]
            benign_idx = [i for i in range(len(online_clients_list)) if i not in evils_idx]
            
            for idx in evils_idx:
                cluster_labels[idx] = 1
            
            print(f'[{self.NAME}] Predicted benign: {benign_idx}, Predicted malicious: {list(evils_idx)}')
            print(f'[{self.NAME}] Actual malicious: {actual_malicious_idx}')
            
            # 计算 filtered ratio
            filtered_ratio = self.compute_filtered_ratio(list(evils_idx), actual_malicious_idx)
            self.filtered_ratio_history.append(filtered_ratio)
            
            # 记录检测结果
            self.detection_results.append({
                'benign_idx': benign_idx,
                'evil_idx': list(evils_idx),
                'actual_malicious_idx': actual_malicious_idx,
                'scores': scores.tolist(),
                'cluster_labels': cluster_labels
            })
            
            # 重新计算权重：将恶意客户端权重置 0，然后归一化
            freq[evils_idx] = 0
            reconstructed_freq = freq / (sum(freq) + self.eps)
        
        # 聚合（使用 reconstructed_freq）
        self.agg_parts(
            online_clients_list=online_clients_list,
            nets_list=nets_list,
            global_net=global_net,
            freq=reconstructed_freq,
            except_part=[],
            global_only=False
        )


class CosineFDCR_Head(CosineFDCR):
    """Cosine-FDCR with head slice"""
    NAME = 'CosineFDCR_Head'
    
    def __init__(self, args, cfg):
        super().__init__(args, cfg)
        self.score_mode = 'cos'
        self.slice_mode = 'head'


class CosineFDCR_LastBlock(CosineFDCR):
    """Cosine-FDCR with last_block slice"""
    NAME = 'CosineFDCR_LastBlock'
    
    def __init__(self, args, cfg):
        super().__init__(args, cfg)
        self.score_mode = 'cos'
        self.slice_mode = 'last_block'


class NormalizedMSE_FDCR(CosineFDCR):
    """Normalized MSE FDCR"""
    NAME = 'NormalizedMSE_FDCR'
    
    def __init__(self, args, cfg):
        super().__init__(args, cfg)
        self.score_mode = 'nmse'
        self.slice_mode = 'head'


class UnitNormMSE_FDCR(CosineFDCR):
    """Unit Norm MSE FDCR"""
    NAME = 'UnitNormMSE_FDCR'
    
    def __init__(self, args, cfg):
        super().__init__(args, cfg)
        self.score_mode = 'unit'
        self.slice_mode = 'head'
