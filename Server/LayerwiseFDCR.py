"""
A2: 最小改动版 FDCR - 层级受限 V_k

用 V_{k,last} 或 V_{k,head+last} 替换全局 V_k 做 FINCH 聚类。

变体:
- Baseline: 原 FDCR (global V_k)
- Variant-1: FINCH 输入换成 V_{k,last}
- Variant-2: FINCH 输入换成 V_{k,head+last}
"""

from Server.OurRandomControlNoCheat import OurRandomControlNoCheat
from utils.utils import row_into_parameters
import numpy as np
import torch
import copy
import torch.nn.functional as F
from utils.finch import FINCH
import os


class LayerwiseFDCR(OurRandomControlNoCheat):
    """
    分层 V_k 版 FDCR
    
    通过 layer_mode 参数控制使用哪个层级的 V_k 进行聚类。
    """
    NAME = 'LayerwiseFDCR'
    
    # 层级参数定义
    LAYER_PARAMS = {
        'head': ['cls.weight', 'cls.bias'],
        'last_block': ['l2.weight', 'l2.bias'],
        'head_last': ['cls.weight', 'cls.bias', 'l2.weight', 'l2.bias'],
        'l1': ['l1.weight', 'l1.bias'],
    }
    
    def __init__(self, args, cfg, layer_mode='global'):
        """
        初始化分层 FDCR
        
        Args:
            layer_mode: 'global', 'last_block', 'head_last', 'head'
        """
        super(LayerwiseFDCR, self).__init__(args, cfg)
        self.layer_mode = layer_mode
        self.layer_indices = None
        
        print(f"[LayerwiseFDCR] V_k 计算层级: {layer_mode}")
    
    def _build_layer_indices(self, net):
        """构建层级索引映射"""
        if self.layer_indices is not None:
            return
        
        self.layer_indices = {}
        self.param_offset_map = {}
        
        offset = 0
        for name, param in net.named_parameters():
            length = param.numel()
            self.param_offset_map[name] = {'offset': offset, 'length': length}
            offset += length
        
        self.total_params = offset
        
        for layer_name, param_names in self.LAYER_PARAMS.items():
            indices = []
            for pname in param_names:
                if pname in self.param_offset_map:
                    info = self.param_offset_map[pname]
                    start = info['offset']
                    end = start + info['length']
                    indices.extend(range(start, end))
            self.layer_indices[layer_name] = torch.tensor(indices, dtype=torch.long, device=self.device)
        
        self.layer_indices['global'] = torch.arange(self.total_params, device=self.device)
    
    def _get_layer_slice(self, vector, layer_name):
        """获取向量的指定层切片"""
        if layer_name == 'global':
            return vector
        
        indices = self.layer_indices.get(layer_name)
        if indices is None or len(indices) == 0:
            return vector
        
        return vector[indices]
    
    def server_update(self, **kwargs):
        """
        服务器端更新 - 使用分层 V_k
        """
        online_clients_list = kwargs['online_clients_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']

        default_net = copy.deepcopy(global_net)

        priloader_list = kwargs['priloader_list']

        freq = self.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)
        self.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                       global_net=default_net, freq=freq, except_part=[], global_only=True)

        # 构建层级索引
        self._build_layer_indices(nets_list[0])

        # 计算 actual_malicious_idx 用于评估
        bad_scale = int(self.cfg.DATASET.parti_num * self.cfg['attack'].bad_client_rate)
        good_scale = self.cfg.DATASET.parti_num - bad_scale
        client_type = np.repeat(True, good_scale).tolist() + (np.repeat(False, bad_scale)).tolist()
        actual_malicious_idx = [i for i, is_benign in enumerate(client_type) if not is_benign]

        local_fish_dict = kwargs['local_fish_dict']
        prev_net = copy.deepcopy(global_net)
        vectorize_nets_list = []
        for query_net in nets_list:
            vectorize_net = torch.cat([p.view(-1) for p in query_net.parameters()]).detach()
            vectorize_nets_list.append(vectorize_net)
        prev_vectorize_net = torch.cat([p.view(-1) for p in prev_net.parameters()]).detach()

        # Step 1: 初始化 Logger
        if self.enable_step1_logging and self.fdcr_logger is None:
            from utils.fdcr_logger import FDCRLogger
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
            self.fdcr_logger = FDCRLogger(
                output_dir=output_dir,
                cfg=self.cfg,
                reference_net=nets_list[0],
                top_k=512
            )
            print(f'[Step1 Logger] Initialized at {self.fdcr_logger.get_output_dir()}')

        grad_list = []
        weight_list = []
        fish_list = []
        for query_index, _ in enumerate(nets_list):
            grad_list.append((prev_vectorize_net - vectorize_nets_list[query_index]) / self.learning_rate)
            query_fish_dict = local_fish_dict[query_index]
            query_fish = torch.cat([p.view(-1) for p in query_fish_dict.values()]).detach()
            
            fish_list.append(query_fish)
            norm_fish = (query_fish - torch.min(query_fish)) / (torch.max(query_fish) - torch.min(query_fish))
            weight_list.append(norm_fish)

        weight_grad_list = []
        for query_index, _ in enumerate(nets_list):
            query_grad = grad_list[query_index]
            query_weight = weight_list[query_index]
            weight_grad_list.append(torch.mul(query_grad, query_weight))

        assert len(weight_grad_list) == len(freq)

        # ========== 关键修改：按层级计算 weight_global_grad ==========
        weight_global_grad_full = torch.zeros_like(weight_grad_list[0])
        for weight_client_grad, client_freq in zip(weight_grad_list, freq):
            weight_global_grad_full += weight_client_grad * client_freq

        # ========== 关键修改：按层级计算 V_k ==========
        div_score = []
        for query_index, _ in enumerate(nets_list):
            # 获取指定层的切片
            weight_grad_layer = self._get_layer_slice(weight_grad_list[query_index], self.layer_mode)
            weight_global_layer = self._get_layer_slice(weight_global_grad_full, self.layer_mode)
            
            # 计算该层的 pairwise distance
            div = F.pairwise_distance(
                weight_grad_layer.view(1, -1), 
                weight_global_layer.view(1, -1), 
                p=2
            )
            div_score.append(div)

        div_score = torch.tensor(div_score).view(-1, 1)
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

            print(f'[LayerwiseFDCR-{self.layer_mode}] benign: {benign_idx}, evil: {list(evils_idx)}')
            print(f'[LayerwiseFDCR-{self.layer_mode}] Actual malicious: {actual_malicious_idx}')
            
            freq[evils_idx] = 0
            reconstructed_freq = freq / sum(freq)

            for i in benign_idx:
                curr_net = nets_list[i]
                norm_weight = weight_list[i]
                index = 0
                for name, curr_param in curr_net.state_dict().items():
                    prev_para = prev_net.state_dict()[name].detach()
                    delta = (prev_para - curr_param.detach())
                    param_number = prev_para.numel()
                    param_size = prev_para.size()

                    weight_para = norm_weight[index:index + param_number].reshape(param_size).to(self.device)
                    weight_para = torch.nn.functional.sigmoid(weight_para) * 2

                    weight_delta = torch.mul(delta, weight_para)
                    index += param_number
                    curr_param.data.copy_(prev_para - weight_delta)
                nets_list[i] = curr_net

        # Step 1 日志记录
        if self.enable_step1_logging and self.fdcr_logger is not None:
            for query_index in range(len(nets_list)):
                delta_w_true = vectorize_nets_list[query_index] - prev_vectorize_net
                
                self.fdcr_logger.log_round_client(
                    round_idx=self.epoch_index,
                    client_idx=query_index,
                    F_raw=fish_list[query_index],
                    I_minmax=weight_list[query_index],
                    min_F=torch.min(fish_list[query_index]),
                    max_F=torch.max(fish_list[query_index]),
                    delta_w_true=delta_w_true,
                    g_weighted=weight_grad_list[query_index]
                )
            
            self.fdcr_logger.log_round_decision(
                round_idx=self.epoch_index,
                ge_global=weight_global_grad_full,
                V_k=div_score,
                cluster_labels=cluster_labels,
                benign_idx=benign_idx,
                evil_idx=list(evils_idx) if isinstance(evils_idx, np.ndarray) else evils_idx,
                alpha_b=reconstructed_freq if isinstance(reconstructed_freq, np.ndarray) else np.array(reconstructed_freq)
            )
            
            self.fdcr_logger.log_ground_truth(
                round_idx=self.epoch_index,
                actual_malicious_idx=actual_malicious_idx
            )
            
            self.fdcr_logger.save_round(self.epoch_index)

        # 计算检测指标
        predicted_malicious = list(evils_idx) if isinstance(evils_idx, np.ndarray) else evils_idx
        filtered_ratio = self.compute_filtered_ratio(predicted_malicious, actual_malicious_idx)
        self.filtered_ratio_history.append(filtered_ratio)
        
        detection_result = {
            'benign_idx': benign_idx,
            'evil_idx': list(evils_idx) if isinstance(evils_idx, np.ndarray) else evils_idx,
            'actual_malicious_idx': actual_malicious_idx,
            'filtered_ratio': filtered_ratio,
            'aggregation_weights': reconstructed_freq.tolist() if isinstance(reconstructed_freq, np.ndarray) else list(reconstructed_freq),
            'layer_mode': self.layer_mode
        }
        self.detection_results.append(detection_result)
        
        self.last_benign_idx = benign_idx
        self.last_evil_idx = list(evils_idx) if isinstance(evils_idx, np.ndarray) else evils_idx
        self.last_aggregation_weights = reconstructed_freq.tolist() if isinstance(reconstructed_freq, np.ndarray) else list(reconstructed_freq)
        
        print(f'[LayerwiseFDCR-{self.layer_mode}] Filtered ratio: {filtered_ratio:.4f}')

        self.div_score = div_score
        self.aggregation_weight = reconstructed_freq
        self.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                       global_net=global_net, freq=reconstructed_freq, except_part=[], global_only=False)
        return freq


# 便捷类：不同层级变体
class LayerwiseFDCR_Last(LayerwiseFDCR):
    """Variant-1: 使用 last_block 层的 V_k"""
    NAME = 'LayerwiseFDCR_Last'
    
    def __init__(self, args, cfg):
        super().__init__(args, cfg, layer_mode='last_block')


class LayerwiseFDCR_HeadLast(LayerwiseFDCR):
    """Variant-2: 使用 head+last_block 层的 V_k"""
    NAME = 'LayerwiseFDCR_HeadLast'
    
    def __init__(self, args, cfg):
        super().__init__(args, cfg, layer_mode='head_last')


class LayerwiseFDCR_Head(LayerwiseFDCR):
    """Variant-3: 使用 head 层的 V_k"""
    NAME = 'LayerwiseFDCR_Head'
    
    def __init__(self, args, cfg):
        super().__init__(args, cfg, layer_mode='head')
