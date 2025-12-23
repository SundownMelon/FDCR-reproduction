from Server.utils.server_methods import ServerMethod
from utils.utils import row_into_parameters
from torch import optim, nn
from tqdm import tqdm
import numpy as np
import torch
import copy
import torch.nn.functional as F
from utils.finch import FINCH


class OurRandomControl(ServerMethod):
    NAME = 'OurRandomControl'

    def __init__(self, args, cfg):
        super(OurRandomControl, self).__init__(args, cfg)
        self.learning_rate = self.cfg.OPTIMIZER.local_train_lr
        self.div_score = None
        self.aggregation_weight = None
        # Filtered ratio tracking for FDCR defense analysis
        self.filtered_ratio_history = []
        self.detection_results = []
        # Store last round's detection info for logging
        self.last_benign_idx = None
        self.last_evil_idx = None
        self.last_aggregation_weights = None

    def compute_filtered_ratio(self, predicted_malicious, actual_malicious):
        """
        Compute the filtered ratio as the proportion of actual malicious clients
        correctly identified by FDCR.
        
        Args:
            predicted_malicious: List of client indices predicted as malicious
            actual_malicious: List of client indices that are actually malicious
        
        Returns:
            float: filtered_ratio = |predicted ∩ actual| / |actual|
                   Returns 0.0 if there are no actual malicious clients
        
        Requirements: 4.3
        """
        if len(actual_malicious) == 0:
            return 0.0
        
        predicted_set = set(predicted_malicious)
        actual_set = set(actual_malicious)
        
        correctly_identified = predicted_set.intersection(actual_set)
        filtered_ratio = len(correctly_identified) / len(actual_set)
        
        return filtered_ratio

    def get_detection_summary(self):
        """
        Get summary statistics for detection performance.
        
        Returns:
            dict: Summary containing mean filtered_ratio and detection history
        """
        if len(self.filtered_ratio_history) == 0:
            return {
                'mean_filtered_ratio': 0.0,
                'filtered_ratio_history': [],
                'detection_results': []
            }
        
        return {
            'mean_filtered_ratio': np.mean(self.filtered_ratio_history),
            'filtered_ratio_history': self.filtered_ratio_history.copy(),
            'detection_results': self.detection_results.copy()
        }

    def server_update(self, **kwargs):

        online_clients_list = kwargs['online_clients_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']

        default_net = copy.deepcopy(global_net) # 获取默认聚合方向

        priloader_list = kwargs['priloader_list']

        freq = self.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)
        # 获取 假定的更新方向
        self.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                       global_net=default_net, freq=freq, except_part=[], global_only=True)

        bad_scale = int(self.cfg.DATASET.parti_num * self.cfg['attack'].bad_client_rate)
        good_scale = self.cfg.DATASET.parti_num - bad_scale
        client_type = np.repeat(True, good_scale).tolist() + (np.repeat(False, bad_scale)).tolist()
        
        # Compute actual malicious client indices (Requirements: 4.1, 4.2)
        actual_malicious_idx = [i for i, is_benign in enumerate(client_type) if not is_benign]

        local_fish_dict = kwargs['local_fish_dict']
        prev_net = copy.deepcopy(global_net)
        vectorize_nets_list = []
        for query_net in nets_list:
            vectorize_net = torch.cat([p.view(-1) for p in query_net.parameters()]).detach()
            vectorize_nets_list.append(vectorize_net)
        # 参数化上一轮的网络
        prev_vectorize_net = torch.cat([p.view(-1) for p in prev_net.parameters()]).detach()

        grad_list = []
        weight_list = []
        fish_list = []
        for query_index, _ in enumerate(nets_list):
            grad_list.append((prev_vectorize_net - vectorize_nets_list[query_index]) / self.learning_rate)
            query_fish_dict = local_fish_dict[query_index]
            query_fish = torch.cat([p.view(-1) for p in query_fish_dict.values()]).detach()
            if not client_type[query_index]:
                query_fish = torch.randn_like(query_fish)
            fish_list.append(query_fish)
            norm_fish = (query_fish - torch.min(query_fish)) / (torch.max(query_fish) - torch.min(query_fish))
            weight_list.append(norm_fish)

        weight_grad_list = []
        for query_index, _ in enumerate(nets_list):
            query_grad = grad_list[query_index]
            query_weight = weight_list[query_index]
            weight_grad_list.append(torch.mul(query_grad, query_weight))

        assert len(weight_grad_list) == len(freq), "张量列表和权重列表的长度必须相同"

        weight_global_grad = torch.zeros_like(weight_grad_list[0])
        # 客户端规模 x 特征长度
        for weight_client_grad, client_freq in zip(weight_grad_list, freq):
            weight_global_grad += weight_client_grad * client_freq

        div_score = []
        for query_index, _ in enumerate(nets_list):
            div_score.append(
                F.pairwise_distance(weight_grad_list[query_index].view(1, -1), weight_global_grad.view(1, -1), p=2))

        div_score = torch.tensor(div_score).view(-1, 1)
        fin = FINCH()
        fin.fit(div_score)

        # Initialize detection tracking variables
        benign_idx = list(range(len(online_clients_list)))
        evils_idx = []
        
        if len(fin.partitions) == 0:
            reconstructed_freq = freq
        else:
            select_partitions = (fin.partitions)['parition_0']
            evils_center = max(select_partitions['cluster_centers'])
            evils_center_idx = np.where(select_partitions['cluster_centers'] == evils_center)[0]
            evils_idx = select_partitions['cluster_core_indices'][int(evils_center_idx)]
            benign_idx = [i for i in range(len(online_clients_list)) if i not in evils_idx]

            # Log benign and malicious client indices (Requirements: 4.1, 4.2)
            print('benign', benign_idx, 'evil', evils_idx)
            print(f'[FDCR Detection] Predicted benign: {benign_idx}, Predicted malicious: {list(evils_idx)}')
            print(f'[FDCR Detection] Actual malicious: {actual_malicious_idx}')
            
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

        # Compute and store filtered_ratio (Requirements: 4.3)
        predicted_malicious = list(evils_idx) if isinstance(evils_idx, np.ndarray) else evils_idx
        filtered_ratio = self.compute_filtered_ratio(predicted_malicious, actual_malicious_idx)
        self.filtered_ratio_history.append(filtered_ratio)
        
        # Store detection results for this round
        detection_result = {
            'benign_idx': benign_idx,
            'evil_idx': list(evils_idx) if isinstance(evils_idx, np.ndarray) else evils_idx,
            'actual_malicious_idx': actual_malicious_idx,
            'filtered_ratio': filtered_ratio,
            'aggregation_weights': reconstructed_freq.tolist() if isinstance(reconstructed_freq, np.ndarray) else list(reconstructed_freq)
        }
        self.detection_results.append(detection_result)
        
        # Store last round's detection info for external logging access
        self.last_benign_idx = benign_idx
        self.last_evil_idx = list(evils_idx) if isinstance(evils_idx, np.ndarray) else evils_idx
        self.last_aggregation_weights = reconstructed_freq.tolist() if isinstance(reconstructed_freq, np.ndarray) else list(reconstructed_freq)
        
        # Log aggregation weights per client (Requirements: 4.2)
        print(f'[FDCR Aggregation] Weights per client: {self.last_aggregation_weights}')
        print(f'[FDCR Metrics] Filtered ratio this round: {filtered_ratio:.4f}')

        self.div_score = (div_score)
        self.aggregation_weight = (reconstructed_freq)
        self.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                       global_net=global_net, freq=reconstructed_freq, except_part=[], global_only=False)
        return freq
