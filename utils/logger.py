import copy
import os
import csv

from utils.utils import create_if_not_exists

import yaml
from yacs.config import CfgNode as CN

except_args = ['result_path', 'csv_log', 'csv_name', 'device_id', 'seed', 'tensorboard', 'conf_jobnum', 'conf_timestamp', 'conf_host', 'opts']


class CsvWriter:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.model_path = self.model_folder_path()
        self.para_path = self.write_para()
        print(self.para_path)

    def model_folder_path(self):
        if self.args.attack_type == 'None':
            model_path = os.path.join(self.args.result_path, self.args.task, self.args.attack_type, self.args.dataset, str(self.cfg.DATASET.beta), self.args.server,
                                      self.args.optim)
        else:
            model_path = os.path.join(self.args.result_path, self.args.task, self.cfg.attack[self.args.attack_type].evils, str(self.cfg.attack.bad_client_rate),
                                      self.args.dataset, str(self.cfg.DATASET.beta), self.args.server, self.args.optim)
        create_if_not_exists(model_path)
        return model_path

    def write_metric(self, metric_list, epoch_index, name):
        metric_path = os.path.join(self.para_path, name + '.csv')
        if epoch_index != 0:
            write_type = 'a'
        else:
            write_type = 'w'
        with open(metric_path, write_type) as result_file:
            result_file.write(str(epoch_index) + ':' + '\n')
            for i in range(len(metric_list)):
                result_file.write(str(metric_list[i]) + ',')
            result_file.write('\n')
    def write_acc(self, acc, name, mode='ALL'):
        if mode == 'ALL':
            acc_path = os.path.join(self.para_path, name + '_all_acc.csv')
            self.write_all_acc(acc_path, acc)
        elif mode == 'MEAN':
            mean_acc_path = os.path.join(self.para_path, name + '_mean_acc.csv')
            self.write_mean_acc(mean_acc_path, acc)

    def cfg_to_dict(self, cfg):
        d = {}
        for k, v in cfg.items():
            if isinstance(v, CN):
                d[k] = self.cfg_to_dict(v)
            else:
                d[k] = v
        return d

    def dict_to_cfg(self, d):
        cfg = CN()
        for k, v in d.items():
            if isinstance(v, dict):
                cfg[k] = self.dict_to_cfg(v)
            else:
                cfg[k] = v
        return cfg

    def write_para(self) -> None:
        from yacs.config import CfgNode as CN

        args = copy.deepcopy((self.args))
        args = vars(args)
        cfg = copy.deepcopy(self.cfg)

        for cc in except_args:
            if cc in args: del args[cc]
        for key, value in args.items():
            args[key] = str(value)
        paragroup_dirs = os.listdir(self.model_path)
        n_para = len(paragroup_dirs)
        final_check = False

        if self.args.csv_name is not None:

            path = os.path.join(self.model_path, self.args.csv_name)
            create_if_not_exists(path)
            args_path = path + '/args.csv'
            cfg_path = path + '/cfg.yaml'
            columns = list(args.keys())
            with open(args_path, 'w') as tmp:
                writer = csv.DictWriter(tmp, fieldnames=columns)

                writer.writeheader()
                writer.writerow(args)
            with open(cfg_path, 'w') as f:
                f.write(yaml.dump(self.cfg_to_dict(cfg)))
        else:

            for para in paragroup_dirs:
                exist_para_args = True
                exist_para_cfg = True
                dict_from_csv = {}
                key_value_list = []
                para_path = os.path.join(self.model_path, para)
                args_path = para_path + '/args.csv'
                with open(args_path, mode='r') as inp:
                    reader = csv.reader(inp)
                    for rows in reader:
                        key_value_list.append(rows)
                for index, _ in enumerate(key_value_list[0]):
                    dict_from_csv[key_value_list[0][index]] = key_value_list[1][index]
                if args != dict_from_csv:
                    exist_para_args = False
                cfg_path = para_path + '/cfg.yaml'
                query_cfg = copy.deepcopy(cfg)
                query_cfg.merge_from_file(cfg_path)
                for name, value1 in cfg.items():
                    if isinstance(value1, CN):
                        if name not in query_cfg or self.cfg_to_dict(query_cfg[name]) != self.cfg_to_dict(value1):
                            exist_para_cfg = False
                if exist_para_args == True and exist_para_cfg == True:
                    final_check = True
                    break

            if not final_check:

                if self.args.csv_name is None:
                    path = os.path.join(self.model_path, 'para' + str(n_para + 1))
                    k = 1
                    while os.path.exists(path):
                        path = os.path.join(self.model_path, 'para' + str(n_para + k))
                        k = k + 1
                else:
                    path = os.path.join(self.model_path, self.args.csv_name)

                create_if_not_exists(path)
                columns = list(args.keys())
                write_headers = True
                args_path = path + '/args.csv'
                cfg_path = path + '/cfg.yaml'
                with open(args_path, 'a') as tmp:
                    writer = csv.DictWriter(tmp, fieldnames=columns)
                    if write_headers:
                        writer.writeheader()
                    writer.writerow(args)
                with open(cfg_path, 'w') as f:
                    f.write(yaml.dump(self.cfg_to_dict(cfg)))
            else:
                path = para_path
        return path

    def write_mean_acc(self, mean_path, acc_list):
        if os.path.exists(mean_path):
            with open(mean_path, 'a') as result_file:
                for i in range(len(acc_list)):
                    result = acc_list[i]
                    result_file.write(str(result))
                    if i != self.cfg.DATASET.communication_epoch - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')
        else:
            with open(mean_path, 'w') as result_file:
                for epoch in range(self.cfg.DATASET.communication_epoch):
                    result_file.write('epoch_' + str(epoch))
                    if epoch != self.cfg.DATASET.communication_epoch - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')
                for i in range(len(acc_list)):
                    result = acc_list[i]
                    result_file.write(str(result))
                    if i != self.cfg.DATASET.communication_epoch - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')

    def write_all_acc(self, all_path, all_acc_list):
        if os.path.exists(all_path):
            with open(all_path, 'a') as result_file:
                for key in all_acc_list:
                    method_result = all_acc_list[key]
                    result_file.write(key + ',')
                    for epoch in range(len(method_result)):
                        result_file.write(str(method_result[epoch]))
                        if epoch != len(method_result) - 1:
                            result_file.write(',')
                        else:
                            result_file.write('\n')
        else:
            with open(all_path, 'w') as result_file:
                result_file.write('domain,')
                for epoch in range(self.cfg.DATASET.communication_epoch):
                    result_file.write('epoch_' + str(epoch))
                    if epoch != self.cfg.DATASET.communication_epoch - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')

                for key in all_acc_list:
                    method_result = all_acc_list[key]
                    result_file.write(key + ',')
                    for epoch in range(len(method_result)):
                        result_file.write(str(method_result[epoch]))
                        if epoch != len(method_result) - 1:
                            result_file.write(',')
                        else:
                            result_file.write('\n')

    def write_filtered_ratio(self, filtered_ratio, epoch_index):
        """
        Write filtered_ratio metric for a given epoch.
        
        Args:
            filtered_ratio: The filtered ratio value (proportion of malicious clients detected)
            epoch_index: The communication round index
        
        Requirements: 4.3, 4.4
        """
        filtered_ratio_path = os.path.join(self.para_path, 'filtered_ratio.csv')
        
        if epoch_index == 0:
            # Create new file with header
            with open(filtered_ratio_path, 'w') as f:
                f.write('epoch,filtered_ratio\n')
                f.write(f'{epoch_index},{filtered_ratio}\n')
        else:
            # Append to existing file
            with open(filtered_ratio_path, 'a') as f:
                f.write(f'{epoch_index},{filtered_ratio}\n')

    def write_detection_results(self, benign_idx, evil_idx, epoch_index):
        """
        Write detection results (benign and malicious client indices) for a given epoch.
        
        Args:
            benign_idx: List of client indices identified as benign
            evil_idx: List of client indices identified as malicious
            epoch_index: The communication round index
        
        Requirements: 4.3, 4.4
        """
        detection_path = os.path.join(self.para_path, 'detection_results.csv')
        
        # Convert lists to string representation
        benign_str = ';'.join(map(str, benign_idx)) if benign_idx else ''
        evil_str = ';'.join(map(str, evil_idx)) if evil_idx else ''
        
        if epoch_index == 0:
            # Create new file with header
            with open(detection_path, 'w') as f:
                f.write('epoch,benign_indices,malicious_indices\n')
                f.write(f'{epoch_index},"{benign_str}","{evil_str}"\n')
        else:
            # Append to existing file
            with open(detection_path, 'a') as f:
                f.write(f'{epoch_index},"{benign_str}","{evil_str}"\n')

    def write_summary(self, acc_list, asr_list, filtered_ratio_list, window_size=10):
        """
        Generate experiment summary with steady-state metrics.
        
        Computes steady-state ACC/ASR from the final window of communication rounds
        and outputs mean filtered_ratio and detection accuracy.
        
        Args:
            acc_list: List of accuracy values per epoch
            asr_list: List of attack success rate values per epoch
            filtered_ratio_list: List of filtered ratio values per epoch
            window_size: Number of final rounds to use for steady-state computation (default: 10)
        
        Requirements: 3.3, 4.4
        """
        summary_path = os.path.join(self.para_path, 'experiment_summary.csv')
        
        # Compute steady-state ACC from final window
        if len(acc_list) >= window_size:
            steady_state_acc = sum(acc_list[-window_size:]) / window_size
        elif len(acc_list) > 0:
            steady_state_acc = sum(acc_list) / len(acc_list)
        else:
            steady_state_acc = 0.0
        
        # Compute steady-state ASR from final window
        if len(asr_list) >= window_size:
            steady_state_asr = sum(asr_list[-window_size:]) / window_size
        elif len(asr_list) > 0:
            steady_state_asr = sum(asr_list) / len(asr_list)
        else:
            steady_state_asr = 0.0
        
        # Compute mean filtered_ratio
        if len(filtered_ratio_list) > 0:
            mean_filtered_ratio = sum(filtered_ratio_list) / len(filtered_ratio_list)
        else:
            mean_filtered_ratio = 0.0
        
        # Compute detection accuracy (same as mean filtered_ratio for recall-based metric)
        # Detection accuracy here represents the average proportion of malicious clients detected
        detection_accuracy = mean_filtered_ratio
        
        # Write summary to CSV
        with open(summary_path, 'w') as f:
            f.write('metric,value\n')
            f.write(f'steady_state_acc,{steady_state_acc:.6f}\n')
            f.write(f'steady_state_asr,{steady_state_asr:.6f}\n')
            f.write(f'mean_filtered_ratio,{mean_filtered_ratio:.6f}\n')
            f.write(f'detection_accuracy,{detection_accuracy:.6f}\n')
            f.write(f'total_epochs,{len(acc_list)}\n')
            f.write(f'window_size,{window_size}\n')
        
        # Also print summary to console
        print('\n' + '='*50)
        print('EXPERIMENT SUMMARY')
        print('='*50)
        print(f'Steady-state ACC (last {window_size} rounds): {steady_state_acc:.4f}')
        print(f'Steady-state ASR (last {window_size} rounds): {steady_state_asr:.4f}')
        print(f'Mean Filtered Ratio: {mean_filtered_ratio:.4f}')
        print(f'Detection Accuracy: {detection_accuracy:.4f}')
        print('='*50 + '\n')
        
        return {
            'steady_state_acc': steady_state_acc,
            'steady_state_asr': steady_state_asr,
            'mean_filtered_ratio': mean_filtered_ratio,
            'detection_accuracy': detection_accuracy
        }
