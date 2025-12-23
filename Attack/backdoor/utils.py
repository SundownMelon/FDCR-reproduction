import copy
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader


def get_dba_trigger_partition(full_trigger_positions, client_index, num_partitions):
    """
    Get the trigger partition for a specific client in DBA attack.
    
    Partitions the full trigger positions evenly across malicious clients.
    When trigger count is not evenly divisible, earlier clients get one extra position.
    
    Args:
        full_trigger_positions: List of all trigger positions (e.g., [[0,0,0], [0,0,1], ...])
        client_index: Index of the client (0-based) within malicious clients
        num_partitions: Total number of partitions (typically equals number of malicious clients)
    
    Returns:
        list: Subset of trigger positions assigned to this client
    
    Raises:
        ValueError: If num_partitions <= 0 or client_index is out of range
    """
    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive")
    if client_index < 0 or client_index >= num_partitions:
        raise ValueError(f"client_index {client_index} out of range [0, {num_partitions})")
    
    n_triggers = len(full_trigger_positions)
    if n_triggers == 0:
        return []
    
    # Calculate base partition size and remainder
    base_size = n_triggers // num_partitions
    remainder = n_triggers % num_partitions
    
    # Earlier clients (indices < remainder) get one extra trigger
    if client_index < remainder:
        start = client_index * (base_size + 1)
        end = start + base_size + 1
    else:
        start = remainder * (base_size + 1) + (client_index - remainder) * base_size
        end = start + base_size
    
    return full_trigger_positions[start:end]


def base_backdoor(cfg, img, target, noise_data_rate):
    if torch.rand(1) < noise_data_rate:
        target = cfg.attack.backdoor.backdoor_label
        for pos_index in range(0, len(cfg.attack.backdoor.trigger_position)):
            pos = cfg.attack.backdoor.trigger_position[pos_index]
            img[pos[0]][pos[1]][pos[2]] = cfg.attack.backdoor.trigger_value[pos_index]
    return img, target


def semantic_backdoor(cfg, img, target, noise_data_rate):
    if torch.rand(1) < noise_data_rate:
        if target == cfg.attack.backdoor.semantic_backdoor_label:
            target = cfg.attack.backdoor.backdoor_label

            # img, _ = dataset.__getitem__(used_index)
            img = img + torch.randn(img.size()) * 0.05

    return img, target


def dba_backdoor(cfg, img, target, noise_data_rate, client_index, num_malicious_clients):
    """
    DBA (Distributed Backdoor Attack) - applies only a subset of trigger to each client.
    
    In DBA, the full trigger pattern is partitioned across malicious clients.
    Each client only applies their assigned trigger subset to poisoned samples.
    During testing, the full trigger is applied to evaluate ASR.
    
    Args:
        cfg: Configuration object containing backdoor settings
        img: Input image tensor
        target: Original label
        noise_data_rate: Probability of poisoning a sample (0.0 to 1.0)
        client_index: Index of this client within malicious clients (0-based)
        num_malicious_clients: Total number of malicious clients
    
    Returns:
        img: Processed image (may have partial trigger applied)
        target: Processed label (may be changed to backdoor target label)
    """
    if torch.rand(1) < noise_data_rate:
        target = cfg.attack.backdoor.backdoor_label
        
        # Get this client's trigger partition
        full_positions = cfg.attack.backdoor.trigger_position
        full_values = cfg.attack.backdoor.trigger_value
        
        # Get partition indices for this client
        partition_indices = []
        n_triggers = len(full_positions)
        base_size = n_triggers // num_malicious_clients
        remainder = n_triggers % num_malicious_clients
        
        if client_index < remainder:
            start = client_index * (base_size + 1)
            end = start + base_size + 1
        else:
            start = remainder * (base_size + 1) + (client_index - remainder) * base_size
            end = start + base_size
        
        # Apply only this client's trigger subset
        for pos_index in range(start, end):
            pos = full_positions[pos_index]
            img[pos[0]][pos[1]][pos[2]] = full_values[pos_index]
    
    return img, target

def backdoor_attack(args, cfg, client_type, private_dataset, is_train, malicious_client_indices=None):
    """
    Apply backdoor attack to the dataset.
    
    Args:
        args: Command line arguments
        cfg: Configuration object
        client_type: List of booleans, True for benign clients, False for malicious
        private_dataset: Dataset object containing train_loaders and test_loader
        is_train: Whether this is for training (True) or testing (False)
        malicious_client_indices: List of malicious client indices (required for DBA)
    """
    noise_data_rate = cfg.attack.noise_data_rate if is_train else 1.0
    attack_type = cfg.attack.backdoor.evils
    
    if is_train:
        if attack_type == 'dba_backdoor':
            # DBA: Each malicious client gets their own dataset with their trigger partition
            _apply_dba_train(args, cfg, client_type, private_dataset, noise_data_rate, malicious_client_indices)
        else:
            # Non-DBA attacks: All malicious clients share the same poisoned dataset
            dataset = copy.deepcopy(private_dataset.train_loaders[0].dataset)

            all_targets = []
            all_imgs = []

            for i in range(len(dataset)):
                img, target = dataset.__getitem__(i)
                if attack_type == 'base_backdoor':
                    img, target = base_backdoor(cfg, (img), (target), noise_data_rate)

                if attack_type == 'semantic_backdoor':
                    img, target = semantic_backdoor(cfg, (img), (target), noise_data_rate)

                all_targets.append(target)
                all_imgs.append(img.numpy())

            new_dataset = BackdoorDataset(all_imgs, all_targets)

            for client_index in tqdm(range(cfg.DATASET.parti_num), desc="Processing Clients"):
                if not client_type[client_index]:
                    train_sampler = private_dataset.train_loaders[client_index].batch_sampler.sampler

                    if args.task == 'label_skew':
                        private_dataset.train_loaders[client_index] = DataLoader(new_dataset, batch_size=cfg.OPTIMIZER.local_train_batch,
                                                                                 sampler=train_sampler, num_workers=4, drop_last=True)

    else:
        # Testing: Always apply full trigger for ASR evaluation
        if args.task == 'label_skew':
            dataset = copy.deepcopy(private_dataset.test_loader.dataset)

            all_targets = []
            all_imgs = []

            for i in range(len(dataset)):
                img, target = dataset.__getitem__(i)
                # For testing, always use base_backdoor (full trigger) regardless of attack type
                # This is because DBA testing requires the full reconstructed trigger
                if attack_type in ['base_backdoor', 'dba_backdoor']:
                    img, target = base_backdoor(cfg, copy.deepcopy(img), copy.deepcopy(target), 1.0)

                    all_targets.append(target)
                    all_imgs.append(img.numpy())
                elif attack_type == 'semantic_backdoor':
                    if target == cfg.attack.backdoor.semantic_backdoor_label:
                        img, target = semantic_backdoor(cfg, copy.deepcopy(img), copy.deepcopy(target), 1.0)
                        all_targets.append(target)
                        all_imgs.append(img.numpy())
            new_dataset = BackdoorDataset(all_imgs, all_targets)
            private_dataset.backdoor_test_loader = DataLoader(new_dataset, batch_size=cfg.OPTIMIZER.local_train_batch, num_workers=4)


def _apply_dba_train(args, cfg, client_type, private_dataset, noise_data_rate, malicious_client_indices):
    """
    Apply DBA attack during training - each malicious client gets their own trigger partition.
    
    Args:
        args: Command line arguments
        cfg: Configuration object
        client_type: List of booleans, True for benign, False for malicious
        private_dataset: Dataset object
        noise_data_rate: Probability of poisoning
        malicious_client_indices: List of malicious client indices
    """
    if malicious_client_indices is None:
        # Derive malicious indices from client_type
        malicious_client_indices = [i for i, is_benign in enumerate(client_type) if not is_benign]
    
    num_malicious = len(malicious_client_indices)
    if num_malicious == 0:
        return
    
    # Create a mapping from global client index to malicious client index (0-based)
    malicious_index_map = {global_idx: local_idx for local_idx, global_idx in enumerate(malicious_client_indices)}
    
    for client_index in tqdm(range(cfg.DATASET.parti_num), desc="Processing DBA Clients"):
        if not client_type[client_index]:
            # This is a malicious client - create their specific poisoned dataset
            local_malicious_idx = malicious_index_map[client_index]
            
            dataset = copy.deepcopy(private_dataset.train_loaders[0].dataset)
            all_targets = []
            all_imgs = []
            
            for i in range(len(dataset)):
                img, target = dataset.__getitem__(i)
                img, target = dba_backdoor(cfg, img, target, noise_data_rate, 
                                          local_malicious_idx, num_malicious)
                all_targets.append(target)
                all_imgs.append(img.numpy())
            
            new_dataset = BackdoorDataset(all_imgs, all_targets)
            train_sampler = private_dataset.train_loaders[client_index].batch_sampler.sampler
            
            if args.task == 'label_skew':
                private_dataset.train_loaders[client_index] = DataLoader(
                    new_dataset, 
                    batch_size=cfg.OPTIMIZER.local_train_batch,
                    sampler=train_sampler, 
                    num_workers=4, 
                    drop_last=True
                )


class BackdoorDataset(torch.utils.data.Dataset):

    def __init__(self, data, labels):
        self.data = np.array(data)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
