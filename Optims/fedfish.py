"""
FedFish Optimizer Implementation.

This module implements the FedFish optimizer which extends FederatedOptim
to compute Fisher Information during local training for the FDCR defense.

Reference:
    FDCR: Fisher Discrepancy Cluster and Rescale for Backdoor Defense in Federated Learning
"""

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from typing import Dict
from Optims.utils.federated_optim import FederatedOptim


class FedFish(FederatedOptim):
    """
    Federated optimizer with Fisher Information computation.
    
    Used for FDCR defense method. Computes Fisher Information for each client
    after local training, which is used by the server to weight parameter updates
    and detect malicious clients based on divergence in parameter importance.
    """
    NAME = 'FedFish'

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(FedFish, self).__init__(nets_list, client_domain_list, args, cfg)
        self.local_fish_dict = {}

    def ini(self):
        """Initialize global model and synchronize all client models."""
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def compute_fisher_information(
        self,
        net: nn.Module,
        data_loader,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information for model parameters.
        
        Fisher Information is approximated as the expected squared gradient:
        F_i = E[(∂L/∂θ_i)²]
        
        This measures the importance of each parameter for the local data distribution.
        
        Args:
            net: The trained neural network model.
            data_loader: Local training data loader.
            
        Returns:
            Dictionary mapping parameter names to Fisher information tensors.
        """
        net.eval()
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        
        # Initialize Fisher information accumulator
        fisher_info = {}
        for name, param in net.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param, device=self.device)
        
        total_samples = 0
        
        # Iterate through local training data
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            batch_size = labels.size(0)
            total_samples += batch_size
            
            # Zero gradients
            net.zero_grad()
            
            # Forward pass
            outputs = net(images)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass to compute gradients
            loss.backward()
            
            # Accumulate squared gradients (Fisher information)
            for name, param in net.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_info[name] += (param.grad ** 2) * batch_size
        
        # Normalize by total number of samples
        if total_samples > 0:
            for name in fisher_info:
                fisher_info[name] /= total_samples
        
        net.train()
        return fisher_info

    def loc_update(self, priloader_list):
        """
        Perform local updates for all clients and compute Fisher information.
        
        For each client:
        1. Train the local model for local_epoch epochs
        2. Compute Fisher information after training
        3. Store Fisher info in self.local_fish_dict[client_index]
        
        Args:
            priloader_list: List of data loaders for each client.
            
        Returns:
            None
        """
        total_clients = list(range(self.cfg.DATASET.parti_num))
        self.online_clients_list = total_clients
        
        # Clear previous round's Fisher information
        self.local_fish_dict = {}

        for i in self.online_clients_list:
            # Train the client's network
            self._train_net(i, self.nets_list[i], priloader_list[i])
            
            # Compute Fisher information after training
            fisher_info = self.compute_fisher_information(
                self.nets_list[i], 
                priloader_list[i]
            )
            
            # Store Fisher info for server access
            self.local_fish_dict[i] = fisher_info
        
        return None

    def _train_net(self, index, net, train_loader):
        """
        Train a single client's network.
        
        Args:
            index: Client index.
            net: Client's neural network.
            train_loader: Client's training data loader.
        """
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(
            net.parameters(), 
            lr=self.local_lr, 
            momentum=0.9, 
            weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        
        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss)
                optimizer.step()
