"""
FDCR Benign Client Implementation.

This module implements the FDCRBenignClient which extends BenignClient
to compute Fisher Information during local training for the FDCR defense.

Reference:
    FDCR: Fisher Discrepancy Cluster and Rescale for Backdoor Defense in Federated Learning
"""

import torch
import torch.nn as nn

from typing import Tuple, Dict, Any
from torch.utils.data import DataLoader
from backfed.const import ModelUpdate, Metrics
from backfed.clients.base_benign_client import BenignClient
from backfed.utils import log
from logging import INFO


class FDCRBenignClient(BenignClient):
    """
    FDCR Benign Client that computes Fisher Information during training.
    
    Fisher Information is used by the FDCR server to weight parameter updates
    and detect malicious clients based on divergence in parameter importance.
    
    Parameters
    ----------
    client_id : int
        Unique identifier for the client.
    dataset : Dataset
        Client's local dataset.
    model : nn.Module
        Model to train.
    client_config : DictConfig
        Client configuration.
    client_type : str, default='fdcr_benign'
        Type identifier for the client.
    verbose : bool, default=False
        Whether to log verbose output.
    **kwargs : dict
        Additional arguments passed to parent class.
    """
    
    def __init__(
        self,
        client_id,
        dataset,
        model,
        client_config,
        client_type: str = "fdcr_benign",
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(
            client_id=client_id,
            dataset=dataset,
            model=model,
            client_config=client_config,
            client_type=client_type,
            verbose=verbose,
            **kwargs
        )


    def compute_fisher_information(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information for model parameters.
        
        Fisher Information is approximated as the expected squared gradient:
        F_i = E[(∂L/∂θ_i)²]
        
        This measures the importance of each parameter for the local data distribution.
        
        Parameters
        ----------
        model : nn.Module
            The trained model.
        dataloader : DataLoader
            Local training data loader.
        device : torch.device
            Computation device.
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary mapping parameter names to Fisher information tensors.
        """
        model.eval()
        
        # Initialize Fisher information accumulator
        fisher_info = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param, device=device)
        
        total_samples = 0
        
        # Iterate through local training data
        for batch_idx, (data, targets) in enumerate(dataloader):
            # Handle different data formats
            if isinstance(data, dict):
                # Transformer models (e.g., ALBERT for sentiment)
                data = {k: v.to(device) for k, v in data.items()}
            else:
                data = data.to(device)
            targets = targets.to(device)
            
            batch_size = targets.size(0) if isinstance(targets, torch.Tensor) else len(targets)
            total_samples += batch_size
            
            # Zero gradients
            model.zero_grad()
            
            # Forward pass
            if isinstance(data, dict):
                outputs = model(**data)
                if isinstance(outputs, dict):
                    outputs = outputs.get('logits', outputs)
                elif hasattr(outputs, 'logits'):
                    outputs = outputs.logits
            else:
                outputs = model(data)
            
            # Compute loss
            loss = self.criterion(outputs.view(-1, outputs.size(-1)) if outputs.dim() > 2 else outputs, 
                                  targets.view(-1) if targets.dim() > 1 else targets)
            
            # Backward pass to compute gradients
            loss.backward()
            
            # Accumulate squared gradients (Fisher information)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_info[name] += (param.grad ** 2) * batch_size
        
        # Normalize by total number of samples
        if total_samples > 0:
            for name in fisher_info:
                fisher_info[name] /= total_samples
        
        model.train()
        return fisher_info


    def train(self, train_package: Dict[str, Any]) -> Tuple[int, ModelUpdate, Metrics, Dict[str, torch.Tensor]]:
        """
        Train the model and compute Fisher Information.
        
        Extends the parent train method to also compute and return Fisher Information
        for use by the FDCR server in malicious client detection.

        Args:
            train_package: Data package received from server to train the model 
                          (e.g., global model weights, learning rate, etc.)
                          
        Returns:
            num_examples (int): number of examples in the training dataset
            weight_diff_dict (ModelUpdate): updated model parameters
            training_metrics (Metrics): training metrics
            fisher_info (Dict[str, torch.Tensor]): Fisher information per parameter
        """
        # Call parent train method to perform standard training
        if self.client_config.dataset.upper() == "SENTIMENT140":
            num_examples, model_updates, training_metrics = self.train_albert_sentiment(train_package)
        elif self.client_config.dataset.upper() == "REDDIT":
            num_examples, model_updates, training_metrics = self.train_lstm_reddit(train_package)
        else:
            num_examples, model_updates, training_metrics = self.train_img_classifier(train_package)
        
        # Compute Fisher Information after training
        fisher_info = self.compute_fisher_information(
            model=self.model,
            dataloader=self.train_loader,
            device=self.device
        )
        
        if self.verbose:
            log(INFO, f"Client [{self.client_id}] ({self.client_type}) computed Fisher Information "
                f"for {len(fisher_info)} parameters")
        
        return num_examples, model_updates, training_metrics, fisher_info
    
    @staticmethod
    def get_client_type():
        return "fdcr_benign"
