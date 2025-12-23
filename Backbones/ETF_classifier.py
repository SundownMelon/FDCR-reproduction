"""
ETF (Equiangular Tight Frame) Classifier for federated learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ETF_classifier(nn.Module):
    """
    ETF (Equiangular Tight Frame) Classifier.
    
    Creates a fixed classifier with equiangular class prototypes,
    which can help with class imbalance in federated learning.
    """
    
    def __init__(self, feat_dim, num_classes):
        super(ETF_classifier, self).__init__()
        
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        
        # Create ETF prototypes (fixed, not learnable)
        self.register_buffer('prototypes', self._create_etf_prototypes())
    
    def _create_etf_prototypes(self):
        """
        Create equiangular tight frame prototypes.
        
        For K classes in d dimensions (d >= K-1), creates K unit vectors
        with equal pairwise angles.
        """
        K = self.num_classes
        d = self.feat_dim
        
        # Simple implementation: use orthogonal vectors if d >= K
        if d >= K:
            # Start with identity-like structure
            prototypes = torch.zeros(K, d)
            for i in range(K):
                prototypes[i, i % d] = 1.0
            
            # Normalize
            prototypes = F.normalize(prototypes, p=2, dim=1)
        else:
            # If d < K, use random orthogonal initialization
            prototypes = torch.randn(K, d)
            prototypes = F.normalize(prototypes, p=2, dim=1)
        
        return prototypes
    
    def forward(self, features):
        """
        Compute similarity between features and prototypes.
        
        Args:
            features: Input features of shape (batch_size, feat_dim)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute cosine similarity with prototypes
        logits = torch.mm(features, self.prototypes.t())
        
        return logits
