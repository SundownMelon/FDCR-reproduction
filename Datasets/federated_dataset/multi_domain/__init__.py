"""
Multi-domain federated dataset module.

This is a placeholder module. The FDCR reproduction experiments
use single_domain datasets (CIFAR-10, etc.) with label skew.
"""

# Multi-domain dataset names (placeholder - not used in current experiments)
multi_domain_dataset_name = []


def get_multi_domain_dataset(args, cfg):
    """
    Get multi-domain federated dataset.
    
    Note: This is a placeholder. Current experiments use single_domain datasets.
    """
    raise NotImplementedError(
        "Multi-domain datasets are not implemented. "
        "Use single_domain datasets (fl_cifar10, fl_mnist, etc.) instead."
    )
