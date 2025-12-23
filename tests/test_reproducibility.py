"""
Property-based tests for experiment reproducibility.

Tests for FDCR reproduction experiments reproducibility functionality.
"""

import pytest
import random
import numpy as np
import torch
from hypothesis import given, settings, strategies as st, assume
from typing import List, Tuple
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.conf import set_random_seed
from run_experiments import (
    generate_experiment_configs,
    ExperimentConfig,
    build_command,
    DEFAULT_ATTACK_PATTERNS,
    DEFAULT_ALPHA_VALUES,
    DEFAULT_SEEDS,
)


# **Feature: fdcr-reproduction, Property 7: Experiment Reproducibility**
# **Validates: Requirements 5.2**
class TestExperimentReproducibility:
    """
    Property 7: Experiment Reproducibility
    
    For any experiment configuration run twice with the same random seed,
    the ACC and ASR sequences SHALL be identical.
    
    This test suite verifies that:
    1. Random seed setting produces deterministic random sequences
    2. Experiment configurations are generated consistently
    3. Command building is deterministic for the same config
    """
    
    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=100)
    def test_random_seed_produces_deterministic_sequence(self, seed: int):
        """
        Property: For any seed, setting the random seed twice produces
        identical random sequences from all random sources.
        
        This is the foundation of experiment reproducibility - if random
        number generation is deterministic, then model training will be too.
        """
        # First run with seed
        set_random_seed(seed)
        random_seq_1 = [random.random() for _ in range(10)]
        np_seq_1 = np.random.rand(10).tolist()
        torch_seq_1 = torch.rand(10).tolist()
        
        # Second run with same seed
        set_random_seed(seed)
        random_seq_2 = [random.random() for _ in range(10)]
        np_seq_2 = np.random.rand(10).tolist()
        torch_seq_2 = torch.rand(10).tolist()
        
        # Property: Sequences must be identical
        assert random_seq_1 == random_seq_2, \
            f"Python random sequences differ for seed {seed}"
        assert np_seq_1 == np_seq_2, \
            f"NumPy random sequences differ for seed {seed}"
        assert torch_seq_1 == torch_seq_2, \
            f"PyTorch random sequences differ for seed {seed}"
    
    @given(
        seed1=st.integers(min_value=0, max_value=2**31 - 1),
        seed2=st.integers(min_value=0, max_value=2**31 - 1)
    )
    @settings(max_examples=100)
    def test_different_seeds_produce_different_sequences(self, seed1: int, seed2: int):
        """
        Property: For any two different seeds, the random sequences
        should be different (with very high probability).
        
        This ensures seeds actually affect the random state.
        """
        assume(seed1 != seed2)
        
        # First seed
        set_random_seed(seed1)
        seq_1 = [random.random() for _ in range(10)]
        
        # Second seed
        set_random_seed(seed2)
        seq_2 = [random.random() for _ in range(10)]
        
        # Property: Sequences should differ (extremely unlikely to be equal)
        assert seq_1 != seq_2, \
            f"Different seeds {seed1} and {seed2} produced identical sequences"
    
    @given(
        attack_patterns=st.lists(
            st.sampled_from(['base_backdoor', 'dba_backdoor']),
            min_size=1,
            max_size=2,
            unique=True
        ),
        alpha_values=st.lists(
            st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=3,
            unique=True
        ),
        seeds=st.lists(
            st.integers(min_value=0, max_value=100),
            min_size=1,
            max_size=5,
            unique=True
        ),
        dataset=st.sampled_from(['fl_cifar10', 'fl_cifar100', 'fl_mnist'])
    )
    @settings(max_examples=100)
    def test_config_generation_is_deterministic(
        self,
        attack_patterns: List[str],
        alpha_values: List[float],
        seeds: List[int],
        dataset: str
    ):
        """
        Property: For any set of input parameters, generate_experiment_configs
        produces identical configuration lists when called multiple times.
        
        This ensures experiment batches are reproducible.
        """
        # First generation
        configs_1 = generate_experiment_configs(
            attack_patterns=attack_patterns,
            alpha_values=alpha_values,
            seeds=seeds,
            dataset=dataset
        )
        
        # Second generation
        configs_2 = generate_experiment_configs(
            attack_patterns=attack_patterns,
            alpha_values=alpha_values,
            seeds=seeds,
            dataset=dataset
        )
        
        # Property: Config lists must be identical
        assert len(configs_1) == len(configs_2), \
            "Config list lengths differ"
        
        for i, (c1, c2) in enumerate(zip(configs_1, configs_2)):
            assert c1.attack_pattern == c2.attack_pattern, \
                f"Attack pattern differs at index {i}"
            assert c1.alpha == c2.alpha, \
                f"Alpha differs at index {i}"
            assert c1.seed == c2.seed, \
                f"Seed differs at index {i}"
            assert c1.dataset == c2.dataset, \
                f"Dataset differs at index {i}"
    
    @given(
        attack_patterns=st.lists(
            st.sampled_from(['base_backdoor', 'dba_backdoor']),
            min_size=1,
            max_size=2,
            unique=True
        ),
        alpha_values=st.lists(
            st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=3,
            unique=True
        ),
        seeds=st.lists(
            st.integers(min_value=0, max_value=100),
            min_size=1,
            max_size=5,
            unique=True
        )
    )
    @settings(max_examples=100)
    def test_config_count_equals_cartesian_product(
        self,
        attack_patterns: List[str],
        alpha_values: List[float],
        seeds: List[int]
    ):
        """
        Property: The number of generated configs equals the Cartesian product
        of all input parameter lists.
        
        This ensures all combinations are generated for reproducibility.
        """
        configs = generate_experiment_configs(
            attack_patterns=attack_patterns,
            alpha_values=alpha_values,
            seeds=seeds
        )
        
        expected_count = len(attack_patterns) * len(alpha_values) * len(seeds)
        
        # Property: Config count must equal Cartesian product size
        assert len(configs) == expected_count, \
            f"Expected {expected_count} configs, got {len(configs)}"
    
    @given(
        attack=st.sampled_from(['base_backdoor', 'dba_backdoor']),
        alpha=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        seed=st.integers(min_value=0, max_value=1000),
        dataset=st.sampled_from(['fl_cifar10', 'fl_cifar100', 'fl_mnist']),
        device_id=st.integers(min_value=0, max_value=7)
    )
    @settings(max_examples=100)
    def test_command_building_is_deterministic(
        self,
        attack: str,
        alpha: float,
        seed: int,
        dataset: str,
        device_id: int
    ):
        """
        Property: For any experiment configuration, build_command produces
        identical command lists when called multiple times.
        
        This ensures the same experiment can be re-run with identical parameters.
        """
        config = ExperimentConfig(
            attack_pattern=attack,
            alpha=alpha,
            seed=seed,
            dataset=dataset
        )
        
        # Build command twice
        cmd_1 = build_command(config, device_id)
        cmd_2 = build_command(config, device_id)
        
        # Property: Commands must be identical
        assert cmd_1 == cmd_2, \
            f"Commands differ for same config: {cmd_1} vs {cmd_2}"
    
    @given(
        attack=st.sampled_from(['base_backdoor', 'dba_backdoor']),
        alpha=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        seed=st.integers(min_value=0, max_value=1000),
        dataset=st.sampled_from(['fl_cifar10', 'fl_cifar100', 'fl_mnist'])
    )
    @settings(max_examples=100)
    def test_command_contains_seed_parameter(
        self,
        attack: str,
        alpha: float,
        seed: int,
        dataset: str
    ):
        """
        Property: For any experiment configuration, the built command
        contains the --seed parameter with the correct value.
        
        This ensures reproducibility is enforced at the command level.
        """
        config = ExperimentConfig(
            attack_pattern=attack,
            alpha=alpha,
            seed=seed,
            dataset=dataset
        )
        
        cmd = build_command(config)
        
        # Property: Command must contain --seed with correct value
        assert '--seed' in cmd, "Command missing --seed parameter"
        seed_idx = cmd.index('--seed')
        assert cmd[seed_idx + 1] == str(seed), \
            f"Seed value mismatch: expected {seed}, got {cmd[seed_idx + 1]}"
    
    @given(
        attack=st.sampled_from(['base_backdoor', 'dba_backdoor']),
        alpha=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        seed=st.integers(min_value=0, max_value=1000),
        dataset=st.sampled_from(['fl_cifar10', 'fl_cifar100', 'fl_mnist'])
    )
    @settings(max_examples=100)
    def test_command_contains_alpha_parameter(
        self,
        attack: str,
        alpha: float,
        seed: int,
        dataset: str
    ):
        """
        Property: For any experiment configuration, the built command
        contains the DATASET.beta parameter with the correct alpha value.
        
        This ensures Non-IID settings are correctly passed.
        """
        config = ExperimentConfig(
            attack_pattern=attack,
            alpha=alpha,
            seed=seed,
            dataset=dataset
        )
        
        cmd = build_command(config)
        
        # Property: Command must contain DATASET.beta with correct value
        beta_param = f'DATASET.beta={alpha}'
        assert beta_param in cmd, \
            f"Command missing DATASET.beta parameter: {beta_param} not in {cmd}"
    
    @given(
        attack=st.sampled_from(['base_backdoor', 'dba_backdoor']),
        alpha=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        seed=st.integers(min_value=0, max_value=1000),
        dataset=st.sampled_from(['fl_cifar10', 'fl_cifar100', 'fl_mnist'])
    )
    @settings(max_examples=100)
    def test_command_contains_attack_pattern(
        self,
        attack: str,
        alpha: float,
        seed: int,
        dataset: str
    ):
        """
        Property: For any experiment configuration, the built command
        contains the attack.backdoor.evils parameter with the correct attack pattern.
        
        This ensures attack configuration is correctly passed.
        """
        config = ExperimentConfig(
            attack_pattern=attack,
            alpha=alpha,
            seed=seed,
            dataset=dataset
        )
        
        cmd = build_command(config)
        
        # Property: Command must contain attack.backdoor.evils with correct value
        attack_param = f'attack.backdoor.evils={attack}'
        assert attack_param in cmd, \
            f"Command missing attack.backdoor.evils parameter: {attack_param} not in {cmd}"
    
    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=100)
    def test_torch_deterministic_operations(self, seed: int):
        """
        Property: For any seed, PyTorch operations that depend on random
        state produce identical results when the seed is reset.
        
        This tests that model initialization and training would be reproducible.
        """
        # First run
        set_random_seed(seed)
        tensor_1 = torch.randn(5, 5)
        linear_1 = torch.nn.Linear(5, 3)
        weight_1 = linear_1.weight.clone()
        
        # Second run with same seed
        set_random_seed(seed)
        tensor_2 = torch.randn(5, 5)
        linear_2 = torch.nn.Linear(5, 3)
        weight_2 = linear_2.weight.clone()
        
        # Property: Tensors and weights must be identical
        assert torch.equal(tensor_1, tensor_2), \
            f"Random tensors differ for seed {seed}"
        assert torch.equal(weight_1, weight_2), \
            f"Linear layer weights differ for seed {seed}"
    
    @given(
        seeds=st.lists(
            st.integers(min_value=0, max_value=100),
            min_size=1,
            max_size=10,
            unique=True
        )
    )
    @settings(max_examples=100)
    def test_each_seed_in_config_is_unique(self, seeds: List[int]):
        """
        Property: When generating configs with multiple seeds, each seed
        appears exactly once per (attack, alpha) combination.
        
        This ensures proper coverage for reproducibility testing.
        """
        configs = generate_experiment_configs(
            attack_patterns=['base_backdoor'],
            alpha_values=[0.5],
            seeds=seeds
        )
        
        # Extract seeds from configs
        config_seeds = [c.seed for c in configs]
        
        # Property: Each input seed appears exactly once
        assert sorted(config_seeds) == sorted(seeds), \
            f"Config seeds {config_seeds} don't match input seeds {seeds}"

