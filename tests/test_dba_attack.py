"""
Property-based tests for DBA (Distributed Backdoor Attack) functionality.

Tests for FDCR reproduction experiments DBA attack module.
"""

import pytest
import copy
import torch
import numpy as np
from hypothesis import given, settings, strategies as st, assume
from typing import List, Tuple
from unittest.mock import MagicMock

from Attack.backdoor.utils import get_dba_trigger_partition, dba_backdoor


# **Feature: fdcr-reproduction, Property 1: DBA Trigger Partition Completeness and Disjointness**
# **Validates: Requirements 1.1, 1.4**
class TestDBAPartitionCompletenessAndDisjointness:
    """
    Property 1: DBA Trigger Partition Completeness and Disjointness
    
    For any set of malicious clients and a full trigger pattern, the union of all 
    client trigger partitions SHALL equal the full trigger pattern, and the 
    intersection of any two client partitions SHALL be empty.
    """
    
    @given(
        num_triggers=st.integers(min_value=1, max_value=100),
        num_partitions=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100)
    def test_partition_completeness(self, num_triggers: int, num_partitions: int):
        """
        Property: The union of all client trigger partitions SHALL equal 
        the full trigger pattern.
        
        This verifies that no trigger positions are lost during partitioning.
        """
        # Create full trigger positions as list of [channel, row, col] positions
        full_trigger_positions = [[0, i, j] for i in range(num_triggers) for j in range(1)]
        full_trigger_positions = full_trigger_positions[:num_triggers]  # Ensure exact count
        
        # Collect all partitions
        all_partitioned_positions = []
        for client_idx in range(num_partitions):
            partition = get_dba_trigger_partition(
                full_trigger_positions, 
                client_idx, 
                num_partitions
            )
            all_partitioned_positions.extend(partition)
        
        # Property: Union of all partitions equals full trigger set
        assert len(all_partitioned_positions) == len(full_trigger_positions), \
            f"Total partitioned positions ({len(all_partitioned_positions)}) " \
            f"should equal full trigger count ({len(full_trigger_positions)})"
        
        # Verify each position appears exactly once
        for i, pos in enumerate(full_trigger_positions):
            assert pos in all_partitioned_positions, \
                f"Position {pos} at index {i} missing from partitions"
    
    @given(
        num_triggers=st.integers(min_value=2, max_value=100),
        num_partitions=st.integers(min_value=2, max_value=20)
    )
    @settings(max_examples=100)
    def test_partition_disjointness(self, num_triggers: int, num_partitions: int):
        """
        Property: The intersection of any two client partitions SHALL be empty.
        
        This verifies that no trigger position is assigned to multiple clients.
        """
        # Create full trigger positions
        full_trigger_positions = [[0, i, 0] for i in range(num_triggers)]
        
        # Get all partitions
        partitions = []
        for client_idx in range(num_partitions):
            partition = get_dba_trigger_partition(
                full_trigger_positions, 
                client_idx, 
                num_partitions
            )
            partitions.append(partition)
        
        # Property: Check pairwise disjointness
        for i in range(num_partitions):
            for j in range(i + 1, num_partitions):
                # Convert to tuples for set operations
                set_i = set(tuple(pos) for pos in partitions[i])
                set_j = set(tuple(pos) for pos in partitions[j])
                
                intersection = set_i & set_j
                assert len(intersection) == 0, \
                    f"Partitions {i} and {j} have non-empty intersection: {intersection}"
    
    @given(
        num_triggers=st.integers(min_value=1, max_value=100),
        num_partitions=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100)
    def test_partition_preserves_order(self, num_triggers: int, num_partitions: int):
        """
        Property: Partitions maintain the original ordering of trigger positions.
        
        When concatenating partitions in client order, the result should match
        the original trigger position list.
        """
        # Create full trigger positions with unique identifiable values
        full_trigger_positions = [[c, r, col] for c, (r, col) in 
                                   enumerate([(i // 10, i % 10) for i in range(num_triggers)])]
        full_trigger_positions = full_trigger_positions[:num_triggers]
        
        # Concatenate all partitions in order
        reconstructed = []
        for client_idx in range(num_partitions):
            partition = get_dba_trigger_partition(
                full_trigger_positions, 
                client_idx, 
                num_partitions
            )
            reconstructed.extend(partition)
        
        # Property: Reconstructed list equals original
        assert reconstructed == full_trigger_positions, \
            "Concatenated partitions should equal original trigger positions"
    
    @given(
        num_triggers=st.integers(min_value=1, max_value=100),
        num_partitions=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100)
    def test_partition_size_balance(self, num_triggers: int, num_partitions: int):
        """
        Property: Partition sizes differ by at most 1.
        
        When triggers cannot be evenly divided, earlier clients get one extra,
        but no client should have more than ceil(n/k) or less than floor(n/k).
        """
        full_trigger_positions = [[0, i, 0] for i in range(num_triggers)]
        
        base_size = num_triggers // num_partitions
        remainder = num_triggers % num_partitions
        
        for client_idx in range(num_partitions):
            partition = get_dba_trigger_partition(
                full_trigger_positions, 
                client_idx, 
                num_partitions
            )
            
            expected_size = base_size + (1 if client_idx < remainder else 0)
            
            assert len(partition) == expected_size, \
                f"Client {client_idx} partition size {len(partition)} " \
                f"should be {expected_size} (base={base_size}, remainder={remainder})"
    
    @given(num_partitions=st.integers(min_value=1, max_value=20))
    @settings(max_examples=100)
    def test_empty_trigger_list(self, num_partitions: int):
        """
        Property: Empty trigger list results in empty partitions for all clients.
        """
        full_trigger_positions = []
        
        for client_idx in range(num_partitions):
            partition = get_dba_trigger_partition(
                full_trigger_positions, 
                client_idx, 
                num_partitions
            )
            
            assert partition == [], \
                f"Client {client_idx} should have empty partition for empty trigger list"


# **Feature: fdcr-reproduction, Property 2: DBA Trigger Application Correctness**
# **Validates: Requirements 1.2**
class TestDBATriggerApplicationCorrectness:
    """
    Property 2: DBA Trigger Application Correctness
    
    For any poisoned image created by a malicious client in DBA mode, only the 
    trigger positions assigned to that specific client SHALL be modified from 
    the original image.
    """
    
    @staticmethod
    def create_mock_cfg(trigger_positions, trigger_values, backdoor_label=2):
        """Create a mock configuration object for testing."""
        cfg = MagicMock()
        cfg.attack.backdoor.trigger_position = trigger_positions
        cfg.attack.backdoor.trigger_value = trigger_values
        cfg.attack.backdoor.backdoor_label = backdoor_label
        return cfg
    
    @given(
        img_channels=st.integers(min_value=1, max_value=3),
        img_height=st.integers(min_value=4, max_value=32),
        img_width=st.integers(min_value=4, max_value=32),
        num_triggers=st.integers(min_value=1, max_value=16),
        num_malicious_clients=st.integers(min_value=1, max_value=8),
        client_index=st.integers(min_value=0, max_value=7),
        original_label=st.integers(min_value=0, max_value=9),
    )
    @settings(max_examples=100)
    def test_only_assigned_positions_modified(
        self, 
        img_channels: int, 
        img_height: int, 
        img_width: int,
        num_triggers: int,
        num_malicious_clients: int,
        client_index: int,
        original_label: int
    ):
        """
        Property: Only the trigger positions assigned to a specific client 
        SHALL be modified from the original image.
        
        This verifies that DBA correctly applies only the client's trigger subset.
        """
        # Ensure client_index is valid
        assume(client_index < num_malicious_clients)
        
        # Ensure we have enough pixels for triggers
        total_pixels = img_channels * img_height * img_width
        assume(num_triggers <= total_pixels)
        
        # Create trigger positions within image bounds
        trigger_positions = []
        trigger_values = []
        for i in range(num_triggers):
            c = i % img_channels
            r = (i // img_channels) % img_height
            col = (i // (img_channels * img_height)) % img_width
            trigger_positions.append([c, r, col])
            trigger_values.append(1.0)  # Use 1.0 as trigger value
        
        cfg = self.create_mock_cfg(trigger_positions, trigger_values)
        
        # Create original image with zeros
        original_img = torch.zeros(img_channels, img_height, img_width)
        img_copy = original_img.clone()
        
        # Force poisoning by using noise_data_rate=1.0
        # We need to mock torch.rand to always return 0 (less than 1.0)
        original_rand = torch.rand
        torch.rand = lambda x: torch.tensor([0.0])
        
        try:
            result_img, result_target = dba_backdoor(
                cfg, img_copy, original_label, 1.0, client_index, num_malicious_clients
            )
        finally:
            torch.rand = original_rand
        
        # Get the expected partition for this client
        expected_partition = get_dba_trigger_partition(
            trigger_positions, client_index, num_malicious_clients
        )
        expected_positions_set = set(tuple(pos) for pos in expected_partition)
        
        # Check each pixel in the image
        for c in range(img_channels):
            for r in range(img_height):
                for col in range(img_width):
                    pos_tuple = (c, r, col)
                    original_value = original_img[c][r][col].item()
                    result_value = result_img[c][r][col].item()
                    
                    if pos_tuple in expected_positions_set:
                        # This position should be modified (trigger applied)
                        assert result_value == 1.0, \
                            f"Position {pos_tuple} should have trigger value 1.0, got {result_value}"
                    else:
                        # This position should NOT be modified
                        assert result_value == original_value, \
                            f"Position {pos_tuple} should be unchanged ({original_value}), got {result_value}"
    
    @given(
        num_triggers=st.integers(min_value=2, max_value=16),
        num_malicious_clients=st.integers(min_value=2, max_value=8),
    )
    @settings(max_examples=100)
    def test_different_clients_modify_different_positions(
        self,
        num_triggers: int,
        num_malicious_clients: int,
    ):
        """
        Property: Different clients in DBA mode modify disjoint sets of positions.
        
        This verifies that no two clients modify the same pixel position.
        """
        # Create a simple 4x4 image with 3 channels
        img_channels, img_height, img_width = 3, 4, 4
        
        # Ensure we have enough pixels
        total_pixels = img_channels * img_height * img_width
        assume(num_triggers <= total_pixels)
        
        # Create trigger positions
        trigger_positions = []
        trigger_values = []
        for i in range(num_triggers):
            c = i % img_channels
            r = (i // img_channels) % img_height
            col = (i // (img_channels * img_height)) % img_width
            trigger_positions.append([c, r, col])
            trigger_values.append(1.0)
        
        cfg = self.create_mock_cfg(trigger_positions, trigger_values)
        
        # Track which positions each client modifies
        client_modified_positions = []
        
        original_rand = torch.rand
        torch.rand = lambda x: torch.tensor([0.0])
        
        try:
            for client_idx in range(num_malicious_clients):
                original_img = torch.zeros(img_channels, img_height, img_width)
                result_img, _ = dba_backdoor(
                    cfg, original_img, 0, 1.0, client_idx, num_malicious_clients
                )
                
                # Find modified positions
                modified = set()
                for c in range(img_channels):
                    for r in range(img_height):
                        for col in range(img_width):
                            if result_img[c][r][col].item() != 0.0:
                                modified.add((c, r, col))
                
                client_modified_positions.append(modified)
        finally:
            torch.rand = original_rand
        
        # Verify disjointness: no two clients modify the same position
        for i in range(num_malicious_clients):
            for j in range(i + 1, num_malicious_clients):
                intersection = client_modified_positions[i] & client_modified_positions[j]
                assert len(intersection) == 0, \
                    f"Clients {i} and {j} both modified positions: {intersection}"
    
    @given(
        num_triggers=st.integers(min_value=1, max_value=16),
        num_malicious_clients=st.integers(min_value=1, max_value=8),
        client_index=st.integers(min_value=0, max_value=7),
    )
    @settings(max_examples=100)
    def test_target_label_changed_on_poisoning(
        self,
        num_triggers: int,
        num_malicious_clients: int,
        client_index: int,
    ):
        """
        Property: When poisoning occurs, the target label is changed to backdoor_label.
        """
        assume(client_index < num_malicious_clients)
        
        img_channels, img_height, img_width = 3, 4, 4
        total_pixels = img_channels * img_height * img_width
        assume(num_triggers <= total_pixels)
        
        trigger_positions = [[i % 3, (i // 3) % 4, (i // 12) % 4] for i in range(num_triggers)]
        trigger_values = [1.0] * num_triggers
        backdoor_label = 7
        
        cfg = self.create_mock_cfg(trigger_positions, trigger_values, backdoor_label)
        
        original_img = torch.zeros(img_channels, img_height, img_width)
        original_label = 3  # Different from backdoor_label
        
        original_rand = torch.rand
        torch.rand = lambda x: torch.tensor([0.0])
        
        try:
            _, result_target = dba_backdoor(
                cfg, original_img, original_label, 1.0, client_index, num_malicious_clients
            )
        finally:
            torch.rand = original_rand
        
        assert result_target == backdoor_label, \
            f"Target should be changed to {backdoor_label}, got {result_target}"
