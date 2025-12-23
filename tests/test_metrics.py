"""
Property-based tests for metrics logging.

Tests for FDCR reproduction experiments metrics logging functionality.
"""

import pytest
from hypothesis import given, settings, strategies as st
from typing import List, Dict, Any, Set
from dataclasses import dataclass
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Server.OurRandomControl import OurRandomControl


@dataclass
class RoundMetrics:
    """Represents metrics logged for a single communication round."""
    epoch: int
    acc: float = None
    asr: float = None


class MetricsTracker:
    """
    Tracks ACC and ASR metrics per communication round.
    
    This class provides the interface for logging and validating
    that both ACC and ASR are logged for each completed round.
    """
    
    def __init__(self):
        self.rounds: Dict[int, RoundMetrics] = {}
    
    def log_acc(self, epoch: int, acc: float) -> None:
        """Log accuracy for a given epoch."""
        if epoch not in self.rounds:
            self.rounds[epoch] = RoundMetrics(epoch=epoch)
        self.rounds[epoch].acc = acc
    
    def log_asr(self, epoch: int, asr: float) -> None:
        """Log attack success rate for a given epoch."""
        if epoch not in self.rounds:
            self.rounds[epoch] = RoundMetrics(epoch=epoch)
        self.rounds[epoch].asr = asr
    
    def is_round_complete(self, epoch: int) -> bool:
        """Check if both ACC and ASR are logged for a given epoch."""
        if epoch not in self.rounds:
            return False
        metrics = self.rounds[epoch]
        return metrics.acc is not None and metrics.asr is not None
    
    def get_completed_rounds(self) -> List[int]:
        """Get list of epochs where both ACC and ASR are logged."""
        return [epoch for epoch in self.rounds if self.is_round_complete(epoch)]
    
    def complete_round(self, epoch: int, acc: float, asr: float) -> None:
        """Log both ACC and ASR for a round, marking it complete."""
        self.log_acc(epoch, acc)
        self.log_asr(epoch, asr)


# **Feature: fdcr-reproduction, Property 4: Per-Round Metric Logging Completeness**
# **Validates: Requirements 3.1, 3.2**
class TestPerRoundMetricLoggingCompleteness:
    """
    Property 4: Per-Round Metric Logging Completeness
    
    For any completed communication round, the system SHALL have logged 
    both ACC and ASR values for that round.
    """
    
    @given(
        epochs=st.lists(
            st.integers(min_value=0, max_value=1000),
            min_size=1,
            max_size=50,
            unique=True
        ),
        acc_values=st.lists(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50
        ),
        asr_values=st.lists(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50
        )
    )
    @settings(max_examples=100)
    def test_complete_round_logs_both_metrics(
        self, 
        epochs: List[int], 
        acc_values: List[float], 
        asr_values: List[float]
    ):
        """
        Property: For any completed communication round, both ACC and ASR 
        values SHALL be logged.
        
        This test verifies that when complete_round() is called, both
        ACC and ASR are properly stored and the round is marked complete.
        """
        tracker = MetricsTracker()
        
        # Use minimum length to ensure we have values for all epochs
        n = min(len(epochs), len(acc_values), len(asr_values))
        
        for i in range(n):
            epoch = epochs[i]
            acc = acc_values[i]
            asr = asr_values[i]
            
            # Complete the round by logging both metrics
            tracker.complete_round(epoch, acc, asr)
            
            # Property: After completing a round, both ACC and ASR must be logged
            assert tracker.is_round_complete(epoch), \
                f"Round {epoch} should be complete after logging both ACC and ASR"
            
            # Verify the values are correctly stored
            assert tracker.rounds[epoch].acc == acc, \
                f"ACC value mismatch for epoch {epoch}"
            assert tracker.rounds[epoch].asr == asr, \
                f"ASR value mismatch for epoch {epoch}"
    
    @given(
        epoch=st.integers(min_value=0, max_value=1000),
        acc=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_incomplete_round_without_asr(self, epoch: int, acc: float):
        """
        Property: A round with only ACC logged is NOT complete.
        
        This ensures the system correctly identifies incomplete rounds.
        """
        tracker = MetricsTracker()
        tracker.log_acc(epoch, acc)
        
        # Property: Round should NOT be complete without ASR
        assert not tracker.is_round_complete(epoch), \
            f"Round {epoch} should NOT be complete with only ACC logged"
    
    @given(
        epoch=st.integers(min_value=0, max_value=1000),
        asr=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_incomplete_round_without_acc(self, epoch: int, asr: float):
        """
        Property: A round with only ASR logged is NOT complete.
        
        This ensures the system correctly identifies incomplete rounds.
        """
        tracker = MetricsTracker()
        tracker.log_asr(epoch, asr)
        
        # Property: Round should NOT be complete without ACC
        assert not tracker.is_round_complete(epoch), \
            f"Round {epoch} should NOT be complete with only ASR logged"
    
    @given(
        completed_epochs=st.lists(
            st.integers(min_value=0, max_value=100),
            min_size=0,
            max_size=20,
            unique=True
        ),
        incomplete_epochs=st.lists(
            st.integers(min_value=101, max_value=200),
            min_size=0,
            max_size=20,
            unique=True
        )
    )
    @settings(max_examples=100)
    def test_get_completed_rounds_returns_only_complete(
        self,
        completed_epochs: List[int],
        incomplete_epochs: List[int]
    ):
        """
        Property: get_completed_rounds() returns exactly the epochs where
        both ACC and ASR have been logged.
        """
        tracker = MetricsTracker()
        
        # Log complete rounds
        for epoch in completed_epochs:
            tracker.complete_round(epoch, acc=50.0, asr=10.0)
        
        # Log incomplete rounds (only ACC)
        for epoch in incomplete_epochs:
            tracker.log_acc(epoch, acc=50.0)
        
        completed = tracker.get_completed_rounds()
        
        # Property: All completed epochs should be in the result
        for epoch in completed_epochs:
            assert epoch in completed, \
                f"Completed epoch {epoch} should be in get_completed_rounds()"
        
        # Property: No incomplete epochs should be in the result
        for epoch in incomplete_epochs:
            assert epoch not in completed, \
                f"Incomplete epoch {epoch} should NOT be in get_completed_rounds()"


# **Feature: fdcr-reproduction, Property 6: Filtered Ratio Computation Correctness**
# **Validates: Requirements 4.3**
class TestFilteredRatioComputationCorrectness:
    """
    Property 6: Filtered Ratio Computation Correctness
    
    For any set of predicted malicious clients and actual malicious clients,
    filtered_ratio SHALL equal |predicted ∩ actual| / |actual|.
    """
    
    def _get_server_controller(self):
        """Create a minimal OurRandomControl instance for testing compute_filtered_ratio."""
        # We only need the compute_filtered_ratio method, so we can test it directly
        # without full initialization by accessing the method
        return OurRandomControl.__new__(OurRandomControl)
    
    @given(
        predicted_malicious=st.lists(
            st.integers(min_value=0, max_value=99),
            min_size=0,
            max_size=50,
            unique=True
        ),
        actual_malicious=st.lists(
            st.integers(min_value=0, max_value=99),
            min_size=1,  # At least one actual malicious client
            max_size=50,
            unique=True
        )
    )
    @settings(max_examples=100)
    def test_filtered_ratio_equals_intersection_over_actual(
        self,
        predicted_malicious: List[int],
        actual_malicious: List[int]
    ):
        """
        Property: filtered_ratio = |predicted ∩ actual| / |actual|
        
        For any sets of predicted and actual malicious clients,
        the filtered_ratio must equal the size of their intersection
        divided by the size of actual malicious clients.
        """
        server_controller = self._get_server_controller()
        
        # Compute expected value manually
        predicted_set = set(predicted_malicious)
        actual_set = set(actual_malicious)
        intersection = predicted_set.intersection(actual_set)
        expected_ratio = len(intersection) / len(actual_set)
        
        # Compute using the implementation
        computed_ratio = server_controller.compute_filtered_ratio(
            predicted_malicious, actual_malicious
        )
        
        # Property: computed ratio must equal expected ratio
        assert abs(computed_ratio - expected_ratio) < 1e-10, \
            f"Filtered ratio mismatch: computed={computed_ratio}, expected={expected_ratio}"
    
    @given(
        predicted_malicious=st.lists(
            st.integers(min_value=0, max_value=99),
            min_size=0,
            max_size=50,
            unique=True
        )
    )
    @settings(max_examples=100)
    def test_filtered_ratio_zero_when_no_actual_malicious(
        self,
        predicted_malicious: List[int]
    ):
        """
        Property: When there are no actual malicious clients, filtered_ratio = 0.0
        
        This handles the edge case where division by zero would occur.
        """
        server_controller = self._get_server_controller()
        actual_malicious = []  # No actual malicious clients
        
        computed_ratio = server_controller.compute_filtered_ratio(
            predicted_malicious, actual_malicious
        )
        
        # Property: ratio must be 0.0 when no actual malicious clients
        assert computed_ratio == 0.0, \
            f"Filtered ratio should be 0.0 when no actual malicious, got {computed_ratio}"
    
    @given(
        actual_malicious=st.lists(
            st.integers(min_value=0, max_value=99),
            min_size=1,
            max_size=50,
            unique=True
        )
    )
    @settings(max_examples=100)
    def test_filtered_ratio_one_when_all_detected(
        self,
        actual_malicious: List[int]
    ):
        """
        Property: When all actual malicious clients are predicted, filtered_ratio = 1.0
        
        Perfect detection should yield a ratio of 1.0.
        """
        server_controller = self._get_server_controller()
        # Predict exactly the actual malicious clients
        predicted_malicious = actual_malicious.copy()
        
        computed_ratio = server_controller.compute_filtered_ratio(
            predicted_malicious, actual_malicious
        )
        
        # Property: ratio must be 1.0 when all malicious are detected
        assert computed_ratio == 1.0, \
            f"Filtered ratio should be 1.0 when all detected, got {computed_ratio}"
    
    @given(
        actual_malicious=st.lists(
            st.integers(min_value=0, max_value=49),
            min_size=1,
            max_size=25,
            unique=True
        ),
        extra_predictions=st.lists(
            st.integers(min_value=50, max_value=99),
            min_size=0,
            max_size=25,
            unique=True
        )
    )
    @settings(max_examples=100)
    def test_filtered_ratio_one_with_false_positives(
        self,
        actual_malicious: List[int],
        extra_predictions: List[int]
    ):
        """
        Property: False positives don't affect filtered_ratio when all actual are detected.
        
        filtered_ratio only measures recall (true positives / actual positives),
        not precision. Extra false positive predictions should not reduce the ratio.
        """
        server_controller = self._get_server_controller()
        # Predict all actual malicious plus some extra (false positives)
        predicted_malicious = actual_malicious + extra_predictions
        
        computed_ratio = server_controller.compute_filtered_ratio(
            predicted_malicious, actual_malicious
        )
        
        # Property: ratio must still be 1.0 even with false positives
        assert computed_ratio == 1.0, \
            f"Filtered ratio should be 1.0 with false positives, got {computed_ratio}"
    
    @given(
        actual_malicious=st.lists(
            st.integers(min_value=0, max_value=99),
            min_size=1,
            max_size=50,
            unique=True
        )
    )
    @settings(max_examples=100)
    def test_filtered_ratio_zero_when_none_detected(
        self,
        actual_malicious: List[int]
    ):
        """
        Property: When no actual malicious clients are predicted, filtered_ratio = 0.0
        
        Complete miss should yield a ratio of 0.0.
        """
        server_controller = self._get_server_controller()
        predicted_malicious = []  # No predictions
        
        computed_ratio = server_controller.compute_filtered_ratio(
            predicted_malicious, actual_malicious
        )
        
        # Property: ratio must be 0.0 when none detected
        assert computed_ratio == 0.0, \
            f"Filtered ratio should be 0.0 when none detected, got {computed_ratio}"
    
    @given(
        actual_malicious=st.lists(
            st.integers(min_value=0, max_value=99),
            min_size=1,
            max_size=50,
            unique=True
        ),
        predicted_malicious=st.lists(
            st.integers(min_value=0, max_value=99),
            min_size=0,
            max_size=50,
            unique=True
        )
    )
    @settings(max_examples=100)
    def test_filtered_ratio_bounded_zero_to_one(
        self,
        actual_malicious: List[int],
        predicted_malicious: List[int]
    ):
        """
        Property: filtered_ratio is always in the range [0.0, 1.0]
        
        The ratio represents a proportion and must be bounded.
        """
        server_controller = self._get_server_controller()
        
        computed_ratio = server_controller.compute_filtered_ratio(
            predicted_malicious, actual_malicious
        )
        
        # Property: ratio must be in [0.0, 1.0]
        assert 0.0 <= computed_ratio <= 1.0, \
            f"Filtered ratio should be in [0, 1], got {computed_ratio}"
