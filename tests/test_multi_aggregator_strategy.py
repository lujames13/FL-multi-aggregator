import sys
import os
import pytest
import numpy as np
import random
import time
from unittest.mock import patch

# Add the fl/fl directory to sys.path for direct import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../fl/fl')))

from multi_aggregator_strategy import MultiAggregatorStrategy
from hybrid_strategy import HybridOptimisticPBFTAggregatorStrategy


def test_round_robin_aggregator_selection():
    strategy = MultiAggregatorStrategy(num_aggregators=3)
    rounds = 6
    expected_ids = [0, 1, 2, 0, 1, 2]
    actual_ids = []
    for r in range(1, rounds + 1):
        strategy.round = r
        strategy.current_aggregator_id = (r - 1) % strategy.num_aggregators
        actual_ids.append(strategy.current_aggregator_id)
    assert actual_ids == expected_ids


def test_server_app_multi_aggregator_config():
    # Simulate server app configuration for multiple aggregators
    num_aggregators = 4
    strategy = MultiAggregatorStrategy(num_aggregators=num_aggregators)
    assert strategy.num_aggregators == num_aggregators
    # Optionally, check that malicious_aggregator_ids can be set
    malicious_ids = [1, 3]
    strategy = MultiAggregatorStrategy(num_aggregators=num_aggregators, malicious_aggregator_ids=malicious_ids)
    assert strategy.malicious_aggregator_ids == malicious_ids


def test_honest_and_malicious_aggregator_behavior():
    # Honest aggregator should return unmodified parameters
    honest_strategy = MultiAggregatorStrategy(num_aggregators=2, malicious_aggregator_ids=[])
    honest_params = [np.ones((2, 2)), np.ones((2, 2))]
    honest_strategy.current_aggregator_id = 0  # Not malicious
    honest_result = honest_strategy._create_malicious_aggregation(honest_params)
    assert all(np.allclose(h, r) for h, r in zip(honest_params, honest_result))

    # Malicious aggregator should modify parameters
    malicious_strategy = MultiAggregatorStrategy(num_aggregators=2, malicious_aggregator_ids=[1])
    malicious_params = [np.ones((2, 2)), np.ones((2, 2))]
    malicious_strategy.current_aggregator_id = 1  # Malicious
    malicious_result = malicious_strategy._create_malicious_aggregation(malicious_params)
    assert any(not np.allclose(m, r) for m, r in zip(malicious_params, malicious_result))


def test_challenge_detection_triggers_on_malicious_aggregation():
    # Setup: 2 aggregators, aggregator 1 is malicious, challenges enabled
    strategy = MultiAggregatorStrategy(num_aggregators=2, malicious_aggregator_ids=[1], enable_challenges=True)
    server_round = 2  # This will select aggregator 1 (malicious)

    # Simulate client results (use simple arrays for deterministic behavior)
    class DummyClient:
        pass
    dummy_client = DummyClient()
    honest_params = [np.ones((2, 2)), np.ones((2, 2))]
    fit_res = type('FitRes', (), {'parameters': None})()
    # Patch super().aggregate_fit to return honest_params as Parameters
    def fake_aggregate_fit(self, server_round, results, failures):
        from flwr.common import ndarrays_to_parameters
        return ndarrays_to_parameters(honest_params), {}
    strategy.__class__.__bases__[0].aggregate_fit = fake_aggregate_fit

    # Patch random.random to always trigger a challenge
    with patch('random.random', return_value=0.0):
        aggregated_parameters, metrics = strategy.aggregate_fit(
            server_round,
            results=[(dummy_client, fit_res)],
            failures=[]
        )

    # Check that a challenge was triggered and was successful
    assert metrics.get('challenged', False), "Challenge was not triggered on malicious aggregation"
    assert metrics.get('challenge_successful', False), "Challenge did not succeed on malicious aggregation"
    # The round should be in challenged_rounds
    assert server_round in strategy.challenged_rounds, "Challenged round not recorded"
    # There should be a challenge metric in metrics_history
    assert any(m['round'] == server_round and m['challenged'] for m in strategy.metrics_history), "Challenge metrics not recorded"


def dummy_results():
    class DummyClient: pass
    class DummyFitRes: pass
    return [(DummyClient(), DummyFitRes()), (DummyClient(), DummyFitRes())], []

def test_rr_mode_no_challenge():
    strategy = HybridOptimisticPBFTAggregatorStrategy(
        num_aggregators=2, challenge_frequency=0.0, challenge_mode='deterministic'
    )
    for r in range(1, 9):
        results, failures = dummy_results()
        _, metrics = strategy.aggregate_fit(r, results, failures)
        assert metrics["challenge_simulated_this_round"] is False

def test_pbft_mode_all_challenged():
    strategy = HybridOptimisticPBFTAggregatorStrategy(
        num_aggregators=2, challenge_frequency=1.0, challenge_mode='deterministic'
    )
    for r in range(1, 5):
        results, failures = dummy_results()
        _, metrics = strategy.aggregate_fit(r, results, failures)
        assert metrics["challenge_simulated_this_round"] is True

def test_hybrid_mode_deterministic():
    strategy = HybridOptimisticPBFTAggregatorStrategy(
        num_aggregators=2, challenge_frequency=0.25, challenge_mode='deterministic'
    )
    for r in range(1, 9):
        results, failures = dummy_results()
        _, metrics = strategy.aggregate_fit(r, results, failures)
        expected = (r % 4 == 0)
        assert metrics["challenge_simulated_this_round"] == expected

def test_processing_time_fit_positive():
    strategy = HybridOptimisticPBFTAggregatorStrategy(
        num_aggregators=2, challenge_frequency=0.25, challenge_mode='deterministic'
    )
    results, failures = dummy_results()
    _, metrics = strategy.aggregate_fit(1, results, failures)
    assert "processing_time_fit" in metrics
    assert metrics["processing_time_fit"] > 0 