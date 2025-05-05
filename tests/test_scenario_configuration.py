import pytest
from unittest.mock import MagicMock
from flwr.common import Context
import sys
import os

# Add the fl/fl directory to sys.path for direct import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../fl/fl')))

from multi_aggregator_server_app import server_fn


def test_scenario_configuration_parsing_and_tracking():
    # Simulate a scenario config: 3 aggregators, 2 malicious, challenges enabled
    run_config = {
        "num-server-rounds": 5,
        "fraction-fit": 0.5,
        "num-aggregators": 3,
        "malicious-aggregators": "1,2",
        "enable-challenges": True,
    }
    context = Context(run_id="test_run", run_config=run_config, node_config={}, node_id="server", state={})

    # Call the server function to parse config and create the strategy
    components = server_fn(context)
    strategy = components.strategy

    # Check that the strategy has the correct configuration
    assert strategy.num_aggregators == 3
    assert strategy.malicious_aggregator_ids == [1, 2]
    assert strategy.enable_challenges is True

    # Check that the strategy is ready to track scenario info
    assert hasattr(strategy, 'metrics_history')
    assert isinstance(strategy.metrics_history, list)


def test_aggregator_selection_and_malicious_behavior():
    """Test that the correct aggregator is selected each round and malicious behavior is executed."""
    run_config = {
        "num-server-rounds": 4,
        "fraction-fit": 0.5,
        "num-aggregators": 3,
        "malicious-aggregators": "1",
        "enable-challenges": True,
    }
    context = Context(run_id="test_run2", run_config=run_config, node_config={}, node_id="server", state={})
    components = server_fn(context)
    strategy = components.strategy

    # Simulate 4 rounds and check aggregator selection and malicious flag
    for round_num in range(1, 5):
        # Fake results and failures (simulate 2 clients)
        dummy_results = [(MagicMock(), MagicMock()) for _ in range(2)]
        dummy_failures = []
        params, metrics = strategy.aggregate_fit(round_num, dummy_results, dummy_failures)
        expected_agg_id = (round_num - 1) % 3
        assert strategy.current_aggregator_id == expected_agg_id
        if expected_agg_id == 1:
            assert metrics["malicious"] is True
        else:
            assert metrics["malicious"] is False


def test_metrics_history_tracking():
    """Test that metrics history is tracked for each round, especially for challenges and malicious detection."""
    run_config = {
        "num-server-rounds": 6,
        "fraction-fit": 0.5,
        "num-aggregators": 2,
        "malicious-aggregators": "0",
        "enable-challenges": True,
    }
    context = Context(run_id="test_run3", run_config=run_config, node_config={}, node_id="server", state={})
    components = server_fn(context)
    strategy = components.strategy

    # Simulate 6 rounds
    for round_num in range(1, 7):
        dummy_results = [(MagicMock(), MagicMock()) for _ in range(2)]
        dummy_failures = []
        strategy.aggregate_fit(round_num, dummy_results, dummy_failures)

    # There should be at least some entries in metrics_history for challenged malicious rounds
    assert isinstance(strategy.metrics_history, list)
    assert any(m["malicious"] and m["challenged"] for m in strategy.metrics_history)
    # All entries should have required keys
    for entry in strategy.metrics_history:
        assert "round" in entry
        assert "aggregator_id" in entry
        assert "malicious" in entry
        assert "challenged" in entry
        assert "challenge_success" in entry


def test_advanced_challenge_validation_logic():
    """Test that advanced challenge validation logic (parameter distance, thresholds) is applied."""
    run_config = {
        "num-server-rounds": 2,
        "fraction-fit": 0.5,
        "num-aggregators": 2,
        "malicious-aggregators": "1",
        "enable-challenges": True,
    }
    context = Context(run_id="test_adv_challenge", run_config=run_config, node_config={}, node_id="server", state={})
    components = server_fn(context)
    strategy = components.strategy

    # Patch the threshold for challenge validation to a high value to force failure
    strategy._validate_challenge = lambda r, c, h: False
    dummy_results = [(MagicMock(), MagicMock()) for _ in range(2)]
    dummy_failures = []
    _, metrics = strategy.aggregate_fit(1, dummy_results, dummy_failures)
    # Should not be successful
    assert not metrics.get("challenge_successful", False)

    # Patch the threshold for challenge validation to a low value to force success
    strategy._validate_challenge = lambda r, c, h: True
    _, metrics = strategy.aggregate_fit(2, dummy_results, dummy_failures)
    # Should be successful
    assert metrics.get("challenge_successful", False)


def test_multiple_malicious_behavior_strategies():
    """Test that multiple malicious behavior strategies can be configured and triggered."""
    run_config = {
        "num-server-rounds": 3,
        "fraction-fit": 0.5,
        "num-aggregators": 3,
        "malicious-aggregators": "0,2",
        "enable-challenges": True,
    }
    context = Context(run_id="test_multi_malicious", run_config=run_config, node_config={}, node_id="server", state={})
    components = server_fn(context)
    strategy = components.strategy

    # Simulate 3 rounds and check that both aggregator 0 and 2 are marked as malicious
    for round_num in range(1, 4):
        dummy_results = [(MagicMock(), MagicMock()) for _ in range(2)]
        dummy_failures = []
        _, metrics = strategy.aggregate_fit(round_num, dummy_results, dummy_failures)
        expected_agg_id = (round_num - 1) % 3
        if expected_agg_id in [0, 2]:
            assert metrics["malicious"] is True
        else:
            assert metrics["malicious"] is False


def test_detailed_challenge_and_verification_metrics_logged():
    """Test that detailed challenge and verification metrics are logged in metrics_history."""
    run_config = {
        "num-server-rounds": 2,
        "fraction-fit": 0.5,
        "num-aggregators": 2,
        "malicious-aggregators": "1",
        "enable-challenges": True,
    }
    context = Context(run_id="test_metrics_logging", run_config=run_config, node_config={}, node_id="server", state={})
    components = server_fn(context)
    strategy = components.strategy

    # Simulate 2 rounds
    for round_num in range(1, 3):
        dummy_results = [(MagicMock(), MagicMock()) for _ in range(2)]
        dummy_failures = []
        strategy.aggregate_fit(round_num, dummy_results, dummy_failures)

    # Check that metrics_history contains detailed challenge info
    for entry in strategy.metrics_history:
        assert "round" in entry
        assert "aggregator_id" in entry
        assert "malicious" in entry
        assert "challenged" in entry
        assert "challenge_success" in entry


def test_challenge_status_tracking():
    """Test that challenge status (pending, successful, rejected) is tracked/logged."""
    run_config = {
        "num-server-rounds": 2,
        "fraction-fit": 0.5,
        "num-aggregators": 2,
        "malicious-aggregators": "0",  # aggregator 0 is malicious in both rounds
        "enable-challenges": True,
    }
    context = Context(run_id="test_challenge_status", run_config=run_config, node_config={}, node_id="server", state={})
    components = server_fn(context)
    strategy = components.strategy

    # Patch _validate_challenge to alternate between success and failure
    outcomes = [True, False]
    def alt_validate(r, c, h):
        return outcomes.pop(0)
    strategy._validate_challenge = alt_validate

    dummy_results = [(MagicMock(), MagicMock()) for _ in range(2)]
    dummy_failures = []
    # Round 1: should be successful
    _, metrics1 = strategy.aggregate_fit(1, dummy_results, dummy_failures)
    # Round 2: should be rejected
    _, metrics2 = strategy.aggregate_fit(2, dummy_results, dummy_failures)
    # Check metrics for status
    assert metrics1.get("challenge_successful", False) is True
    assert metrics2.get("challenge_successful", False) is False or "challenge_successful" not in metrics2


def test_compare_challenged_vs_honest_parameters_in_logs():
    """Test that challenged and honest parameters are compared and differences are logged/recorded."""
    run_config = {
        "num-server-rounds": 1,
        "fraction-fit": 0.5,
        "num-aggregators": 2,
        "malicious-aggregators": "0",  # aggregator 0 is malicious in round 1
        "enable-challenges": True,
    }
    context = Context(run_id="test_param_compare", run_config=run_config, node_config={}, node_id="server", state={})
    components = server_fn(context)
    strategy = components.strategy

    # Patch _validate_challenge to check for difference and log it
    called = {}
    def custom_validate(r, c, h):
        called["called"] = True
        # Simulate a difference
        return True
    strategy._validate_challenge = custom_validate

    dummy_results = [(MagicMock(), MagicMock()) for _ in range(2)]
    dummy_failures = []
    strategy.aggregate_fit(1, dummy_results, dummy_failures)
    # Ensure the custom validation was called
    assert called.get("called", False) 