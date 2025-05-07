"""fl: A Flower / PyTorch app with multiple virtual aggregators."""

import copy
import json
import logging
import random
import time # Import time for processing time measurement
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from flwr.common import (
    Context,
    EvaluateRes,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, Strategy

logger = logging.getLogger(__name__)


class HybridOptimisticPBFTAggregatorStrategy(FedAvg):
    """Strategy for simulating multiple aggregators and configurable challenge mechanisms (RR, Hybrid, PBFT)
    for processing time analysis.
    """

    def __init__(
        self,
        num_aggregators: int = 3,
        malicious_aggregator_ids: List[int] = None, # Kept for basic malicious simulation if needed
        enable_challenges: bool = True, # Now primarily controls if challenge logic is active
        challenge_frequency: float = 0.25, # 0 for RR, 1 for PBFT, 0 < freq < 1 for Hybrid
        challenge_mode: str = 'deterministic', # 'deterministic' or 'random'
        **kwargs,
    ):
        """Initialize the MultiAggregatorStrategy.

        Args:
            num_aggregators: Number of virtual aggregators to simulate.
            malicious_aggregator_ids: List of aggregator IDs that will produce malicious results.
            enable_challenges: If False, challenge_frequency is ignored, and no challenges occur.
            challenge_frequency: Likelihood of a challenge. 0 for RR, 1 for PBFT.
            challenge_mode: 'deterministic' or 'random' for hybrid mode.
            **kwargs: Additional arguments to pass to FedAvg.
        """
        super().__init__(**kwargs)
        self.num_aggregators = num_aggregators
        self.malicious_aggregator_ids = malicious_aggregator_ids or []
        self.enable_challenges = enable_challenges
        self.challenge_frequency = challenge_frequency if self.enable_challenges else 0.0
        self.challenge_mode = challenge_mode
        self.current_aggregator_id = 0
        self.round = 0
        self.metrics_history = [] # To store per-round metrics including processing time
        self._honest_parameters_for_round: Optional[Parameters] = None # Simplified storage

        # Validation for malicious aggregator IDs
        for agg_id in self.malicious_aggregator_ids:
            if agg_id < 0 or agg_id >= num_aggregators:
                raise ValueError(f"Malicious aggregator ID {agg_id} is out of range [0, {num_aggregators-1}]")

        logger.info(f"Initialized HybridOptimisticPBFTAggregatorStrategy with {num_aggregators} aggregators.")
        logger.info(f"Malicious aggregators: {self.malicious_aggregator_ids}")
        logger.info(f"Challenge mechanism enabled: {self.enable_challenges}")
        if self.enable_challenges:
            logger.info(f"Challenge frequency: {self.challenge_frequency}, Challenge mode: {self.challenge_mode}")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates, simulate challenge logic, and measure processing time."""
        start_time = time.time()
        self.round = server_round

        self.current_aggregator_id = (server_round - 1) % self.num_aggregators
        is_malicious_aggregator = self.current_aggregator_id in self.malicious_aggregator_ids

        logger.info(f"Round {server_round}: Using aggregator {self.current_aggregator_id}" +
                   (" (MALICIOUS)" if is_malicious_aggregator else ""))

        honest_parameters, honest_metrics = super().aggregate_fit(server_round, results, failures)
        if honest_parameters is None:
            # Record processing time even on failure
            metrics = {"processing_time_fit": time.time() - start_time}
            return None, metrics
        
        self._honest_parameters_for_round = honest_parameters # Store for potential use if challenged

        current_params_to_use = honest_parameters
        metrics = {**honest_metrics} # Start with metrics from super().aggregate_fit
        metrics["aggregator_id"] = self.current_aggregator_id
        metrics["malicious_aggregator_this_round"] = is_malicious_aggregator
        
        challenge_this_round = False
        challenge_assumed_successful = False

        if self.enable_challenges:
            if self.challenge_frequency >= 1.0: # PBFT mode
                challenge_this_round = True
            elif self.challenge_frequency > 0: # Hybrid mode
                if self.challenge_mode == 'deterministic':
                    N = int(1.0 / self.challenge_frequency)
                    challenge_this_round = (server_round % N == 0)
                elif self.challenge_mode == 'random':
                    challenge_this_round = (random.random() < self.challenge_frequency)
            # If challenge_frequency is 0 (RR mode), challenge_this_round remains False

        if is_malicious_aggregator:
            honest_ndarrays = parameters_to_ndarrays(honest_parameters)
            malicious_ndarrays = self._create_malicious_aggregation(honest_ndarrays)
            current_params_to_use = ndarrays_to_parameters(malicious_ndarrays)
            metrics["malicious_behavior_simulated"] = True
            # In this simplified strategy, if malicious, a challenge (if active by frequency)
            # is assumed to catch it and revert to honest parameters.
            if challenge_this_round:
                logger.info(f"Round {server_round}: Malicious aggregator {self.current_aggregator_id} was challenged (simulated). Using honest parameters.")
                current_params_to_use = self._honest_parameters_for_round
                challenge_assumed_successful = True 
        elif challenge_this_round: # Not malicious, but challenged by frequency
             logger.info(f"Round {server_round}: Aggregator {self.current_aggregator_id} was challenged (simulated by frequency). Using honest parameters.")
             # No change needed, current_params_to_use is already honest_parameters
             challenge_assumed_successful = True # Mark that a challenge occurred and was 'successful' by definition

        metrics["challenge_simulated_this_round"] = challenge_this_round
        metrics["challenge_assumed_successful"] = challenge_assumed_successful
        
        end_time = time.time()
        metrics["processing_time_fit"] = end_time - start_time

        round_metrics_log = {
            "round": int(server_round),
            "aggregator_id": int(self.current_aggregator_id),
            "is_malicious_aggregator": bool(is_malicious_aggregator),
            "malicious_behavior_simulated": metrics.get("malicious_behavior_simulated", False),
            "challenge_simulated_this_round": bool(challenge_this_round),
            "challenge_assumed_successful": bool(challenge_assumed_successful),
            "processing_time_fit": metrics["processing_time_fit"],
        }
        self.metrics_history.append(round_metrics_log)
        logger.info(f"Round {server_round}: Metrics - {json.dumps(round_metrics_log)}")

        return current_params_to_use, metrics

    def _create_malicious_aggregation(self, honest_ndarrays: List[np.ndarray]) -> List[np.ndarray]:
        """Create a malicious aggregation by modifying the honest aggregation (simplified)."""
        malicious_ndarrays = [np.copy(arr) for arr in honest_ndarrays]
        for i in range(len(malicious_ndarrays)):
            scale = np.mean(np.abs(malicious_ndarrays[i])) * 0.5 # Example noise
            malicious_ndarrays[i] += np.random.normal(0, scale, size=malicious_ndarrays[i].shape)
        return malicious_ndarrays

    def _store_honest_parameters(self, server_round: int, honest_parameters: Parameters) -> None:
        # This method is effectively replaced by self._honest_parameters_for_round
        # but kept to avoid breaking calls from the parent class if any (though unlikely for FedAvg)
        self._honest_parameters_for_round = honest_parameters

    def _validate_challenge(
        self, 
        server_round: int, 
        challenged_parameters: Parameters,
        honest_parameters: Parameters
    ) -> bool:
        # This method is not actively used in the simplified challenge logic of this class.
        # Challenge success is assumed if a challenge occurs.
        logger.warning("_validate_challenge called but not used in this simplified strategy.")
        return True # Default to true if ever called by mistake

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results and measure processing time."""
        start_time = time.time()
        aggregated_loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        end_time = time.time()
        
        metrics["processing_time_evaluate"] = end_time - start_time
        # Add aggregator info to metrics if not already present
        if "aggregator_id" not in metrics:
            metrics["aggregator_id"] = self.current_aggregator_id # current_aggregator_id is set in aggregate_fit
        
        # Log research metrics periodically
        if server_round % 5 == 0 or server_round == 1: # Example: log every 5 rounds or on first round
            self._log_research_metrics()
        
        return aggregated_loss, metrics

    def _log_research_metrics(self) -> None:
        """Log metrics relevant for the research paper (simplified for this class)."""
        if not self.metrics_history:
            return
        
        # Example: Log average processing time for fit
        avg_fit_time = np.mean([m["processing_time_fit"] for m in self.metrics_history if "processing_time_fit" in m])
        
        logger.info(f"\n----- RESEARCH METRICS (Round {self.round}) -----")
        logger.info(f"Average Fit Processing Time: {avg_fit_time:.4f}s")
        # Add more derived metrics as needed
        logger.info("--------------------------------------------\n")