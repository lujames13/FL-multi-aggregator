"""fl: A Flower / PyTorch app with multiple virtual aggregators."""

import copy
import json
import logging
import random
import time
import math
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
        malicious_aggregator_ids: List[int] = None,
        enable_challenges: bool = True,
        challenge_frequency: float = 0.25,  # 0 for RR, 1 for PBFT, 0 < freq < 1 for Hybrid
        challenge_mode: str = 'deterministic',  # 'deterministic' or 'random'
        pbft_simulation_factor: float = 0.1,  # Factor to control PBFT computation simulation intensity
        **kwargs,
    ):
        """Initialize the HybridOptimisticPBFTAggregatorStrategy.

        Args:
            num_aggregators: Number of virtual aggregators to simulate.
            malicious_aggregator_ids: List of aggregator IDs that will produce malicious results.
            enable_challenges: If False, challenge_frequency is ignored, and no challenges occur.
            challenge_frequency: Likelihood of a challenge. 0 for RR, 1 for PBFT.
            challenge_mode: 'deterministic' or 'random' for hybrid mode.
            pbft_simulation_factor: Factor to control intensity of PBFT computation simulation.
            **kwargs: Additional arguments to pass to FedAvg.
        """
        super().__init__(**kwargs)
        self.num_aggregators = num_aggregators
        self.malicious_aggregator_ids = malicious_aggregator_ids or []
        self.enable_challenges = enable_challenges
        self.challenge_frequency = challenge_frequency if self.enable_challenges else 0.0
        self.challenge_mode = challenge_mode
        self.pbft_simulation_factor = pbft_simulation_factor
        self.current_aggregator_id = 0
        self.round = 0
        self.metrics_history = []  # To store per-round metrics including processing time
        self._honest_parameters_for_round: Optional[Parameters] = None  # Simplified storage
        self.challenged_rounds = set()  # Track rounds where challenges occurred

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
        
        self._honest_parameters_for_round = honest_parameters  # Store for potential use if challenged

        current_params_to_use = honest_parameters
        metrics = {**honest_metrics}  # Start with metrics from super().aggregate_fit
        metrics["aggregator_id"] = self.current_aggregator_id
        metrics["malicious_aggregator_this_round"] = is_malicious_aggregator
        
        # Determine if this round will include a challenge based on mode and frequency
        challenge_this_round = False
        challenge_assumed_successful = False

        if self.enable_challenges:
            if self.challenge_frequency >= 1.0:  # PBFT mode
                challenge_this_round = True
            elif self.challenge_frequency > 0:  # Hybrid mode
                if self.challenge_mode == 'deterministic':
                    N = int(1.0 / self.challenge_frequency)
                    challenge_this_round = (server_round % N == 0)
                elif self.challenge_mode == 'random':
                    challenge_this_round = (random.random() < self.challenge_frequency)
            # If challenge_frequency is 0 (RR mode), challenge_this_round remains False

        # Create malicious parameters if this aggregator is malicious
        if is_malicious_aggregator:
            honest_ndarrays = parameters_to_ndarrays(honest_parameters)
            malicious_ndarrays = self._create_malicious_aggregation(honest_ndarrays)
            current_params_to_use = ndarrays_to_parameters(malicious_ndarrays)
            metrics["malicious_behavior_simulated"] = True
        
        # If this round includes a challenge, validate the parameters
        if challenge_this_round:
            logger.info(f"Round {server_round}: Challenging aggregator {self.current_aggregator_id}")
            self.challenged_rounds.add(server_round)
            
            # Simulate PBFT challenge validation with validators = num_aggregators
            params_to_validate = current_params_to_use
            validation_start_time = time.time()
            
            # Simulate the PBFT consensus process with multiple validators
            pbft_success = self._simulate_pbft_validation(
                server_round=server_round,
                params_to_validate=params_to_validate,
                honest_parameters=honest_parameters,
                num_validators=self.num_aggregators
            )
            
            metrics["pbft_validation_time"] = time.time() - validation_start_time
            metrics["challenge_simulated_this_round"] = True
            
            # If validation was successful and the aggregator was malicious, revert to honest parameters
            if pbft_success and is_malicious_aggregator:
                logger.info(f"Round {server_round}: Malicious aggregator {self.current_aggregator_id} " +
                           f"was challenged and PBFT consensus rejected the parameters. Using honest parameters.")
                current_params_to_use = self._honest_parameters_for_round
                challenge_assumed_successful = True
            elif pbft_success:
                logger.info(f"Round {server_round}: Aggregator {self.current_aggregator_id} was challenged " +
                           f"and PBFT consensus validated the parameters.")
                challenge_assumed_successful = True
                
            metrics["challenge_assumed_successful"] = challenge_assumed_successful
        else:
            metrics["challenge_simulated_this_round"] = False
            metrics["challenge_assumed_successful"] = False
        
        # Record final processing time
        end_time = time.time()
        metrics["processing_time_fit"] = end_time - start_time

        # Log metrics for this round
        round_metrics_log = {
            "round": int(server_round),
            "aggregator_id": int(self.current_aggregator_id),
            "is_malicious_aggregator": bool(is_malicious_aggregator),
            "malicious_behavior_simulated": metrics.get("malicious_behavior_simulated", False),
            "challenge_simulated_this_round": bool(challenge_this_round),
            "challenge_assumed_successful": bool(challenge_assumed_successful),
            "processing_time_fit": metrics["processing_time_fit"],
            "pbft_validation_time": metrics.get("pbft_validation_time", 0.0),
        }
        self.metrics_history.append(round_metrics_log)
        logger.info(f"Round {server_round}: Metrics - {json.dumps(round_metrics_log)}")

        return current_params_to_use, metrics

    def _create_malicious_aggregation(self, honest_ndarrays: List[np.ndarray]) -> List[np.ndarray]:
        """Create a malicious aggregation by modifying the honest aggregation."""
        malicious_ndarrays = [np.copy(arr) for arr in honest_ndarrays]
        for i in range(len(malicious_ndarrays)):
            scale = np.mean(np.abs(malicious_ndarrays[i])) * 0.5  # Example noise
            malicious_ndarrays[i] += np.random.normal(0, scale, size=malicious_ndarrays[i].shape)
        return malicious_ndarrays

    def _simulate_pbft_validation(
        self, 
        server_round: int, 
        params_to_validate: Parameters, 
        honest_parameters: Parameters,
        num_validators: int
    ) -> bool:
        """Simulate PBFT validation process with multiple validators.
        
        Args:
            server_round: The current server round
            params_to_validate: Parameters to validate
            honest_parameters: Known honest parameters for comparison
            num_validators: Number of validators participating in PBFT consensus
            
        Returns:
            bool: True if validation successful, False otherwise
        """
        # Convert parameters to ndarrays for validation
        validate_ndarrays = parameters_to_ndarrays(params_to_validate)
        honest_ndarrays = parameters_to_ndarrays(honest_parameters)
        
        # Track votes from validators
        votes = []
        
        # Simulate communication and computation overhead for each validator
        for validator_id in range(num_validators):
            # Simulate validator computation time based on parameter size
            total_params = sum(arr.size for arr in validate_ndarrays)
            
            # Simulate computation (more intensive for PBFT)
            self._simulate_validator_computation(validate_ndarrays, honest_ndarrays)
            
            # Determine this validator's vote
            if validator_id in self.malicious_aggregator_ids:
                # Malicious validators might vote incorrectly
                vote = random.random() > 0.7  # 30% chance to vote against honest parameters
            else:
                # Honest validators compare parameters and vote
                vote = self._validate_parameters(validate_ndarrays, honest_ndarrays)
            
            votes.append(vote)
        
        # Simulate prepare and commit phases with message passing
        # In PBFT, we need 2f+1 votes to reach consensus where f is max faulty nodes
        max_faulty = (num_validators - 1) // 3
        min_votes_needed = num_validators - max_faulty
        
        # Count positive votes
        positive_votes = sum(votes)
        
        # Determine if consensus was reached
        consensus_reached = positive_votes >= min_votes_needed
        
        logger.info(f"PBFT consensus: {positive_votes}/{num_validators} votes, " +
                   f"needed {min_votes_needed}, consensus {'reached' if consensus_reached else 'failed'}")
        
        return consensus_reached

    def _simulate_validator_computation(self, validate_ndarrays: List[np.ndarray], honest_ndarrays: List[np.ndarray]) -> None:
        """Simulate the computational work done by a validator."""
        # Simulate computational work proportional to parameter size
        # This is a realistic simulation of the computation needed to verify parameters
        
        # Calculate total parameter size to scale computation
        total_size = sum(arr.size for arr in validate_ndarrays)
        
        # Scale computation based on model size and simulation factor
        computation_intensity = int(total_size * self.pbft_simulation_factor)
        computation_intensity = min(computation_intensity, 1000000)  # Cap to avoid excessive computation
        
        # Perform actual computation to simulate validator work
        # This includes calculating differences between parameters
        diff_sum = 0
        for i in range(min(len(validate_ndarrays), len(honest_ndarrays))):
            # Calculate element-wise absolute differences
            if validate_ndarrays[i].shape == honest_ndarrays[i].shape:
                diff = np.abs(validate_ndarrays[i] - honest_ndarrays[i])
                diff_sum += np.sum(diff)
            
        # Simulate additional computation based on intensity
        for _ in range(computation_intensity // 10000 + 1):
            # Simple computation to simulate work
            _ = math.sqrt(random.random() * diff_sum + 1.0)

    def _validate_parameters(self, validate_ndarrays: List[np.ndarray], honest_ndarrays: List[np.ndarray]) -> bool:
        """Validate parameters by comparing with honest parameters."""
        # Calculate differences between parameters
        diffs = []
        for i in range(min(len(validate_ndarrays), len(honest_ndarrays))):
            if validate_ndarrays[i].shape == honest_ndarrays[i].shape:
                # Using MSE as distance metric
                mse = np.mean((validate_ndarrays[i] - honest_ndarrays[i])**2)
                diffs.append(mse)
        
        if not diffs:
            return False
        
        # Average MSE across all parameters
        avg_mse = sum(diffs) / len(diffs)
        
        # Define threshold for validation
        threshold = 1e-6
        
        # Parameters are valid if average MSE is below threshold
        return avg_mse < threshold

    def _store_honest_parameters(self, server_round: int, honest_parameters: Parameters) -> None:
        """Store honest parameters for future use."""
        self._honest_parameters_for_round = honest_parameters

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
            metrics["aggregator_id"] = self.current_aggregator_id
            
        # Add challenge info to metrics
        metrics["challenged_round"] = server_round in self.challenged_rounds
        
        # Log research metrics periodically
        if server_round % 5 == 0 or server_round == 1:
            self._log_research_metrics()
        
        return aggregated_loss, metrics

    def _log_research_metrics(self) -> None:
        """Log metrics relevant for the research paper."""
        if not self.metrics_history:
            return
        
        # Calculate average processing times
        fit_times = [m["processing_time_fit"] for m in self.metrics_history if "processing_time_fit" in m]
        avg_fit_time = np.mean(fit_times) if fit_times else 0
        
        # Calculate PBFT validation times (only for challenged rounds)
        pbft_times = [m.get("pbft_validation_time", 0) for m in self.metrics_history 
                      if m.get("challenge_simulated_this_round", False)]
        avg_pbft_time = np.mean(pbft_times) if pbft_times else 0
        
        # Calculate percentage of time spent in PBFT validation
        total_fit_time = sum(fit_times) if fit_times else 0
        total_pbft_time = sum(pbft_times) if pbft_times else 0
        pbft_percentage = (total_pbft_time / total_fit_time * 100) if total_fit_time > 0 else 0
        
        logger.info(f"\n----- RESEARCH METRICS (Round {self.round}) -----")
        logger.info(f"Average Fit Processing Time: {avg_fit_time:.4f}s")
        logger.info(f"Average PBFT Validation Time: {avg_pbft_time:.4f}s")
        logger.info(f"PBFT Validation Time Percentage: {pbft_percentage:.2f}%")
        logger.info(f"Total Challenged Rounds: {len(self.challenged_rounds)}")
        logger.info("--------------------------------------------\n")