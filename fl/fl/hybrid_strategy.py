"""fl: A Flower / PyTorch app with multiple virtual aggregators."""

import copy
import json
import logging
import random
import time
import math
import hashlib
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
        network_delay_factor: float = 0.05,  # Network delay factor in seconds per validator
        computation_intensity: float = 0.1,  # Computational intensity factor
        **kwargs,
    ):
        """Initialize the HybridOptimisticPBFTAggregatorStrategy.

        Args:
            num_aggregators: Number of virtual aggregators to simulate.
            malicious_aggregator_ids: List of aggregator IDs that will produce malicious results.
            enable_challenges: If False, challenge_frequency is ignored, and no challenges occur.
            challenge_frequency: Likelihood of a challenge. 0 for RR, 1 for PBFT.
            challenge_mode: 'deterministic' or 'random' for hybrid mode.
            network_delay_factor: Factor to control simulated network delay (seconds per validator).
            computation_intensity: Factor to control computational work intensity.
            **kwargs: Additional arguments to pass to FedAvg.
        """
        super().__init__(**kwargs)
        self.num_aggregators = num_aggregators
        self.malicious_aggregator_ids = malicious_aggregator_ids or []
        self.enable_challenges = enable_challenges
        self.challenge_frequency = challenge_frequency if self.enable_challenges else 0.0
        self.challenge_mode = challenge_mode
        self.network_delay_factor = network_delay_factor
        self.computation_intensity = computation_intensity
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
            logger.info(f"Network delay factor: {self.network_delay_factor}s per validator")
            logger.info(f"Computation intensity factor: {self.computation_intensity}")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates, simulate challenge logic, and measure processing time."""
        start_time = time.time()
        self.round = server_round

        # Determine current aggregator using round-robin
        self.current_aggregator_id = (server_round - 1) % self.num_aggregators
        is_malicious_aggregator = self.current_aggregator_id in self.malicious_aggregator_ids

        logger.info(f"Round {server_round}: Using aggregator {self.current_aggregator_id}" +
                   (" (MALICIOUS)" if is_malicious_aggregator else ""))

        # Call parent FedAvg implementation to get honest parameters
        honest_parameters, honest_metrics = super().aggregate_fit(server_round, results, failures)
        if honest_parameters is None:
            # Record processing time even on failure
            metrics = {"processing_time_fit": time.time() - start_time}
            return None, metrics
        
        # Store honest parameters for potential use if challenged
        self._honest_parameters_for_round = honest_parameters

        # Start with honest parameters and update metrics
        current_params_to_use = honest_parameters
        metrics = {**honest_metrics}  # Start with metrics from super().aggregate_fit
        metrics["aggregator_id"] = self.current_aggregator_id
        metrics["malicious_aggregator_this_round"] = is_malicious_aggregator
        
        # Detailed metrics for different phases
        metrics["pre_prepare_time"] = 0.0
        metrics["prepare_time"] = 0.0
        metrics["commit_time"] = 0.0
        metrics["reply_time"] = 0.0
        metrics["total_network_delay"] = 0.0
        metrics["computation_time"] = 0.0
        
        # Determine if this round will include a challenge based on mode and frequency
        challenge_this_round = False

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
            validation_start_time = time.time()
            
            # Run the full PBFT consensus process with network delays
            pbft_result = self._simulate_pbft_consensus(
                server_round=server_round,
                params_to_validate=current_params_to_use,
                honest_parameters=honest_parameters,
                num_validators=self.num_aggregators,
                metrics=metrics
            )
            
            validation_time = time.time() - validation_start_time
            metrics["pbft_validation_time"] = validation_time
            metrics["challenge_simulated_this_round"] = True
            metrics["pbft_consensus_reached"] = pbft_result["consensus_reached"]
            
            # Extend metrics with detailed PBFT phase timings
            metrics.update(pbft_result["phase_timings"])
            
            # If validation rejected parameters and aggregator was malicious, revert to honest parameters
            if pbft_result["consensus_reached"] and is_malicious_aggregator and not pbft_result["parameters_accepted"]:
                logger.info(f"Round {server_round}: Malicious aggregator {self.current_aggregator_id} " +
                           f"was challenged and PBFT consensus rejected the parameters. Using honest parameters.")
                current_params_to_use = self._honest_parameters_for_round
                metrics["challenge_successful"] = True
            else:
                metrics["challenge_successful"] = False
                
        else:
            metrics["challenge_simulated_this_round"] = False
            metrics["challenge_successful"] = False
        
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
            "challenge_successful": metrics.get("challenge_successful", False),
            "processing_time_fit": metrics["processing_time_fit"],
            "pbft_validation_time": metrics.get("pbft_validation_time", 0.0),
            "total_network_delay": metrics.get("total_network_delay", 0.0),
            "computation_time": metrics.get("computation_time", 0.0),
        }
        
        # Add PBFT phase timings if available
        for phase in ["pre_prepare_time", "prepare_time", "commit_time", "reply_time"]:
            if phase in metrics:
                round_metrics_log[phase] = metrics[phase]
                
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

    def _simulate_network_delay(self, num_validators: int, delay_factor: float) -> float:
        """Simulate network delay proportional to number of validators."""
        delay = num_validators * delay_factor
        time.sleep(delay)
        return delay

    def _simulate_computation(self, data_size: int, intensity: float) -> float:
        """Simulate computation with hash operations proportional to data size."""
        start_time = time.time()
        
        # Scale computation based on data size and intensity
        iterations = max(1, int(data_size * intensity / 10000))
        
        # Perform actual cryptographic operations
        data = b"initial data for hashing"
        for _ in range(iterations):
            data = hashlib.sha256(data).digest()
            
        computation_time = time.time() - start_time
        return computation_time

    def _simulate_pbft_consensus(
        self, 
        server_round: int, 
        params_to_validate: Parameters, 
        honest_parameters: Parameters,
        num_validators: int,
        metrics: Dict[str, Scalar]
    ) -> Dict:
        """Simulate the full PBFT consensus process with network delays.
        
        Args:
            server_round: The current server round
            params_to_validate: Parameters to validate
            honest_parameters: Known honest parameters for comparison
            num_validators: Number of validators participating in PBFT consensus
            metrics: Dictionary to store timing metrics
            
        Returns:
            Dict with consensus result and timing metrics
        """
        validate_ndarrays = parameters_to_ndarrays(params_to_validate)
        honest_ndarrays = parameters_to_ndarrays(honest_parameters)
        
        # Calculate total data size for computation scaling
        total_data_size = sum(arr.size * arr.itemsize for arr in validate_ndarrays)
        
        # Track total network delay and computation time
        total_network_delay = 0.0
        total_computation_time = 0.0
        
        # Maximum Byzantine failures we can tolerate
        max_faulty = (num_validators - 1) // 3
        min_votes_needed = 2 * max_faulty + 1
        
        logger.info(f"PBFT consensus requires {min_votes_needed}/{num_validators} votes " +
                   f"(max faulty nodes: {max_faulty})")
        
        # Phase 1: Pre-prepare phase (primary validator broadcasts to all)
        pre_prepare_start = time.time()
        
        # Primary validator (0) broadcasts to all others
        network_delay = self._simulate_network_delay(num_validators - 1, self.network_delay_factor)
        total_network_delay += network_delay
        
        # Simulate primary computation (verify signature, validate request)
        computation_time = self._simulate_computation(total_data_size, self.computation_intensity)
        total_computation_time += computation_time
        
        pre_prepare_end = time.time()
        pre_prepare_time = pre_prepare_end - pre_prepare_start
        metrics["pre_prepare_time"] = pre_prepare_time
        
        # Phase 2: Prepare phase (all validators verify and broadcast prepare messages)
        prepare_start = time.time()
        
        prepare_votes = []
        for validator_id in range(num_validators):
            # Each validator verifies and computes
            computation_time = self._simulate_computation(total_data_size, self.computation_intensity)
            total_computation_time += computation_time
            
            # Determine vote based on validator's honesty
            if validator_id in self.malicious_aggregator_ids:
                # Malicious validators vote randomly
                vote = random.random() > 0.3  # 70% chance to vote correctly
            else:
                # Honest validators compare parameters with a threshold
                param_diff = self._compute_parameter_difference(validate_ndarrays, honest_ndarrays)
                vote = param_diff < 1e-6
                
            prepare_votes.append(vote)
            
            # Broadcast prepare message to all other validators (n-1 messages)
            network_delay = self._simulate_network_delay(num_validators - 1, self.network_delay_factor * 0.5)
            total_network_delay += network_delay
        
        prepare_end = time.time()
        prepare_time = prepare_end - prepare_start
        metrics["prepare_time"] = prepare_time
        
        # Phase 3: Commit phase (validators broadcast commit messages)
        commit_start = time.time()
        
        prepare_quorum_reached = sum(prepare_votes) >= min_votes_needed
        
        if prepare_quorum_reached:
            commit_votes = []
            for validator_id in range(num_validators):
                # Each validator computes and broadcasts commit
                computation_time = self._simulate_computation(total_data_size/2, self.computation_intensity)
                total_computation_time += computation_time
                
                # In commit phase, validators mostly follow their prepare vote
                vote = prepare_votes[validator_id]
                if vote and validator_id in self.malicious_aggregator_ids and random.random() < 0.1:
                    # Small chance for malicious node to change vote
                    vote = False
                    
                commit_votes.append(vote)
                
                # Broadcast commit message to all other validators
                network_delay = self._simulate_network_delay(num_validators - 1, self.network_delay_factor * 0.5)
                total_network_delay += network_delay
        else:
            # Prepare quorum not reached, skip commit
            commit_votes = [False] * num_validators
        
        commit_end = time.time()
        commit_time = commit_end - commit_start
        metrics["commit_time"] = commit_time
        
        # Phase 4: Reply phase (final decision and reply to client)
        reply_start = time.time()
        
        commit_quorum_reached = sum(commit_votes) >= min_votes_needed
        
        if commit_quorum_reached:
            # Final verification by primary
            computation_time = self._simulate_computation(total_data_size/4, self.computation_intensity)
            total_computation_time += computation_time
            
            # Send reply to client
            network_delay = self._simulate_network_delay(1, self.network_delay_factor)
            total_network_delay += network_delay
            
            # Determine if parameters are accepted
            param_diff = self._compute_parameter_difference(validate_ndarrays, honest_ndarrays)
            parameters_accepted = param_diff < 1e-6
        else:
            # Consensus failed
            parameters_accepted = False
        
        reply_end = time.time()
        reply_time = reply_end - reply_start
        metrics["reply_time"] = reply_time
        
        # Update total network delay and computation time
        metrics["total_network_delay"] = total_network_delay
        metrics["computation_time"] = total_computation_time
        
        # Log consensus results
        prepare_votes_count = sum(prepare_votes)
        commit_votes_count = sum(commit_votes)
        logger.info(f"PBFT consensus: Prepare: {prepare_votes_count}/{num_validators}, " +
                   f"Commit: {commit_votes_count}/{num_validators}, " +
                   f"Result: {'Success' if commit_quorum_reached else 'Failed'}")
        
        # Return consensus results and detailed timings
        return {
            "consensus_reached": commit_quorum_reached,
            "parameters_accepted": parameters_accepted if commit_quorum_reached else False,
            "prepare_votes": sum(prepare_votes),
            "commit_votes": sum(commit_votes),
            "phase_timings": {
                "pre_prepare_time": pre_prepare_time,
                "prepare_time": prepare_time,
                "commit_time": commit_time,
                "reply_time": reply_time,
                "total_network_delay": total_network_delay,
                "computation_time": total_computation_time
            }
        }

    def _compute_parameter_difference(self, params1: List[np.ndarray], params2: List[np.ndarray]) -> float:
        """Compute the average parameter difference (MSE) between two sets of parameters."""
        diffs = []
        for i in range(min(len(params1), len(params2))):
            if params1[i].shape == params2[i].shape:
                # Using MSE as distance metric
                mse = np.mean((params1[i] - params2[i])**2)
                diffs.append(mse)
        
        if not diffs:
            return float('inf')
        
        # Average MSE across all parameters
        avg_mse = sum(diffs) / len(diffs)
        return avg_mse

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
        
        # Network delay and computation times
        network_delays = [m.get("total_network_delay", 0) for m in self.metrics_history 
                         if "total_network_delay" in m]
        avg_network_delay = np.mean(network_delays) if network_delays else 0
        
        computation_times = [m.get("computation_time", 0) for m in self.metrics_history 
                           if "computation_time" in m]
        avg_computation_time = np.mean(computation_times) if computation_times else 0
        
        # Phase timings (PBFT only)
        phase_times = {}
        for phase in ["pre_prepare_time", "prepare_time", "commit_time", "reply_time"]:
            times = [m.get(phase, 0) for m in self.metrics_history if phase in m]
            phase_times[phase] = np.mean(times) if times else 0
        
        # Calculate percentage of time spent in different components
        total_fit_time = sum(fit_times) if fit_times else 0
        total_pbft_time = sum(pbft_times) if pbft_times else 0
        total_network_delay = sum(network_delays) if network_delays else 0
        total_computation_time = sum(computation_times) if computation_times else 0
        
        pbft_percentage = (total_pbft_time / total_fit_time * 100) if total_fit_time > 0 else 0
        network_percentage = (total_network_delay / total_pbft_time * 100) if total_pbft_time > 0 else 0
        compute_percentage = (total_computation_time / total_pbft_time * 100) if total_pbft_time > 0 else 0
        
        logger.info(f"\n----- RESEARCH METRICS (Round {self.round}) -----")
        logger.info(f"Average Fit Processing Time: {avg_fit_time:.4f}s")
        logger.info(f"Average PBFT Validation Time: {avg_pbft_time:.4f}s")
        logger.info(f"Average Network Delay: {avg_network_delay:.4f}s")
        logger.info(f"Average Computation Time: {avg_computation_time:.4f}s")
        
        logger.info(f"PBFT Phase Timings:")
        for phase, time_value in phase_times.items():
            logger.info(f"  - {phase}: {time_value:.4f}s")
        
        logger.info(f"Time Distribution:")
        logger.info(f"  - PBFT Validation: {pbft_percentage:.2f}% of total fit time")
        logger.info(f"  - Network Delay: {network_percentage:.2f}% of PBFT time")
        logger.info(f"  - Computation: {compute_percentage:.2f}% of PBFT time")
        
        logger.info(f"Total Challenged Rounds: {len(self.challenged_rounds)}")
        logger.info("--------------------------------------------\n")