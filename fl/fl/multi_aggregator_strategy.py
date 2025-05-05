"""fl: A Flower / PyTorch app with multiple virtual aggregators."""

import copy
import json
import logging
import random
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


class MultiAggregatorStrategy(FedAvg):
    """Strategy for simulating multiple aggregators in Flower.

    This strategy extends FedAvg to simulate multiple aggregators, where:
    - Each round, a different aggregator is selected (round-robin)
    - Some aggregators can be configured as "malicious"
    - A challenge mechanism can detect malicious aggregations
    - Metrics are tracked for research purposes
    """

    def __init__(
        self,
        num_aggregators: int = 3,
        malicious_aggregator_ids: List[int] = None,
        enable_challenges: bool = True,
        malicious_strategies: Optional[Dict[int, str]] = None,
        challenge_threshold: float = 1e-6,
        challenge_metric: str = "mse",
        **kwargs,
    ):
        """Initialize the MultiAggregatorStrategy.

        Args:
            num_aggregators: Number of virtual aggregators to simulate
            malicious_aggregator_ids: List of aggregator IDs (0-indexed) that will produce malicious results
            enable_challenges: Whether to enable the challenge mechanism
            malicious_strategies: Dict mapping aggregator_id to strategy name (e.g., 'noise', 'scaling', 'zeroing')
            challenge_threshold: Threshold for challenge validation
            challenge_metric: Metric for challenge validation ('mse', 'mae', etc.)
            **kwargs: Additional arguments to pass to FedAvg
        """
        super().__init__(**kwargs)
        self.num_aggregators = num_aggregators
        self.malicious_aggregator_ids = malicious_aggregator_ids or []
        self.enable_challenges = enable_challenges
        self.current_aggregator_id = 0
        self.round = 0
        self.metrics_history = []
        self.challenged_rounds = set()
        self.malicious_strategies = malicious_strategies or {i: "noise" for i in self.malicious_aggregator_ids}
        self.challenge_threshold = challenge_threshold
        self.challenge_metric = challenge_metric

        # Validation for malicious aggregator IDs
        for agg_id in self.malicious_aggregator_ids:
            if agg_id < 0 or agg_id >= num_aggregators:
                raise ValueError(f"Malicious aggregator ID {agg_id} is out of range [0, {num_aggregators-1}]")

        logger.info(f"Initialized MultiAggregatorStrategy with {num_aggregators} aggregators")
        logger.info(f"Malicious aggregators: {self.malicious_aggregator_ids}")
        logger.info(f"Malicious strategies: {self.malicious_strategies}")
        logger.info(f"Challenge mechanism enabled: {self.enable_challenges}")
        logger.info(f"Challenge threshold: {self.challenge_threshold}, metric: {self.challenge_metric}")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates from clients using the current aggregator.

        Args:
            server_round: Current server round
            results: List of (client, fit_res) tuples with client updates
            failures: List of failures that occurred during fitting

        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        self.round = server_round

        # Determine the current aggregator using round-robin
        self.current_aggregator_id = (server_round - 1) % self.num_aggregators
        is_malicious = self.current_aggregator_id in self.malicious_aggregator_ids

        logger.info(f"Round {server_round}: Using aggregator {self.current_aggregator_id}" +
                   (" (MALICIOUS)" if is_malicious else ""))

        # First, compute honest aggregation (we'll use this for comparison and challenges)
        honest_parameters, honest_metrics = super().aggregate_fit(server_round, results, failures)
        if honest_parameters is None:
            return None, {}

        # If this is a malicious aggregator, manipulate the results
        if is_malicious:
            honest_ndarrays = parameters_to_ndarrays(honest_parameters)
            malicious_ndarrays = self._create_malicious_aggregation(honest_ndarrays)
            aggregated_parameters = ndarrays_to_parameters(malicious_ndarrays)
            
            # Add a tag to identify this as malicious (for challenge verification)
            metrics = {
                **honest_metrics,
                "aggregator_id": self.current_aggregator_id,
                "malicious": True
            }
        else:
            aggregated_parameters = honest_parameters
            metrics = {
                **honest_metrics,
                "aggregator_id": self.current_aggregator_id,
                "malicious": False
            }

        # Store honest parameters for challenge validation
        self._store_honest_parameters(server_round, honest_parameters)

        # If challenges are enabled, determine if this aggregation is challenged
        if self.enable_challenges and is_malicious:
            challenged = random.random() < 0.8  # 80% chance of catching malicious aggregation
            status = "pending"
            challenge_distance = None
            challenge_threshold = None
            if challenged:
                challenge_success = self._validate_challenge(server_round, aggregated_parameters, honest_parameters)
                self.challenged_rounds.add(server_round)
                challenge_distance = getattr(self, '_last_challenge_distance', None)
                challenge_threshold = getattr(self, '_last_challenge_threshold', None)
                status = "successful" if challenge_success else "rejected"
                # Track challenge metrics
                challenge_metrics = {
                    "round": int(server_round),
                    "aggregator_id": int(self.current_aggregator_id),
                    "malicious": bool(is_malicious),
                    "malicious_strategy": self.malicious_strategies.get(self.current_aggregator_id, "noise"),
                    "challenged": bool(challenged),
                    "challenge_success": bool(challenge_success),
                    "challenge_status": status,
                    "challenge_distance": challenge_distance,
                    "challenge_threshold": challenge_threshold,
                }
                # Log parameter differences for challenged rounds
                if challenge_distance is not None:
                    challenge_metrics["param_diff"] = float(challenge_distance)
                self.metrics_history.append(challenge_metrics)
                logger.info(f"Round {server_round}: Challenge result - {json.dumps(challenge_metrics)}")
                if challenge_success:
                    logger.info(f"Round {server_round}: Challenge successful! Using honest parameters instead")
                    return honest_parameters, {
                        **metrics,
                        "challenged": True,
                        "challenge_successful": True,
                        "challenge_status": status,
                        "challenge_distance": challenge_distance,
                        "challenge_threshold": challenge_threshold,
                        "malicious_strategy": self.malicious_strategies.get(self.current_aggregator_id, "noise"),
                    }
                else:
                    return aggregated_parameters, {
                        **metrics,
                        "challenged": True,
                        "challenge_successful": False,
                        "challenge_status": status,
                        "challenge_distance": challenge_distance,
                        "challenge_threshold": challenge_threshold,
                        "malicious_strategy": self.malicious_strategies.get(self.current_aggregator_id, "noise"),
                    }
            else:
                # Not challenged
                challenge_metrics = {
                    "round": int(server_round),
                    "aggregator_id": int(self.current_aggregator_id),
                    "malicious": bool(is_malicious),
                    "malicious_strategy": self.malicious_strategies.get(self.current_aggregator_id, "noise"),
                    "challenged": False,
                    "challenge_success": None,
                    "challenge_status": status,
                    "challenge_distance": None,
                    "challenge_threshold": None,
                }
                self.metrics_history.append(challenge_metrics)
        return aggregated_parameters, metrics

    def _create_malicious_aggregation(self, honest_ndarrays: List[np.ndarray]) -> List[np.ndarray]:
        """Create a malicious aggregation by modifying the honest aggregation.

        Supports multiple strategies: 'noise', 'scaling', 'zeroing'.
        """
        if hasattr(self, 'current_aggregator_id') and self.current_aggregator_id in self.malicious_aggregator_ids:
            strategy = self.malicious_strategies.get(self.current_aggregator_id, "noise")
            malicious_ndarrays = [np.copy(arr) for arr in honest_ndarrays]
            if strategy == "noise":
                for i in range(len(malicious_ndarrays)):
                    scale = np.mean(np.abs(malicious_ndarrays[i])) * 0.5
                    malicious_ndarrays[i] += np.random.normal(0, scale, size=malicious_ndarrays[i].shape)
            elif strategy == "scaling":
                for i in range(len(malicious_ndarrays)):
                    malicious_ndarrays[i] *= 2.0  # Arbitrary scaling
            elif strategy == "zeroing":
                for i in range(len(malicious_ndarrays)):
                    malicious_ndarrays[i].fill(0)
            # Add more strategies as needed
            return malicious_ndarrays
        else:
            return [np.copy(arr) for arr in honest_ndarrays]

    def _store_honest_parameters(self, server_round: int, honest_parameters: Parameters) -> None:
        """Store the honest parameters for later comparison in challenges.

        Args:
            server_round: Current server round
            honest_parameters: Honestly computed parameters
        """
        # In a real implementation, these would be securely stored or committed to the blockchain
        # Here we just store them as an attribute
        self._honest_parameters = honest_parameters

    def _validate_challenge(
        self, 
        server_round: int, 
        challenged_parameters: Parameters,
        honest_parameters: Parameters
    ) -> bool:
        """Validate a challenge by comparing challenged parameters with honest ones.

        Supports configurable metric and threshold.
        Logs the distance and threshold used.
        """
        challenged_ndarrays = parameters_to_ndarrays(challenged_parameters)
        honest_ndarrays = parameters_to_ndarrays(honest_parameters)
        # Calculate distance
        if self.challenge_metric == "mse":
            total_diff = sum(np.mean(np.square(c - h)) for c, h in zip(challenged_ndarrays, honest_ndarrays))
        elif self.challenge_metric == "mae":
            total_diff = sum(np.mean(np.abs(c - h)) for c, h in zip(challenged_ndarrays, honest_ndarrays))
        else:
            total_diff = sum(np.mean(np.square(c - h)) for c, h in zip(challenged_ndarrays, honest_ndarrays))
        result = total_diff > self.challenge_threshold
        logger.info(f"Challenge validation - metric: {self.challenge_metric}, difference: {total_diff}, threshold: {self.challenge_threshold}, result: {result}")
        # Store for metrics logging
        self._last_challenge_distance = total_diff
        self._last_challenge_threshold = self.challenge_threshold
        return result

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results from clients.

        Args:
            server_round: Current server round
            results: List of (client, eval_res) tuples with evaluation results
            failures: List of failures that occurred during evaluation

        Returns:
            Tuple of (aggregated_loss, metrics)
        """
        aggregated_loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        
        # Add aggregator info to metrics
        metrics["aggregator_id"] = self.current_aggregator_id
        metrics["malicious"] = self.current_aggregator_id in self.malicious_aggregator_ids
        metrics["challenged"] = server_round in self.challenged_rounds
        
        # Generate summary metrics for the paper
        if server_round % 5 == 0 or server_round == 1:
            self._log_research_metrics()
        
        return aggregated_loss, metrics

    def _log_research_metrics(self) -> None:
        """Log metrics relevant for the research paper."""
        if not self.metrics_history:
            return
        
        # Calculate metrics
        total_rounds = self.round
        total_malicious = len([m for m in self.metrics_history if m["malicious"]])
        total_challenges = len([m for m in self.metrics_history if m["challenged"]])
        successful_challenges = len([m for m in self.metrics_history if m.get("challenge_success", False)])
        
        # Compute rates
        challenge_success_rate = successful_challenges / total_challenges if total_challenges > 0 else 0
        malicious_detection_rate = successful_challenges / total_malicious if total_malicious > 0 else 0
        
        logger.info(f"\n----- RESEARCH METRICS (Round {self.round}) -----")
        logger.info(f"Total rounds: {total_rounds}")
        logger.info(f"Total malicious aggregations: {total_malicious}")
        logger.info(f"Total challenges: {total_challenges}")
        logger.info(f"Successful challenges: {successful_challenges}")
        logger.info(f"Challenge success rate: {challenge_success_rate:.2f}")
        logger.info(f"Malicious detection rate: {malicious_detection_rate:.2f}")
        logger.info("--------------------------------------------\n")

    def get_research_data(self) -> Dict:
        """Get research data for paper generation.

        Returns:
            Dictionary with research metrics
        """
        if not self.metrics_history:
            return {}
        
        # Calculate overall metrics
        total_rounds = self.round
        malicious_rounds = [m for m in self.metrics_history if m["malicious"]]
        total_malicious = len(malicious_rounds)
        total_challenges = len([m for m in self.metrics_history if m["challenged"]])
        successful_challenges = len([m for m in self.metrics_history if m.get("challenge_success", False)])
        
        # Compute rates
        challenge_success_rate = successful_challenges / total_challenges if total_challenges > 0 else 0
        malicious_detection_rate = successful_challenges / total_malicious if total_malicious > 0 else 0
        
        # Return data for research
        return {
            "total_rounds": total_rounds,
            "total_aggregators": self.num_aggregators,
            "malicious_aggregators": len(self.malicious_aggregator_ids),
            "total_malicious_rounds": total_malicious,
            "total_challenges": total_challenges,
            "successful_challenges": successful_challenges,
            "challenge_success_rate": challenge_success_rate,
            "malicious_detection_rate": malicious_detection_rate,
            "detailed_history": self.metrics_history
        }