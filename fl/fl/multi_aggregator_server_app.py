"""fl: A Flower / PyTorch app with multiple virtual aggregators."""

import logging
import json
import os
import sys
from typing import Dict, List
import argparse

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_manager import SimpleClientManager

from .hybrid_strategy import HybridOptimisticPBFTAggregatorStrategy
from .hybrid_strategy_rollback import HybridOptimisticPBFTAggregatorStrategy_Rollback
from .server import MultiAggregatorResultsSaverServer, save_results_and_research_data
from fl.task import Net, get_weights

logger = logging.getLogger(__name__)

def server_fn(context: Context):
    """Server function for the Flower server."""
    # 解析 CLI 傳入的 --run-config
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", type=str, default="")
    args, _ = parser.parse_known_args()
    if args.run_config:
        cli_config = json.loads(args.run_config)
        print("DEBUG: Detected CLI run-config:", cli_config)
        # 強制 merge，CLI 參數優先
        context.run_config.update(cli_config)
    print("DEBUG: (after merge) context.run_config =", context.run_config)

    logger.info(f"DEBUG: context.run_config = {context.run_config}")

    # Read from config
    num_rounds = context.run_config.get("num_rounds", 3)
    fraction_fit = context.run_config.get("fraction_fit", 0.5)
    
    # Get multi-aggregator settings from config or use defaults
    num_aggregators = context.run_config.get("num_aggregators", 3)
    enable_challenges = context.run_config.get("enable_challenges", True)
    challenge_frequency = context.run_config.get("challenge_frequency", 0.25)
    challenge_mode = context.run_config.get("challenge_mode", 'deterministic')
    strategy_type = context.run_config.get("strategy_type", "hybrid")
    network_delay_factor = context.run_config.get("network_delay_factor", 0.05)

    # Determine which aggregators are malicious (if any)
    malicious_aggregator_str = context.run_config.get("malicious_aggregators", "")
    malicious_aggregator_ids = []
    if malicious_aggregator_str:
        malicious_aggregator_ids = [int(x) for x in malicious_aggregator_str.split(",")]
    
    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)
    
    # Log configuration
    logger.info(f"Server starting with {num_aggregators} aggregators")
    logger.info(f"Malicious aggregators: {malicious_aggregator_ids}")
    logger.info(f"Challenge mechanism enabled: {enable_challenges}")
    logger.info(f"Challenge frequency: {challenge_frequency}, Challenge mode: {challenge_mode}")
    logger.info(f"Running for {num_rounds} rounds with fraction_fit={fraction_fit}")
    logger.info(f"Strategy type selected: {strategy_type}")

    # Define strategy based on type
    if strategy_type == "rollback":
        strategy = HybridOptimisticPBFTAggregatorStrategy_Rollback(
            num_aggregators=num_aggregators,
            malicious_aggregator_ids=malicious_aggregator_ids,
            enable_challenges=enable_challenges,
            challenge_frequency=challenge_frequency,
            challenge_mode=challenge_mode,
            detection_delay=context.run_config.get("detection-delay", 2),
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            network_delay_factor=network_delay_factor,
        )
    else:
        strategy = HybridOptimisticPBFTAggregatorStrategy(
            num_aggregators=num_aggregators,
            malicious_aggregator_ids=malicious_aggregator_ids,
            enable_challenges=enable_challenges,
            challenge_frequency=challenge_frequency,
            challenge_mode=challenge_mode,
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            network_delay_factor=network_delay_factor,
        )
    
    # Create client manager
    client_manager = SimpleClientManager()
    
    # Create custom server with results saving capabilities
    server = MultiAggregatorResultsSaverServer(
        client_manager=client_manager,
        strategy=strategy,
        results_saver_fn=save_results_and_research_data,
        run_config=context.run_config,
    )
    
    # Server config
    config = ServerConfig(num_rounds=num_rounds)
    
    # Return components with custom server
    return ServerAppComponents(server=server, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)