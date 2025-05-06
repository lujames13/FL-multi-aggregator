"""fl: A Flower / PyTorch app with multiple virtual aggregators."""

import logging
import json
import os
from typing import Dict, List

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_manager import SimpleClientManager

from .multi_aggregator_strategy import MultiAggregatorStrategy
from .server import MultiAggregatorResultsSaverServer, save_results_and_research_data
from fl.task import Net, get_weights

logger = logging.getLogger(__name__)

def server_fn(context: Context):
    """Server function for the Flower server."""
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    
    # Get multi-aggregator settings from config or use defaults
    num_aggregators = context.run_config.get("num-aggregators", 3)
    enable_challenges = context.run_config.get("enable-challenges", True)
    
    # Determine which aggregators are malicious (if any)
    malicious_aggregator_str = context.run_config.get("malicious-aggregators", "")
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
    logger.info(f"Running for {num_rounds} rounds with fraction_fit={fraction_fit}")
    
    # Define strategy
    strategy = MultiAggregatorStrategy(
        num_aggregators=num_aggregators,
        malicious_aggregator_ids=malicious_aggregator_ids,
        enable_challenges=enable_challenges,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
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