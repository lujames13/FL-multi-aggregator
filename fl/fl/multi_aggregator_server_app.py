"""fl: A Flower / PyTorch app with multiple virtual aggregators."""

import logging
import json
import os
from typing import Dict, List

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from .multi_aggregator_strategy import MultiAggregatorStrategy
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
    
    # Server config
    config = ServerConfig(num_rounds=num_rounds)
    
    # Save the strategy for later data collection
    # We'll store it in a global variable that can be accessed after simulation
    global multi_aggregator_strategy
    multi_aggregator_strategy = strategy
    
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)

# Global variable to store the strategy instance for data collection
multi_aggregator_strategy = None


def save_research_data(output_path: str = "research_data.json") -> None:
    """Save research data to a file after simulation completes.
    
    Args:
        output_path: Path to save the research data JSON file
    
    Returns:
        None
    """
    if multi_aggregator_strategy is None:
        logger.warning("No research data available. Run simulation first.")
        return
    
    # Get research data from the strategy
    data = multi_aggregator_strategy.get_research_data()
    
    # Save to file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Research data saved to {output_path}")
    
    # Print summary
    print("\n=== RESEARCH SUMMARY ===")
    print(f"Total rounds: {data['total_rounds']}")
    print(f"Total aggregators: {data['total_aggregators']}")
    print(f"Malicious aggregators: {data['malicious_aggregators']}")
    print(f"Challenge success rate: {data['challenge_success_rate']:.2f}")
    print(f"Malicious detection rate: {data['malicious_detection_rate']:.2f}")
    print("========================\n")