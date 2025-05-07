#!/usr/bin/env python3
"""
Run simulations with multiple virtual aggregators for research.

Outputs:
- research_data_<scenario>_<challenges>_<timestamp>.json: Per-scenario results
- research_summary.json: Summary of all scenarios and their result files
- visualizations/: Directory with generated plots (if --visualize is used)

Usage:
    python run_multi_aggregator_simulation.py --scenario all
    python run_multi_aggregator_simulation.py --scenario single --clients 10 --rounds 5 --aggregators 3 --malicious 1

    # Or use the Flower CLI directly:
    flwr run  \
        --run-config '{"num-aggregators": 3, "malicious-aggregators": "1", "enable-challenges": true}' \
        --federation-config '{"num-supernodes": 10}'

See README.md for more details and advanced configuration.
"""

import argparse
import json
import logging
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Scenario/experiment logic ---

def run_simulation(
    num_clients: int,
    num_rounds: int,
    num_aggregators: int,
    malicious_aggregator_ids: Optional[List[int]] = None,
    enable_challenges: bool = True,
    output_dir: str = "results",
    challenge_frequency: float = 0.25,
    challenge_mode: str = 'deterministic',
):
    """Run a single simulation with specified parameters using flwr run.
    
    Args:
        num_clients: Number of federated learning clients
        num_rounds: Number of federated learning rounds
        num_aggregators: Number of virtual aggregators
        malicious_aggregator_ids: List of aggregator IDs (0-indexed) that will produce malicious results
        enable_challenges: Whether to enable the challenge mechanism
        output_dir: Directory to save results
        challenge_frequency: Challenge frequency: 0 for RR, 1 for PBFT, 0.25 for Hybrid
        challenge_mode: Challenge mode: 'deterministic' or 'random'
        
    Returns:
        Path to the output file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Format malicious aggregators as comma-separated string
    malicious_str = ",".join(map(str, malicious_aggregator_ids)) if malicious_aggregator_ids else ""
    
    # Create unique scenario name
    scenario_name = f"aggs{num_aggregators}_mal{malicious_str.replace(',', '_')}_chal{'on' if enable_challenges else 'off'}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Output file path
    output_file = os.path.join(output_dir, f"research_data_{scenario_name}_{timestamp}.json")
    
    # Construct run_config JSON
    run_config = {
        "num-server-rounds": num_rounds,
        "fraction-fit": 0.5,
        "local-epochs": 1,
        "num-aggregators": num_aggregators,
        "malicious-aggregators": malicious_str,
        "enable-challenges": enable_challenges,
        "output-path": output_file,
        "challenge-frequency": challenge_frequency,
        "challenge-mode": challenge_mode,
    }
    
    # Construct federation_config JSON
    federation_config = {
        "num-supernodes": num_clients
    }
    
    # Convert to JSON strings
    run_config_json = json.dumps(run_config)
    federation_config_json = json.dumps(federation_config)
    
    # Log the configuration
    logger.info(f"Running simulation with:")
    logger.info(f"  - Clients: {num_clients}")
    logger.info(f"  - Rounds: {num_rounds}")
    logger.info(f"  - Aggregators: {num_aggregators}")
    logger.info(f"  - Malicious: {malicious_str if malicious_str else 'None'}")
    logger.info(f"  - Challenges: {'Enabled' if enable_challenges else 'Disabled'}")
    logger.info(f"  - Output file: {output_file}")
    
    # Construct the flwr run command
    cmd = [
        "flwr", "run",
        # "--server-app", "fl/fl/multi_aggregator_server_app.py",
        # "--client-app", "fl/fl/client_app.py",
        "--run-config", run_config_json,
        "--federation-config", federation_config_json
    ]
    
    # Run the command
    logger.info(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Simulation completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Simulation failed with error: {e}")
        return None
    
    # Check if output file was created
    if not os.path.exists(output_file):
        logger.warning(f"Output file {output_file} not found. Make sure your server app saves research data.")
    
    return output_file


def run_research_scenarios(
    output_dir: str = "research_results",
    base_clients: int = 10,
    base_rounds: int = 3,
):
    """Run multiple research scenarios to collect data for the paper.
    Args:
        output_dir: Directory to save all simulation results
        base_clients: Base number of clients for simulations
        base_rounds: Base number of rounds for simulations
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    # Scenario 1: RR (no challenge)
    logger.info("\n\n=== SCENARIO 1: RR (no challenge) ===\n")
    results["rr"] = run_simulation(
        num_clients=base_clients,
        num_rounds=base_rounds,
        num_aggregators=3,
        malicious_aggregator_ids=None,
        enable_challenges=False,
        output_dir=output_dir,
        challenge_frequency=0.0,
        challenge_mode='deterministic',
    )
    
    # Scenario 2: Hybrid (1/4 deterministic)
    logger.info("\n\n=== SCENARIO 2: Hybrid (1/4 deterministic) ===\n")
    results["hybrid"] = run_simulation(
        num_clients=base_clients,
        num_rounds=base_rounds,
        num_aggregators=3,
        malicious_aggregator_ids=None,
        enable_challenges=True,
        output_dir=output_dir,
        challenge_frequency=0.25,
        challenge_mode='deterministic',
    )
    
    # Scenario 3: PBFT (all challenged)
    logger.info("\n\n=== SCENARIO 3: PBFT (all challenged) ===\n")
    results["pbft"] = run_simulation(
        num_clients=base_clients,
        num_rounds=base_rounds,
        num_aggregators=3,
        malicious_aggregator_ids=None,
        enable_challenges=True,
        output_dir=output_dir,
        challenge_frequency=1.0,
        challenge_mode='deterministic',
    )
    
    # Scenario 4: One malicious aggregator without challenges
    logger.info("\n\n=== SCENARIO 4: One malicious aggregator (no challenges) ===\n")
    results["malicious_no_challenges"] = run_simulation(
        num_clients=base_clients,
        num_rounds=base_rounds,
        num_aggregators=3,
        malicious_aggregator_ids=[1],  # Make aggregator 1 malicious
        enable_challenges=False,
        output_dir=output_dir,
        challenge_frequency=0.0,
        challenge_mode='deterministic',
    )
    
    # Scenario 5: One malicious aggregator with challenges
    logger.info("\n\n=== SCENARIO 5: One malicious aggregator (with challenges) ===\n")
    results["malicious_with_challenges"] = run_simulation(
        num_clients=base_clients,
        num_rounds=base_rounds,
        num_aggregators=3,
        malicious_aggregator_ids=[1],  # Make aggregator 1 malicious
        enable_challenges=True,
        output_dir=output_dir,
        challenge_frequency=0.25,
        challenge_mode='deterministic',
    )
    
    # Scenario 6: Multiple malicious aggregators with challenges
    logger.info("\n\n=== SCENARIO 6: Multiple malicious aggregators (with challenges) ===\n")
    results["multiple_malicious"] = run_simulation(
        num_clients=base_clients,
        num_rounds=base_rounds,
        num_aggregators=5,
        malicious_aggregator_ids=[1, 3],  # Make aggregators 1 and 3 malicious
        enable_challenges=True,
        output_dir=output_dir,
        challenge_frequency=0.25,
        challenge_mode='deterministic',
    )
    
    # Save summary of all scenarios
    summary = {
        "timestamp": datetime.now().isoformat(),
        "scenarios": {
            "rr": {"description": "RR (no challenge)", "result_file": results.get("rr")},
            "hybrid": {"description": "Hybrid (1/4 deterministic)", "result_file": results.get("hybrid")},
            "pbft": {"description": "PBFT (all challenged)", "result_file": results.get("pbft")},
            "malicious_no_challenges": {
                "description": "One malicious aggregator without challenges",
                "result_file": results.get("malicious_no_challenges"),
            },
            "malicious_with_challenges": {
                "description": "One malicious aggregator with challenges",
                "result_file": results.get("malicious_with_challenges"),
            },
            "multiple_malicious": {
                "description": "Multiple malicious aggregators with challenges",
                "result_file": results.get("multiple_malicious"),
            },
        },
        "config": {
            "base_clients": base_clients,
            "base_rounds": base_rounds,
        },
    }
    with open(os.path.join(output_dir, "research_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("\n=== RESEARCH SCENARIOS COMPLETED ===")
    logger.info(f"Results saved to {output_dir}")
    logger.info("Summary of scenarios:")
    for scenario, info in summary["scenarios"].items():
        logger.info(f"- {scenario}: {info['description']}")
        logger.info(f"  Result file: {info['result_file']}")

# --- Command line interface ---

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run multi-aggregator simulations")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["single", "all"],
        default="single",
        help="Scenario to run: 'single' for a single simulation, 'all' for predefined research scenarios",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=10,
        help="Number of clients for the simulation",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of rounds for the simulation",
    )
    parser.add_argument(
        "--aggregators",
        type=int,
        default=3,
        help="Number of virtual aggregators",
    )
    parser.add_argument(
        "--malicious",
        type=str,
        default="",
        help="Comma-separated list of malicious aggregator IDs (0-indexed)",
    )
    parser.add_argument(
        "--challenges",
        type=str,
        choices=["enable", "disable"],
        default="enable",
        help="Enable or disable challenge mechanism",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations after simulation",
    )
    parser.add_argument(
        "--challenge-frequency",
        type=float,
        default=0.25,
        help="Challenge frequency: 0 for RR, 1 for PBFT, 0.25 for Hybrid",
    )
    parser.add_argument(
        "--challenge-mode",
        type=str,
        choices=["deterministic", "random"],
        default="deterministic",
        help="Challenge mode: 'deterministic' or 'random'",
    )
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    if args.scenario == "all":
        # Run all predefined research scenarios
        run_research_scenarios(
            output_dir=args.output_dir,
            base_clients=args.clients,
            base_rounds=args.rounds,
        )
    else:
        # Run a single simulation with the specified parameters
        malicious_ids = [int(x) for x in args.malicious.split(",")] if args.malicious else None
        enable_challenges = args.challenges == "enable"
        
        output_file = run_simulation(
            num_clients=args.clients,
            num_rounds=args.rounds,
            num_aggregators=args.aggregators,
            malicious_aggregator_ids=malicious_ids,
            enable_challenges=enable_challenges,
            output_dir=args.output_dir,
            challenge_frequency=args.challenge_frequency,
            challenge_mode=args.challenge_mode,
        )
        
        if output_file and args.visualize:
            # Import and run the visualization function if available
            try:
                # Update this import path as needed
                from fl.analyze_results import generate_plots
                generate_plots(args.output_dir)
                logger.info(f"Visualizations generated in {args.output_dir}/visualizations")
            except ImportError:
                logger.warning("Could not import generate_plots function. Make sure analyze_results.py is available.")
            except Exception as e:
                logger.error(f"Error generating visualizations: {e}")


if __name__ == "__main__":
    main()