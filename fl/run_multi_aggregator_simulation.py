#!/usr/bin/env python3
"""
Run simulations with multiple virtual aggregators for research.

Outputs:
- research_data_<scenario>_<challenges>_<timestamp>.json: Per-scenario results
- research_summary.json: Summary of all scenarios and their result files
- visualizations/: Directory with generated plots (if --visualize is used)

Usage (new, recommended):
    flwr run --server-app fl/fl/multi_aggregator_server_app.py --client-app fl/fl/client_app.py --num-supernodes 10

Advanced usage:
    # For scenario/experiment automation, call run_research_scenarios() from server_app.py

See README.md for more details and advanced configuration.
"""

# This script is now a utility module for scenario/experiment logic.
# Use 'flwr run' CLI to launch the simulation. Do NOT run this file directly.

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# --- Scenario/experiment logic for import/use in server_app.py ---

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
    
    # Scenario 1: All honest aggregators
    logger.info("\n\n=== SCENARIO 1: All honest aggregators ===\n")
    results["honest"] = run_simulation(
        num_clients=base_clients,
        num_rounds=base_rounds,
        num_aggregators=3,
        malicious_aggregator_ids=None,
        enable_challenges=True,
        output_dir=output_dir,
    )
    
    # Scenario 2: One malicious aggregator without challenges
    logger.info("\n\n=== SCENARIO 2: One malicious aggregator (no challenges) ===\n")
    results["malicious_no_challenges"] = run_simulation(
        num_clients=base_clients,
        num_rounds=base_rounds,
        num_aggregators=3,
        malicious_aggregator_ids=[1],  # Make aggregator 1 malicious
        enable_challenges=False,
        output_dir=output_dir,
    )
    
    # Scenario 3: One malicious aggregator with challenges
    logger.info("\n\n=== SCENARIO 3: One malicious aggregator (with challenges) ===\n")
    results["malicious_with_challenges"] = run_simulation(
        num_clients=base_clients,
        num_rounds=base_rounds,
        num_aggregators=3,
        malicious_aggregator_ids=[1],  # Make aggregator 1 malicious
        enable_challenges=True,
        output_dir=output_dir,
    )
    
    # Scenario 4: Multiple malicious aggregators with challenges
    logger.info("\n\n=== SCENARIO 4: Multiple malicious aggregators (with challenges) ===\n")
    results["multiple_malicious"] = run_simulation(
        num_clients=base_clients,
        num_rounds=base_rounds,
        num_aggregators=5,
        malicious_aggregator_ids=[1, 3],  # Make aggregators 1 and 3 malicious
        enable_challenges=True,
        output_dir=output_dir,
    )
    
    # Save summary of all scenarios
    summary = {
        "timestamp": datetime.now().isoformat(),
        "scenarios": {
            "honest": {"description": "All honest aggregators", "result_file": results.get("honest")},
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

# --- CLI entrypoint: print usage info ---

def main():
    print("[INFO] This script is now a utility module for scenario/experiment logic.")
    print("[INFO] Please use 'flwr run' to launch the simulation, e.g.:")
    print("    flwr run --server-app fl/fl/multi_aggregator_server_app.py --client-app fl/fl/client_app.py --num-supernodes 10")
    print("[INFO] For automated research scenarios, import and call run_research_scenarios() from your server_app.py.")

if __name__ == "__main__":
    run_research_scenarios(output_dir="research_results")
    main()