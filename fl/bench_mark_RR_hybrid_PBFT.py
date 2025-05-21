#!/usr/bin/env python3
"""
Benchmark RR, Optimistic, PBFT processing time vs. number of aggregators.

Runs simulations with clients=40 and aggregators in [5, 10, 15, 20, 25],
for each of RR (no challenge), Optimistic (0.25), PBFT (1.0).
Plots processing time (s) vs. aggregators for all three methods.

Usage:
    python bench_mark_RR_Optimistic_PBFT.py --rounds 3 --output-dir results
"""
import os
import subprocess
import json
import time
import argparse
import matplotlib.pyplot as plt
from fl.server import save_research_data  # 集中數據收集模組

AGGREGATOR_COUNTS = [5, 10, 15, 20, 25]
SCENARIOS = [
    ("RR", False, 0.0),
    ("Optimistic", True, 0.25),
    ("PBFT", True, 1.0),
]


def run_simulation(clients, rounds, aggregators, enable_challenges, challenge_frequency, output_dir, network_delay_factor, scenario_name):
    os.makedirs(output_dir, exist_ok=True)
    malicious_str = ""
    run_config = {
        "num_rounds": rounds,
        "fraction_fit": 0.5,
        "local_epochs": 1,
        "num_aggregators": aggregators,
        "malicious_aggregators": malicious_str,
        "enable_challenges": enable_challenges,
        "output_dir": output_dir,
        "challenge_frequency": challenge_frequency,
        "challenge_mode": "deterministic",
        "network_delay_factor": network_delay_factor,
        "scenario": scenario_name,
    }
    federation_config = {
        "num_supernodes": clients
    }
    run_config_json = json.dumps(run_config)
    federation_config_json = json.dumps(federation_config)

    cmd = [
        "flwr", "run",
        "--run-config", run_config_json,
        "--federation-config", federation_config_json
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    # 只回傳 output_dir，數據收集交由集中模組
    return output_dir, run_config


def collect_processing_time_by_scenario(output_dir, run_config):
    scenario = run_config["scenario"]
    num_aggregators = run_config["num_aggregators"]
    # 檔名格式: research_data_{scenario}_aggs{num_aggregators}_mal...json
    prefix = f"research_data_{scenario}_aggs{num_aggregators}_"
    files = [f for f in os.listdir(output_dir) if f.startswith(prefix) and f.endswith(".json")]
    if not files:
        raise RuntimeError(f"No research_data file found for scenario={scenario}, aggs={num_aggregators} in {output_dir}")
    # 若有多個，取最新
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
    latest_json = os.path.join(output_dir, files[0])
    with open(latest_json, "r") as f:
        data = json.load(f)
    proc_time = data.get("total_elapsed_time_sec", None)
    metadata = data.get("metadata", {})
    scenario_from_metadata = metadata.get("scenario", metadata.get("name", None))
    return proc_time, scenario_from_metadata


def main():
    parser = argparse.ArgumentParser(description="Benchmark RR, Optimistic, PBFT processing time vs. aggregators.")
    parser.add_argument("--clients", type=int, default=40, help="Number of clients (default: 40)")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds (default: 3)")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--network-delay-factor", type=float, default=0.05, help="Network delay factor (default: 0.05)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}  # scenario_name: {agg_count: proc_time}

    for agg_count in AGGREGATOR_COUNTS:
        for scenario_name, enable_challenges, challenge_freq in SCENARIOS:
            print(f"\n=== {scenario_name} | Aggregators: {agg_count} ===")
            output_dir, run_config = run_simulation(
                clients=args.clients,
                rounds=args.rounds,
                aggregators=agg_count,
                enable_challenges=enable_challenges,
                challenge_frequency=challenge_freq,
                output_dir=args.output_dir,
                network_delay_factor=args.network_delay_factor,
                scenario_name=scenario_name,
            )
            proc_time, scenario_from_metadata = collect_processing_time_by_scenario(output_dir, run_config)
            print(f"Processing time: {proc_time}s (scenario: {scenario_from_metadata})")
            if scenario_from_metadata not in results:
                results[scenario_from_metadata] = {}
            results[scenario_from_metadata][agg_count] = proc_time
            time.sleep(1)  # Avoid file timestamp collision

    # Plotting
    plt.figure(figsize=(10, 6))
    for scenario_name in ["RR", "Optimistic", "PBFT"]:
        if scenario_name in results:
            y = [results[scenario_name].get(agg, None) for agg in AGGREGATOR_COUNTS]
            plt.plot(AGGREGATOR_COUNTS, y, marker='o', label=scenario_name)
    plt.xlabel("Number of Aggregators")
    plt.ylabel("Total Elapsed Time (s)")
    plt.title("Processing Time vs. Number of Aggregators (clients=40)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "processing_time_vs_aggregators.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main() 