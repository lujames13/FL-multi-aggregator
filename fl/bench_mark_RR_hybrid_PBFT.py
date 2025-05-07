#!/usr/bin/env python3
"""
Benchmark RR, Hybrid, PBFT processing time vs. number of aggregators.

Runs simulations with clients=40 and aggregators in [5, 10, 15, 20, 25],
for each of RR (no challenge), Hybrid (0.25), PBFT (1.0).
Plots processing time (s) vs. aggregators for all three methods.

Usage:
    python bench_mark_RR_hybrid_PBFT.py --rounds 3 --output-dir results
"""
import os
import subprocess
import json
import time
import argparse
import matplotlib.pyplot as plt

AGGREGATOR_COUNTS = [5, 10, 15, 20, 25]
SCENARIOS = [
    ("RR", False, 0.0),
    ("Hybrid", True, 0.25),
    ("PBFT", True, 1.0),
]


def run_simulation(clients, rounds, aggregators, enable_challenges, challenge_frequency, output_dir, network_delay_factor, scenario_name):
    os.makedirs(output_dir, exist_ok=True)
    malicious_str = ""
    run_config = {
        "num-server-rounds": rounds,
        "fraction-fit": 0.5,
        "local-epochs": 1,
        "num-aggregators": aggregators,
        "malicious-aggregators": malicious_str,
        "enable-challenges": enable_challenges,
        "output-dir": output_dir,
        "challenge-frequency": challenge_frequency,
        "challenge-mode": "deterministic",
        "network-delay-factor": network_delay_factor,
        "scenario": scenario_name,
    }
    federation_config = {
        "num-supernodes": clients
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
    files = [f for f in os.listdir(output_dir) if f.startswith("research_data_") and f.endswith(".json")]
    if not files:
        raise RuntimeError(f"No research_data_*.json found in {output_dir}")
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
    return os.path.join(output_dir, files[0])


def extract_processing_time(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data.get("total_elapsed_time_sec", None)


def main():
    parser = argparse.ArgumentParser(description="Benchmark RR, Hybrid, PBFT processing time vs. aggregators.")
    parser.add_argument("--clients", type=int, default=40, help="Number of clients (default: 40)")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds (default: 3)")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--network-delay-factor", type=float, default=0.05, help="Network delay factor (default: 0.05)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = {scenario[0]: [] for scenario in SCENARIOS}

    for agg_count in AGGREGATOR_COUNTS:
        for scenario_name, enable_challenges, challenge_freq in SCENARIOS:
            print(f"\n=== {scenario_name} | Aggregators: {agg_count} ===")
            json_file = run_simulation(
                clients=args.clients,
                rounds=args.rounds,
                aggregators=agg_count,
                enable_challenges=enable_challenges,
                challenge_frequency=challenge_freq,
                output_dir=args.output_dir,
                network_delay_factor=args.network_delay_factor,
                scenario_name=scenario_name,
            )
            proc_time = extract_processing_time(json_file)
            print(f"Processing time: {proc_time}s")
            results[scenario_name].append(proc_time)
            time.sleep(1)  # Avoid file timestamp collision

    # Plotting
    plt.figure(figsize=(10, 6))
    for scenario_name in results:
        plt.plot(AGGREGATOR_COUNTS, results[scenario_name], marker='o', label=scenario_name)
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