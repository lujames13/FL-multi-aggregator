#!/usr/bin/env python3
"""
Analyze results from multi-aggregator simulations and generate visualizations.

Outputs:
- challenge_effectiveness.png: Bar plot of challenge/detection rates per scenario
- aggregator_performance.png: Scatter plot of aggregator activity and challenges
- challenge_timeline.png: Timeline of challenge outcomes per scenario
- aggregator_exclusion.png: Analysis of aggregator exclusion rates
- rollback_overhead.png: Analysis of training efficiency and rollback overhead
- attack_detection.png: Performance of attack detection mechanism
- challenge_delay_correlation.png: Correlation between challenge success and detection delay
- comparative_analysis.json: Summary of key metrics and scenario comparison
- research_report.txt: Human-readable report for research papers

Usage:
    python analyze_results.py --results-dir <results_dir> [--output-dir <output_dir>]

The output directory defaults to <results_dir>/visualizations if not specified.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.ticker import MaxNLocator
except ImportError:
    logger.error("Failed to import required packages. Please install with:")
    logger.error("pip install matplotlib numpy pandas")
    sys.exit(1)


def load_results(results_dir: str) -> Dict:
    """Load all simulation results from a directory.
    
    Args:
        results_dir: Directory containing simulation results
    
    Returns:
        Dictionary mapping scenario names to result data
    """
    all_data = {}
    
    # Find all result files
    result_files = list(Path(results_dir).glob("research_data_*.json"))
    if not result_files:
        logger.warning(f"No result files found in {results_dir}")
        return all_data
    
    # Load data from all result files
    for file_path in result_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                scenario_name = file_path.stem.replace("research_data_", "")
                all_data[scenario_name] = data
                logger.info(f"Loaded data from {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    return all_data


def generate_challenge_effectiveness_plot(
    all_data: Dict,
    output_dir: str,
    filename: str = "challenge_effectiveness.png"
):
    """Generate a plot showing challenge effectiveness across scenarios.
    
    Args:
        all_data: Dictionary mapping scenario names to result data
        output_dir: Directory to save the plot
        filename: Filename for the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    scenarios = []
    challenge_rates = []
    detection_rates = []
    
    for scenario, data in all_data.items():
        if "challenge_success_rate" in data and "malicious_detection_rate" in data:
            scenarios.append(scenario)
            challenge_rates.append(data["challenge_success_rate"] * 100)  # Convert to percentage
            detection_rates.append(data["malicious_detection_rate"] * 100)  # Convert to percentage
    
    if not scenarios:
        logger.warning("No challenge data found in results")
        return
    
    # Set up bar width and positions
    bar_width = 0.35
    x = np.arange(len(scenarios))
    
    # Create bars
    plt.bar(x - bar_width/2, challenge_rates, bar_width, label='Challenge Success Rate (%)')
    plt.bar(x + bar_width/2, detection_rates, bar_width, label='Malicious Detection Rate (%)')
    
    # Add labels and legend
    plt.xlabel('Scenario')
    plt.ylabel('Rate (%)')
    plt.title('Challenge Mechanism Effectiveness')
    plt.xticks(x, scenarios, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    logger.info(f"Generated challenge effectiveness plot: {output_path}")


def generate_aggregator_performance_plot(
    all_data: Dict,
    output_dir: str,
    filename: str = "aggregator_performance.png"
):
    """Generate a plot showing per-aggregator performance across rounds.
    
    Args:
        all_data: Dictionary mapping scenario names to result data
        output_dir: Directory to save the plot
        filename: Filename for the plot
    """
    # Find a scenario with detailed history
    scenario_data = None
    scenario_name = None
    
    for name, data in all_data.items():
        if "detailed_history" in data and data["detailed_history"]:
            scenario_data = data
            scenario_name = name
            break
    
    if scenario_data is None:
        logger.warning("No detailed history found in results")
        return
    
    # Extract per-aggregator data
    history = scenario_data["detailed_history"]
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(history)
    
    if "aggregator_id" not in df.columns or "round" not in df.columns:
        logger.warning("Missing required columns in history data")
        return
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Group by aggregator_id
    aggregator_ids = df["aggregator_id"].unique()
    
    for agg_id in aggregator_ids:
        # Filter data for this aggregator
        agg_data = df[df["aggregator_id"] == agg_id]
        
        # Check if this aggregator is malicious
        is_malicious = any(agg_data["malicious"]) if "malicious" in agg_data.columns else False
        
        # Count challenged rounds for this aggregator
        challenged_count = agg_data["challenged"].sum() if "challenged" in agg_data.columns else 0
        
        # Plot point for each round this aggregator was active
        rounds = agg_data["round"].tolist()
        
        # Use different markers for malicious vs honest aggregators
        marker = 'x' if is_malicious else 'o'
        color = 'red' if is_malicious else 'green'
        label = f"Aggregator {agg_id} ({'Malicious' if is_malicious else 'Honest'}, Challenged: {challenged_count})"
        
        plt.scatter(rounds, [agg_id] * len(rounds), marker=marker, s=100, color=color, label=label)
    
    # Add labels and legend
    plt.xlabel('Round')
    plt.ylabel('Aggregator ID')
    plt.title(f'Aggregator Activity and Challenges ({scenario_name})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    # Set y-axis to show only integer values
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Adjust the round ticks to integers
    plt.xticks(np.arange(min(df["round"]), max(df["round"])+1, 1.0))
    
    # Ensure enough space for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    
    # Save the plot
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    logger.info(f"Generated aggregator performance plot: {output_path}")


def generate_challenge_timeline_plot(
    all_data: Dict,
    output_dir: str,
    filename: str = "challenge_timeline.png"
):
    """Generate a timeline plot showing challenges across rounds.
    
    Args:
        all_data: Dictionary mapping scenario names to result data
        output_dir: Directory to save the plot
        filename: Filename for the plot
    """
    # Find scenarios with challenges
    scenarios_with_challenges = {}
    
    for name, data in all_data.items():
        if "detailed_history" in data and data["detailed_history"]:
            history = data["detailed_history"]
            
            # Check if there are any challenges
            challenges = [entry for entry in history if entry.get("challenged", False)]
            
            if challenges:
                scenarios_with_challenges[name] = challenges
    
    if not scenarios_with_challenges:
        logger.warning("No challenges found in any scenario")
        return
    
    # Set up the plot (one subplot per scenario)
    num_scenarios = len(scenarios_with_challenges)
    fig, axes = plt.subplots(num_scenarios, 1, figsize=(12, 5 * num_scenarios), sharex=True)
    
    # Handle case with only one scenario
    if num_scenarios == 1:
        axes = [axes]
    
    # Plot each scenario
    for i, (scenario_name, challenges) in enumerate(scenarios_with_challenges.items()):
        ax = axes[i]
        
        # Extract data for plotting
        rounds = [c["round"] for c in challenges]
        success = [c.get("challenge_success", False) for c in challenges]
        
        # Create a colormap based on success
        colors = ['green' if s else 'red' for s in success]
        
        # Plot challenges as points
        ax.scatter(rounds, [1] * len(rounds), c=colors, s=100, marker='o')
        
        # Add labels for each point
        for j, r in enumerate(rounds):
            ax.annotate(
                f"Round {r}\n{'Success' if success[j] else 'Failed'}",
                (r, 1),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5)
            )
        
        # Set titles and labels
        ax.set_title(f'Challenge Timeline - {scenario_name}')
        ax.set_xlabel('Round')
        ax.set_yticks([])  # Hide y-axis ticks
        
        # Set x-axis to show only integer values
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add a horizontal line for the timeline
        ax.axhline(y=1, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    logger.info(f"Generated challenge timeline plot: {output_path}")


def generate_aggregator_exclusion_plot(
    all_data: Dict,
    output_dir: str,
    filename: str = "aggregator_exclusion.png"
):
    """Generate a plot showing aggregator exclusion rates across scenarios.
    
    Args:
        all_data: Dictionary mapping scenario names to result data
        output_dir: Directory to save the plot
        filename: Filename for the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    scenarios = []
    exclusion_rates = []
    effective_aggregators = []
    
    for scenario, data in all_data.items():
        if "aggregator_exclusion_percentage" in data:
            scenarios.append(scenario)
            exclusion_rates.append(data["aggregator_exclusion_percentage"])
            effective_aggregators.append(data.get("effective_aggregators", 0))
    
    if not scenarios:
        logger.warning("No aggregator exclusion data found in results")
        return
    
    # Primary plot: Exclusion rates
    ax1 = plt.gca()
    bars = ax1.bar(scenarios, exclusion_rates, color='red', alpha=0.7)
    ax1.set_ylabel('Exclusion Rate (%)', color='red')
    ax1.set_ylim(0, 100)  # Percentage scale
    
    # Add value labels above the bars
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Secondary plot: Effective aggregators
    ax2 = ax1.twinx()
    ax2.plot(scenarios, effective_aggregators, 'bo-', linewidth=2, markersize=8)
    ax2.set_ylabel('Effective Aggregators', color='blue')
    ax2.set_ylim(0, max(effective_aggregators) * 1.5 if effective_aggregators else 10)
    
    # Add title and adjust layout
    plt.title('Aggregator Exclusion Analysis')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    logger.info(f"Generated aggregator exclusion plot: {output_path}")


def generate_rollback_overhead_plot(
    all_data: Dict,
    output_dir: str,
    filename: str = "rollback_overhead.png"
):
    """Generate a plot showing rollback overhead and effective training.
    
    Args:
        all_data: Dictionary mapping scenario names to result data
        output_dir: Directory to save the plot
        filename: Filename for the plot
    """
    plt.figure(figsize=(12, 7))
    
    # Prepare data
    scenarios = []
    total_rounds = []
    effective_rounds = []
    rollback_overhead = []
    
    for scenario, data in all_data.items():
        if "total_rounds" in data and "effective_training_rounds" in data:
            scenarios.append(scenario)
            total_rounds.append(data["total_rounds"])
            effective_rounds.append(data["effective_training_rounds"])
            rollback_overhead.append(data.get("rollback_overhead_percentage", 0))
    
    if not scenarios:
        logger.warning("No rollback data found in results")
        return
    
    # Set up bar width and positions
    bar_width = 0.35
    x = np.arange(len(scenarios))
    
    # Create stacked bars for total and effective rounds
    plt.bar(x, total_rounds, bar_width, label='Total Rounds', color='skyblue')
    plt.bar(x, effective_rounds, bar_width, label='Effective Training Rounds', color='darkblue')
    
    # Add a line for rollback overhead percentage
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(x, rollback_overhead, 'ro-', linewidth=2, markersize=8, label='Rollback Overhead (%)')
    ax2.set_ylabel('Rollback Overhead (%)', color='red')
    ax2.set_ylim(0, max(rollback_overhead) * 1.2 if rollback_overhead else 100)
    
    # Add data annotations
    for i, (tr, er, ro) in enumerate(zip(total_rounds, effective_rounds, rollback_overhead)):
        ax1.annotate(f'{tr}',
                   xy=(i, tr),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom')
        
        ax1.annotate(f'{er}',
                   xy=(i, er / 2),  # Middle of the blue bar
                   xytext=(0, 0),
                   textcoords="offset points",
                   ha='center', va='center',
                   color='white')
        
        ax2.annotate(f'{ro:.1f}%',
                   xy=(i, ro),
                   xytext=(0, -15),
                   textcoords="offset points",
                   ha='center', va='top',
                   color='red')
    
    # Add labels and legend
    ax1.set_xlabel('Scenario')
    ax1.set_ylabel('Number of Rounds')
    plt.title('Training Efficiency and Rollback Overhead Analysis')
    plt.xticks(x, scenarios, rotation=45)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    logger.info(f"Generated rollback overhead plot: {output_path}")


def generate_attack_detection_plot(
    all_data: Dict,
    output_dir: str,
    filename: str = "attack_detection.png"
):
    """Generate a plot analyzing attack detection performance.
    
    Args:
        all_data: Dictionary mapping scenario names to result data
        output_dir: Directory to save the plot
        filename: Filename for the plot
    """
    plt.figure(figsize=(12, 7))
    
    # Prepare data
    scenarios = []
    total_attacks = []
    detected_attacks = []
    detection_delays = []
    
    for scenario, data in all_data.items():
        if "total_rounds_with_attack" in data or "detected_attacks" in data:
            scenarios.append(scenario)
            total_attacks.append(data.get("total_rounds_with_attack", 0))
            detected_attacks.append(len(data.get("detected_attacks", [])))
            detection_delays.append(data.get("detection_delay", 0))
    
    if not scenarios:
        logger.warning("No attack detection data found in results")
        return
    
    # Calculate detection rates
    detection_rates = [detected / total * 100 if total > 0 else 0 
                       for detected, total in zip(detected_attacks, total_attacks)]
    
    # Primary plot: Attacks and detection
    x = np.arange(len(scenarios))
    bar_width = 0.35
    
    ax1 = plt.subplot(211)
    rects1 = ax1.bar(x - bar_width/2, total_attacks, bar_width, label='Total Attacks', color='red', alpha=0.7)
    rects2 = ax1.bar(x + bar_width/2, detected_attacks, bar_width, label='Detected Attacks', color='green', alpha=0.7)
    
    # Add detection rate as text
    for i, rate in enumerate(detection_rates):
        ax1.annotate(f'{rate:.1f}%',
                    xy=(x[i], max(total_attacks[i], detected_attacks[i]) + 0.5),
                    ha='center', va='bottom')
    
    ax1.set_ylabel('Count')
    ax1.set_title('Attack Detection Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.legend()
    
    # Secondary plot: Detection delay
    ax2 = plt.subplot(212)
    ax2.bar(scenarios, detection_delays, color='orange')
    ax2.set_ylabel('Rounds')
    ax2.set_title('Average Detection Delay')
    
    # Add value labels
    for i, delay in enumerate(detection_delays):
        ax2.annotate(f'{delay:.1f}',
                    xy=(i, delay),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    logger.info(f"Generated attack detection plot: {output_path}")


def generate_challenge_delay_correlation_plot(
    all_data: Dict,
    output_dir: str,
    filename: str = "challenge_delay_correlation.png"
):
    """Generate a scatter plot showing correlation between challenge success and detection delay.
    
    Args:
        all_data: Dictionary mapping scenario names to result data
        output_dir: Directory to save the plot
        filename: Filename for the plot
    """
    # Extract data points
    points = []
    labels = []
    
    for scenario, data in all_data.items():
        if "challenge_success_rate" in data and "detection_delay" in data:
            success_rate = data["challenge_success_rate"] * 100  # Convert to percentage
            delay = data["detection_delay"]
            points.append((success_rate, delay))
            labels.append(scenario)
    
    if not points:
        logger.warning("No challenge-delay correlation data found in results")
        return
    
    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    
    # Unpack the points
    success_rates, delays = zip(*points)
    
    # Create scatter plot
    plt.scatter(success_rates, delays, s=100, c=range(len(points)), cmap='viridis')
    
    # Add labels to points
    for i, (sr, d, label) in enumerate(zip(success_rates, delays, labels)):
        plt.annotate(
            label,
            (sr, d),
            xytext=(5, 5),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    # Add best fit line if we have enough points
    if len(points) > 2:
        z = np.polyfit(success_rates, delays, 1)
        p = np.poly1d(z)
        plt.plot(
            np.linspace(min(success_rates), max(success_rates), 100),
            p(np.linspace(min(success_rates), max(success_rates), 100)),
            "r--", alpha=0.8
        )
        
        # Calculate correlation coefficient
        corr = np.corrcoef(success_rates, delays)[0, 1]
        plt.title(f'Challenge Success Rate vs. Detection Delay (r = {corr:.2f})')
    else:
        plt.title('Challenge Success Rate vs. Detection Delay')
    
    # Add labels and grid
    plt.xlabel('Challenge Success Rate (%)')
    plt.ylabel('Detection Delay (rounds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    logger.info(f"Generated challenge-delay correlation plot: {output_path}")


def generate_comparative_analysis(all_data: Dict, output_dir: str):
    """Generate a comparative analysis of all scenarios.
    
    Args:
        all_data: Dictionary mapping scenario names to result data
        output_dir: Directory to save the analysis
    """
    if not all_data:
        logger.warning("No data to analyze")
        return
    
    # Create a comparative summary
    summary = {
        "scenarios": {},
        "comparison": {
            "challenge_mechanism": {},
            "malicious_detection": {},
            "overall_performance": {}
        }
    }
    
    # Extract key metrics for each scenario
    for scenario, data in all_data.items():
        summary["scenarios"][scenario] = {
            # Original metrics
            "total_rounds": data.get("total_rounds", 0),
            "total_aggregators": data.get("total_aggregators", 0),
            "malicious_aggregators": data.get("malicious_aggregators", 0),
            "total_challenges": data.get("total_challenges", 0),
            "successful_challenges": data.get("successful_challenges", 0),
            "challenge_success_rate": data.get("challenge_success_rate", 0),
            "malicious_detection_rate": data.get("malicious_detection_rate", 0),
            
            # New metrics
            "effective_aggregators": data.get("effective_aggregators", 0),
            "aggregator_exclusion_percentage": data.get("aggregator_exclusion_percentage", 0),
            "effective_training_rounds": data.get("effective_training_rounds", 0),
            "rollback_overhead_percentage": data.get("rollback_overhead_percentage", 0),
            "total_rounds_with_attack": data.get("total_rounds_with_attack", 0),
            "detected_attacks": len(data.get("detected_attacks", [])),
            "detection_delay": data.get("detection_delay", 0)
        }
    
    # Compute comparative metrics
    if len(all_data) > 1:
        # Compare scenarios with and without challenges
        challenge_enabled = [s for s, d in all_data.items() if d.get("total_challenges", 0) > 0]
        challenge_disabled = [s for s, d in all_data.items() if d.get("total_challenges", 0) == 0]
        
        summary["comparison"]["challenge_mechanism"]["enabled_scenarios"] = challenge_enabled
        summary["comparison"]["challenge_mechanism"]["disabled_scenarios"] = challenge_disabled
        
        # Compare malicious detection rates
        detection_rates = {s: d.get("malicious_detection_rate", 0) for s, d in all_data.items()}
        max_detection = max(detection_rates.items(), key=lambda x: x[1]) if detection_rates else (None, 0)
        
        summary["comparison"]["malicious_detection"]["best_scenario"] = max_detection[0]
        summary["comparison"]["malicious_detection"]["best_rate"] = max_detection[1]
        summary["comparison"]["malicious_detection"]["all_rates"] = detection_rates
        
        # Overall performance assessment
        # (Simple heuristic: scenarios with high detection rates and challenge success rates)
        scores = {}
        for s, d in all_data.items():
            detection_rate = d.get("malicious_detection_rate", 0)
            challenge_rate = d.get("challenge_success_rate", 0)
            malicious_count = d.get("malicious_aggregators", 0)
            
            # Only score scenarios with malicious aggregators
            if malicious_count > 0:
                scores[s] = (detection_rate * 0.6) + (challenge_rate * 0.4)
        
        best_scenario = max(scores.items(), key=lambda x: x[1]) if scores else (None, 0)
        summary["comparison"]["overall_performance"]["best_scenario"] = best_scenario[0]
        summary["comparison"]["overall_performance"]["best_score"] = best_scenario[1]
        summary["comparison"]["overall_performance"]["all_scores"] = scores
    
    # Save the summary
    output_path = os.path.join(output_dir, "comparative_analysis.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Generated comparative analysis: {output_path}")
    
    # Also generate a text report
    report_path = os.path.join(output_dir, "research_report.txt")
    with open(report_path, "w") as f:
        f.write("# Multi-Aggregator Federated Learning Research Report\n\n")
        
        f.write("## Scenario Summaries\n\n")
        for scenario, metrics in summary["scenarios"].items():
            f.write(f"### {scenario}\n\n")
            f.write(f"- Total rounds: {metrics['total_rounds']}\n")
            f.write(f"- Total aggregators: {metrics['total_aggregators']}\n")
            f.write(f"- Malicious aggregators: {metrics['malicious_aggregators']}\n")
            f.write(f"- Total challenges: {metrics['total_challenges']}\n")
            f.write(f"- Successful challenges: {metrics['successful_challenges']}\n")
            f.write(f"- Challenge success rate: {metrics['challenge_success_rate']:.2%}\n")
            f.write(f"- Malicious detection rate: {metrics['malicious_detection_rate']:.2%}\n\n")
        
        f.write("## Training Efficiency Analysis\n\n")
        for scenario, metrics in summary["scenarios"].items():
            effective_training_pct = (metrics['effective_training_rounds'] / metrics['total_rounds'] * 100) if metrics['total_rounds'] > 0 else 0
            f.write(f"### {scenario}\n\n")
            f.write(f"- Effective training rounds: {metrics['effective_training_rounds']} of {metrics['total_rounds']} ({effective_training_pct:.1f}%)\n")
            f.write(f"- Rollback overhead: {metrics['rollback_overhead_percentage']:.2f}%\n")
            f.write(f"- Effective aggregators: {metrics['effective_aggregators']} of {metrics['total_aggregators']}\n")
            f.write(f"- Aggregator exclusion rate: {metrics['aggregator_exclusion_percentage']:.2f}%\n\n")

        f.write("## Attack Detection Analysis\n\n")
        for scenario, metrics in summary["scenarios"].items():
            detection_rate = (metrics['detected_attacks'] / metrics['total_rounds_with_attack'] * 100) if metrics['total_rounds_with_attack'] > 0 else 0
            f.write(f"### {scenario}\n\n")
            f.write(f"- Attacks detected: {metrics['detected_attacks']} of {metrics['total_rounds_with_attack']} ({detection_rate:.1f}%)\n")
            f.write(f"- Average detection delay: {metrics['detection_delay']:.1f} rounds\n\n")
        
        if len(all_data) > 1:
            f.write("## Comparative Analysis\n\n")
            
            f.write("### Challenge Mechanism Effectiveness\n\n")
            f.write(f"- Scenarios with challenges enabled: {', '.join(summary['comparison']['challenge_mechanism']['enabled_scenarios'])}\n")
            f.write(f"- Scenarios with challenges disabled: {', '.join(summary['comparison']['challenge_mechanism']['disabled_scenarios'])}\n\n")
            
            f.write("### Malicious Detection Performance\n\n")
            best_scenario = summary["comparison"]["malicious_detection"]["best_scenario"]
            best_rate = summary["comparison"]["malicious_detection"]["best_rate"]
            if best_scenario:
                f.write(f"- Best detection performance: {best_scenario} with {best_rate:.2%} detection rate\n")
                f.write("- All detection rates:\n")
                for s, r in summary["comparison"]["malicious_detection"]["all_rates"].items():
                    f.write(f"  - {s}: {r:.2%}\n")
                f.write("\n")
            
            f.write("### Overall Scenario Performance\n\n")
            best_overall = summary["comparison"]["overall_performance"]["best_scenario"]
            best_score = summary["comparison"]["overall_performance"]["best_score"]
            if best_overall:
                f.write(f"- Best overall scenario: {best_overall} with score {best_score:.2f}\n")
                f.write("- All scenario scores:\n")
                for s, score in summary["comparison"]["overall_performance"]["all_scores"].items():
                    f.write(f"  - {s}: {score:.2f}\n")
                f.write("\n")
            
            f.write("## Research Conclusions\n\n")
            
            # Generate some basic conclusions
            has_malicious = any(d.get("malicious_aggregators", 0) > 0 for d in all_data.values())
            has_challenges = any(d.get("total_challenges", 0) > 0 for d in all_data.values())
            
            if has_malicious and has_challenges:
                # Compare scenarios with/without challenges
                with_challenge_rates = [d.get("malicious_detection_rate", 0) for s, d in all_data.items() 
                                       if d.get("total_challenges", 0) > 0 and d.get("malicious_aggregators", 0) > 0]
                without_challenge_rates = [d.get("malicious_detection_rate", 0) for s, d in all_data.items() 
                                          if d.get("total_challenges", 0) == 0 and d.get("malicious_aggregators", 0) > 0]
                
                avg_with = sum(with_challenge_rates) / len(with_challenge_rates) if with_challenge_rates else 0
                avg_without = sum(without_challenge_rates) / len(without_challenge_rates) if without_challenge_rates else 0
                
                if avg_with > avg_without:
                    f.write("1. The challenge mechanism significantly improves detection of malicious aggregators compared to scenarios without challenges.\n\n")
                else:
                    f.write("1. The challenge mechanism did not show significant improvement in detection rates compared to scenarios without challenges.\n\n")
                
                # Effectiveness of the approach
                best_rate = max((d.get("malicious_detection_rate", 0) for d in all_data.values()), default=0)
                if best_rate > 0.7:  # >70% detection rate
                    f.write("2. The approach demonstrates high effectiveness in detecting malicious aggregators, with detection rates exceeding 70% in the best scenarios.\n\n")
                elif best_rate > 0.4:  # >40% detection rate
                    f.write("2. The approach shows moderate effectiveness in detecting malicious aggregators, with detection rates between 40-70% in the best scenarios.\n\n")
                else:
                    f.write("2. The approach shows limited effectiveness in detecting malicious aggregators, with detection rates below 40% in all scenarios.\n\n")
            
            # Add analysis of the training overhead from rollbacks
            avg_rollback_overhead = sum(d.get("rollback_overhead_percentage", 0) for d in all_data.values()) / len(all_data)
            if avg_rollback_overhead < 10:
                f.write("3. The training overhead from rollbacks is minimal (<10%), suggesting that the challenge mechanism is efficient.\n\n")
            elif avg_rollback_overhead < 30:
                f.write("3. The training overhead from rollbacks is moderate (10-30%), suggesting a reasonable tradeoff between security and efficiency.\n\n")
            else:
                f.write("3. The training overhead from rollbacks is significant (>30%), suggesting that optimizations to the challenge mechanism may be needed.\n\n")
                
            # Add analysis of detection delays
            avg_detection_delay = sum(d.get("detection_delay", 0) for d in all_data.values()) / len(all_data)
            if avg_detection_delay < 2:
                f.write("4. Malicious behavior is detected quickly (< 2 rounds on average), enabling rapid response to attacks.\n\n")
            elif avg_detection_delay < 5:
                f.write("4. Malicious behavior detection has moderate delay (2-5 rounds on average), which may be acceptable for most applications.\n\n")
            else:
                f.write("4. Malicious behavior detection has significant delay (> 5 rounds on average), which may allow attackers to cause substantial damage before detection.\n\n")
            
            # Final recommendation
            f.write("5. Further research is recommended to explore different challenge mechanisms and improve detection rates in more complex scenarios, with particular focus on minimizing detection delays and training overhead.\n")
    
    logger.info(f"Generated research report: {report_path}")
    

def generate_all_visualizations(results_dir: str, output_dir: str = None):
    """Generate all visualizations for research data.
    
    Args:
        results_dir: Directory containing research results
        output_dir: Directory to save visualizations (defaults to results_dir/visualizations)
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, "visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    all_data = load_results(results_dir)
    
    if not all_data:
        logger.error(f"No result data found in {results_dir}")
        return
    
    # Generate original visualizations
    generate_challenge_effectiveness_plot(all_data, output_dir)
    generate_aggregator_performance_plot(all_data, output_dir)
    generate_challenge_timeline_plot(all_data, output_dir)
    
    # Generate new visualizations
    generate_aggregator_exclusion_plot(all_data, output_dir)
    generate_rollback_overhead_plot(all_data, output_dir)
    generate_attack_detection_plot(all_data, output_dir)
    generate_challenge_delay_correlation_plot(all_data, output_dir)
    
    # Generate comparative analysis
    generate_comparative_analysis(all_data, output_dir)
    
    logger.info(f"All visualizations generated in {output_dir}")


def main(results_dir: str, output_dir: str = None):
    """Main entry point for analysis (for test and script usage)."""
    generate_all_visualizations(results_dir, output_dir)

# Alias for test compatibility
generate_plots = generate_all_visualizations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze multi-aggregator simulation results")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing simulation results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save visualizations (defaults to results-dir/visualizations)",
    )
    
    args = parser.parse_args()
    
    main(args.results_dir, args.output_dir)