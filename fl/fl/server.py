# fl/fl/server.py

"""Multi-aggregator federated learning server with results saving functionality."""

import json
import os
import pickle
from datetime import datetime
from logging import INFO
from pathlib import Path
from secrets import token_hex
from typing import Dict, Optional, Union

from flwr.common import log
from flwr.server import Server
from flwr.server.history import History

# Get project directory
PROJECT_DIR = Path(os.path.abspath(__file__)).parent.parent

class MultiAggregatorResultsSaverServer(Server):
    """Server to save history and research data to disk."""
    
    def __init__(
        self,
        *,
        client_manager,
        strategy=None,
        results_saver_fn=None,
        run_config=None,
    ):
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.results_saver_fn = results_saver_fn
        self.run_config = run_config
    
    def fit(self, num_rounds, timeout):
        """Run federated learning for a number of rounds and save results."""
        history, elapsed = super().fit(num_rounds, timeout)
        
        if self.results_saver_fn:
            log(INFO, "Results saver function provided. Executing")
            self.results_saver_fn(history, self.strategy, self.run_config, elapsed_time=elapsed)
        
        return history, elapsed


def save_research_data(
    history: History,
    strategy,
    run_config: Dict,
    file_path: Optional[Union[str, Path]] = None,
    elapsed_time: float = None,
):
    """Save research data to JSON file, including all relevant metrics and curves."""
    research_data = {}

    # 1. Collect metrics history (per-round)
    if hasattr(strategy, 'metrics_history'):
        detailed_history = getattr(strategy, "metrics_history", [])
        research_data["detailed_history"] = detailed_history
        
        # Add aggregated network delay metrics
        if detailed_history:
            total_network_delays = [m.get("total_network_delay", 0) for m in detailed_history 
                                   if "total_network_delay" in m]
            if total_network_delays:
                research_data["avg_network_delay"] = float(sum(total_network_delays) / len(total_network_delays))
                research_data["total_network_delay"] = float(sum(total_network_delays))
            
            # Add computation time metrics
            computation_times = [m.get("computation_time", 0) for m in detailed_history 
                                if "computation_time" in m]
            if computation_times:
                research_data["avg_computation_time"] = float(sum(computation_times) / len(computation_times))
                research_data["total_computation_time"] = float(sum(computation_times))

    # 2. Collect accuracy/loss per round if available in metrics_history
    accuracy_curve = []
    loss_curve = []
    if "detailed_history" in research_data:
        for m in research_data["detailed_history"]:
            if "accuracy" in m:
                accuracy_curve.append({"round": m.get("round"), "accuracy": m["accuracy"]})
            if "loss" in m:
                loss_curve.append({"round": m.get("round"), "loss": m["loss"]})
    if accuracy_curve:
        research_data["accuracy_round_curve"] = accuracy_curve
    if loss_curve:
        research_data["loss_round_curve"] = loss_curve

    # 3. Collect rollback/attack/exclusion info if available
    if hasattr(strategy, 'total_rollbacks'):
        research_data["total_rollbacks"] = getattr(strategy, "total_rollbacks", 0)
    if hasattr(strategy, 'detected_attacks'):
        research_data["detected_attacks"] = list(getattr(strategy, "detected_attacks", set()))
    if hasattr(strategy, 'excluded_aggregators'):
        research_data["excluded_aggregators"] = list(getattr(strategy, "excluded_aggregators", set()))

    # 4. Collect processing time statistics
    fit_times = []
    eval_times = []
    if "detailed_history" in research_data:
        for m in research_data["detailed_history"]:
            if "processing_time_fit" in m:
                fit_times.append(m["processing_time_fit"])
            if "processing_time_evaluate" in m:
                eval_times.append(m["processing_time_evaluate"])
    if fit_times:
        research_data["avg_processing_time_fit"] = float(sum(fit_times) / len(fit_times))
    if eval_times:
        research_data["avg_processing_time_evaluate"] = float(sum(eval_times) / len(eval_times))

    # 5. Collect other strategy attributes as before
    if hasattr(strategy, 'challenged_rounds'):
        research_data["challenged_rounds"] = list(getattr(strategy, "challenged_rounds", set()))
    if hasattr(strategy, 'original_rounds_map'):
        research_data["original_rounds_map"] = {str(k): v for k, v in getattr(strategy, "original_rounds_map", {}).items()}
    if hasattr(strategy, 'num_aggregators'):
        research_data["num_aggregators"] = getattr(strategy, "num_aggregators", None)
    if hasattr(strategy, 'malicious_aggregator_ids'):
        research_data["malicious_aggregator_ids"] = list(getattr(strategy, "malicious_aggregator_ids", []))
    if hasattr(strategy, 'total_rounds_with_attack'):
        research_data["total_rounds_with_attack"] = getattr(strategy, "total_rounds_with_attack", 0)
    if hasattr(strategy, 'detection_delay'):
        research_data["detection_delay"] = getattr(strategy, "detection_delay", 0)
    if hasattr(strategy, 'round'):
        research_data["total_rounds"] = getattr(strategy, "round", 0)
    
    # 6. Add PBFT-specific metrics if available
    if hasattr(strategy, 'challenge_frequency'):
        research_data["challenge_frequency"] = getattr(strategy, "challenge_frequency", 0)
    if hasattr(strategy, 'challenge_mode'):
        research_data["challenge_mode"] = getattr(strategy, "challenge_mode", "")
    if hasattr(strategy, 'network_delay_factor'):
        research_data["network_delay_factor"] = getattr(strategy, "network_delay_factor", 0)

    # Derived metrics (as before)
    num_aggregators = research_data.get("num_aggregators", 0)
    excluded_aggregators = research_data.get("excluded_aggregators", [])
    total_rollbacks = research_data.get("total_rollbacks", 0)
    detection_delay = research_data.get("detection_delay", 0)
    total_rounds = research_data.get("total_rounds", 0)
    research_data["effective_aggregators"] = num_aggregators - len(excluded_aggregators)
    research_data["aggregator_exclusion_percentage"] = (len(excluded_aggregators) / num_aggregators * 100) if num_aggregators > 0 else 0
    research_data["effective_training_rounds"] = total_rounds - total_rollbacks * detection_delay
    research_data["rollback_overhead_percentage"] = (total_rollbacks * detection_delay / total_rounds * 100) if total_rounds > 0 else 0

    # Add history metrics (from flwr History)
    if history and hasattr(history, 'metrics_distributed') and history.metrics_distributed:
        research_data["history"] = {
            "loss": history.metrics_distributed.get("loss", {}),
            "metrics_fit": history.metrics_distributed.get("metrics_fit", {}),
            "metrics_evaluate": history.metrics_distributed.get("metrics_evaluate", {}),
        }

    # Add run configuration
    research_data["run_config"] = run_config

    # Ensure scenario parameter is included
    if "scenario" in run_config:
        research_data["scenario"] = run_config["scenario"]
    
    # Add total elapsed time if provided
    if elapsed_time is not None:
        research_data["total_elapsed_time_sec"] = elapsed_time

    # Determine output path (as before)
    if file_path is None:
        if "output-path" in run_config:
            file_path = run_config["output-path"]
        else:
            output_dir = run_config.get("output-dir", "results")
            num_aggregators = run_config.get("num-aggregators", 3)
            malicious_str = run_config.get("malicious-aggregators", "")
            enable_challenges = run_config.get("enable-challenges", True)
            scenario_name = f"aggs{num_aggregators}_mal{malicious_str.replace(',', '_')}_chal{'on' if enable_challenges else 'off'}"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(output_dir, f"research_data_{scenario_name}_{timestamp}.json")

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(research_data, f, indent=2)

    log(INFO, f"Research data saved to {file_path}")
    return file_path


def save_history_as_pickle(
    history: History,
    file_path: Union[str, Path],
    extra_data: Optional[Dict] = None,
):
    """Save history and extra data to pickle file.
    
    Parameters
    ----------
    history: History
        History returned by server.fit()
    file_path: Union[str, Path]
        Path to file to create and store history
    extra_data: Optional[Dict]
        Additional data to include in the pickle file
    """
    path = Path(file_path)
    
    # Ensure directory exists
    path.parent.mkdir(exist_ok=True, parents=True)
    
    # Add suffix if file exists to avoid overwriting
    if path.exists():
        suffix = token_hex(4)
        log(INFO, f"File {path} exists. Adding suffix: {suffix}")
        path = path.parent / (path.stem + "_" + suffix + path.suffix)
    
    # Prepare data
    data = {"history": history}
    if extra_data:
        data.update(extra_data)
    
    # Save to pickle
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    log(INFO, f"History saved to {path}")
    return path


def save_results_and_research_data(history, strategy, run_config, elapsed_time=None):
    """Save both history and research data."""
    # Save research data as JSON
    json_path = save_research_data(history, strategy, run_config, elapsed_time=elapsed_time)
    
    # Optionally save full history as pickle (useful for detailed analysis)
    output_dir = run_config.get("output-dir", "results")
    pickle_path = os.path.join(output_dir, "history_pickles")
    
    # Extract scenario name from json_path
    json_filename = os.path.basename(json_path)
    scenario_name = json_filename.replace("research_data_", "").replace(".json", "")
    
    # Save history pickle
    pickle_file = os.path.join(pickle_path, f"history_{scenario_name}.pkl")
    save_history_as_pickle(history, pickle_file)
    
    # Save run configuration separately for reference
    config_dir = os.path.join(output_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, f"config_{scenario_name}.json")
    
    with open(config_file, "w") as f:
        json.dump(run_config, f, indent=2)