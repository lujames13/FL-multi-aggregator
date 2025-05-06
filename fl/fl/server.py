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
            self.results_saver_fn(history, self.strategy, self.run_config)
        
        return history, elapsed


def save_research_data(
    history: History,
    strategy,
    run_config: Dict,
    file_path: Optional[Union[str, Path]] = None,
):
    """Save research data to JSON file.
    
    Parameters
    ----------
    history: History
        History returned by server.fit()
    strategy: Strategy
        The strategy instance (should be MultiAggregatorStrategy)
    run_config: Dict
        Configuration parameters for the run
    file_path: Optional[Union[str, Path]]
        Optional explicit file path. If not provided, it will be generated
        from run_config or default location.
    """
    # Get research data from strategy if it's a MultiAggregatorStrategy
    research_data = {}
    if hasattr(strategy, 'get_research_data'):
        research_data = strategy.get_research_data()
        log(INFO, f"Retrieved research data from strategy: {len(research_data)} metrics")
    else:
        log(INFO, "Strategy does not provide research data")
    
    # Determine output path
    if file_path is None:
        # Use output-path from run_config if available
        if "output-path" in run_config:
            file_path = run_config["output-path"]
        else:
            # Generate default path based on scenario parameters
            output_dir = run_config.get("output-dir", "results")
            num_aggregators = run_config.get("num-aggregators", 3)
            malicious_str = run_config.get("malicious-aggregators", "")
            enable_challenges = run_config.get("enable-challenges", True)
            
            scenario_name = f"aggs{num_aggregators}_mal{malicious_str.replace(',', '_')}_chal{'on' if enable_challenges else 'off'}"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            file_path = os.path.join(output_dir, f"research_data_{scenario_name}_{timestamp}.json")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Add history metrics to research data
    if history and history.metrics_distributed:
        research_data["history"] = {
            "loss": history.metrics_distributed.get("loss", {}),
            "metrics_fit": history.metrics_distributed.get("metrics_fit", {}),
            "metrics_evaluate": history.metrics_distributed.get("metrics_evaluate", {}),
        }
    
    # Add run configuration
    research_data["run_config"] = run_config
    
    # Save to file
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


def save_results_and_research_data(history, strategy, run_config):
    """Save both history and research data."""
    # Save research data as JSON
    json_path = save_research_data(history, strategy, run_config)
    
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