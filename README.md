# Multi-Aggregator Federated Learning Simulation

This project provides a framework for simulating multi-aggregator federated learning with a challenge mechanism. It's designed to help researchers study the effectiveness of challenge-based validation in federated learning environments where some aggregators may be malicious.

## Overview

The simulation extends the Flower federated learning framework to support multiple virtual aggregators in a single simulation. Key features include:

- **Multiple Virtual Aggregators**: Simulate multiple aggregators using a round-robin approach
- **Malicious Behavior Modeling**: Configure specific aggregators to produce malicious results, with support for multiple strategies (noise, scaling, zeroing, etc.)
- **Challenge Mechanism**: Simulate challenge-based validation to detect malicious aggregation, with advanced validation logic (configurable metric and threshold)
- **Research Metrics**: Collect and analyze detailed challenge and verification metrics, including challenge status tracking and parameter difference logging
- **Visualization Tools**: Generate visualizations for paper-ready results

## Installation

1. Clone the repository:
```bash
git clone https://github.com/lujames13/FL-multi-aggregator.git
cd FL-multi-aggregator
```

2. Install the required dependencies:
```bash
pip install -e .
```

## Usage

### Running a Single Simulation

```bash
python run_multi_aggregator_simulation.py --scenario single --clients 10 --rounds 5 --aggregators 3 --malicious 1 --challenges enable
```

Parameters:
- `--scenario`: Choose `single` for a single simulation or `all` for predefined research scenarios
- `--clients`: Number of federated learning clients
- `--rounds`: Number of federated learning rounds
- `--aggregators`: Number of virtual aggregators
- `--malicious`: Comma-separated list of malicious aggregator IDs (0-indexed)
- `--challenges`: Enable or disable challenge mechanism (`enable` or `disable`)
- `--output-dir`: Directory to save results (default: `results`)
- `--visualize`: Add this flag to generate visualizations after simulation

### Running Predefined Research Scenarios

```bash
python run_multi_aggregator_simulation.py --scenario all --clients 10 --rounds 5 --visualize
```

This will run multiple predefined scenarios:
1. All honest aggregators
2. One malicious aggregator without challenges
3. One malicious aggregator with challenges
4. Multiple malicious aggregators with challenges

### Analyzing Results

After running simulations, analyze the results:

```bash
python analyze_results.py --results-dir results
```

This will generate:
- Challenge effectiveness plots
- Aggregator performance visualizations
- Challenge timeline plots
- Comparative analysis of different scenarios
- Research report with key findings

## Project Structure

- `fl/task.py`: Core federated learning components (model, data loading, training)
- `fl/client_app.py`: Flower client implementation
- `fl/server_app.py`: Original Flower server implementation
- `fl/multi_aggregator_server_app.py`: Extended server with multi-aggregator support
- `multi_aggregator_strategy.py`: Custom strategy implementing the multi-aggregator simulation
- `run_multi_aggregator_simulation.py`: Script to run simulations
- `analyze_results.py`: Script to analyze simulation results

## Extending the Project

### Adding New Malicious Behavior

Modify the `_create_malicious_aggregation` method in `multi_aggregator_strategy.py` to implement different types of malicious behavior:

```python
def _create_malicious_aggregation(self, honest_ndarrays: List[np.ndarray]) -> List[np.ndarray]:
    strategy = self.malicious_strategies.get(self.current_aggregator_id, "noise")
    if strategy == "noise":
        ...
    elif strategy == "scaling":
        ...
    elif strategy == "zeroing":
        ...
    # Add your custom strategy here
```

### Implementing New Challenge Mechanisms

Extend the `_validate_challenge` method in `multi_aggregator_strategy.py` to implement more sophisticated challenge validation:

```python
def _validate_challenge(self, server_round, challenged_parameters, honest_parameters):
    if self.challenge_metric == "mse":
        ...
    elif self.challenge_metric == "mae":
        ...
    # Add your custom metric or logic here
```

## Research Applications

This project is designed for research on:

1. **Security in Federated Learning**: Study how challenge mechanisms can protect against malicious aggregators
2. **Byzantine-Robust Aggregation**: Compare challenge-based approaches with other Byzantine-robust aggregation methods
3. **Decentralized Validation**: Explore decentralized approaches to validation in federated learning
4. **Blockchain Integration**: Develop and test concepts for blockchain-based federated learning systems

## Citation

If you use this project in your research, please cite:

```
@misc{multi-aggregator-fl,
  author = {Your Name},
  title = {Multi-Aggregator Federated Learning Simulation},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/multi-aggregator-fl}}
}
```

### Advanced Configuration

You can further customize the simulation using advanced options:

- `--malicious-strategies`: JSON string mapping aggregator IDs to strategies (e.g., '{"0": "noise", "2": "scaling"}')
- `--challenge-threshold`: Set the threshold for challenge validation (default: 1e-6)
- `--challenge-metric`: Set the metric for challenge validation (`mse` or `mae`, default: `mse`)

Example:
```bash
python run_multi_aggregator_simulation.py --clients 10 --rounds 5 --aggregators 3 --malicious 0,2 --malicious-strategies '{"0": "noise", "2": "scaling"}' --challenge-threshold 0.01 --challenge-metric mae
```

## Interpreting Challenge Metrics

After running simulations, the results will include detailed challenge and verification metrics:
- `challenge_status`: Status of each challenge (`pending`, `successful`, `rejected`)
- `challenge_distance`: The computed distance between challenged and honest parameters
- `challenge_threshold`: The threshold used for validation
- `malicious_strategy`: The strategy used by the malicious aggregator
- `param_diff`: The parameter difference for challenged rounds

These metrics are logged in the research data and can be used for in-depth analysis and visualization.