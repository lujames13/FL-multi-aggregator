# Tech Context

## Stack & Dependencies
- Python 3.8+
- Flower federated learning framework (>=1.17.0)
- PyTorch for model training
- flwr_datasets for federated data partitioning
- NumPy for numerical operations
- Visualization: Matplotlib/Seaborn

## Environment
- Local simulation, extensible to distributed/cloud
- Designed for reproducible research experiments
- Supports configurable experiment scales (clients, aggregators)

## Key Files/Components
- MultiAggregatorStrategy (custom strategy)
- Server/client apps for simulation
- Challenge mechanism and metrics logging
- Analysis and visualization scripts 

## Testing
- All tests are written using pytest (version 8.3.5)
- Run tests with `pytest` from the project root 