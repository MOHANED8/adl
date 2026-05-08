# Example Results

This file summarizes the reproducible example run generated in this repository.

## Run

- Experiment script: [scripts/run_example_experiment.py](C:/Users/lenovo/Desktop/JupyterProject/scripts/run_example_experiment.py)
- Output directory: [runs/example_synthetic_cnn](C:/Users/lenovo/Desktop/JupyterProject/runs/example_synthetic_cnn)
- Visualization script: [scripts/generate_example_visualizations.py](C:/Users/lenovo/Desktop/JupyterProject/scripts/generate_example_visualizations.py)

## Produced Artifacts

- Event log: [events.jsonl](C:/Users/lenovo/Desktop/JupyterProject/runs/example_synthetic_cnn/events.jsonl)
- Result summary: [example_results.json](C:/Users/lenovo/Desktop/JupyterProject/runs/example_synthetic_cnn/example_results.json)
- Best checkpoint: [best.pt](C:/Users/lenovo/Desktop/JupyterProject/runs/example_synthetic_cnn/checkpoints/best.pt)
- Last checkpoint: [last.pt](C:/Users/lenovo/Desktop/JupyterProject/runs/example_synthetic_cnn/checkpoints/last.pt)
- Training curves: [training_curves.png](C:/Users/lenovo/Desktop/JupyterProject/visualization/example_results/training_curves.png)
- Uncertainty curves: [uncertainty_curves.png](C:/Users/lenovo/Desktop/JupyterProject/visualization/example_results/uncertainty_curves.png)

## Interpretation

This example is an engineering smoke run, not a scientific benchmark. It is intended to prove that:

- training executes end to end
- advanced monitors log events
- checkpoints are written
- evaluation runs on the resulting artifacts
- plots can be rendered from the event log

The current synthetic run is stable and small. Because it is intentionally easy and short:

- overfitting targets are sparse
- causal rankings are low-information
- calibration curves are best interpreted as pipeline validation rather than model-quality evidence

For substantive research conclusions, run longer experiments on CIFAR or WILDS datasets with controlled ablations.
