# Reproducibility Guide

## Scope

This guide describes how to reproduce the example artifacts and how to keep experimental behavior stable across machines.

## Environment

Recommended setup:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

For Linux or Docker environments:

```bash
bash setup.sh
```

## Deterministic Controls

The repository uses several deterministic controls:

- explicit runtime seed in `training.runtime.seed`
- bounded dataloader worker count in debug and reproducibility modes
- deterministic synthetic dataset generation
- deterministic pytest seeding in [tests/conftest.py](C:/Users/lenovo/Desktop/JupyterProject/tests/conftest.py)
- stable cache directories for deterministic preprocessing paths

Recommended reproducibility settings for local runs:

```bash
python main.py \
  training.runtime.seed=42 \
  training.runtime.device=cpu \
  dataset.synthetic_shapes \
  model=cnn
```

## Reproducing the Example Run

Generate the example experiment:

```bash
python scripts/run_example_experiment.py
python scripts/generate_example_visualizations.py
```

Expected artifacts:

- [runs/example_synthetic_cnn/events.jsonl](C:/Users/lenovo/Desktop/JupyterProject/runs/example_synthetic_cnn/events.jsonl)
- [runs/example_synthetic_cnn/example_results.json](C:/Users/lenovo/Desktop/JupyterProject/runs/example_synthetic_cnn/example_results.json)
- [runs/example_synthetic_cnn/checkpoints/best.pt](C:/Users/lenovo/Desktop/JupyterProject/runs/example_synthetic_cnn/checkpoints/best.pt)
- [visualization/example_results/training_curves.png](C:/Users/lenovo/Desktop/JupyterProject/visualization/example_results/training_curves.png)
- [visualization/example_results/uncertainty_curves.png](C:/Users/lenovo/Desktop/JupyterProject/visualization/example_results/uncertainty_curves.png)

## Reproducing the Test Suite

Run:

```bash
pytest -q
```

Current validated result:

- `44 passed`
- coverage gate satisfied above `70%`

## Multi-GPU and Distributed Reproducibility

For multi-GPU and DDP runs:

- keep the same `training.runtime.seed`
- keep the same world size when comparing runs
- avoid changing sampler settings between runs
- keep monitor cadence identical
- compare only runs with matching augmentation regimes

Launch commands:

```bash
bash scripts/launch_local.sh
bash scripts/launch_multigpu.sh
bash scripts/launch_distributed.sh
```

## Practical Limits

Perfect bitwise reproducibility across devices, CUDA versions, and backend libraries is not guaranteed. In practice:

- CPU example runs are the best path for exact local regression checks.
- GPU runs should be treated as statistically reproducible rather than bitwise identical.
- calibration and Granger-style outputs are sensitive to small-data regimes and should be interpreted accordingly.
