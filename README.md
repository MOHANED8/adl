# Deep Training Dynamics Framework

Deep Training Dynamics Framework (`dtd`) is a research-oriented training and monitoring stack for image classification experiments. It combines standard supervised training with internal model instrumentation, activation and gradient diagnostics, representation shift tracking, and a lightweight early-warning workflow for overfitting and collapse analysis.

The repository is organized around three usage modes:

1. Training vision models on benchmark datasets such as CIFAR-10, CIFAR-100, Tiny ImageNet, optional ImageNet layouts, and WILDS datasets.
2. Capturing internal signals through hooks for activations, gradients, attention maps, and intermediate embeddings.
3. Evaluating whether internal training signals can predict later degradation in validation behavior.

## Project Overview

Core components:

- `src/dtd/datasets`: dataset builders, transforms, label noise wrappers, deterministic caching, Tiny ImageNet support, WILDS wrapper.
- `src/dtd/models`: model factory, instrumented wrapper, CNN baseline, torchvision and timm-backed model entrypoints.
- `src/dtd/hooks`: unified activation, gradient, attention, and input capture.
- `src/dtd/analysis`: activation statistics, gradient diagnostics, representation similarity, numerical summary utilities.
- `src/dtd/training`: trainer loop, optimizer factory, structured JSONL logging.
- `src/dtd/experiments`: downstream evaluation of internal signals on saved run logs.
- `dashboard/app.py`: Streamlit dashboard for inspecting run artifacts.
- `configs/`: Hydra configuration tree for datasets, models, training, diagnostics, and tracking.

## Installation

### Environment

Python `>=3.10` is required by the package metadata. The current workspace has been tested with Python 3.13.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Optional packages

Some modules depend on optional integrations:

- `timm` for ViT and DeiT model creation.
- `wilds` for WILDS dataset access.
- `wandb` and `mlflow` for external experiment tracking.
- `streamlit` for the dashboard.

These are already listed in `requirements.txt`.

## Setup

### Data layout

Default root is `data/`.

- CIFAR-10: `data/cifar10`
- CIFAR-100: `data/cifar100`
- Tiny ImageNet: `data/tiny-imagenet-200`
- ImageNet: `data/imagenet/train` and `data/imagenet/val`
- WILDS: `data/wilds/...`

The framework can auto-download some datasets when the upstream library supports it. For licensed datasets such as ImageNet, the directory must already exist locally.

### Basic configuration

The default Hydra entrypoint is [configs/config.yaml](C:/Users/lenovo/Desktop/JupyterProject/configs/config.yaml). It composes:

- `dataset: cifar10`
- `model: resnet18`
- `training: base`
- `tracking: base`
- `diagnostics: base`

The default experiment output directory resolves to:

```text
runs/${exp.name}
```

## Usage

### Run the default experiment

```bash
python main.py
```

### Override configuration values

```bash
python main.py training.epochs=5 model=resnet18 dataset=cifar10
python main.py exp.name=cifar10_debug training.log_every_steps=10
```

### Verify hooks only

```bash
python scripts/demo_hooks.py
```

This prints the number of captured activation, gradient, and attention entries for the configured model.

### Evaluate a completed run

```bash
python scripts/evaluate_run.py runs/cifar10_resnet18
```

The evaluation script reads `events.jsonl`, builds a meta-sequence dataset from epoch-level diagnostics, and reports risk-quality statistics such as internal-signal AUC, false positive rate, lead-time proxy, and Granger-ranked features.

## Experiments

Hydra composition makes it straightforward to sweep:

```bash
python main.py -m model=resnet18,efficientnet_b0 training.epochs=2
python main.py -m dataset=cifar10,cifar100 training.optim.name=adamw,sgd
```

Experiment design recommendations:

- Keep `exp.name` unique per run to avoid artifact collisions.
- Use `cache_dir` for deterministic preprocessing reuse.
- Keep `num_workers=0` while debugging reproducibility.
- Change one factor at a time when studying causal signal quality.

See [experiments/sweeps.md](C:/Users/lenovo/Desktop/JupyterProject/experiments/sweeps.md) for multirun patterns.

## Dashboard Instructions

Launch the Streamlit dashboard with:

```bash
streamlit run dashboard/app.py
```

In the sidebar:

1. Enter the path to a run log such as `runs/cifar10_resnet18/events.jsonl`.
2. Select the record kind to inspect.
3. Review training curves, recent records, and diagnostics panels.

The dashboard currently supports:

- Training loss and validation accuracy line charts
- Activation statistics summary bars
- Gradient diagnostics plots
- Representation similarity traces

Reference visuals:

- Dashboard view: [docs/assets/dashboard_screenshot.svg](C:/Users/lenovo/Desktop/JupyterProject/docs/assets/dashboard_screenshot.svg)
- Workflow overview: [docs/assets/workflow_diagram.svg](C:/Users/lenovo/Desktop/JupyterProject/docs/assets/workflow_diagram.svg)

## Expected Outputs

Each run directory typically contains:

- `events.jsonl`: structured event log for steps, validation epochs, diagnostics, and warnings
- `tb/`: TensorBoard event files
- `checkpoints/`: saved model and optimizer states
- optional tracker artifacts from W&B or MLflow

Typical event kinds:

- `train_step`
- `val_epoch`
- `activation_stats`
- `gradient_stats`
- `repr_epoch`
- `early_warning_epoch`

Expected evaluation outputs:

- `auc_internal_signals`
- `false_positive_rate@0.5`
- `lead_time`
- `risk_valacc_corr`
- `granger_top`

Included example artifacts:

- [runs/example_synthetic_cnn](C:/Users/lenovo/Desktop/JupyterProject/runs/example_synthetic_cnn)
- [experiments/EXAMPLE_RESULTS.md](C:/Users/lenovo/Desktop/JupyterProject/experiments/EXAMPLE_RESULTS.md)
- [docs/REPRODUCIBILITY_GUIDE.md](C:/Users/lenovo/Desktop/JupyterProject/docs/REPRODUCIBILITY_GUIDE.md)
- [docs/DEVELOPMENT_STAGES.md](C:/Users/lenovo/Desktop/JupyterProject/docs/DEVELOPMENT_STAGES.md)

## Troubleshooting

| Problem | Likely Cause | Action |
| --- | --- | --- |
| `Unknown model` | `model.name` is not mapped in the factory | Use `cnn`, `resnet18`, `resnet34`, `resnet50`, `densenet121`, `efficientnet_b0`, `vit`, `deit_small`, `convnext_tiny`, or `swin_t` |
| `Unknown dataset` | Unsupported dataset key | Use `cifar10`, `cifar100`, `tiny_imagenet`, `imagenet`, or `wilds:<name>` |
| Empty dashboard | Wrong `events.jsonl` path or no completed run | Point the dashboard to a real run directory and rerun training if needed |
| Missing WILDS dependency | `wilds` is not installed | Install dependencies from `requirements.txt` |
| Tiny ImageNet validation error | Missing `val_annotations.txt` or `wnids.txt` | Re-extract the dataset and verify the canonical folder layout |
| Unstable reproducibility | Worker nondeterminism, random transforms, or changing seeds | Fix seeds, disable stochastic augmentations while debugging, keep worker count low |

## Documentation Map

- Technical report: [docs/TECHNICAL_REPORT.md](C:/Users/lenovo/Desktop/JupyterProject/docs/TECHNICAL_REPORT.md)
- User manual and catalog: [docs/USER_MANUAL.md](C:/Users/lenovo/Desktop/JupyterProject/docs/USER_MANUAL.md)
- Docs index: [docs/README.md](C:/Users/lenovo/Desktop/JupyterProject/docs/README.md)
- Tests and coverage: [tests/README.md](C:/Users/lenovo/Desktop/JupyterProject/tests/README.md)
