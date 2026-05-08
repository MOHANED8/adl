# User Manual and Catalog

## System Usage

This manual explains how to install, configure, train, monitor, and interpret outputs from the Deep Training Dynamics Framework.

### Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Configuration

The main configuration entrypoint is [configs/config.yaml](C:/Users/lenovo/Desktop/JupyterProject/configs/config.yaml). Runtime behavior is composed from dataset, model, training, diagnostics, and tracking config groups.

Configuration checklist:

- set `exp.name`
- set dataset name and root
- set model name and class count
- set training epochs and optimizer
- set monitor frequency
- set cache directory if deterministic preprocessing is desired

### Training

Run a standard experiment:

```bash
python main.py
```

Run a custom experiment:

```bash
python main.py exp.name=custom_run training.epochs=10 dataset=cifar100 model=efficientnet_b0
```

### Monitoring

During or after training, inspect:

- `runs/<exp>/events.jsonl`
- `runs/<exp>/tb`
- optional W&B or MLflow outputs

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

### Interpreting outputs

- `train_step` records show optimization progress.
- `val_epoch` records show validation loss and accuracy.
- `activation_stats` records summarize layer dynamics.
- `gradient_stats` records capture optimization stability.
- `repr_epoch` records measure representation drift.
- `early_warning_epoch` records provide coarse recommendations.

## Input Requirements

### Dataset format

Supported dataset keys:

- `cifar10`
- `cifar100`
- `tiny_imagenet`
- `imagenet`
- `wilds:<dataset_name>`

Expected layouts:

| Dataset | Required Layout |
| --- | --- |
| CIFAR-10/100 | standard torchvision root under `data/` |
| Tiny ImageNet | `tiny-imagenet-200/train`, `val/images`, `val/val_annotations.txt`, `wnids.txt` |
| ImageNet | `imagenet/train/<class>` and `imagenet/val/<class>` |
| WILDS | managed by the `wilds` package under `data/wilds` |

### Configuration requirements

Minimum fields to review before training:

- dataset root
- dataset split names
- `num_classes`
- `image_size`
- `training.epochs`
- optimizer name and learning rate
- hook module globs
- output directory

### Hardware requirements

Recommended baselines:

| Scenario | CPU | RAM | GPU |
| --- | --- | --- | --- |
| quick functional test | 4 cores | 8 GB | optional |
| CIFAR-scale monitored run | 8 cores | 16 GB | 8 GB VRAM |
| larger torchvision model sweeps | 8 to 16 cores | 32 GB | 12 to 24 GB VRAM |
| WILDS or larger image experiments | 16 cores | 32+ GB | 24+ GB VRAM |

## Output Explanation

### Metrics

Common metrics:

- `loss`
- `val_loss`
- `val_acc`
- `grad_norm`
- `grad_var`
- `grad_cos_prev`
- `sharpness_proxy`
- `hessian_trace_proxy`
- `cka`
- `svcca`
- `mean_cosine_drift`

### Warnings

`early_warning_epoch` is currently a scaffolding signal. Treat it as advisory until a trained warning model is integrated.

### Graphs

Available graph surfaces:

- TensorBoard scalar curves
- Streamlit line charts
- activation stat bar chart for the latest record

### Dashboards

Dashboard reference:

![Dashboard Screenshot](C:/Users/lenovo/Desktop/JupyterProject/docs/assets/dashboard_screenshot.svg)

Use the sidebar to select the JSONL path and record kind. Review training curves first, then drill into a single diagnostic family.

### Prediction scores

The post-run evaluator reports:

- internal-signal AUC
- thresholded false positive rate
- lead-time proxy
- risk-to-validation correlation
- top Granger-ranked features

### Saved checkpoints

The current trainer logs metrics and TensorBoard summaries but does not implement a checkpoint manager by default. If launch readiness depends on checkpoint recovery, add explicit checkpoint saving before production use.

## Success Guidelines

### Stable training

- keep learning rates conservative when first enabling monitoring
- validate the plain training path before turning on every diagnostic
- use deterministic cache directories for expensive preprocessing
- keep `monitor_every_steps` sparse enough to avoid signal flooding

### Avoid noisy diagnostics

- do not mix stochastic augmentations with deterministic cache directories unless frozen behavior is intended
- prefer `shuffle=False` during reproducibility debugging
- use bounded hook history via `max_per_module`
- start with major blocks only in `module_globs`

### Tune thresholds

Practical threshold guidance:

| Signal | Starting Rule | Tuning Direction |
| --- | --- | --- |
| exploding gradients | `grad_norm > 10x EMA` | raise threshold if false alarms are frequent |
| activation drift | inspect sustained multi-step elevation | tighten only after a stable baseline is known |
| representation drift | compare against baseline CKA trajectory | lower concern if validation stays stable |
| overfit risk threshold | begin at `0.5` in evaluator | recalibrate on historical runs |

### Validate causal signals

- compare signals across repeated seeds
- test the same diagnostic on clean and label-noise settings
- verify that ranked features remain relevant under modest hyperparameter changes
- avoid interpreting a single p-value ranking as proof of mechanism

### Best hyperparameter settings

Recommended starting point for CIFAR-scale runs:

| Setting | Recommendation |
| --- | --- |
| optimizer | `adamw` |
| learning rate | `3e-4` |
| weight decay | `1e-4` |
| batch size | `64` to `128` |
| monitor frequency | every `20` to `50` steps |
| epochs for debug | `2` to `5` |
| epochs for study | dataset-dependent, after baseline validation |

### Recommended GPU settings

- start on a single GPU
- keep batch sizes below memory saturation
- prefer stable CUDA environments over aggressive throughput tuning
- validate CPU execution paths for diagnostics before scaling

### Memory optimization techniques

- narrow hook `module_globs`
- set `max_per_module=1`
- capture on CPU when long retention is needed
- reduce `image_size` during exploratory runs
- reduce `sample_batches` in representation monitoring

## Failure Cases

### Unstable training

Symptoms:

- loss oscillation
- poor validation accuracy
- repeated drift spikes

Actions:

- reduce learning rate
- switch to `adamw`
- lower augmentation strength
- validate dataset labels and class count

### Exploding gradients

Symptoms:

- abrupt `grad_norm` jumps
- NaNs or divergence

Actions:

- lower learning rate
- inspect normalization layers
- verify batch size and optimizer configuration

### False overfitting predictions

Symptoms:

- evaluator flags risk but validation remains stable

Actions:

- recalibrate thresholds on historical runs
- increase horizon only with enough epochs
- compare signal stability across seeds

### Poor calibration

Symptoms:

- risk scores cluster around one regime
- threshold behavior is brittle

Actions:

- collect more run diversity
- normalize features consistently across runs
- separate debug runs from production-quality runs

### Dataset imbalance

Symptoms:

- misleading validation trajectories
- unstable class-wise performance hidden by aggregate accuracy

Actions:

- inspect class distribution
- add per-class metrics externally
- tune sampling or loss weighting outside the current baseline

### Domain shift failures

Symptoms:

- diagnostics appear normal but validation on shifted data collapses

Actions:

- compare in-domain and shifted validation splits
- use WILDS datasets where appropriate
- avoid over-interpreting in-domain internal-signal success

## Workflow Diagrams

### Workflow diagram

![Workflow Diagram](C:/Users/lenovo/Desktop/JupyterProject/docs/assets/workflow_diagram.svg)

### Pipeline diagram

![Pipeline Diagram](C:/Users/lenovo/Desktop/JupyterProject/docs/assets/pipeline_diagram.svg)

## Troubleshooting Tables

| Issue | Diagnosis | Resolution |
| --- | --- | --- |
| dashboard shows no data | wrong JSONL path or empty run directory | point to `runs/<exp>/events.jsonl` from a completed run |
| no activation stats | hook patterns do not match modules | widen `module_globs` and validate with `scripts/demo_hooks.py` |
| no representation stats | selected module name does not exist | inspect captured keys and update `module_name` |
| cache is inconsistent | stochastic transform frozen into cache | separate deterministic and stochastic preprocessing paths |
| evaluator returns `not_enough_epochs_for_window_horizon` | run is too short | increase training epochs or reduce window and horizon |
| WILDS split error | requested split missing from dataset | use an available split from `split_dict` |

## Catalog Summary

This system is best used as a controlled experimentation platform rather than a turnkey production training service. Its strengths are:

- internal visibility during training
- structured run artifacts
- reproducible pytest validation
- support for downstream causal-signal experiments

Its operational constraints are:

- monitoring overhead if hook scope is too broad
- incomplete live warning policy
- no default checkpoint management
- causal conclusions require disciplined experimental design
