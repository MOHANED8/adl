# Development Stages

This document records the project in the requested staged format. Each stage includes objectives, implementation details, engineering decisions, testing procedures, and expected outputs.

## 1. Project Planning

- Objectives: define a research repository that supports supervised training, internal monitoring, causal-style evaluation, deployment, and documentation.
- Implementation details: selected a modular `src/dtd` package layout with separate dataset, model, hook, training, analysis, monitoring, experiment, and dashboard modules.
- Engineering decisions: kept the baseline stack in plain PyTorch rather than introducing a heavier framework first, because direct control over hooks and event logging mattered more than abstraction.
- Testing procedures: initial import tests and structural smoke tests.
- Expected outputs: package structure, config tree, requirements list, repository docs plan.

## 2. Environment Setup

- Objectives: create a reproducible local and containerized environment.
- Implementation details: maintained `requirements.txt`, added editable install support, added [setup.sh](C:/Users/lenovo/Desktop/JupyterProject/setup.sh), and added [Dockerfile](C:/Users/lenovo/Desktop/JupyterProject/Dockerfile).
- Engineering decisions: used the PyTorch CUDA runtime base image to reduce custom GPU image maintenance.
- Testing procedures: dependency installation checks and pytest execution in the local environment.
- Expected outputs: runnable Python environment, Docker build context, launch scripts.

## 3. Dataset Pipeline

- Objectives: support benchmark datasets plus a deterministic synthetic dataset for example runs and regression checks.
- Implementation details: implemented torchvision-backed datasets, Tiny ImageNet, WILDS wrapper, deterministic cache wrapper, label noise wrapper, and [SyntheticShapesDataset](C:/Users/lenovo/Desktop/JupyterProject/src/dtd/datasets/synthetic.py).
- Engineering decisions: synthetic data was added so example artifacts can be regenerated quickly without external downloads.
- Testing procedures: dataset factory tests, cache tests, Tiny ImageNet indexing tests, WILDS wrapper tests, synthetic dataset tests.
- Expected outputs: dataloaders, cache artifacts, synthetic example dataset path.

## 4. Model Implementation

- Objectives: support CNN, torchvision backbones, and timm transformer models behind one interface.
- Implementation details: implemented [build_model](C:/Users/lenovo/Desktop/JupyterProject/src/dtd/models/factory.py), `InstrumentedModel`, `SmallCNN`, and timm attention patching.
- Engineering decisions: wrapped models uniformly so hook behavior stays consistent across architectures.
- Testing procedures: model factory tests, CNN forward tests, timm attention patch tests.
- Expected outputs: trainable models with a common monitoring surface.

## 5. Hook System

- Objectives: capture activations, gradients, inputs, and attention-like tensors.
- Implementation details: implemented `HookSpec` and `HookManager` with glob filtering, bounded per-module storage, CPU detachment, and gradient capture.
- Engineering decisions: used forward hooks to avoid invasive model rewrites.
- Testing procedures: hook capture tests, storage bound tests, glob-filter tests.
- Expected outputs: activation, gradient, attention, and embedding stores attached to instrumented models.

## 6. Metrics Engine

- Objectives: compute internal scalar summaries suitable for logs, plots, and downstream evaluation.
- Implementation details: implemented moments, entropy, sparsity, dead-neuron ratio, covariance trace, spectral summaries, gradient norm, cosine similarity, sharpness proxy, and Hessian trace proxy.
- Engineering decisions: preferred compact summaries over full tensor dumps to keep logging practical.
- Testing procedures: unit tests for stats helpers and gradient diagnostics paths.
- Expected outputs: JSONL records and TensorBoard scalars.

## 7. Representation Analysis

- Objectives: study how internal representations drift over time and under perturbation.
- Implementation details: implemented CKA, SVCCA approximation, mean cosine drift, and a self-supervised consistency monitor using augmented views.
- Engineering decisions: used lightweight self-supervised probes instead of training a separate SSL model to keep runtime bounded.
- Testing procedures: representation monitor tests, SVCCA tests, self-supervised consistency tests.
- Expected outputs: `repr_epoch` and `ssl_consistency` events.

## 8. Meta-Learning Predictor

- Objectives: build a temporal dataset of run diagnostics for early-warning style evaluation.
- Implementation details: implemented `MetaSequenceDataset`, LSTM and transformer predictor baselines, and post-run evaluation utilities.
- Engineering decisions: kept the learned predictors modular and separate from the online trainer because they depend on cohorts of completed runs.
- Testing procedures: meta dataset window construction tests, predictor shape tests, evaluator tests.
- Expected outputs: temporal feature windows and predictor-ready tensors.

## 9. Causal Inference Module

- Objectives: provide a practical first-pass causal-signal ranking workflow.
- Implementation details: implemented Granger-style feature ranking and integrated it into run evaluation.
- Engineering decisions: treated Granger output as hypothesis generation rather than definitive causal identification.
- Testing procedures: evaluator tests and synthetic short-run validation.
- Expected outputs: ranked internal features in `granger_top`.

## 10. Visualization System

- Objectives: convert event logs into operator-facing plots and publication-friendly assets.
- Implementation details: kept the dashboard on JSONL and added [scripts/generate_example_visualizations.py](C:/Users/lenovo/Desktop/JupyterProject/scripts/generate_example_visualizations.py) for Matplotlib and Seaborn plots.
- Engineering decisions: used file-based artifacts rather than a service dependency to keep the repo portable.
- Testing procedures: dashboard load tests and generated plot smoke runs.
- Expected outputs: TensorBoard events, PNG plots, SVG documentation diagrams.

## 11. Dashboard Development

- Objectives: make training curves and diagnostics inspectable without manual parsing.
- Implementation details: implemented [dashboard/app.py](C:/Users/lenovo/Desktop/JupyterProject/dashboard/app.py) with views for training, activation, gradient, representation, uncertainty, collapse, specialization, and warning records.
- Engineering decisions: kept the UI import-safe and single-file for easy deployment with Streamlit.
- Testing procedures: dashboard import tests and event loading tests.
- Expected outputs: Streamlit dashboard application.

## 12. Experiment Execution

- Objectives: produce concrete deliverables such as checkpoints, logs, plots, and result summaries.
- Implementation details: added [scripts/run_example_experiment.py](C:/Users/lenovo/Desktop/JupyterProject/scripts/run_example_experiment.py), checkpoint saving in the trainer, and synthetic example results.
- Engineering decisions: example execution uses a deterministic synthetic dataset so artifact generation is reliable in constrained environments.
- Testing procedures: trainer integration tests and manual example-run verification.
- Expected outputs: [runs/example_synthetic_cnn](C:/Users/lenovo/Desktop/JupyterProject/runs/example_synthetic_cnn) with checkpoints, logs, and result JSON.

## 13. Testing and Debugging

- Objectives: enforce a launch gate with broad functional coverage.
- Implementation details: expanded pytest coverage across datasets, hooks, training, runtime, monitoring, dashboard, stress, and reproducibility paths.
- Engineering decisions: kept tests synthetic and bounded so they run quickly and deterministically.
- Testing procedures: full `pytest -q` with coverage enforcement.
- Expected outputs: passing suite and enforced coverage threshold.

## 14. Documentation

- Objectives: deliver operator docs, technical report, user manual, reproducibility guide, and staged development record.
- Implementation details: updated `README.md`, technical report, user manual, reproducibility guide, and this staged document.
- Engineering decisions: documentation is tied to current code paths and example artifacts rather than speculative future features.
- Testing procedures: manual link and artifact verification.
- Expected outputs: professional repository documentation set.

## 15. Final Optimization

- Objectives: add higher-value advanced features without destabilizing the core system.
- Implementation details: added uncertainty estimation, Bayesian calibration summaries, adaptive thresholding, spectral collapse detection, neuron specialization tracking, self-supervised consistency analysis, and online warning records.
- Engineering decisions: prioritized features that reuse the existing event pipeline so they remain observable and testable.
- Testing procedures: advanced feature unit and integration tests.
- Expected outputs: richer online diagnostics and warning signals.

## 16. Deployment Preparation

- Objectives: make the project runnable locally, across multiple GPUs, under DDP, and in containers.
- Implementation details: added runtime config, DDP and DataParallel support, AMP support, Dockerfile, setup script, and launch scripts.
- Engineering decisions: extended the existing trainer instead of maintaining separate deployment-specific code paths.
- Testing procedures: runtime helper tests, trainer regression tests, full suite validation.
- Expected outputs: local, multi-GPU, distributed, and container deployment entrypoints.
