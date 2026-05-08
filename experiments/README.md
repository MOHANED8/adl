# Experiments

Run default:

```bash
python main.py
```

Run a faster smoke experiment:

```bash
python main.py model=cnn training.monitor_every_steps=50 training.epochs=1
```

## Sweeps

See `experiments/sweeps.md`.

## Evaluate internal-signal predictive power

After a run finishes (needs enough epochs to form windows), compute metrics:

```bash
python scripts/evaluate_run.py runs/<exp_name>
```

