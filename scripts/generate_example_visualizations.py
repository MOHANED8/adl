from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_events(path: Path) -> pd.DataFrame:
    """Load JSONL experiment events into a dataframe."""

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def main() -> None:
    """Render example training and diagnostics plots for the example run."""

    run_dir = ROOT / "runs" / "example_synthetic_cnn"
    out_dir = ROOT / "visualization" / "example_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_events(run_dir / "events.jsonl")
    sns.set_theme(style="whitegrid")

    train = df[df["kind"] == "train_step"].copy()
    val = df[df["kind"] == "val_epoch"].copy()
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    if not train.empty:
        sns.lineplot(data=train, x="global_step", y="loss", ax=ax1, label="train loss")
    if not val.empty:
        sns.lineplot(data=val, x="global_step", y="val_acc", ax=ax1, label="val acc")
    ax1.set_title("Example Synthetic Training Curves")
    ax1.set_xlabel("Global step")
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    uncertainty = df[df["kind"] == "uncertainty_stats"].copy()
    if not uncertainty.empty:
        fig, ax2 = plt.subplots(figsize=(8, 4.5))
        cols = [c for c in ["predictive_entropy", "ece", "ece_temperature_scaled"] if c in uncertainty.columns]
        melted = uncertainty.melt(id_vars=["global_step"], value_vars=cols, var_name="metric", value_name="value")
        sns.lineplot(data=melted, x="global_step", y="value", hue="metric", ax=ax2)
        ax2.set_title("Example Uncertainty and Calibration")
        fig.tight_layout()
        fig.savefig(out_dir / "uncertainty_curves.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
