import json
import os
import numpy as np

class TrajectoryLogger:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.history = []

    def log_epoch(self, epoch, metrics):
        metrics["epoch"] = epoch
        self.history.append(metrics)
        self._save_json()

    def _save_json(self):
        with open(os.path.join(self.run_dir, "trajectory.json"), "w") as f:
            json.dump(self.history, f, indent=4)

    def save_final(self):
        # Also save as npz for efficient loading in analysis
        keys = self.history[0].keys()
        data = {k: np.array([h[k] for h in self.history if k in h]) for k in keys}
        np.savez(os.path.join(self.run_dir, "trajectory.npz"), **data)
