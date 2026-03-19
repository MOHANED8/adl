import torch
from torch.utils.data import Dataset
import json
import os
import numpy as np
import torch.nn as nn

class MetaDataset(Dataset):
    """
    Custom Dataset for meta-learning, loading temporal sequences of metrics.
    
    Args:
        run_dirs (List[str]): List of directories containing training trajectories.
        seq_len (int): Length of the input sequence (temporal window).
        horizon (int): Number of steps ahead to predict gen_gap.
    """
    def __init__(self, run_dirs, seq_len=10, horizon=5, feature_keys=None):
        self.samples = []
        self.seq_len = seq_len
        self.horizon = horizon
        
        for run_dir in run_dirs:
            traj_path = os.path.join(run_dir, "trajectory.json")
            if not os.path.exists(traj_path): continue
            
            with open(traj_path, "r") as f:
                traj = json.load(f)
            
            # Select features
            if feature_keys is None:
                # Automate: grab all numeric keys except epoch/losses/acc
                exclude = {"epoch", "train_loss", "val_loss", "train_acc", "val_acc", "gen_gap"}
                feature_keys = [k for k in traj[0].keys() if isinstance(traj[0][k], (int, float)) and k not in exclude]
            
            data = np.array([[t[k] for k in feature_keys] for t in traj])
            targets = np.array([t["val_loss"] for t in traj])
            
            # Create sliding windows
            for i in range(len(data) - seq_len - horizon):
                x = data[i : i + seq_len]
                # Target: delta val_loss at horizon h
                y = targets[i + seq_len + horizon] - targets[i + seq_len]
                self.samples.append((torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

