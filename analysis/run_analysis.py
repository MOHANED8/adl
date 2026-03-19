import os
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from analysis.metrics import center_kernel_alignment

def run_analysis(run_id, epochs_to_compare=[0, 2, 4, 6, 8]):
    run_dir = os.path.join("results", run_id)
    snapshot_dir = os.path.join(run_dir, "snapshots")
    
    # Load trajectory for correlation
    with open(os.path.join(run_dir, "trajectory.json"), "r") as f:
        traj = json.load(f)
    
    val_losses = [t["val_loss"] for t in traj]
    
    # Load snapshots
    snapshots = {}
    for epoch in epochs_to_compare:
        # Note: allow_pickle=False for security
        snapshots[epoch] = np.load(os.path.join(snapshot_dir, f"epoch_{epoch}.npz"), allow_pickle=False)
    
    # Pick a sample layer (e.g., the last one)
    layer_names = list(snapshots[0].keys())
    target_layer = layer_names[-1]
    
    # Compute similarity matrix across epochs
    n = len(epochs_to_compare)
    sim_matrix = np.zeros((n, n))
    for i, e1 in enumerate(epochs_to_compare):
        for j, e2 in enumerate(epochs_to_compare):
            x = snapshots[e1][target_layer].reshape(snapshots[e1][target_layer].shape[0], -1)
            y = snapshots[e2][target_layer].reshape(snapshots[e2][target_layer].shape[0], -1)
            sim_matrix[i, j] = center_kernel_alignment(x, y)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, annot=True, xticklabels=epochs_to_compare, yticklabels=epochs_to_compare)
    plt.title(f"CKA Similarity Matrix - Layer: {target_layer}")
    plt.savefig(os.path.join(run_dir, "cka_heatmap.png"))
    plt.close()
    
    print(f"Analysis complete for {run_id}. Heatmap saved.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="smoke_run_0")
    args = parser.parse_args()
    run_analysis(args.run_id)
