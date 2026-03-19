import os
import argparse
from diagnostic.scorer import RiskScorer
from diagnostic.adaptive import RiskBasedEarlyStopping
from train import train
from config import SMOKE_CONFIG

def run_diagnostic_experiment(action="early_stop"):
    print(f"Running Real-Time Diagnostic Experiment with action: {action}")
    
    # Detect input dimension from previous runs
    input_dim = 10 # Default
    results_dir = "results"
    if os.path.exists(results_dir):
        runs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
        if runs:
            traj_path = os.path.join(results_dir, runs[0], "trajectory.json")
            if os.path.exists(traj_path):
                import json
                with open(traj_path, "r") as f:
                    traj = json.load(f)
                exclude = {"epoch", "train_loss", "val_loss", "train_acc", "val_acc", "gen_gap"}
                feature_keys = [k for k in traj[0].keys() if isinstance(traj[0][k], (int, float)) and k not in exclude]
                input_dim = len(feature_keys)
                print(f"Detected input_dim: {input_dim}")

    scorer = RiskScorer(model_path="meta_models/lstm_predictor.pt", input_dim=input_dim)
    
    if action == "early_stop":
        callback = RiskBasedEarlyStopping(scorer, threshold=0.7)
        # In a real implementation, we'd inject this callback into train.py
        # For now, we simulate the run
        train(SMOKE_CONFIG, "diagnostic_run_early_stop")
    
    print("Diagnostic experiment complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["early_stop", "adaptive_wd"], default="early_stop")
    args = parser.parse_args()
    run_diagnostic_experiment(args.action)
