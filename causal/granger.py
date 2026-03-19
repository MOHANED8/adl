import numpy as np
import json
import os
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd

def compute_granger_for_run(run_id: str, target: str = "val_loss", max_lag: int = 1):
    """
    Computes Granger causality between internal signals and a target metric.
    
    Args:
        run_id (str): ID of the run in the 'results' directory.
        target (str): The dependent variable for the causality test.
        max_lag (int): Maximum lag to consider (default: 1 for short trajectories).
    """
    run_dir = os.path.join("results", run_id)
    path = os.path.join(run_dir, "trajectory.json")
    with open(path, "r") as f:
        traj = json.load(f)
    
    df = pd.DataFrame(traj)
    
    results = {}
    exclude = {"epoch", "train_loss", "val_loss", "train_acc", "val_acc", "gen_gap"}
    features = [k for k in df.columns if isinstance(df[k][0], (int, float)) and k not in exclude]
    
    for feat in features:
        # Stationary check (simplified): diff the series
        # We test if 'feat' Granger-causes 'target'
        data = df[[target, feat]].diff().dropna()
        if len(data) <= max_lag + 1: continue
        
        try:
            test_result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            # SSR F-test p-values
            p_values = [test_result[i][0]['ssr_ftest'][1] for i in range(1, max_lag + 1)]
            results[feat] = min(p_values)
        except Exception:
            # For smoke tests with short trajectories, statistical tests might be infeasible
            results[feat] = 1.0
        
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="smoke_run_0")
    args = parser.parse_args()
    res = compute_granger_for_run(args.run_id)
    print("Granger Causality Results (p-values, lower is more causal):")
    for k, v in sorted(res.items(), key=lambda x: x[1]):
        if v < 0.05:
            print(f"  {k}: {v:.6f} ***")
        else:
            print(f"  {k}: {v:.6f}")
