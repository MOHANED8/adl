import subprocess
import os
import sys

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {cmd}:")
        print(result.stderr)
        return False
    return True

def test_all():
    steps = [
        ("Induction (Smoke)", "python run_experiments.py --smoke"),
        ("Analysis (CKA)", "python -m analysis.run_analysis --run_id smoke_run_0"),
        ("Causal (Granger)", "python -m causal.granger --run_id smoke_run_0"),
        ("Meta-Training", "python -m meta.train_meta"),
        ("Diagnostic Deploy", "python -m diagnostic.run_diagnostic --action early_stop")
    ]
    
    overall_success = True
    for name, cmd in steps:
        print(f"--- Testing {name} ---")
        if run_cmd(cmd):
            print(f"SUCCESS: {name}")
        else:
            print(f"FAILED: {name}")
            overall_success = False
    
    if overall_success:
        print("\nALL PROJECT TESTS PASSED SUCCESSFULLY!")
    else:
        print("\nSOME PROJECT TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    test_all()
