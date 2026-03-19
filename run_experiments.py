import argparse
from config import SMOKE_CONFIG, ExperimentConfig, DatasetName, ModelType, NoiseType
from train import train

def main():
    parser = argparse.ArgumentParser(description="Overfitting Induction Runner")
    parser.add_argument("--smoke", action="store_true", help="Run a fast smoke test")
    args = parser.parse_args()

    if args.smoke:
        print("Starting Smoke Test...")
        train(SMOKE_CONFIG, "smoke_run_0")
    else:
        # Here we would define the full run matrix as planned
        print("Full run matrix implementation would go here.")

if __name__ == "__main__":
    main()
