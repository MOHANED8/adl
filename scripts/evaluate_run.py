from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dtd.experiments import evaluate_run


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/cifar10_resnet18/events.jsonl").parent
    res = evaluate_run(run_dir)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

