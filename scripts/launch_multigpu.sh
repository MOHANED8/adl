#!/usr/bin/env bash
set -euo pipefail

python main.py training.runtime.multi_gpu=true training.runtime.device=cuda "$@"
