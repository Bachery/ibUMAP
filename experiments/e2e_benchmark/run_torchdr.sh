#!/bin/bash
# TorchDR variants; run in the ibumap-torchdr environment.
set -euo pipefail
cd "$(dirname "$0")"

algorithm_args=(
  --algorithm torchdr_umap_seed_42
  --algorithm torchdr_umap_seed_none
)

python -u scripts/00_check_environment.py "${algorithm_args[@]}" --probe --strict
python -u scripts/01_run_e2e_benchmark.py "${algorithm_args[@]}" --run
