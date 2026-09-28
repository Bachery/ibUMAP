#!/bin/bash
# umap-learn, cuML and ibUMAP variants; run in the ibumap-cuda environment.
set -euo pipefail
cd "$(dirname "$0")"

algorithm_args=(
  --algorithm umap_learn_seed_42
  --algorithm umap_learn_seed_none
  --algorithm cuml_umap_seed_42
  --algorithm cuml_umap_seed_none
  --algorithm ibumap_cpu_seed_42
  --algorithm ibumap_cpu_seed_none
  --algorithm ibumap_cuda_seed_42
  --algorithm ibumap_cuda_seed_none
)

python -u scripts/00_check_environment.py "${algorithm_args[@]}" --probe --strict
python -u scripts/01_run_e2e_benchmark.py "${algorithm_args[@]}" --run
