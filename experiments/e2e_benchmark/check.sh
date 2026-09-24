#!/bin/bash
# Configuration, data availability, task grid and resource-skip inspection.
# Dry run: writes nothing and needs neither CUDA nor TorchDR.
set -euo pipefail
cd "$(dirname "$0")"
python -u scripts/01_run_e2e_benchmark.py
