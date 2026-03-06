#!/bin/bash
# ===========================================================================
# 01_run_baseline.sh - Evaluate the default FFMP policy (no optimization).
#
# Runs a single Pywr-DRB simulation with baseline decision variable values
# and saves full HDF5 output for analysis. This provides the "status quo"
# reference point against which optimized solutions are compared.
#
# Usage:
#     bash 01_run_baseline.sh [--formulation ffmp] [--test-inmemory]
#
# Outputs:
#     outputs/baseline/{formulation}_baseline.hdf5
#     outputs/baseline/{formulation}_baseline_objectives.csv
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  01: Baseline Evaluation"
echo "============================================"

python3 scripts/run_baseline.py "$@"
