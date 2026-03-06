#!/bin/bash
# ===========================================================================
# 03_run_diagnostics.sh - Run MOEA runtime diagnostics (MOEAFramework v5.0).
#
# Executes the three-step diagnostic pipeline:
#   1. ResultFileMerger:     runtime files -> per-seed .set files
#   2. ResultFileSeedMerger: per-seed .set  -> cross-seed .ref file
#   3. MetricsEvaluator:     runtime + .ref -> .metrics files
#
# Usage:
#     bash 03_run_diagnostics.sh [formulation]
#
# Prerequisites:
#     - MOEAFramework v5.0 CLI installed (see DIAGNOSTICS_SETTINGS in config.py)
#     - Optimization runtime files from step 02
#
# Outputs:
#     outputs/optimization/{formulation}/sets/seed_XX_{formulation}.set
#     outputs/reference_sets/{formulation}.ref
#     outputs/diagnostics/{formulation}/*.metrics
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FORMULATION="${1:-ffmp}"

echo "============================================"
echo "  03: MOEA Diagnostics"
echo "  Formulation: ${FORMULATION}"
echo "============================================"

python3 - <<PYEOF
import sys
sys.path.insert(0, ".")
from src.diagnostics import run_full_diagnostics
run_full_diagnostics("${FORMULATION}")
PYEOF
