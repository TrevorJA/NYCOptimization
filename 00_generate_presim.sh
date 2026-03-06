#!/bin/bash
# ===========================================================================
# 00_generate_presim.sh - One-time setup: generate pre-simulated releases.
#
# Runs a single full Pywr-DRB simulation (no trimming) and extracts releases
# from the independent STARFIT reservoirs. These are saved to:
#     outputs/presim/presimulated_releases_mgd.csv
#
# The trimmed model used during Borg optimization reads this CSV to replace
# the independent STARFIT reservoir nodes, reducing per-evaluation runtime
# from ~30s to ~5-10s.
#
# This step only needs to be run ONCE per inflow_type / date range.
# Re-run with --force if you change START_DATE, END_DATE, or INFLOW_TYPE.
#
# Usage:
#     bash 00_generate_presim.sh [--force]
#
# Estimated runtime: 3-10 minutes (full 1945-2022 simulation).
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  00: Generate Pre-Simulated Releases"
echo "============================================"

python3 scripts/generate_presim.py "$@"
