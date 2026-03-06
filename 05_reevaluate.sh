#!/bin/bash
# ===========================================================================
# 05_reevaluate.sh - Re-evaluate Pareto-approximate solutions.
#
# Takes the reference set from step 03 and re-simulates each solution
# with full HDF5 output. This enables detailed post-hoc analysis,
# scenario discovery, and robustness assessment.
#
# Usage:
#     bash 05_reevaluate.sh [formulation] [--max-solutions N]
#
# Arguments:
#     formulation    : Formulation name (default: "ffmp")
#     --max-solutions: Limit number of solutions to re-evaluate (default: all)
#
# Outputs:
#     outputs/reevaluation/{formulation}/solution_XXXX.hdf5
#     outputs/reevaluation/{formulation}/objectives_summary.csv
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FORMULATION="${1:-ffmp}"
MAX_SOLUTIONS="${2:-0}"

echo "============================================"
echo "  05: Re-evaluation"
echo "  Formulation: ${FORMULATION}"
echo "============================================"

python3 - <<PYEOF
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, ".")

from config import OUTPUTS_DIR, get_n_vars, get_obj_names
from src.load.reference_set import load_reference_set
from src.simulation import dvs_to_config, run_simulation_to_disk
from src.objectives import DEFAULT_OBJECTIVES

formulation = "${FORMULATION}"
max_solutions = int("${MAX_SOLUTIONS}")

ref_file = OUTPUTS_DIR / "reference_sets" / f"{formulation}.ref"
reeval_dir = OUTPUTS_DIR / "reevaluation" / formulation
reeval_dir.mkdir(parents=True, exist_ok=True)

n_vars = get_n_vars(formulation)
dv_data, obj_data = load_reference_set(ref_file, n_vars)
n_solutions = dv_data.shape[0]

if max_solutions > 0:
    n_solutions = min(n_solutions, max_solutions)

print(f"Re-evaluating {n_solutions} solutions from {ref_file}")

obj_names = get_obj_names()
all_objectives = []

for i in range(n_solutions):
    print(f"\n--- Solution {i+1}/{n_solutions} ---")
    dv_vector = dv_data[i, :]
    config = dvs_to_config(dv_vector, formulation)

    output_file = reeval_dir / f"solution_{i:04d}.hdf5"
    data = run_simulation_to_disk(config, output_file)

    objs = DEFAULT_OBJECTIVES.compute(data)
    all_objectives.append(objs)
    for name, val in zip(obj_names, objs):
        print(f"  {name} = {val:.6f}")

# Save summary
summary_df = pd.DataFrame(all_objectives, columns=obj_names)
summary_csv = reeval_dir / "objectives_summary.csv"
summary_df.to_csv(summary_csv, index_label="solution_id")
print(f"\nObjectives summary: {summary_csv}")
print(f"\n--- Re-evaluation complete ({n_solutions} solutions) ---")
PYEOF
