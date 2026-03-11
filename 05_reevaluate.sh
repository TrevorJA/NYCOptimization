#!/bin/bash
# Step 5: Re-simulate Pareto-optimal solutions with the full (untrimmed)
# Pywr-DRB model and export per-solution HDF5 files + objective summary.
#
# Usage:
#   bash 05_reevaluate.sh [FORMULATION] [MAX_SOLUTIONS]
#   bash 05_reevaluate.sh ffmp 50
#   sbatch 05_reevaluate.sh
#
#SBATCH --job-name=reevaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/reevaluate_%j.out
#SBATCH --error=logs/reevaluate_%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p logs
source venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

FORMULATION="${1:-ffmp}"
MAX_SOLUTIONS="${2:-0}"

python3 -c "
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, '.')

from config import OUTPUTS_DIR, get_n_vars, get_obj_names
from src.load.reference_set import load_reference_set
from src.simulation import dvs_to_config, run_simulation_to_disk
from src.objectives import DEFAULT_OBJECTIVES

formulation = '${FORMULATION}'
max_solutions = int('${MAX_SOLUTIONS}')
ref_file = OUTPUTS_DIR / 'reference_sets' / f'{formulation}.ref'
reeval_dir = OUTPUTS_DIR / 'reevaluation' / formulation
reeval_dir.mkdir(parents=True, exist_ok=True)

n_vars = get_n_vars(formulation)
dv_data, obj_data = load_reference_set(ref_file, n_vars)
n_solutions = min(dv_data.shape[0], max_solutions) if max_solutions > 0 else dv_data.shape[0]
obj_names = get_obj_names()
all_objectives = []

for i in range(n_solutions):
    print(f'--- Solution {i+1}/{n_solutions} ---')
    config = dvs_to_config(dv_data[i, :], formulation)
    data = run_simulation_to_disk(config, reeval_dir / f'solution_{i:04d}.hdf5')
    objs = DEFAULT_OBJECTIVES.compute(data)
    all_objectives.append(objs)
    for name, val in zip(obj_names, objs):
        print(f'  {name} = {val:.6f}')

summary_df = pd.DataFrame(all_objectives, columns=obj_names)
summary_csv = reeval_dir / 'objectives_summary.csv'
summary_df.to_csv(summary_csv, index_label='solution_id')
print(f'Re-evaluation complete ({n_solutions} solutions) -> {summary_csv}')
"
