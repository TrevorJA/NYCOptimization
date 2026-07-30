#!/bin/bash
# Node-class test for the shard-vs-regeneration mismatch: run the partition
# diagnostic pinned to a specific node (sbatch --nodelist=...) and print the
# node's system-library versions. If a shard-class node reproduces the staged
# image exactly while a merge-class node does not, the cause is heterogeneous
# node system software (libm-level FP), not the generation code.
#
#SBATCH --job-name=nestedp_node
#SBATCH --account=ees260021
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/nestedp_node_%j.out
#SBATCH --error=logs/nestedp_node_%j.err
set -uo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/workflow/_common.sh"
nycopt_setup_env

echo "[node] host=$(hostname) kernel=$(uname -r)"
echo "[node] glibc: $(rpm -q glibc 2>/dev/null | head -1 || ldd --version | head -1)"
echo "[node] libm resolves to: $(ldd "$(command -v python3)" 2>/dev/null | grep libm || true)"
python3 -c "import numpy, scipy; print('[node] numpy', numpy.__version__, 'scipy', scipy.__version__)"

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python3 -u scripts/supplemental/diagnose_partition_mismatch.py || echo "[node] diag rc=$?"
