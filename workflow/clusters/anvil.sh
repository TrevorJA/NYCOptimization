# workflow/clusters/anvil.sh — cluster profile: Anvil (Purdue RCAC).
#
# Sourced by workflow/cluster.sh; never run directly. Assignments use
# ${VAR=default}, so a value already set in the environment wins. The #SBATCH
# headers of every workflow script are written for this cluster, so nothing
# here redirects them (SBATCH_PARTITION and NYCOPT_SBATCH_ARGS stay unset).

_nycopt_profile_begin "anvil"

: "${NYCOPT_PARTITION_SERIAL=shared}"      # per-core billing, max 1 node
: "${NYCOPT_PARTITION_MPI=wholenode}"      # node-exclusive, up to 16 nodes
: "${NYCOPT_PARTITION_DEBUG=debug}"        # 2 nodes, 2 h
: "${NYCOPT_SLURM_ACCOUNT=ees260021}"
: "${NYCOPT_RANKS_PER_NODE=128}"           # measured wholenode packing
: "${NYCOPT_CPUS_PER_RANK=1}"              # one Slurm cpu is one physical core
: "${NYCOPT_NODE_MEMORY_GB=256}"
: "${NYCOPT_WALL_CAP_HOURS=96}"            # wholenode per-job maximum
: "${NYCOPT_REEVAL_UNIT_MINUTES=80}"       # rank-minutes per policy per 500-realization chunk, measured
: "${NYCOPT_PYTHON_MODULE=}"               # none: python comes from the conda env
: "${NYCOPT_CONDA_MODULE=anaconda/2024.02-py311}"
: "${NYCOPT_CONDA_ENV=venv}"
: "${NYCOPT_MPI_MCA_FLAGS=}"               # default fabric (InfiniBand/UCX)
