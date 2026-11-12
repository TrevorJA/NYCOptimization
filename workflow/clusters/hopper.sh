# workflow/clusters/hopper.sh — cluster profile: Hopper (Cornell CAC).
#
# Sourced by workflow/cluster.sh; never run directly. Assignments use
# ${VAR=default}, so a value already set in the environment wins. Facts and
# measurements behind these values: HOPPER_NOTES.md.
#
# Hopper has one partition and two node classes whose core counts differ, so a
# submission targets ONE class:
#   epyc  c0023-c0026, 96 cores, 571 GB   (default on the login node)
#   xeon  c0001-c0022, 40 cores, 191 GB
# NYCOPT_NODE_CLASS picks it at submission; inside a job the class is that of
# the node the job landed on. The #SBATCH headers name Anvil partitions:
# SBATCH_PARTITION (an sbatch input variable, which overrides headers)
# redirects them, and NYCOPT_SBATCH_ARGS, which the sbatch wrapper of
# workflow/cluster.sh prepends to every submission, keeps jobs off the other
# node class (Slurm 20.11 has no SBATCH_EXCLUDE input variable).

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    case "$(hostname -s)" in
        c00[01][0-9]|c002[0-2]) _nycopt_class=xeon ;;
        *)                      _nycopt_class=epyc ;;
    esac
else
    _nycopt_class="${NYCOPT_NODE_CLASS:-epyc}"
fi
_nycopt_profile_begin "hopper:${_nycopt_class}"

case "${_nycopt_class}" in
    epyc)
        : "${NYCOPT_RANKS_PER_NODE=96}"
        : "${NYCOPT_NODE_MEMORY_GB=571}"
        : "${NYCOPT_REEVAL_UNIT_MINUTES=60}"   # estimate from the eval timing; not yet measured
        : "${NYCOPT_SBATCH_ARGS=--exclude=c[0001-0022]}"
        ;;
    xeon)
        : "${NYCOPT_RANKS_PER_NODE=40}"
        : "${NYCOPT_NODE_MEMORY_GB=191}"
        : "${NYCOPT_REEVAL_UNIT_MINUTES=130}"  # estimate (healthy node); not yet measured
        : "${NYCOPT_SBATCH_ARGS=--exclude=c[0023-0026]}"
        ;;
    *)
        echo "[cluster] ERROR: NYCOPT_NODE_CLASS='${_nycopt_class}' (expected epyc or xeon)" >&2
        unset _nycopt_class
        return 2
        ;;
esac
unset _nycopt_class

: "${NYCOPT_PARTITION_SERIAL=normal}"
: "${NYCOPT_PARTITION_MPI=normal}"
: "${NYCOPT_PARTITION_DEBUG=normal}"
: "${NYCOPT_SLURM_ACCOUNT=}"               # accounts are not enforced
: "${NYCOPT_WALL_CAP_HOURS=0}"             # 0 = no wall-time limit
: "${NYCOPT_PYTHON_MODULE=python/3.11.5}"  # ./venv needs this module's libpython
: "${NYCOPT_CONDA_MODULE=}"
: "${NYCOPT_CONDA_ENV=}"
# Hopper's default fabric providers hang on large mpi4py gathers; force TCP.
: "${NYCOPT_MPI_MCA_FLAGS=--mca pml ob1 --mca btl self,vader,tcp}"
: "${SBATCH_PARTITION=normal}"
