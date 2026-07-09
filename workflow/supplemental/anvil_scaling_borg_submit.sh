#!/bin/bash
# anvil_scaling_borg_submit.sh — login-node helper that submits the Stage B
# strong-scaling batch: one anvil_scaling_borg.sh job per scale_* geometry in
# supplemental_config.BORG_SCALE_GEOMETRIES, each an --array over the seed
# replicates. NOT an sbatch script — run it directly from the repo root (the
# allocation account is set in anvil_scaling_borg.sh's #SBATCH header):
#
#   ./workflow/supplemental/anvil_scaling_borg_submit.sh
#
#   # smoke first (single tiny geometry, 1 seed):
#   NYCOPT_BORG_SUBMIT_ONLY=scale_smoke ./workflow/supplemental/anvil_scaling_borg_submit.sh
#
# For each geometry it (1) asserts the (ranks, time) row in
# supplemental_config.py still matches MOEAConfig.total_ntasks_mpi in
# src/moea_config.py — the two files must not drift — and (2) submits with
# the geometry-sized --ntasks/--time and NYCOPT_MOEA_CONFIG exported (the env
# file deliberately omits it).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

ENV_FILE="workflow/envs/anvil_scaling_borg.env"
ONLY="${NYCOPT_BORG_SUBMIT_ONLY:-}"

# GEOMETRY RANKS TIME SEEDS per line, from the single source of truth.
# Captured first so a failing python3 aborts (process substitution into
# mapfile would silently yield an empty array and submit nothing).
GEOMS_RAW="$(python3 -c "
import supplemental_config as scfg
for name, (ranks, time) in scfg.BORG_SCALE_GEOMETRIES.items():
    seeds = 1 if name == 'scale_smoke' else scfg.BORG_SCALE_SEEDS
    print(name, ranks, time, seeds)
")"
mapfile -t GEOMS <<< "${GEOMS_RAW}"
(( ${#GEOMS[@]} > 0 )) || { echo "ERROR: no geometries resolved" >&2; exit 2; }

for GEOM in "${GEOMS[@]}"; do
    read -r CFG RANKS TIME SEEDS <<< "${GEOM}"
    if [[ -n "${ONLY}" && "${CFG}" != "${ONLY}" ]]; then
        continue
    fi
    WANT_RANKS="${RANKS}" NYCOPT_MOEA_CONFIG="${CFG}" python3 -c "
import os
import config
mc = config.ACTIVE_MOEA_CONFIG
want = int(os.environ['WANT_RANKS'])
assert mc.total_ntasks_mpi == want, (
    f'geometry drift: supplemental_config.BORG_SCALE_GEOMETRIES[{mc.name!r}] '
    f'says {want} ranks but MOEAConfig.total_ntasks_mpi is '
    f'{mc.total_ntasks_mpi} — realign the two registries.')
print(f'geometry OK: {mc.name} = {mc.total_ntasks_mpi} ranks')
"
    # Memory: the smoke's largest process was ~1.92 GB (the master rank);
    # plain evaluator ranks measure ~0.8 GB (Stage A shards). Above shared's
    # MaxMemPerCPU=1896M, a 3G request is satisfied by charging 2 CPUs per
    # rank — fine to 64 ranks, but >64 ranks would exceed the 128-core node
    # and shared's QOS forbids multi-node. The big 64-slot arms therefore run
    # on the default 1896M/CPU: their aggregate cgroup budget (ranks x 1.9 GB)
    # comfortably covers one heavy master + light evaluator ranks.
    MEM_ARGS=(--mem-per-cpu=3G)
    (( RANKS > 64 )) && MEM_ARGS=()
    sbatch \
        --job-name="ansb_${CFG}" \
        --nodes=1 \
        --ntasks="${RANKS}" \
        --time="${TIME}" \
        "${MEM_ARGS[@]}" \
        --array="1-${SEEDS}" \
        --export=ALL,NYCOPT_ENV_FILE="${ENV_FILE}",NYCOPT_MOEA_CONFIG="${CFG}",DEBUG_SIM=true \
        workflow/supplemental/anvil_scaling_borg.sh
    echo "submitted ${CFG}: ${RANKS} ranks, ${TIME}, seeds 1-${SEEDS}"
done
