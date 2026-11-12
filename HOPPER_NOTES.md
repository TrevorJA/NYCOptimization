# Hopper notes

Running record of what is needed to run this project on Cornell's Hopper
cluster, with measured timings. The Anvil counterpart is `ANVIL_NOTES.md`.
Hopper has no allocation or billing, so this file tracks wall time only.

## Cluster facts

| Item | Value |
|---|---|
| Scheduler | Slurm 20.11, cluster name `hopper`, one partition `normal`, no wall-time limit |
| Accounts | not enforced: an Anvil `--account=ees260021` header is accepted and ignored |
| Partitions | `shared`, `wholenode` and `debug` do not exist; every Anvil header needs `--partition=normal` |
| Core accounting | `CR_CORE`: one task occupies one physical core (2 hardware threads), so `--cpus-per-task=8` takes 4 cores |
| Memory | not a consumable resource; `--mem` is accepted and not enforced |
| Xeon nodes | c0001–c0022: 2 × Xeon Gold 5218R, 40 cores, 191 GB |
| EPYC nodes | c0023–c0026: 2 × EPYC 9454, 96 cores, 571 GB |
| Python | `module load python/3.11.5`, then `source venv/bin/activate` (the venv needs the module's `libpython`) |
| MPI | OpenMPI 4.0.5 (`openmpi4/4.0.5`, default module) |
| Storage | repo and `outputs/` live on `/home` (shared ZFS pool); there is no separate scratch |

## Node health: most Xeon nodes are degraded (2026-11-12)

21 of the 22 Xeon nodes report a constant load of ~40 while idle, with ~50 %
user CPU and no owning process visible in `ps` or `top`. Work on those nodes
runs at about 40 % speed. c0013 was the only clean Xeon node; the EPYC nodes
are clean.

| Node | State | Pure-Python loop (s) | Historic evaluation, 1 rank (s) |
|---|---|---|---|
| c0025 (EPYC) | clean | 2.2 | 21 |
| c0013 (Xeon) | clean | 3.3 | not measured |
| c0001, c0014 (Xeon) | loaded | 8.7–9.8 | 109 |

Check before choosing nodes: `sinfo -N -o "%N %t %O"` (the third column is the
CPU load; an idle node should read near 0). Not yet reported to CAC.

## Measured timings

Historic single-trace evaluation (`src.simulation.evaluate`, trimmed model,
1945-12-01 to 2023-11-30, `ffmp_obj8`, Pywr-DRB `nyc_opt` at `9619a90`, repo at
`b6d3fb8`), one exclusive node, jobs 1484195 and 1484196 on 2026-11-12:

| Node | Ranks on the node | Seconds per evaluation | Evaluations per second per node |
|---|---|---|---|
| EPYC c0025 | 1 | 21 | |
| EPYC c0025 | 96 (one per core) | 24 | 4.0 |
| EPYC c0025 | 192 (hyperthreads) | 46 | 4.2 |
| Xeon c0001, loaded | 1 | 109 | |
| Xeon c0001, loaded | 40 (one per core) | 121 | 0.33 |
| Xeon c0001, loaded | 80 (hyperthreads) | 236 | 0.34 |

- Hyperthreads add about 4 % throughput: use one rank per physical core.
- Peak RSS is 0.44 GB per rank, so memory never binds for the historic search.
- Reference points: Anvil runs the same evaluation in 33.8 s at 128 ranks per
  node (2026-11-11); a healthy Xeon node ran it in 52.8 s at 33 ranks per node
  under the June code (job 322915).
- One clean EPYC node does the work of about 12 loaded Xeon nodes, or about 5
  healthy ones.

## Sizing rules

MM-Borg wall time = NFE × simulated fraction × seconds per evaluation ÷ workers,
plus about 9 % for the slowest island. The simulated fraction is the share of
NFE that passes the DV constraints and is actually simulated: 0.73 in the
2026-11-11 Anvil HIST run (current ε) and 0.52 in the 2026-11-06 run.

| 1.2M NFE HIST search on | Workers | Wall (simulated fraction 0.73) |
|---|---|---|
| 2 clean EPYC nodes | 184 | ~34 h |
| 10 Xeon nodes, loaded as on 2026-11-12 | 392 | ~82 h |
| 10 Xeon nodes, healthy (June speed, not re-measured) | 392 | ~36 h |
| Anvil, 12 × 128 (for reference) | 1,528 | ~6 h |

There is no resume, so a search must finish in one job; Hopper has no wall cap.

## Running the workflow here

Machine settings come from the cluster profile `workflow/clusters/hopper.sh`,
loaded by `workflow/cluster.sh` (`workflow/README.md`, "Cluster profiles").
Jobs detect the cluster themselves; the login shell needs one line so that
hand-typed `sbatch` lines stop pointing at Anvil partitions:

```bash
source workflow/cluster.sh                         # EPYC nodes (default)
NYCOPT_NODE_CLASS=xeon source workflow/cluster.sh  # Xeon nodes instead
```

This exports `SBATCH_PARTITION=normal` and wraps `sbatch` in a shell function
that prepends an `--exclude` of the other node class (Slurm 20.11 ignores
`SBATCH_EXCLUDE`), for the rest of the shell session and for every sbatch
typed in it; `command sbatch` bypasses the wrapper. Without it Slurm fills the
lowest-numbered nodes first, which are the Xeon nodes. Profile values: 96 ranks per node and 571 GB (EPYC) or 40 and 191 GB
(Xeon), TCP MPI transport for the re-evaluation steps, no wall cap.

- Submit helpers (`submit_etest_staging.sh`, `submit_reeval_chunks.sh`,
  `submit_search_staging.sh`, `submit_smoke.sh`) load the profile themselves.
- `submit_production_search.sh` refuses to run here: `production` is 1,533
  ranks. The Hopper historic search is `hopper_2x96` (189 ranks on two EPYC
  nodes); the sbatch line is in the header of
  `workflow/envs/ffmp_obj8_historic_hopper_2x96.env`.
- MPI steps whose header asks for 128 ranks per node need
  `--nodes/--ntasks-per-node` on the sbatch line.
- The generation shards are single-process but reserve 8 cpus (4 cores) each,
  so 24 of them fill one EPYC node.

Sibling repos must match Anvil: Pywr-DRB on `nyc_opt` (`9619a90`), scengen
`main` (`c61fef8`), SynHydro `main`.

## Log

- 2026-11-12: Pywr-DRB switched from `master` to `nyc_opt` (`9619a90`). Step-01
  presim regenerated at the current path `outputs/reference/presim/` (the old
  copy under `outputs/presim/` predates the path change). Timing jobs 1484195
  (Xeon c0001) and 1484196 (EPYC c0025) run with the temporary scripts
  `scripts/temporary/hopper_eval_timing.{sh,py}`; Xeon degradation found with
  `scripts/temporary/hopper_cpu_check.sh`.
- 2026-11-12: cluster profiles added (`workflow/cluster.sh`, `workflow/clusters/`);
  `NYCOPT_CLUSTER` removed from the env files; `hopper_2x96` MOEA config and
  `workflow/envs/ffmp_obj8_historic_hopper_2x96.env` registered (not run).
- 2026-11-12: E_test direct build (500-SOW Latin hypercube at the registry size,
  the default since this date; `TODO.md` rerun chain, item 0). Generation array
  1484206 (25 shards, 13:16 EST: 24 on c0025, 1 on c0026), then
  `submit_etest_staging.sh all --from merge --after 1484206`: merge 1484235,
  utilization 1484236, per-chunk inputs 1484237 (25 tasks x 33 ranks), realized
  utilization 1484238, hazard-image shards 1484239 and merge 1484240.
