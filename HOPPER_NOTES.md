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
| Core accounting | `CR_CORE` with 2 hardware threads per core, and Slurm cpus are threads: `--cpus-per-task=8` takes 4 cores, and MPI ranks at 1 cpu each are packed two per core unless the job is `--exclusive`. Give each rank 2 cpus (`NYCOPT_CPUS_PER_RANK`) |
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

E_test staging (`etest_kn_50yr_n12500`, 12,500 realizations x 50 yr in 25
chunks; `submit_etest_staging.sh all` on the EPYC nodes, 2026-11-12, jobs
1484419 to 1484425), 59 min end to end:

| Step | Tasks | Wall per task |
|---|---|---|
| Generation shards (single process each) | 25 | 45-47 min, of which ~2,670 s is the daily disaggregation |
| Merge | 1 | 16 s |
| Utilization column | 1 | 6 s |
| Per-chunk pywrdrb inputs (33 MPI ranks each, default transport) | 25 | 3.6-5.8 min |
| Realized utilization and contract check | 1 | 5 s |
| Hazard-image shards | 25 | 1.3-1.4 min |
| Hazard-image merge | 1 | 3 s |

Staged size 117 GB under `outputs/synthetic_ensembles/`. Anvil's generation
shards took 92 min each for the same 500 realizations.

Baseline re-evaluation on that E_test (one policy, 500 (policy, chunk, block)
units of 25 realizations x 50 yr; `submit_reeval_chunks.sh baseline --nodes 3`,
job 1484562, 2026-11-12): 14 min 48 s wall including the in-job merge, no
failed unit, peak RSS 0.89 GB per rank. The job asked for 1 cpu per rank, so
its 288 ranks shared 144 physical cores, two per core: unit wall median 489 s
(min 162 s, max 539 s). At one rank per core (`NYCOPT_CPUS_PER_RANK=2`, now in
the profile) a unit takes 173 s on an EPYC node at full packing (next table).

Design re-evaluation unit times at full packing (`ffmp_obj8_merged.set`, 240
policies, on `etest_kn_50yr_n12500`; a unit is one (policy, chunk, block) of 25
realizations x 50 yr; one node each, one rank per core, through
`submit_reeval_chunks.sh historic --nodes 1 --no-merge`; jobs 1485081 to
1485083, 2026-11-16, no failed unit):

| Node | Ranks | First unit per rank (s) | Later units (s): mean, min to max | Units per node-hour | Rank-minutes per (policy, chunk) |
|---|---|---|---|---|---|
| EPYC c0023 | 96 | 398 | 173, 168 to 184 | 1,998 | 58 |
| Xeon c0013, clean | 40 | 475 | 325, 314 to 391 | 443 | 108 |
| Xeon c0002, loaded | 40 | 805 | 661, 626 to 804 | 218 | 220 |

- The first unit of a rank carries a one-time startup (imports, first model
  build, cold reads): about 225 s on EPYC, 150 s on Xeon. With all 544 ranks of
  the production job starting at once it was 535 s on EPYC.
- The same times hold in the 8-node production job (1485100, 544 ranks): EPYC
  174 s, clean Xeon 325 s. One EPYC node does the work of 4.5 clean or 9 loaded
  Xeon nodes.
- Profile values (`NYCOPT_REEVAL_UNIT_MINUTES`): 60 on EPYC holds (58
  measured); 130 on Xeon is 108 on a clean node and 220 on a loaded one, so the
  helper's wall request is about 1.7x too short on the loaded nodes.
- Peak RSS per rank steps 0.89 -> 1.32 -> 1.74 GB over the first four units and
  held at 1.74 through unit 6 (c0013) and through the 52 minutes of job
  1485095: 167 GB of 571 on an EPYC node at 96 ranks, 70 GB of 191 on a Xeon
  node at 40. The loaded Xeon nodes' phantom CPU load uses about 13 GB.
- Each rank keeps one model JSON in `/dev/shm/pywrdrb_opt_r*` (overwritten per
  evaluation, about 0.25 MB; the directories are left behind after the job).
- Step-10 stdout is about 45 KB per unit (Pywr-DRB's "fully constrained" line
  per timestep): about 5.5 GB of `.out` for 120,000 units.

**HDF5 file locking must be off on Hopper** (`HDF5_USE_FILE_LOCKING=FALSE` in
the job environment). `/home` is NFSv3, HDF5 takes a shared lock at every
open, and the Linux NFS client drops its cached pages of a file whenever a lock
on it is granted, so with locking on every unit re-reads its inputs (about
240 MB) from the file server. The single-node tests above ran with locking on
and stayed under the server's ceiling; the first 8-node job did not:

| Job | Locking | EPYC unit (s) | Clean Xeon unit (s) | Loaded Xeon unit (s) | State on c0023 |
|---|---|---|---|---|---|
| 1485095 (544 ranks, cancelled after 52 min) | on | 574 | 394 | 860 | 93 of 96 ranks in uninterruptible I/O wait, CPUs 80 % idle, page cache 0.8 GB, 40 MB/s of NFS reads |
| 1485100 (same nodes) | off | 174 | 325 | not yet measured | 96 of 96 ranks running, page cache 14 GB and growing, no NFS read in 10 s |

Job 1485095 ran at about 0.8 units per second in total, the same as the three
single-node tests together (0.74), which is the file server's ceiling for this
read pattern (about 200 MB/s across the nodes); every other user of `/home`
shares it. The staged inputs are read-only during a run, so turning locking off
is safe; Pywr-DRB sets the same variable in `pywrdrb/pre/_mpi_utils.py`. No
workflow script sets it yet: it was passed through `sbatch --export`. The
MM-Borg ensemble searches read the same kind of files and have not been
checked for this on Hopper.

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

E_test re-evaluation wall time = units ÷ (sum of the nodes' units per
node-hour, table above), with 500 units per policy on `etest_kn_50yr_n12500`
and file locking off. For the 240-policy HIST set (120,000 units):

| Nodes | Units per hour | Wall |
|---|---|---|
| 4 EPYC | 7,990 | 15.0 h |
| 4 EPYC + c0013 + 3 loaded Xeon (job 1485100) | 9,090 | 13.2 h |
| 8 loaded Xeon | 1,740 | 69 h |

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
- Login-node scripts that call `nycopt_setup_env` under `set -u`
  (`00_setup_borg_jars.sh`) exit silently: `/etc/profile.d/lmod.sh` reads the
  unset `SLURM_NODELIST`. Run them as
  `SLURM_NODELIST= PBS_NODEFILE= bash workflow/00_setup_borg_jars.sh ffmp`.
  Jobs are unaffected. The problem JAR here was stale (24 variables x 7
  objectives) until rebuilt on 2026-11-16; step 09 needs the current counts.
- One step-10 job across both node classes: `submit_reeval_chunks.sh` targets
  one class, and two step-10 jobs on the same set duplicate units (claims are
  per job), so a mixed run is one hand-typed job that takes every cpu of each
  node, with `--ntasks` equal to the sum of the cores and no
  `--ntasks-per-node`. The step-10 header's `--ntasks-per-node=16` cannot be
  overridden by `--ntasks`, so that header line is dropped at submission:

  ```bash
  sed '/^#SBATCH --ntasks-per-node/d' workflow/10_simulate_test_chunks.sh | \
    command sbatch --job-name=reeval_hist --partition=normal --nodes=8 --ntasks=544 \
      --cpus-per-task=2 --nodelist=c0002,c0011,c0013,c0014,c[0023-0026] --time=24:00:00 \
      --export=ALL,HDF5_USE_FILE_LOCKING=FALSE,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env,DRAW=0,NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n12500,NYCOPT_CHUNK_POLICIES=outputs/historic/ffmp_obj8/sets/ffmp_obj8_merged.set,NYCOPT_SEARCH_REALIZATION_BATCH=25,NYCOPT_CHUNK_UNIT_SECONDS=720,NYCOPT_CHUNK_MERGE=off
  bash workflow/submit_reeval_chunks.sh historic --merge-only --dependency=afterok:<jobid>
  ```

  Slurm places 96 ranks on each EPYC node and 40 on each Xeon node, OpenMPI
  binds each rank to a socket, and MPI connects across the classes (probe job
  1485094). `NYCOPT_CHUNK_UNIT_SECONDS` is sized for the slowest node class.
- Memory is not enforced by Slurm, so a long step-10 job runs with a watchdog:
  `sbatch scripts/temporary/reeval_memwatch.sh <jobid> 3.0 <merge jobid>` reads
  the peak RSS of every rank from the job's `[unit]` lines every 2 minutes and
  cancels the listed jobs above 3.0 GB per rank (63 % of a Xeon node at 40
  ranks, 50 % of an EPYC node at 96). It logs to
  `logs/reeval_memwatch_<jobid>.out`. `ssh <node>` works while a job of yours
  runs there (`ps`, `/proc/meminfo`, `/proc/self/mountstats` for the NFS rates).

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
- 2026-11-12: the forcing box was restricted to the mid-century CMIP6 projections
  (`config.ENSEMBLE_FORCING_PERIOD`) after that chain was submitted, so its
  output samples another box and is not the campaign E_test. Not yet cancelled
  or resubmitted from the updated code (`TODO.md` rerun chain, item 0).
- 2026-11-12: that chain (1484206, 1484235 to 1484240) was cancelled at ~13:40 EST
  before any chunk was written, and its empty staging directory and shard logs
  were deleted. E_test resubmitted at 15:15 EST from the mid-century forcing box
  (repo `a3aede7`, scengen `b882693`) with `submit_etest_staging.sh all`: shards
  1484419 (14 on c0023, 11 on c0024), merge 1484420, utilization 1484421,
  per-chunk inputs 1484422, realized utilization 1484423, hazard-image shards
  1484424 and merge 1484425.
- 2026-11-12: E_test chain completed at 16:14 EST, every job COMPLETED (timings
  above). `etest_kn_50yr_n12500` passes the staged contract (LHS, R = 25,
  `forcing_period` 2020_2059, utilization column; 500 of 500 SOWs realized).
- 2026-11-12: re-evaluation work unit changed to (policy, chunk, block)
  (`src/chunk_reeval.py`); baseline cube built by job 1484562 on c0023 to c0025
  (timings above) at `outputs/baseline/reeval/etest_kn_50yr_n12500/` (500 SOWs,
  8 objectives, no NaN, 25 survivors per SOW). The venv has no `pyarrow`, so
  unit files and the cube are `csv.gz`.
- 2026-11-16: HIST post-processing on Hopper from the Anvil sets and runtime
  files (`outputs/historic/ffmp_obj8/`), all on c0023, under a minute each:
  problem JAR rebuilt (step 00, 36 x 8); step 08 `--seed 2` job 1485059 (734
  island rows at 125,000 NFE per island -> 239); step 08 `--merge --install`
  job 1485060 (462 -> 240, `ffmp_obj8_merged.set`); step 09 job 1485061 (both
  seeds, controller boxes reproduced, two-seed diagnostics reference set of 198
  rows, figures `outputs/figures/historic/ffmp_obj8/search_0{1,2,3}_*.png`);
  step 06 historic baseline job 1485062 (`outputs/baseline/historic/`; replaced
  the June `ffmp_baseline.hdf5`); explore suite jobs 1485063 (seed-2 final set,
  `explore_seed02/`), 1485064 (merged set, `explore/`) and 1485065 (seed-1
  final set, `explore_seed01/`: one policy dominates the baseline, row 53, as
  on Anvil). Coordinator HV at 500k total NFE is 92 % (seed 1, of its 750k
  final) and 88 % (seed 2, of its 1.25M final); seed 2 gains under 1 % per 100k
  NFE after 750k except +1.9 % from 1.0M to 1.1M.
- 2026-11-16: re-evaluation timing tests, one node each (jobs 1485081 to
  1485083, table above), written to `reeval/etest_kn_50yr_n12500/seed_97..99`
  through `SEED=` and deleted afterwards.
- 2026-11-16: HIST E_test re-evaluation (240 policies, 120,000 units) on 4 EPYC
  + c0013 + c0002, c0011, c0014 (544 ranks). Job 1485095 (21:42 EST) ran at a
  third of the tested speed on EPYC because of HDF5 file locking on NFS (section
  above) and was cancelled at 22:35 with its merge 1485096 and watchdog
  1485097; its 2,398 units were kept. Resubmitted with
  `HDF5_USE_FILE_LOCKING=FALSE` as job 1485100 (22:36 EST, 24 h limit, resumes
  from the kept units), merge 1485101 (`afterok`), watchdog 1485102: EPYC units
  back to 174 s; expected to finish about 11:40 EST on 2026-11-17.
- 2026-11-17: constant-800 MGD NYC demand sensitivity re-evaluation added
  (`scripts/supplemental/constant_demand_reeval.py`,
  `workflow/supplemental/results_diagnostics/constant_demand_reeval.sh`): every
  row of the 7 HIST `.set` files (1,526 rows, 1,046 unique policies + baseline)
  on the historic trace. Smoke jobs 1485136 (control, `DEMAND=historical`) and
  1485137 (constant) on loaded Xeon c0003/c0004, 8 ranks: 107-111 s per
  evaluation, 2.3 min wall. The control reproduces the set scores to 1e-8 on the
  reliability objectives and within 0.19 (p99 deficit), 0.16 (storage p01) and
  0.075 (Trenton p99) on the continuous ones: LP jitter between runs, not a code
  difference. Full run job 1485138 (4 EPYC x 96) queued behind 1485100.
